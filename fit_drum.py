"""Fit a bank of parallel ZDF bandpass filters (plus a short noise burst) to a drum sample.

Each mode of the drum is one ZDF / TPT state-variable bandpass filter with three learned
settings: centre frequency, decay time (T60) and level. A short burst of filtered noise
models the hand or stick contact. Everything is fixed per hit, so the filters are computed
exactly in the frequency domain (identical to running the ZDF filters sample by sample on an
impulse, just much faster) and fitted to the sample by gradient descent.

Two ways to start the fit:
  --init peaks   modes start at the strongest peaks in the sample's spectrum (best match).
  --init bessel  modes start at the ideal-membrane Bessel ratios times a fundamental f01,
                 and may only drift a little from them (shows how "ideal" the drum is).

Usage (on woodforde, in ~/gpu-tests/tone-chase-mue with env gpu active):
    python fit_drum.py samples/FPC_Conga_001a.wav
    python fit_drum.py samples/FPC_Conga_001a.wav --init bessel
    python fit_drum.py samples/*.wav                  # fit several in one go
    python fit_drum.py --check                         # prove the fast filter maths matches a real ZDF loop

Results go to out_fit/<sample name>_<init>/ : original.wav, recreation.wav, modes_only.wav,
params.json and plot.png.
"""
import argparse
import json
import math
import os
import time

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, find_peaks, sosfiltfilt
from scipy.special import jn_zeros

J01 = jn_zeros(0, 1)[0]


# --------------------------------------------------------------------------------------
# Audio helpers
# --------------------------------------------------------------------------------------
def load_sample(path, hp_hz, max_s):
    sr, x = wavfile.read(path)
    if x.dtype.kind == "i":
        x = x / float(np.iinfo(x.dtype).max)
    elif x.dtype.kind == "u":
        x = (x.astype(np.float64) - 128) / 128.0
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if hp_hz > 0:  # removes sub-bass wobble that is not a membrane mode
        x = sosfiltfilt(butter(2, hp_hz, "highpass", fs=sr, output="sos"), x)
    peak = np.abs(x).max() + 1e-12
    onset = int(np.argmax(np.abs(x) > 0.05 * peak))
    x = x[onset: onset + int(max_s * sr)]
    return sr, x / (np.abs(x).max() + 1e-12)


def write_wav(path, sr, y):
    wavfile.write(path, sr, (np.clip(y, -1, 1) * 32767).astype(np.int16))


def spectrum_db(x, sr, nfft):
    w = np.ones(len(x))
    taper = min(len(x) // 4, int(0.05 * sr))
    if taper > 1:
        w[-taper:] = np.hanning(2 * taper)[taper:]
    S = np.abs(np.fft.rfft(x * w, nfft))
    f = np.fft.rfftfreq(nfft, 1 / sr)
    return f, S


def decay_t60_guess(x, sr):
    """Rough T60 of the whole hit from the time it takes to fall 40 dB."""
    win = max(1, int(0.005 * sr))
    env = np.sqrt(np.convolve(x ** 2, np.ones(win) / win, "same"))
    db = 20 * np.log10(env / env.max() + 1e-12)
    pk = int(np.argmax(env))
    below = np.where(db[pk:] < -40)[0]
    t40 = below[0] / sr if len(below) else len(x) / sr
    return float(np.clip(t40 * 1.5, 0.05, 5.0))


def bessel_table(count, m_max=10, n_max=8):
    rows = [(z / J01, m, n) for m in range(m_max + 1)
            for n, z in enumerate(jn_zeros(m, n_max), start=1)]
    rows.sort()
    return rows[:count]


# --------------------------------------------------------------------------------------
# Reference ZDF filter, run sample by sample (used only by --check)
# --------------------------------------------------------------------------------------
def zdf_bandpass_impulse(f_hz, t60, sr, n):
    """Impulse response of a TPT state-variable bandpass (Zavalishin), scaled to unit start level."""
    g = math.tan(math.pi * f_hz / sr)
    R = 6.9078 / (t60 * 2 * math.pi * f_hz)
    s1 = s2 = 0.0
    out = np.zeros(n)
    for i in range(n):
        x = 1.0 if i == 0 else 0.0
        hp = (x - (2 * R + g) * s1 - s2) / (1 + 2 * R * g + g * g)
        v1 = g * hp
        bp = v1 + s1
        s1 = bp + v1
        v2 = g * bp
        lp = v2 + s2
        s2 = lp + v2
        out[i] = bp
    return out / (2 * g)


def zdf_bandpass_freqdomain(f_hz, t60, sr, n, nfft):
    """The same filter evaluated from its transfer function, as the fitting code does."""
    g = math.tan(math.pi * f_hz / sr)
    R = 6.9078 / (t60 * 2 * math.pi * f_hz)
    w = np.linspace(0, np.pi, nfft // 2 + 1)
    z1 = np.exp(-1j * w)
    z2 = z1 * z1
    den = (1 + 2 * R * g + g * g) + (2 * g * g - 2) * z1 + (1 - 2 * R * g + g * g) * z2
    H = g * (1 - z2) / den / (2 * g)
    return np.fft.irfft(H, nfft)[:n]


def run_check():
    sr, n, nfft = 44100, 8000, 1 << 17
    worst = 0.0
    for f, t60 in [(110, 1.0), (277, 0.4), (715, 0.15), (3000, 0.05)]:
        a = zdf_bandpass_impulse(f, t60, sr, n)
        b = zdf_bandpass_freqdomain(f, t60, sr, n, nfft)
        err = np.abs(a - b).max() / np.abs(a).max()
        worst = max(worst, err)
        print(f"  {f:>5} Hz, T60 {t60:>4} s: max difference {err:.2e}")
    print("PASS: frequency-domain maths matches the sample-by-sample ZDF filter"
          if worst < 1e-6 else "FAIL: mismatch")


# --------------------------------------------------------------------------------------
# The trainable drum
# --------------------------------------------------------------------------------------
def build_model(torch, sr, n, freqs, amps, t60s, init, ratios, f01, device, n_bands=16):
    nn = torch.nn
    dt = torch.float64

    class ModalDrum(nn.Module):
        def __init__(self):
            super().__init__()
            self.sr, self.n = sr, n
            self.nfft = 1 << int(math.ceil(math.log2(2 * n)))
            self.init = init
            K = len(t60s)
            if init == "bessel":
                self.log_f01 = nn.Parameter(torch.tensor(math.log(f01), dtype=dt))
                self.dev = nn.Parameter(torch.zeros(K, dtype=dt))
                self.register_buffer("ratios", torch.tensor(ratios, dtype=dt))
            else:
                self.log_f = nn.Parameter(torch.tensor(np.log(freqs), dtype=dt))
            self.log_t60 = nn.Parameter(torch.tensor(np.log(t60s), dtype=dt))
            self.amp = nn.Parameter(torch.tensor(amps, dtype=dt))
            # noise burst: fixed white noise, learned decay and learned spectral shape
            gen = torch.Generator().manual_seed(0)
            self.register_buffer("noise", torch.randn(n, generator=gen, dtype=dt))
            self.noise_log_decay = nn.Parameter(torch.tensor(math.log(1 / 0.02), dtype=dt))
            self.noise_bands = nn.Parameter(torch.full((n_bands,), -4.0, dtype=dt))
            bins = np.fft.rfftfreq(self.nfft, 1 / sr)
            centers = np.geomspace(50, sr / 2, n_bands)
            pos = np.interp(np.log(np.maximum(bins, 1)), np.log(centers), np.arange(n_bands))
            lo = np.floor(pos).astype(int)
            hi = np.minimum(lo + 1, n_bands - 1)
            frac = pos - lo
            W = np.zeros((len(bins), n_bands))
            W[np.arange(len(bins)), lo] += 1 - frac
            W[np.arange(len(bins)), hi] += frac
            self.register_buffer("band_w", torch.tensor(W, dtype=dt))
            w = torch.linspace(0, math.pi, self.nfft // 2 + 1, dtype=dt)
            self.register_buffer("z1", torch.exp(-1j * w))
            self.register_buffer("t", torch.arange(n, dtype=dt) / sr)

        def freqs(self):
            if self.init == "bessel":
                f = torch.exp(self.log_f01) * self.ratios * torch.exp(self.dev)
            else:
                f = torch.exp(self.log_f)
            return f.clamp(20.0, 0.45 * self.sr)

        def t60(self):
            # cap the decay so the tail has died away (-120 dB) before the FFT wraps around
            return torch.exp(self.log_t60).clamp(0.005, self.nfft / self.sr / 2)

        def modes_only(self):
            f = self.freqs()[:, None]
            g = torch.tan(math.pi * f / self.sr)
            R = (6.9078 / (self.t60()[:, None] * 2 * math.pi * f)).clamp(1e-5, 2.0)
            z1 = self.z1[None, :]
            z2 = z1 * z1
            den = (1 + 2 * R * g + g * g) + (2 * g * g - 2) * z1 + (1 - 2 * R * g + g * g) * z2
            H = (1 - z2) / (2 * den)          # ZDF bandpass, scaled so each mode starts at level 1
            return torch.fft.irfft((self.amp[:, None] * H).sum(0), self.nfft)[: self.n]

        def noise_only(self):
            env = torch.exp(-torch.exp(self.noise_log_decay) * self.t)
            NZ = torch.fft.rfft(self.noise * env, self.nfft)
            shape = torch.exp(self.band_w @ self.noise_bands)
            return torch.fft.irfft(NZ * shape, self.nfft)[: self.n]

        def forward(self):
            return self.modes_only() + self.noise_only()

    return ModalDrum().to(device)


def mrstft_loss(torch, y, x, ffts=(2048, 1024, 512, 256, 128)):
    loss = 0.0
    for nf in ffts:
        if nf > len(x):
            continue
        win = torch.hann_window(nf, dtype=x.dtype, device=x.device)
        Y = torch.stft(y, nf, nf // 4, window=win, return_complex=True).abs()
        X = torch.stft(x, nf, nf // 4, window=win, return_complex=True).abs()
        sc = torch.linalg.norm(X - Y) / (torch.linalg.norm(X) + 1e-12)
        lg = (torch.log(Y + 1e-5) - torch.log(X + 1e-5)).abs().mean()
        loss = loss + sc + lg
    return loss


# --------------------------------------------------------------------------------------
# Fitting one sample
# --------------------------------------------------------------------------------------
def fit_one(torch, path, args, device):
    name = os.path.splitext(os.path.basename(path))[0]
    sr, x = load_sample(path, args.hp, args.max_seconds)
    n = len(x)
    nfft = 1 << int(math.ceil(math.log2(2 * n)))
    f, S = spectrum_db(x, sr, nfft)
    Sdb = 20 * np.log10(S / S.max() + 1e-12)
    t60_0 = decay_t60_guess(x, sr)
    K = args.modes

    # clear spectral peaks (standing at least 6 dB above their surroundings), at least 8 Hz apart,
    # most prominent first; this spreads the filters across real modes instead of one big peak's skirt
    pk, props = find_peaks(Sdb, height=-55, prominence=6, distance=max(1, int(8 / f[1])))
    sel = (f[pk] > args.hp + 10) & (f[pk] < 0.45 * sr)
    pk, prom = pk[sel], props["prominences"][sel]
    pk = pk[np.argsort(prom + 0.5 * Sdb[pk])[::-1]]

    ratios, f01, mn = None, None, None
    if args.init == "bessel":
        f01 = args.f01 if args.f01 else float(f[pk[0]])   # assume the loudest peak is mode (0,1)
        table = bessel_table(K)
        ratios = [r for r, _, _ in table]
        mn = [(m, n_) for _, m, n_ in table]
        freqs = np.array([f01 * r for r in ratios])
        keep = freqs < 0.45 * sr
        freqs, ratios, mn = freqs[keep], list(np.array(ratios)[keep]), [p for p, k in zip(mn, keep) if k]
    else:
        freqs = f[pk[:K]]
        if len(freqs) < K:  # pad with quiet filler modes if the spectrum has few peaks
            extra = np.geomspace(100, 0.4 * sr, K - len(freqs))
            freqs = np.concatenate([freqs, extra])
    bin_idx = np.clip(np.round(freqs / f[1]).astype(int), 0, len(S) - 1)
    sigma = 6.9078 / t60_0
    amps = 2 * sigma * S[bin_idx] / sr          # level of a decaying sine with that spectral peak
    t60s = np.full(len(freqs), t60_0)

    model = build_model(torch, sr, n, freqs, amps, t60s, args.init, ratios, f01, device)
    xt = torch.tensor(x, dtype=torch.float64, device=device)

    freq_params = [model.log_f01, model.dev] if args.init == "bessel" else [model.log_f]
    other = [model.log_t60, model.amp, model.noise_log_decay, model.noise_bands]
    opt = torch.optim.Adam([{"params": freq_params, "lr": args.lr * 0.1},
                            {"params": other, "lr": args.lr}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps, eta_min=args.lr * 0.05)

    t0 = time.time()
    for step in range(args.steps + 1):
        opt.zero_grad()
        loss = mrstft_loss(torch, model(), xt)
        if args.init == "bessel":
            loss = loss + args.bessel_penalty * torch.mean((model.dev / 0.05) ** 2)
        if step % max(1, args.steps // 5) == 0:
            print(f"  step {step:>5}  loss {loss.item():.4f}")
        if step == args.steps:
            break
        loss.backward()
        opt.step()
        sched.step()
    secs = time.time() - t0

    # ---------------- save results ----------------
    out_dir = os.path.join(args.out, f"{name}_{args.init}")
    os.makedirs(out_dir, exist_ok=True)
    with torch.no_grad():
        y = model().cpu().numpy()
        ym = model.modes_only().cpu().numpy()
        fr = model.freqs().cpu().numpy()
        t60 = model.t60().cpu().numpy()
        amp = model.amp.cpu().numpy()
        final = mrstft_loss(torch, model(), xt).item()
    scale = 0.99 / max(1.0, np.abs(y).max(), np.abs(x).max())
    write_wav(os.path.join(out_dir, "original.wav"), sr, x * scale)
    write_wav(os.path.join(out_dir, "recreation.wav"), sr, y * scale)
    write_wav(os.path.join(out_dir, "modes_only.wav"), sr, ym * scale)

    order = np.argsort(fr)
    loud = np.abs(amp) * t60                       # rough "how much this mode matters"
    f01_fit = math.exp(model.log_f01.item()) if args.init == "bessel" else None
    modes = []
    for i in order:
        d = dict(freq_hz=round(float(fr[i]), 2), t60_s=round(float(t60[i]), 4),
                 level=round(float(amp[i]), 6), q=round(float(math.pi * fr[i] * t60[i] / 6.9078), 1),
                 weight_db=round(float(20 * np.log10(loud[i] / loud.max() + 1e-12)), 1))
        if args.init == "bessel":
            d.update(m=mn[i][0], n=mn[i][1], ideal_ratio=round(float(ratios[i]), 4),
                     fitted_ratio=round(float(fr[i] / f01_fit), 4))
        modes.append(d)

    # ratios of the important modes relative to the loudest one (taken as the (0,1) mode),
    # each compared with the nearest ideal-membrane ratio
    strong = [d for d in modes if d["weight_db"] > -25]
    base = max(modes, key=lambda d: d["weight_db"])["freq_hz"]
    btab = bessel_table(40)
    ratio_report = []
    for d in strong:
        r = d["freq_hz"] / base
        if r < 0.97:
            ratio_report.append(dict(freq_hz=d["freq_hz"], ratio=round(r, 3), nearest_bessel=None,
                                     mode="below", off_by_pct=None, weight_db=d["weight_db"]))
            continue
        nr, m, n_ = min(btab, key=lambda row: abs(row[0] - r))
        ratio_report.append(dict(freq_hz=d["freq_hz"], ratio=round(r, 3),
                                 nearest_bessel=round(nr, 3), mode=f"({m},{n_})",
                                 off_by_pct=round(100 * (r - nr) / nr, 1), weight_db=d["weight_db"]))

    result = dict(sample=path, init=args.init, sr=sr, seconds=round(n / sr, 3), final_loss=round(final, 4),
                  fit_seconds=round(secs, 1), steps=args.steps, highpass_hz=args.hp,
                  f01_hz=round(f01_fit, 2) if f01_fit else None,
                  noise=dict(decay_ms=round(1000 / math.exp(model.noise_log_decay.item()), 2),
                             band_log_gains=[round(v, 3) for v in model.noise_bands.tolist()]),
                  strong_mode_ratios=ratio_report, modes=modes)
    with open(os.path.join(out_dir, "params.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    make_plot(x, y, sr, fr, amp, t60, result, os.path.join(out_dir, "plot.png"))

    print(f"  done in {secs:.0f} s, final loss {final:.4f} -> {out_dir}/")
    print("  strong modes (ratio to the loudest mode, nearest ideal-membrane ratio, level):")
    for r in ratio_report[:12]:
        if r["nearest_bessel"] is None:
            print(f"    {r['freq_hz']:>8.1f} Hz  ratio {r['ratio']:>6.3f}  below the main mode "
                  f"(shell/air resonance?)  {r['weight_db']:>6.1f} dB")
        else:
            print(f"    {r['freq_hz']:>8.1f} Hz  ratio {r['ratio']:>6.3f}  nearest {r['mode']:>6} "
                  f"{r['nearest_bessel']:>6.3f}  ({r['off_by_pct']:+5.1f}%)  {r['weight_db']:>6.1f} dB")
    return result


def make_plot(x, y, sr, fr, amp, t60, result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import spectrogram

    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    vmax = None
    for a, sig, title in [(ax[0], x, "original"), (ax[1], y, "recreation")]:
        ff, tt, SS = spectrogram(sig, sr, nperseg=1024, noverlap=896)
        SS = 10 * np.log10(SS + 1e-14)
        vmax = SS.max() if vmax is None else vmax
        a.pcolormesh(tt, ff, SS, vmin=vmax - 80, vmax=vmax, shading="auto")
        a.set_ylim(0, min(4000, sr / 2))
        a.set_ylabel("Hz")
        a.set_title(f"{os.path.basename(result['sample'])}: {title}")
    nfft = 1 << int(math.ceil(math.log2(2 * len(x))))
    f, Sx = spectrum_db(x, sr, nfft)
    _, Sy = spectrum_db(y, sr, nfft)
    ref = Sx.max()
    ax[2].plot(f, 20 * np.log10(Sx / ref + 1e-12), lw=1, label="original")
    ax[2].plot(f, 20 * np.log10(Sy / ref + 1e-12), lw=1, alpha=0.8, label="recreation")
    loud = np.abs(amp) * t60
    for fi, li in zip(fr, loud):
        if li > 0.01 * loud.max():
            ax[2].axvline(fi, color="k", lw=0.5, alpha=0.3)
    ax[2].set_xlim(0, min(4000, sr / 2))
    ax[2].set_ylim(-70, 5)
    ax[2].set_xlabel("Hz")
    ax[2].set_ylabel("dB")
    ax[2].legend()
    ax[2].set_title(f"spectrum (grey lines = fitted modes), init={result['init']}, loss={result['final_loss']}")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("samples", nargs="*", help="wav files to fit")
    p.add_argument("--init", choices=["peaks", "bessel"], default="peaks")
    p.add_argument("--modes", type=int, default=24, help="number of bandpass filters")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--f01", type=float, default=None, help="bessel init: fundamental in Hz (default: loudest peak)")
    p.add_argument("--bessel-penalty", type=float, default=0.01, help="bessel init: cost of drifting from ideal ratios")
    p.add_argument("--hp", type=float, default=40.0, help="high-pass the sample at this many Hz (0 = off)")
    p.add_argument("--max-seconds", type=float, default=1.0)
    p.add_argument("--out", default="out_fit")
    p.add_argument("--device", default=None, help="e.g. cuda:0, cuda:1 or cpu (default: cuda if available)")
    p.add_argument("--check", action="store_true", help="only verify the filter maths and exit")
    args = p.parse_args()

    if args.check:
        run_check()
        return
    if not args.samples:
        p.error("give at least one wav file (or --check)")

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    for path in args.samples:
        print(f"\n== {path} ({args.init} init, {args.modes} modes)")
        fit_one(torch, path, args, device)


if __name__ == "__main__":
    main()
