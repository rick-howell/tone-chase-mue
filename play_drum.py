"""Play a fitted drum in ways the original sample can't: new strike positions, hand softness,
tuning and damping, plus a short groove, all rendered from the fitted ZDF filter bank.

It reads a params.json written by fit_drum.py. Each fitted mode is matched to its ideal
membrane mode (m, n). Moving the strike point then reweights the modes the way the Bessel
solution says it should: a centre hit excites only the round (m = 0) modes, while a hit near
the edge brings in the others.

Usage (on woodforde, in ~/gpu-tests/tone-chase-mue with env gpu active; runs on the CPU, takes seconds):
    python play_drum.py out_fit/FPC_Bongo_004_peaks/params.json
    python play_drum.py out_fit/FPC_Bongo_004_peaks/params.json --low out_fit/FPC_Conga_011b_peaks/params.json

Writes to out_play/<drum name>/:
    variations.wav   one file: as fitted, strike positions, soft vs hard, tuning, damping,
                     with short gaps between hits, all at matched loudness (listen to this first)
    strike_*.wav, soft.wav, hard.wav, tuned_*.wav, damped.wav, ringing.wav   the same hits, one per file
    groove.wav       a short bongo-style pattern played by the model (uses --low as the second drum
                     if given, otherwise the same drum tuned down 5 semitones)
"""
import argparse
import json
import math
import os

import numpy as np
from scipy.io import wavfile
from scipy.special import jn_zeros, jv

J01 = jn_zeros(0, 1)[0]


def bessel_table(count=60, m_max=12, n_max=8):
    rows = [(z / J01, m, n, z) for m in range(m_max + 1)
            for n, z in enumerate(jn_zeros(m, n_max), start=1)]
    rows.sort()
    return rows[:count]


class FittedDrum:
    """A drum rebuilt from fit_drum.py's params.json."""

    def __init__(self, params_path, ref_strike=0.7, tolerance=0.08):
        with open(params_path) as fh:
            p = json.load(fh)
        self.name = os.path.basename(os.path.dirname(os.path.abspath(params_path)))
        self.sr = p["sr"]
        self.noise_decay_ms = p["noise"]["decay_ms"]
        self.noise_bands = np.array(p["noise"]["band_log_gains"])
        modes = p["modes"]
        self.freq = np.array([d["freq_hz"] for d in modes])
        self.t60 = np.array([d["t60_s"] for d in modes])
        self.level = np.array([d["level"] for d in modes])

        # Label every mode with its ideal (m, n), measured against the loudest mode as (0,1).
        weight = np.abs(self.level) * self.t60
        base = self.freq[np.argmax(weight)]
        table = bessel_table()
        self.m = np.full(len(modes), -1)          # -1 = no ideal match (shell/air): position-independent
        self.z = np.zeros(len(modes))
        for i, d in enumerate(modes):
            if "m" in d:                          # bessel-init fits already carry their labels
                self.m[i] = d["m"]
                self.z[i] = jn_zeros(d["m"], d["n"])[-1]
                continue
            r = self.freq[i] / base
            ratio, m, n, z = min(table, key=lambda row: abs(row[0] - r))
            if abs(r - ratio) / ratio < tolerance and r > 0.97:
                self.m[i], self.z[i] = m, z
        # The sample was hit at an unknown spot; assume ref_strike (fraction of the radius).
        self.ref_strike = ref_strike

    def position_gain(self, strike_r):
        """How much louder or quieter each mode gets when moving from the reference strike point."""
        g = np.ones(len(self.freq))
        has = self.m >= 0
        floor = 0.15  # keeps modes near a node at the reference point from blowing up
        ref = jv(self.m[has], self.z[has] * self.ref_strike)
        new = jv(self.m[has], self.z[has] * strike_r)
        ref = np.sign(ref) * np.maximum(np.abs(ref), floor) + (ref == 0) * floor
        g[has] = new / ref
        return g

    def render(self, strike_r=None, softness=0.0, semitones=0.0, damping=1.0, velocity=1.0,
               dur=1.0, seed=0):
        """softness: 0 = as fitted, >0 = softer hand (darker, less slap), <0 = harder (brighter).
        damping: multiplies ring time (0.5 = choked, 2 = rings longer)."""
        sr = self.sr
        n = int(dur * sr)
        nfft = 1 << int(math.ceil(math.log2(2 * n)))
        f = self.freq * 2 ** (semitones / 12)
        t60 = np.minimum(self.t60 * damping, nfft / sr / 2)
        amp = self.level.copy()
        if strike_r is not None:
            amp = amp * self.position_gain(strike_r)
        f_main = f[np.argmax(np.abs(self.level) * self.t60)]
        tilt = -1.5 * softness + 0.5 * (velocity - 1.0)  # harder hits are brighter
        amp = amp * np.where(f > f_main, (f / f_main) ** tilt, 1.0)  # only reshape above the main pitch

        # the same ZDF bandpass bank as fit_drum.py, evaluated exactly in the frequency domain
        g = np.tan(np.pi * np.minimum(f, 0.45 * sr) / sr)[:, None]
        R = np.clip(6.9078 / (t60[:, None] * 2 * np.pi * f[:, None]), 1e-5, 2.0)
        z1 = np.exp(-1j * np.linspace(0, np.pi, nfft // 2 + 1))[None, :]
        z2 = z1 * z1
        den = (1 + 2 * R * g + g * g) + (2 * g * g - 2) * z1 + (1 - 2 * R * g + g * g) * z2
        y = np.fft.irfft((amp[:, None] * (1 - z2) / (2 * den)).sum(0), nfft)[:n]

        # noise burst for the hand contact, shaped like the fitted one
        rng = np.random.default_rng(seed)
        t = np.arange(n) / sr
        env = np.exp(-t / (self.noise_decay_ms / 1000))
        NZ = np.fft.rfft(rng.standard_normal(n) * env, nfft)
        bins = np.fft.rfftfreq(nfft, 1 / sr)
        centers = np.geomspace(50, sr / 2, len(self.noise_bands))
        shape = np.exp(np.interp(np.log(np.maximum(bins, 1)), np.log(centers), self.noise_bands))
        shape *= np.where(bins > f_main, (np.maximum(bins, 1) / f_main) ** tilt, 1.0)
        noise_gain = 2 ** (-2 * softness) * velocity
        y = y + noise_gain * np.fft.irfft(NZ * shape, nfft)[:n]
        return velocity * y


def write(path, sr, y, peak):
    wavfile.write(path, sr, (np.clip(y / peak * 0.9, -1, 1) * 32767).astype(np.int16))


def join(sr, hits, gap_s=0.25):
    gap = np.zeros(int(gap_s * sr))
    return np.concatenate([np.concatenate([h, gap]) for h in hits])


def groove(sr, hi, lo, bpm=96, bars=4):
    """A martillo-style bongo pattern: steady 8th notes on the high drum with varied strokes,
    the low drum on the 'and' of beats 2 and 4. Each tuple: (8th-note index, drum, strike, softness, velocity)."""
    eighth = 60 / bpm / 2
    bar = [(0, "hi", 0.75, 0.0, 1.0), (1, "hi", 0.30, 0.6, 0.55), (2, "hi", 0.75, 0.2, 0.75),
           (3, "lo", 0.70, 0.0, 0.95), (4, "hi", 0.75, 0.0, 0.9), (5, "hi", 0.30, 0.6, 0.55),
           (6, "hi", 0.85, -0.3, 0.8), (7, "lo", 0.70, 0.0, 1.0)]
    total = int((bars * 8 * eighth + 1.0) * sr)
    out = np.zeros(total)
    cache = {}
    for b in range(bars):
        for k, (step, which, strike, soft, vel) in enumerate(bar):
            key = (which, strike, soft, vel)
            if key not in cache:
                drum, semis = (hi, 0.0) if which == "hi" else lo
                cache[key] = drum.render(strike_r=strike, softness=soft, semitones=semis,
                                         velocity=vel, dur=0.8, seed=k)
            start = int((b * 8 + step) * eighth * sr)
            h = cache[key]
            out[start:start + len(h)] += h[: total - start]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("params", help="params.json from fit_drum.py")
    p.add_argument("--low", default=None, help="optional second drum (params.json) for the groove")
    p.add_argument("--ref-strike", type=float, default=0.7,
                   help="where we assume the sample was hit, as a fraction of the radius (default 0.7)")
    p.add_argument("--bpm", type=float, default=96)
    p.add_argument("--out", default="out_play")
    args = p.parse_args()

    drum = FittedDrum(args.params, ref_strike=args.ref_strike)
    sr = drum.sr
    out_dir = os.path.join(args.out, drum.name)
    os.makedirs(out_dir, exist_ok=True)
    labelled = int((drum.m >= 0).sum())
    print(f"{drum.name}: {len(drum.freq)} modes, {labelled} matched to ideal membrane modes")

    hits = {
        "strike_center": drum.render(strike_r=0.0),
        "strike_halfway": drum.render(strike_r=0.5),
        "strike_near_edge": drum.render(strike_r=0.9),
        "as_fitted": drum.render(),
        "soft": drum.render(softness=1.0),
        "hard": drum.render(softness=-0.5),
        "tuned_up_3": drum.render(semitones=3),
        "tuned_down_5": drum.render(semitones=-5),
        "damped": drum.render(damping=0.4),
        "ringing": drum.render(damping=2.0),
    }
    # level-match every hit to the fitted one (RMS of the first 0.2 s), so you hear timbre, not loudness
    def rms(h):
        return np.sqrt(np.mean(h[: int(0.2 * sr)] ** 2)) + 1e-12
    target = rms(hits["as_fitted"])
    hits = {k: h * target / rms(h) for k, h in hits.items()}
    peak = max(np.abs(h).max() for h in hits.values())
    for name, h in hits.items():
        write(os.path.join(out_dir, f"{name}.wav"), sr, h, peak)
    order = ["as_fitted", "strike_center", "strike_halfway", "strike_near_edge", "soft", "hard",
             "tuned_up_3", "tuned_down_5", "damped", "ringing"]
    write(os.path.join(out_dir, "variations.wav"), sr, join(sr, [hits[k] for k in order], 0.35), peak)
    print("  variations.wav order: " + ", ".join(order))

    if args.low:
        low = (FittedDrum(args.low, ref_strike=args.ref_strike), 0.0)
        if low[0].sr != sr:
            raise SystemExit("both drums must have the same sample rate")
    else:
        low = (drum, -5.0)
    g = groove(sr, drum, low, bpm=args.bpm)
    write(os.path.join(out_dir, "groove.wav"), sr, g, np.abs(g).max())
    print(f"  wrote {len(hits) + 2} files to {out_dir}/")


if __name__ == "__main__":
    main()
