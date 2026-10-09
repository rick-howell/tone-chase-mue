"""Ideal circular drum with a fixed edge, synthesized from Bessel-function modes.

Each mode (m, n) rings at f_mn = f01 * j_mn / j_01, where j_mn is the n-th zero of J_m.
How loudly a mode rings depends on where the drum is struck (strike_r) and where it is
"heard" (mic_r), both given as a fraction of the radius from the centre.
This is a simplified textbook model: point strike, point pickup, no air loading, one head.

Writes a few example hits to out_ideal/ plus a modes table (modes.json), so we can
hear the ideal membrane and have known right answers to test the fitting code against.

Usage:  python ideal_drum.py
"""
import json
import os

import numpy as np
from scipy.io import wavfile
from scipy.special import jn_zeros, jv

SR = 48000
OUT_DIR = "out_ideal"
J01 = jn_zeros(0, 1)[0]  # first zero of J_0, about 2.405


def render(f01=110.0, strike_r=0.3, mic_r=0.5, decay0=4.0, decay2=2e-5,
           mallet_ms=1.0, dur=2.0, m_max=5, n_max=4):
    """Return (audio, modes). decay rate per mode = decay0 + decay2 * f^2 (per second)."""
    t = np.arange(int(dur * SR)) / SR
    y = np.zeros_like(t)
    modes = []
    for m in range(m_max + 1):
        for n, z in enumerate(jn_zeros(m, n_max), start=1):
            f = f01 * z / J01
            if f >= 0.45 * SR:
                continue
            radial_norm = jv(m + 1, z) ** 2 / 2          # integral of J_m(z r)^2 r dr over the disc
            angular = 1.0 if m == 0 else 2.0             # m = 0 modes have no angular pattern
            amp = angular * jv(m, z * strike_r) * jv(m, z * mic_r) / radial_norm
            sigma = decay0 + decay2 * f ** 2
            y += amp * np.exp(-sigma * t) * np.sin(2 * np.pi * f * t)
            modes.append(dict(m=m, n=n, ratio=round(z / J01, 4), freq_hz=round(f, 2),
                              amp=round(float(amp), 5), decay_per_s=round(sigma, 3)))

    # Mallet: a raised-cosine contact pulse. Longer contact = softer, darker hit.
    L = int(mallet_ms * 1e-3 * SR)
    if L > 1:
        pulse = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(L) / L)
        y = np.convolve(y, pulse / pulse.sum())[: len(t)]

    y = y / (np.max(np.abs(y)) + 1e-12) * 10 ** (-1 / 20)  # normalize to -1 dBFS
    modes.sort(key=lambda d: d["freq_hz"])
    return y, modes


PRESETS = {
    "tom_offcenter":   dict(f01=110, strike_r=0.3),
    "tom_center":      dict(f01=110, strike_r=0.0),   # only m = 0 modes get excited
    "tom_near_edge":   dict(f01=110, strike_r=0.8),
    "floor_tom":       dict(f01=70, strike_r=0.3, decay0=3.0),
    "tom_hard_mallet": dict(f01=110, strike_r=0.3, mallet_ms=0.3),
    "tom_soft_mallet": dict(f01=110, strike_r=0.3, mallet_ms=4.0),
}


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    all_modes = {}
    for name, params in PRESETS.items():
        y, modes = render(**params)
        wavfile.write(os.path.join(OUT_DIR, f"{name}.wav"), SR, (y * 32767).astype(np.int16))
        all_modes[name] = dict(params=params, modes=modes)
        print(f"wrote {OUT_DIR}/{name}.wav")

    with open(os.path.join(OUT_DIR, "modes.json"), "w") as fh:
        json.dump(all_modes, fh, indent=2)

    print("\nLowest 10 modes of tom_offcenter (f01 = 110 Hz):")
    print(f"{'(m,n)':>6} {'ratio':>7} {'Hz':>8} {'amp':>9}")
    for d in all_modes["tom_offcenter"]["modes"][:10]:
        print(f"({d['m']},{d['n']}){'':>1} {d['ratio']:>7.3f} {d['freq_hz']:>8.1f} {d['amp']:>9.3f}")
