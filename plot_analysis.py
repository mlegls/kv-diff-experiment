#!/usr/bin/env python3
"""
Analysis of K curve shapes: FFT, regression, and roll-vs-truncate decomposition.

Tests the hypothesis that:
1. K curve shape is dominated by positional encoding (RoPE periodicity)
2. Roll vs truncate difference is a near-constant offset (not slope change)
3. Roll curve may show periodic structure from RoPE
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal, stats

OUT = Path("results_comparison")
OUT.mkdir(exist_ok=True)

# Load the step-5 rolling data (finer granularity than the original)
MODELS = {
    "7B Base": "results_7b_base",
    "14B Base": "results_14b_base",
    "14B Instruct": "results_14b_instruct",
}
COLORS = {
    "7B Base": "#2ca02c",
    "14B Base": "#1f77b4",
    "14B Instruct": "#d62728",
}

data = {}
for name, d in MODELS.items():
    with open(Path(d) / "roll_truncate.json") as f:
        data[name] = json.load(f)


# ── 1. FFT of roll K curve ──────────────────────────────────────────────────

fig, axes = plt.subplots(len(MODELS), 2, figsize=(16, 4 * len(MODELS)))

for row, (name, d) in enumerate(data.items()):
    ns = np.array(d["ns"])
    roll_k = np.array(d["roll_last_k"])
    seq_len = d["seq_len"]

    # Detrend first (remove linear trend to isolate oscillations)
    detrended = signal.detrend(roll_k)

    # FFT
    N = len(detrended)
    freqs = np.fft.rfftfreq(N, d=ns[1] - ns[0] if len(ns) > 1 else 1)
    fft_mag = np.abs(np.fft.rfft(detrended))

    # Time domain (detrended)
    ax = axes[row][0]
    ax.plot(ns, detrended, color=COLORS[name], lw=0.8, alpha=0.7)
    ax.set_xlabel("Tokens rolled")
    ax.set_ylabel("Detrended K cos sim (last pos)")
    ax.set_title(f"{name} -- detrended roll K")

    # Frequency domain
    ax = axes[row][1]
    # Skip DC component (index 0)
    ax.plot(freqs[1:], fft_mag[1:], color=COLORS[name], lw=1.2)
    ax.set_xlabel("Frequency (cycles per token)")
    ax.set_ylabel("FFT Magnitude")
    ax.set_title(f"{name} -- FFT of detrended roll K")
    # Mark top peaks
    peak_idx = np.argsort(fft_mag[1:])[-5:] + 1
    for pi in peak_idx:
        if fft_mag[pi] > fft_mag[1:].mean() + 2 * fft_mag[1:].std():
            period = 1.0 / freqs[pi] if freqs[pi] > 0 else float("inf")
            ax.annotate(f"T={period:.0f}",
                        (freqs[pi], fft_mag[pi]),
                        fontsize=7, ha="center", va="bottom")

plt.suptitle("FFT Analysis: Periodicity in Rolling K Curves", fontsize=14)
plt.tight_layout()
plt.savefig(OUT / "fft_roll_k.png", dpi=150)
plt.close()
print("Saved fft_roll_k.png")


# ── 2. FFT of V curves too ──────────────────────────────────────────────────

fig, axes = plt.subplots(len(MODELS), 2, figsize=(16, 4 * len(MODELS)))

for row, (name, d) in enumerate(data.items()):
    ns = np.array(d["ns"])
    roll_v = np.array(d["roll_last_v"])

    detrended = signal.detrend(roll_v)
    N = len(detrended)
    freqs = np.fft.rfftfreq(N, d=ns[1] - ns[0] if len(ns) > 1 else 1)
    fft_mag = np.abs(np.fft.rfft(detrended))

    ax = axes[row][0]
    ax.plot(ns, detrended, color=COLORS[name], lw=0.8, alpha=0.7)
    ax.set_xlabel("Tokens rolled")
    ax.set_ylabel("Detrended V cos sim (last pos)")
    ax.set_title(f"{name} -- detrended roll V")

    ax = axes[row][1]
    ax.plot(freqs[1:], fft_mag[1:], color=COLORS[name], lw=1.2)
    ax.set_xlabel("Frequency (cycles per token)")
    ax.set_ylabel("FFT Magnitude")
    ax.set_title(f"{name} -- FFT of detrended roll V")

plt.suptitle("FFT Analysis: Periodicity in Rolling V Curves", fontsize=14)
plt.tight_layout()
plt.savefig(OUT / "fft_roll_v.png", dpi=150)
plt.close()
print("Saved fft_roll_v.png")


# ── 3. Roll vs Truncate regression ───────────────────────────────────────────

fig, axes = plt.subplots(len(MODELS), 2, figsize=(16, 4 * len(MODELS)))

for row, (name, d) in enumerate(data.items()):
    ns = np.array(d["ns"])
    roll_k = np.array(d["roll_last_k"])
    trunc_k = np.array(d["trunc_last_k"])
    seq_len = d["seq_len"]
    frac = ns / seq_len

    # Fit linear regression to both
    slope_r, intercept_r, r_r, _, _ = stats.linregress(frac, roll_k)
    slope_t, intercept_t, r_t, _, _ = stats.linregress(frac, trunc_k)

    # Plot with fits
    ax = axes[row][0]
    ax.scatter(frac, roll_k, s=2, alpha=0.3, color="steelblue", label="Roll")
    ax.scatter(frac, trunc_k, s=2, alpha=0.3, color="seagreen", label="Truncate")
    ax.plot(frac, intercept_r + slope_r * frac, "steelblue", lw=2,
            label=f"Roll fit: {slope_r:.3f}x + {intercept_r:.3f} (R²={r_r**2:.3f})")
    ax.plot(frac, intercept_t + slope_t * frac, "seagreen", lw=2,
            label=f"Trunc fit: {slope_t:.3f}x + {intercept_t:.3f} (R²={r_t**2:.3f})")
    ax.set_xlabel("Fraction removed")
    ax.set_ylabel("K cos sim (last pos)")
    ax.set_title(f"{name} -- linear regression")
    ax.legend(fontsize=7)

    # Difference curve
    ax = axes[row][1]
    diff = roll_k - trunc_k
    ax.plot(frac, diff, color="purple", lw=0.8, alpha=0.7)
    slope_d, intercept_d, r_d, _, _ = stats.linregress(frac, diff)
    ax.plot(frac, intercept_d + slope_d * frac, "purple", lw=2, ls="--",
            label=f"slope={slope_d:.4f}, intercept={intercept_d:.3f}, R²={r_d**2:.3f}")
    ax.axhline(np.mean(diff), color="gray", ls=":", alpha=0.5,
               label=f"mean diff = {np.mean(diff):.3f}")
    ax.set_xlabel("Fraction removed")
    ax.set_ylabel("Roll - Truncate (K cos sim)")
    ax.set_title(f"{name} -- difference")
    ax.legend(fontsize=8)

plt.suptitle("Roll vs Truncate K: Regression & Difference Analysis", fontsize=14)
plt.tight_layout()
plt.savefig(OUT / "regression_roll_trunc.png", dpi=150)
plt.close()
print("Saved regression_roll_trunc.png")


# ── 4. Autocorrelation of roll K (another way to check periodicity) ─────────

fig, axes = plt.subplots(1, len(MODELS), figsize=(5 * len(MODELS), 5))

for i, (name, d) in enumerate(data.items()):
    roll_k = np.array(d["roll_last_k"])
    detrended = signal.detrend(roll_k)

    # Autocorrelation
    acf = np.correlate(detrended, detrended, mode="full")
    acf = acf[len(acf) // 2:]  # positive lags only
    acf = acf / acf[0]  # normalize

    ax = axes[i]
    step = d["ns"][1] - d["ns"][0] if len(d["ns"]) > 1 else 1
    lags = np.arange(len(acf)) * step
    ax.plot(lags[:200], acf[:200], color=COLORS[name], lw=1.2)
    ax.axhline(0, color="gray", ls="--", alpha=0.3)
    ax.set_xlabel("Lag (tokens)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title(f"{name}")

    # Find first significant peak after lag 0
    peaks, props = signal.find_peaks(acf[1:100], height=0.1, prominence=0.05)
    if len(peaks) > 0:
        peak_lag = (peaks[0] + 1) * step
        ax.axvline(peak_lag, color="red", ls=":", alpha=0.5)
        ax.annotate(f"peak @ {peak_lag}", (peak_lag, acf[peaks[0] + 1]),
                    fontsize=8, color="red")

plt.suptitle("Autocorrelation of Detrended Roll K (last pos)", fontsize=14)
plt.tight_layout()
plt.savefig(OUT / "autocorrelation_roll_k.png", dpi=150)
plt.close()
print("Saved autocorrelation_roll_k.png")


# ── 5. Overlay: Roll avg-pos vs last-pos to isolate position effect ─────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for name, d in data.items():
    ns = np.array(d["ns"])
    frac = ns / d["seq_len"]
    c = COLORS[name]

    axes[0].plot(frac, d["roll_k"], color=c, lw=1.5, alpha=0.7,
                 label=f"{name} avg")
    axes[0].plot(frac, d["roll_last_k"], color=c, lw=1.5, alpha=0.4,
                 ls="--", label=f"{name} last")
    axes[1].plot(frac, d["roll_v"], color=c, lw=1.5, alpha=0.7,
                 label=f"{name} avg")
    axes[1].plot(frac, d["roll_last_v"], color=c, lw=1.5, alpha=0.4,
                 ls="--", label=f"{name} last")

for ax, kv in zip(axes, ["K", "V"]):
    ax.set_xlabel("Fraction of context rolled")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title(f"Rolling {kv}: All-Position Avg vs Last-Position")
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(0, 1.05)

plt.suptitle("Average vs Terminal State Under Rolling", fontsize=14)
plt.tight_layout()
plt.savefig(OUT / "avg_vs_last_rolling.png", dpi=150)
plt.close()
print("Saved avg_vs_last_rolling.png")


print(f"\nAll analysis plots saved to {OUT}/")
