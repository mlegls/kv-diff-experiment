#!/usr/bin/env python3
"""
Comprehensive FFT analysis of roll vs truncate curves.

1. FFT of all curves: roll/trunc x K/V x avg/last
2. Side-by-side roll vs truncate frequency spectra
3. Autocorrelation comparison
4. Spectrograms: how frequency content changes with roll amount
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

OUT = Path("results_comparison")
OUT.mkdir(exist_ok=True)

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


def compute_fft(values, step):
    """Detrend, window, and FFT."""
    arr = np.array(values)
    detrended = signal.detrend(arr)
    # Apply Hann window to reduce spectral leakage
    window = np.hanning(len(detrended))
    windowed = detrended * window
    N = len(windowed)
    freqs = np.fft.rfftfreq(N, d=step)
    magnitudes = np.abs(np.fft.rfft(windowed)) * 2 / N  # normalized
    return freqs, magnitudes, detrended


def compute_autocorr(values):
    """Normalized autocorrelation."""
    arr = np.array(values)
    detrended = signal.detrend(arr)
    acf = np.correlate(detrended, detrended, mode="full")
    acf = acf[len(acf) // 2:]
    return acf / (acf[0] + 1e-10)


# ── Plot 1: All 8 curves' FFTs for 14B Base ─────────────────────────────────

d = data["14B Base"]
ns = np.array(d["ns"])
step = ns[1] - ns[0]
curves = {
    "roll_k (avg)": d["roll_k"],
    "roll_v (avg)": d["roll_v"],
    "roll_k (last)": d["roll_last_k"],
    "roll_v (last)": d["roll_last_v"],
    "trunc_k (avg)": d["trunc_k"],
    "trunc_v (avg)": d["trunc_v"],
    "trunc_k (last)": d["trunc_last_k"],
    "trunc_v (last)": d["trunc_last_v"],
}

fig, axes = plt.subplots(4, 2, figsize=(16, 16))
curve_list = list(curves.items())
for i, (name, vals) in enumerate(curve_list):
    ax = axes[i // 2][i % 2]
    freqs, mags, _ = compute_fft(vals, step)
    ax.plot(freqs[1:], mags[1:], lw=0.8,
            color="steelblue" if "roll" in name else "seagreen")
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("Frequency (cycles/token)")
    ax.set_ylabel("FFT Magnitude")
    # Annotate top peaks
    peak_idx, _ = signal.find_peaks(mags[1:], prominence=mags[1:].std())
    peak_idx += 1  # offset for skipping DC
    if len(peak_idx) > 0:
        top3 = peak_idx[np.argsort(mags[peak_idx])[-3:]]
        for pi in top3:
            if freqs[pi] > 0:
                period = 1.0 / freqs[pi]
                ax.annotate(f"T={period:.0f}", (freqs[pi], mags[pi]),
                            fontsize=7, color="red")

plt.suptitle("FFT of All Similarity Curves (14B Base)", fontsize=14)
plt.tight_layout()
plt.savefig(OUT / "fft_all_curves.png", dpi=150)
plt.close()
print("Saved fft_all_curves.png")


# ── Plot 2: Roll vs Truncate FFT comparison (K and V, last-pos) ─────────────

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

for col, kv in enumerate(["k", "v"]):
    for row, pos in enumerate(["last", "avg"]):
        ax = axes[row][col]
        suffix = f"_{kv}" if pos == "avg" else f"_last_{kv}"
        roll_key = f"roll{suffix}"
        trunc_key = f"trunc{suffix}"

        for model_name, md in data.items():
            freqs_r, mags_r, _ = compute_fft(md[roll_key], step)
            freqs_t, mags_t, _ = compute_fft(md[trunc_key], step)

            ax.plot(freqs_r[1:], mags_r[1:], color=COLORS[model_name],
                    lw=1.0, alpha=0.7, label=f"{model_name} roll")
            ax.plot(freqs_t[1:], mags_t[1:], color=COLORS[model_name],
                    lw=1.0, alpha=0.4, ls="--", label=f"{model_name} trunc")

        ax.set_xlabel("Frequency (cycles/token)")
        ax.set_ylabel("FFT Magnitude")
        label = "last-pos" if pos == "last" else "all-pos avg"
        ax.set_title(f"{kv.upper()} ({label})")
        ax.legend(fontsize=6, ncol=2)

plt.suptitle("FFT: Roll vs Truncate Across Models", fontsize=14)
plt.tight_layout()
plt.savefig(OUT / "fft_roll_vs_trunc.png", dpi=150)
plt.close()
print("Saved fft_roll_vs_trunc.png")


# ── Plot 3: Detrended time-domain overlay (roll vs trunc) ───────────────────

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
d = data["14B Base"]

for col, kv in enumerate(["k", "v"]):
    for row, pos in enumerate(["last", "avg"]):
        ax = axes[row][col]
        suffix = f"_{kv}" if pos == "avg" else f"_last_{kv}"

        roll_detrended = signal.detrend(np.array(d[f"roll{suffix}"]))
        trunc_detrended = signal.detrend(np.array(d[f"trunc{suffix}"]))

        ax.plot(ns, roll_detrended, lw=0.7, alpha=0.7, color="steelblue",
                label="Roll (detrended)")
        ax.plot(ns, trunc_detrended, lw=0.7, alpha=0.7, color="seagreen",
                label="Truncate (detrended)")
        label = "last-pos" if pos == "last" else "all-pos avg"
        ax.set_title(f"{kv.upper()} ({label}) detrended")
        ax.set_xlabel("Tokens removed")
        ax.set_ylabel("Detrended cos sim")
        ax.legend(fontsize=8)

plt.suptitle("Detrended Curves: Roll vs Truncate (14B Base)", fontsize=14)
plt.tight_layout()
plt.savefig(OUT / "detrended_roll_vs_trunc.png", dpi=150)
plt.close()
print("Saved detrended_roll_vs_trunc.png")


# ── Plot 4: Autocorrelation comparison ──────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
d = data["14B Base"]

for col, kv in enumerate(["k", "v"]):
    for row, pos in enumerate(["last", "avg"]):
        ax = axes[row][col]
        suffix = f"_{kv}" if pos == "avg" else f"_last_{kv}"

        acf_roll = compute_autocorr(d[f"roll{suffix}"])
        acf_trunc = compute_autocorr(d[f"trunc{suffix}"])

        max_lag = min(200, len(acf_roll))
        lags = np.arange(max_lag) * step

        ax.plot(lags, acf_roll[:max_lag], color="steelblue", lw=1.2,
                label="Roll")
        ax.plot(lags, acf_trunc[:max_lag], color="seagreen", lw=1.2,
                label="Truncate")
        ax.axhline(0, color="gray", ls="--", alpha=0.3)
        label = "last-pos" if pos == "last" else "all-pos avg"
        ax.set_title(f"{kv.upper()} ({label})")
        ax.set_xlabel("Lag (tokens)")
        ax.set_ylabel("Autocorrelation")
        ax.legend(fontsize=8)

plt.suptitle("Autocorrelation: Roll vs Truncate (14B Base)", fontsize=14)
plt.tight_layout()
plt.savefig(OUT / "autocorr_roll_vs_trunc.png", dpi=150)
plt.close()
print("Saved autocorr_roll_vs_trunc.png")


# ── Plot 5: Spectrogram — how frequency content evolves with position ───────
# Use a sliding window FFT along the curve

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
d = data["14B Base"]

window_size = 80  # tokens (in step units, so 80*5=400 token window)
hop = 10

for col, kv in enumerate(["k", "v"]):
    for row, (prefix, label) in enumerate([
        ("roll_last", "Roll (last-pos)"),
        ("trunc_last", "Trunc (last-pos)"),
    ]):
        ax = axes[row][col]
        key = f"{prefix}_{kv}"
        arr = np.array(d[key])

        if len(arr) < window_size:
            ax.set_title(f"{label} {kv.upper()} — too short")
            continue

        n_windows = (len(arr) - window_size) // hop + 1
        spectrogram = []
        centers = []

        for i in range(n_windows):
            start = i * hop
            chunk = arr[start:start + window_size]
            detrended = signal.detrend(chunk)
            windowed = detrended * np.hanning(len(detrended))
            fft_mag = np.abs(np.fft.rfft(windowed))
            spectrogram.append(fft_mag)
            centers.append(ns[start + window_size // 2]
                           if start + window_size // 2 < len(ns)
                           else ns[-1])

        spec = np.array(spectrogram).T
        freqs = np.fft.rfftfreq(window_size, d=step)

        im = ax.pcolormesh(centers, freqs[1:], spec[1:],
                           shading="auto", cmap="viridis")
        ax.set_xlabel("Center position (tokens removed)")
        ax.set_ylabel("Frequency (cycles/token)")
        ax.set_title(f"{label} — {kv.upper()}")
        plt.colorbar(im, ax=ax, label="Magnitude")

plt.suptitle("Spectrogram: Frequency Content vs Position (14B Base)",
             fontsize=14)
plt.tight_layout()
plt.savefig(OUT / "spectrogram_roll_trunc.png", dpi=150)
plt.close()
print("Saved spectrogram_roll_trunc.png")


# ── Plot 6: Power spectrum ratio (roll / trunc) ────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for col, kv in enumerate(["k", "v"]):
    ax = axes[col]
    for model_name, md in data.items():
        _, mags_r, _ = compute_fft(md[f"roll_last_{kv}"], step)
        freqs, mags_t, _ = compute_fft(md[f"trunc_last_{kv}"], step)

        # Power ratio (in dB)
        ratio = 10 * np.log10((mags_r[1:] + 1e-10) / (mags_t[1:] + 1e-10))
        # Smooth with moving average
        kernel = np.ones(5) / 5
        ratio_smooth = np.convolve(ratio, kernel, mode="same")

        ax.plot(freqs[1:], ratio_smooth, color=COLORS[model_name],
                lw=1.2, alpha=0.8, label=model_name)

    ax.axhline(0, color="gray", ls="--", alpha=0.3)
    ax.set_xlabel("Frequency (cycles/token)")
    ax.set_ylabel("Power Ratio Roll/Trunc (dB)")
    ax.set_title(f"{kv.upper()} — Roll vs Truncate Power Ratio")
    ax.legend(fontsize=8)

plt.suptitle("Which frequencies are stronger in Roll vs Truncate?", fontsize=14)
plt.tight_layout()
plt.savefig(OUT / "fft_power_ratio.png", dpi=150)
plt.close()
print("Saved fft_power_ratio.png")


print(f"\nAll FFT analysis plots saved to {OUT}/")
