#!/usr/bin/env python3
"""
FFT analysis comparing different roll amounts.

For each roll amount, we have a per-position similarity curve.
This script FFTs those curves to see how frequency content changes
with roll amount.

Also does per-layer FFT for each roll amount (from original experiment).
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
    "7B Base": "orig_7b_base",
    "14B Base": "orig_14b_base",
    "14B Instruct": "orig_14b_instruct",
}
COLORS_ROLL = {
    "roll_1": "#1f77b4",
    "roll_2": "#2ca02c",
    "roll_3": "#ff7f0e",
    "roll_5": "#d62728",
    "roll_10": "#9467bd",
    "roll_20": "#8c564b",
    "scrambled": "#e377c2",
    "unrelated": "#7f7f7f",
}

ROLL_CONDITIONS = ["roll_1", "roll_2", "roll_3", "roll_5", "roll_10", "roll_20"]
ALL_CONDITIONS = ROLL_CONDITIONS + ["scrambled", "unrelated"]


def compute_fft(values):
    arr = np.array(values)
    detrended = signal.detrend(arr)
    window = np.hanning(len(detrended))
    windowed = detrended * window
    N = len(windowed)
    freqs = np.fft.rfftfreq(N, d=1)  # 1 token spacing
    magnitudes = np.abs(np.fft.rfft(windowed)) * 2 / N
    return freqs, magnitudes


# ── Load data ────────────────────────────────────────────────────────────────

data = {}
for name, d in MODELS.items():
    with open(Path(d) / "results.json") as f:
        data[name] = json.load(f)


# ── Plot 1: Per-position K similarity at different roll amounts (14B Base) ──

d = data["14B Base"]

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Top-left: raw per-position curves
ax = axes[0][0]
for cond in ALL_CONDITIONS:
    pp = d[cond]["cos_k"]["per_position"]
    ax.plot(pp, label=cond, color=COLORS_ROLL[cond], lw=1.2, alpha=0.8)
ax.set_xlabel("Position (in overlapping region)")
ax.set_ylabel("Cosine Similarity")
ax.set_title("K per-position similarity")
ax.legend(fontsize=7)

# Top-right: same for V
ax = axes[0][1]
for cond in ALL_CONDITIONS:
    pp = d[cond]["cos_v"]["per_position"]
    ax.plot(pp, label=cond, color=COLORS_ROLL[cond], lw=1.2, alpha=0.8)
ax.set_xlabel("Position")
ax.set_ylabel("Cosine Similarity")
ax.set_title("V per-position similarity")
ax.legend(fontsize=7)

# Bottom-left: FFT of K per-position for each roll amount
ax = axes[1][0]
for cond in ALL_CONDITIONS:
    pp = d[cond]["cos_k"]["per_position"]
    if len(pp) < 10:
        continue
    freqs, mags = compute_fft(pp)
    ax.plot(freqs[1:], mags[1:], label=cond, color=COLORS_ROLL[cond],
            lw=1.2, alpha=0.8)
ax.set_xlabel("Frequency (cycles/token)")
ax.set_ylabel("FFT Magnitude")
ax.set_title("FFT of K per-position curves")
ax.legend(fontsize=7)

# Bottom-right: FFT of V per-position
ax = axes[1][1]
for cond in ALL_CONDITIONS:
    pp = d[cond]["cos_v"]["per_position"]
    if len(pp) < 10:
        continue
    freqs, mags = compute_fft(pp)
    ax.plot(freqs[1:], mags[1:], label=cond, color=COLORS_ROLL[cond],
            lw=1.2, alpha=0.8)
ax.set_xlabel("Frequency (cycles/token)")
ax.set_ylabel("FFT Magnitude")
ax.set_title("FFT of V per-position curves")
ax.legend(fontsize=7)

plt.suptitle("Per-Position Similarity & FFT by Roll Amount (14B Base, 87 tok)",
             fontsize=13)
plt.tight_layout()
plt.savefig(OUT / "fft_per_position_by_roll.png", dpi=150)
plt.close()
print("Saved fft_per_position_by_roll.png")


# ── Plot 2: Per-layer K/V for each roll amount, and their FFTs ──────────────

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Top: per-layer curves
for col, kv in enumerate(["k", "v"]):
    ax = axes[0][col]
    for cond in ALL_CONDITIONS:
        pl = d[cond][f"cos_{kv}"]["per_layer"]
        ax.plot(pl, label=cond, color=COLORS_ROLL[cond], lw=1.2, alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title(f"{kv.upper()} per-layer similarity")
    ax.legend(fontsize=7)

# Bottom: FFT of per-layer curves
for col, kv in enumerate(["k", "v"]):
    ax = axes[1][col]
    for cond in ALL_CONDITIONS:
        pl = d[cond][f"cos_{kv}"]["per_layer"]
        freqs, mags = compute_fft(pl)
        ax.plot(freqs[1:], mags[1:], label=cond, color=COLORS_ROLL[cond],
                lw=1.2, alpha=0.8)
    ax.set_xlabel("Frequency (cycles/layer)")
    ax.set_ylabel("FFT Magnitude")
    ax.set_title(f"FFT of {kv.upper()} per-layer curves")
    ax.legend(fontsize=7)

plt.suptitle("Per-Layer Similarity & FFT by Roll Amount (14B Base)", fontsize=13)
plt.tight_layout()
plt.savefig(OUT / "fft_per_layer_by_roll.png", dpi=150)
plt.close()
print("Saved fft_per_layer_by_roll.png")


# ── Plot 3: Cross-model per-position FFT (roll_5 as representative) ────────

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
model_colors = {"7B Base": "#2ca02c", "14B Base": "#1f77b4",
                "14B Instruct": "#d62728"}

for col, kv in enumerate(["k", "v"]):
    # Per-position curves
    ax = axes[0][col]
    for name, md in data.items():
        for cond in ["roll_5", "scrambled"]:
            pp = md[cond][f"cos_{kv}"]["per_position"]
            ls = "-" if cond == "roll_5" else "--"
            ax.plot(pp, label=f"{name} {cond}", color=model_colors[name],
                    ls=ls, lw=1.2, alpha=0.8)
    ax.set_xlabel("Position")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title(f"{kv.upper()} per-position (roll_5 vs scrambled)")
    ax.legend(fontsize=6)

    # FFT
    ax = axes[1][col]
    for name, md in data.items():
        for cond in ["roll_5", "scrambled"]:
            pp = md[cond][f"cos_{kv}"]["per_position"]
            freqs, mags = compute_fft(pp)
            ls = "-" if cond == "roll_5" else "--"
            ax.plot(freqs[1:], mags[1:], label=f"{name} {cond}",
                    color=model_colors[name], ls=ls, lw=1.2, alpha=0.8)
    ax.set_xlabel("Frequency (cycles/token)")
    ax.set_ylabel("FFT Magnitude")
    ax.set_title(f"FFT of {kv.upper()} per-position")
    ax.legend(fontsize=6)

plt.suptitle("Per-Position FFT: Roll vs Scrambled Across Models", fontsize=13)
plt.tight_layout()
plt.savefig(OUT / "fft_per_position_cross_model.png", dpi=150)
plt.close()
print("Saved fft_per_position_cross_model.png")


# ── Plot 4: How dominant frequency magnitude changes with roll amount ───────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for col, kv in enumerate(["k", "v"]):
    ax = axes[col]
    for name, md in data.items():
        total_powers = []
        roll_ns = []
        for cond in ROLL_CONDITIONS:
            pp = md[cond][f"cos_{kv}"]["per_position"]
            _, mags = compute_fft(pp)
            total_power = np.sum(mags[1:] ** 2)  # total spectral power
            total_powers.append(total_power)
            roll_ns.append(int(cond.split("_")[1]))
        ax.plot(roll_ns, total_powers, "o-", label=name,
                color=model_colors[name], lw=2)
    ax.set_xlabel("Roll amount (tokens)")
    ax.set_ylabel("Total spectral power (excl. DC)")
    ax.set_title(f"{kv.upper()} — oscillation energy vs roll amount")
    ax.legend()

plt.suptitle("Does Rolling More Create More Oscillation?", fontsize=14)
plt.tight_layout()
plt.savefig(OUT / "spectral_power_vs_roll.png", dpi=150)
plt.close()
print("Saved spectral_power_vs_roll.png")


print(f"\nAll plots saved to {OUT}/")
