#!/usr/bin/env python3
"""
Cross-model comparison plots for the original KV cache diff experiment.
Produces:
  1. Summary bar chart (like results/summary.png) with 3 model variants
  2. Per-layer cosine similarity for roll_5, scrambled, unrelated across models
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODELS = {
    "7B Base": "orig_7b_base",
    "14B Base": "orig_14b_base",
    "14B Instruct": "orig_14b_instruct",
}
COLORS = {
    "7B Base": "#2ca02c",
    "14B Base": "#1f77b4",
    "14B Instruct": "#d62728",
}
OUT = Path("results_comparison")
OUT.mkdir(exist_ok=True)


def load(d):
    with open(Path(d) / "results.json") as f:
        return json.load(f)


data = {name: load(d) for name, d in MODELS.items()}
CONDITIONS = list(data[list(MODELS.keys())[0]].keys())


# ── Plot 1: Summary bar chart with 3 model variants ─────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(18, 11))
model_names = list(MODELS.keys())
n_models = len(model_names)
n_conds = len(CONDITIONS)
x = np.arange(n_conds)
width = 0.25

for row, (prefix, label) in enumerate([
    ("cos", "Cosine Similarity"), ("l2", "Norm. L2 Distance")
]):
    for col, kv in enumerate(["k", "v"]):
        ax = axes[row][col]
        metric = f"{prefix}_{kv}"
        for i, name in enumerate(model_names):
            vals = [data[name][c][metric]["mean"] for c in CONDITIONS]
            ax.bar(x + (i - 1) * width, vals, width,
                   label=name, color=COLORS[name], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(CONDITIONS, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(label)
        ax.set_title(f"{label} -- {kv.upper()} vectors")
        ax.legend(fontsize=8)
        if prefix == "cos":
            ax.set_ylim(0, 1.05)
            ax.axhline(1.0, color="gray", ls="--", alpha=0.3)

plt.suptitle("KV Cache Distance: 3-Model Comparison", fontsize=15)
plt.tight_layout()
plt.savefig(OUT / "summary_3model.png", dpi=150)
plt.close()
print("Saved summary_3model.png")


# ── Plot 2: Per-layer comparison — key conditions ────────────────────────────
# Compare roll_5, roll_20, scrambled, unrelated across models

KEY_CONDITIONS = ["roll_5", "roll_20", "scrambled", "unrelated"]
COND_STYLES = {
    "roll_5": ("solid", 1.5),
    "roll_20": ("dashed", 1.5),
    "scrambled": ("solid", 2.0),
    "unrelated": ("dotted", 2.0),
}

fig, axes = plt.subplots(2, 2, figsize=(16, 11))

for col, kv in enumerate(["k", "v"]):
    metric = f"cos_{kv}"
    # Top row: all conditions, one line per (model, condition)
    ax = axes[0][col]
    for name in model_names:
        for cond in KEY_CONDITIONS:
            per_layer = data[name][cond][metric]["per_layer"]
            ls, lw = COND_STYLES[cond]
            ax.plot(per_layer, label=f"{name} {cond}",
                    color=COLORS[name], ls=ls, lw=lw, alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title(f"Per-Layer Cosine Sim -- {kv.upper()}")
    ax.legend(fontsize=5.5, ncol=3, loc="lower left")

    # Bottom row: normalized layer position (0-1) so different depths align
    ax = axes[1][col]
    for name in model_names:
        for cond in KEY_CONDITIONS:
            per_layer = data[name][cond][metric]["per_layer"]
            n_layers = len(per_layer)
            frac = np.linspace(0, 1, n_layers)
            ls, lw = COND_STYLES[cond]
            ax.plot(frac, per_layer, label=f"{name} {cond}",
                    color=COLORS[name], ls=ls, lw=lw, alpha=0.8)
    ax.set_xlabel("Normalized Layer Position (0=first, 1=last)")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title(f"Per-Layer (depth-normalized) -- {kv.upper()}")
    ax.legend(fontsize=5.5, ncol=3, loc="lower left")

plt.suptitle("Per-Layer KV Similarity: Model Comparison", fontsize=15)
plt.tight_layout()
plt.savefig(OUT / "per_layer_3model.png", dpi=150)
plt.close()
print("Saved per_layer_3model.png")


# ── Plot 3: Per-layer, one subplot per condition ─────────────────────────────

fig, axes = plt.subplots(len(KEY_CONDITIONS), 2,
                          figsize=(14, 4 * len(KEY_CONDITIONS)))

for row, cond in enumerate(KEY_CONDITIONS):
    for col, kv in enumerate(["k", "v"]):
        ax = axes[row][col]
        metric = f"cos_{kv}"
        for name in model_names:
            per_layer = data[name][cond][metric]["per_layer"]
            ax.plot(per_layer, label=name, color=COLORS[name], lw=2)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Cosine Similarity")
        ax.set_title(f"{cond} -- {kv.upper()}")
        ax.legend(fontsize=8)

plt.suptitle("Per-Layer KV Similarity by Condition", fontsize=15)
plt.tight_layout()
plt.savefig(OUT / "per_layer_by_condition.png", dpi=150)
plt.close()
print("Saved per_layer_by_condition.png")


# ── Plot 4: Focus on rolling vs scrambled vs unrelated per layer ─────────────
# One plot per model, showing how these 3 conditions differ

fig, axes = plt.subplots(len(model_names), 2,
                          figsize=(14, 4 * len(model_names)))

FOCUS_CONDS = ["roll_5", "scrambled", "unrelated"]
FOCUS_COLORS = {"roll_5": "steelblue", "scrambled": "darkorange",
                "unrelated": "crimson"}

for row, name in enumerate(model_names):
    for col, kv in enumerate(["k", "v"]):
        ax = axes[row][col]
        metric = f"cos_{kv}"
        for cond in FOCUS_CONDS:
            per_layer = data[name][cond][metric]["per_layer"]
            ax.plot(per_layer, label=cond, color=FOCUS_COLORS[cond], lw=2)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Cosine Similarity")
        ax.set_title(f"{name} -- {kv.upper()}")
        ax.legend(fontsize=9)
        ax.set_ylim(-0.1, 1.05)

plt.suptitle("Roll vs Scrambled vs Unrelated: Per-Layer by Model", fontsize=15)
plt.tight_layout()
plt.savefig(OUT / "roll_scrambled_unrelated_per_layer.png", dpi=150)
plt.close()
print("Saved roll_scrambled_unrelated_per_layer.png")


print(f"\nAll plots saved to {OUT}/")
