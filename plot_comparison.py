#!/usr/bin/env python3
"""Cross-model comparison plots for KV cache diff experiments."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
OUT = Path("results_comparison")
OUT.mkdir(exist_ok=True)


def load(model_dir, filename):
    with open(Path(model_dir) / filename) as f:
        return json.load(f)


# ── Load all data ─────────────────────────────────────────────────────────────

roll_data = {name: load(d, "roll_truncate.json") for name, d in MODELS.items()}
summary_data = {name: load(d, "summary.json") for name, d in MODELS.items()}


# ── Plot 1: Rolling K & V (all-position avg) across models ───────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for name, d in roll_data.items():
    fracs = [n / d["seq_len"] for n in d["ns"]]
    axes[0].plot(fracs, d["roll_k"], label=name, color=COLORS[name], lw=1.5, alpha=0.85)
    axes[1].plot(fracs, d["roll_v"], label=name, color=COLORS[name], lw=1.5, alpha=0.85)

for ax, kv in zip(axes, ["K", "V"]):
    ax.set_xlabel("Fraction of context rolled")
    ax.set_ylabel("Mean Cosine Similarity")
    ax.set_title(f"{kv} vectors (all-position avg)")
    ax.legend()
    ax.set_ylim(0, 1.05)

plt.suptitle("Rolling Context: Model Comparison", fontsize=14)
plt.tight_layout()
plt.savefig(OUT / "rolling_avg_comparison.png", dpi=150)
plt.close()
print("Saved rolling_avg_comparison.png")


# ── Plot 2: Terminal state (last-pos) rolling across models ──────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for name, d in roll_data.items():
    fracs = [n / d["seq_len"] for n in d["ns"]]
    axes[0].plot(fracs, d["roll_last_k"], label=name,
                 color=COLORS[name], lw=1.2, alpha=0.7)
    axes[1].plot(fracs, d["roll_last_v"], label=name,
                 color=COLORS[name], lw=1.2, alpha=0.7)

for ax, kv in zip(axes, ["K", "V"]):
    ax.set_xlabel("Fraction of context rolled")
    ax.set_ylabel("Cosine Similarity (last position)")
    ax.set_title(f"{kv} vectors (terminal state)")
    ax.legend()
    ax.set_ylim(0, 1.05)

plt.suptitle("Terminal Generating State Under Rolling: Model Comparison", fontsize=14)
plt.tight_layout()
plt.savefig(OUT / "rolling_terminal_comparison.png", dpi=150)
plt.close()
print("Saved rolling_terminal_comparison.png")


# ── Plot 3: Roll vs Truncate terminal state, all models in one ───────────────

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for name, d in roll_data.items():
    fracs = [n / d["seq_len"] for n in d["ns"]]
    c = COLORS[name]
    axes[0][0].plot(fracs, d["roll_last_k"], label=f"{name} roll",
                    color=c, lw=1.2, alpha=0.7)
    axes[0][0].plot(fracs, d["trunc_last_k"], label=f"{name} trunc",
                    color=c, lw=1.2, alpha=0.35, ls="--")
    axes[0][1].plot(fracs, d["roll_last_v"], label=f"{name} roll",
                    color=c, lw=1.2, alpha=0.7)
    axes[0][1].plot(fracs, d["trunc_last_v"], label=f"{name} trunc",
                    color=c, lw=1.2, alpha=0.35, ls="--")
    axes[1][0].plot(fracs, d["roll_k"], label=f"{name} roll",
                    color=c, lw=1.2, alpha=0.7)
    axes[1][0].plot(fracs, d["trunc_k"], label=f"{name} trunc",
                    color=c, lw=1.2, alpha=0.35, ls="--")
    axes[1][1].plot(fracs, d["roll_v"], label=f"{name} roll",
                    color=c, lw=1.2, alpha=0.7)
    axes[1][1].plot(fracs, d["trunc_v"], label=f"{name} trunc",
                    color=c, lw=1.2, alpha=0.35, ls="--")

titles = [
    ("K -- terminal state", "V -- terminal state"),
    ("K -- all-position avg", "V -- all-position avg"),
]
for row in range(2):
    for col in range(2):
        ax = axes[row][col]
        ax.set_xlabel("Fraction removed")
        ax.set_ylabel("Cosine Similarity")
        ax.set_title(titles[row][col])
        ax.legend(fontsize=7, ncol=2)
        ax.set_ylim(-0.05, 1.05)

plt.suptitle("Roll vs Truncate: All Models", fontsize=14)
plt.tight_layout()
plt.savefig(OUT / "roll_trunc_all_models.png", dpi=150)
plt.close()
print("Saved roll_trunc_all_models.png")


# ── Plot 4: Compaction — centroid V (the key finding) across models ──────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for col, kv in enumerate(["k", "v"]):
    ax = axes[col]
    x = np.arange(len(summary_data[list(MODELS.keys())[0]]))
    width = 0.25
    offsets = [-width, 0, width]

    for i, (name, items) in enumerate(summary_data.items()):
        comp_vals = [it[f"summary_cent_{kv}"] for it in items]
        unrel_vals = [np.mean(it[f"unrel_cent_{kv}"]) for it in items]
        ax.bar(x + offsets[i] - 0.06, comp_vals, width * 0.45,
               label=f"{name} compacted", color=COLORS[name], alpha=0.9)
        ax.bar(x + offsets[i] + 0.06, unrel_vals, width * 0.45,
               label=f"{name} unrelated", color=COLORS[name], alpha=0.35,
               hatch="//")

    ax.set_xlabel("Conversation")
    ax.set_ylabel("Centroid Cosine Similarity")
    ax.set_title(f"{kv.upper()} vectors -- centroid")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i + 1) for i in x], fontsize=8)
    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=6, ncol=2)

plt.suptitle("Compaction: Centroid Similarity Across Models", fontsize=14)
plt.tight_layout()
plt.savefig(OUT / "compaction_centroid_comparison.png", dpi=150)
plt.close()
print("Saved compaction_centroid_comparison.png")


# ── Plot 5: Compaction summary — aggregated bar chart ────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
model_names = list(MODELS.keys())
x = np.arange(len(model_names))
width = 0.3

for col, kv in enumerate(["k", "v"]):
    ax = axes[col]
    comp_means = []
    unrel_means = []
    for name in model_names:
        items = summary_data[name]
        comp_means.append(np.mean([it[f"summary_cent_{kv}"] for it in items]))
        unrel_means.append(np.mean([np.mean(it[f"unrel_cent_{kv}"]) for it in items]))

    bars1 = ax.bar(x - width / 2, comp_means, width, label="Compacted",
                   color=[COLORS[n] for n in model_names], alpha=0.9)
    bars2 = ax.bar(x + width / 2, unrel_means, width, label="Unrelated",
                   color=[COLORS[n] for n in model_names], alpha=0.35,
                   hatch="//")

    ax.set_xlabel("Model")
    ax.set_ylabel("Mean Centroid Cosine Similarity")
    ax.set_title(f"{kv.upper()} vectors")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    if kv == "v":
        ax.set_ylim(0, 1.0)

plt.suptitle("Compaction Effectiveness: Mean Centroid Similarity", fontsize=14)
plt.tight_layout()
plt.savefig(OUT / "compaction_summary_comparison.png", dpi=150)
plt.close()
print("Saved compaction_summary_comparison.png")


# ── Plot 6: Per-layer centroid V for compacted (avg across conversations) ────

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for col, kv in enumerate(["k", "v"]):
    ax = axes[col]
    for name, items in summary_data.items():
        # Average per-layer centroid across all conversations
        all_layers = np.array([it[f"summary_cent_{kv}_layers"] for it in items])
        mean_layers = all_layers.mean(axis=0)
        ax.plot(mean_layers, label=name, color=COLORS[name], lw=2)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Centroid Cosine Similarity")
    ax.set_title(f"{kv.upper()} -- compacted centroid per layer")
    ax.legend()

plt.suptitle("Compaction Centroid Similarity Per Layer (avg over conversations)",
             fontsize=14)
plt.tight_layout()
plt.savefig(OUT / "compaction_per_layer_comparison.png", dpi=150)
plt.close()
print("Saved compaction_per_layer_comparison.png")

print(f"\nAll comparison plots saved to {OUT}/")
