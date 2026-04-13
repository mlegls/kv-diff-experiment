#!/usr/bin/env python3
"""Plot RoPE-corrected roll/truncate K similarity (raw vs derotated)."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IN_PATH = Path("results_rope_2000/roll_truncate_rope.json")
OUT_DIR = Path("results_rope_2000")
OUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    data = json.loads(IN_PATH.read_text())
    ns = data["ns"]
    seq_len = data.get("seq_len", None)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Roll: raw vs derotated K
    ax = axes[0]
    ax.plot(ns, data["roll_last_k_raw"], label="Raw K (with RoPE)",
            color="steelblue", lw=1.2, alpha=0.8)
    ax.plot(ns, data["roll_last_k_derot"], label="De-rotated K (content only)",
            color="darkorange", lw=1.2)
    ax.plot(ns, data["roll_last_v"], label="V (no RoPE)",
            color="seagreen", lw=1.2, alpha=0.6)
    ax.set_xlabel("Tokens rolled")
    ax.set_ylabel("Cosine Similarity (last position)")
    title = "Rolling: Raw vs RoPE-Corrected K"
    if seq_len:
        title += f" ({seq_len} tok)"
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.05)

    # Truncate: raw vs derotated K
    ax = axes[1]
    ax.plot(ns, data["trunc_last_k_raw"], label="Raw K (with RoPE)",
            color="steelblue", lw=1.2, alpha=0.8)
    ax.plot(ns, data["trunc_last_k_derot"], label="De-rotated K (content only)",
            color="darkorange", lw=1.2)
    ax.plot(ns, data["trunc_last_v"], label="V (no RoPE)",
            color="seagreen", lw=1.2, alpha=0.6)
    ax.set_xlabel("Tokens truncated")
    ax.set_ylabel("Cosine Similarity (last position)")
    title = "Truncation: Raw vs RoPE-Corrected K"
    if seq_len:
        title += f" ({seq_len} tok)"
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.05)

    plt.suptitle("RoPE Correction: Terminal State K Similarity", fontsize=13)
    plt.tight_layout()

    out = OUT_DIR / "roll_vs_truncate_last_rope_k.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out.name}")


if __name__ == "__main__":
    main()
