#!/usr/bin/env python3
"""
Attention pattern comparison under rolling.

Compares attention weights and downstream outputs between:
  - Original sequence
  - Rolled sequences (first N tokens removed)

Tests whether K vector changes from rolling actually affect attention routing,
or are cosmetic (positional encoding shifts that cancel out in Q*K^T).
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASELINE_TEXT = (
    "The history of artificial intelligence began in antiquity, with myths, "
    "stories and rumors of artificial beings endowed with intelligence or "
    "consciousness by master craftsmen. The seeds of modern AI were planted by "
    "philosophers who attempted to describe the process of human thinking as "
    "the mechanical manipulation of symbols. This work culminated in the "
    "invention of the programmable digital computer in the 1940s, a machine "
    "based on the abstract essence of mathematical reasoning."
)


def load_model(model_name, device, dtype):
    print(f"Loading {model_name} -> {device} ({dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True,
        attn_implementation="eager")
    model = model.to(device)
    model.eval()
    return model, tokenizer


def forward_with_attention(model, input_ids, device):
    """Returns (attentions, logits). attentions: list of (1, heads, seq, seq)."""
    ids = input_ids.unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(input_ids=ids, output_attentions=True)
    return [a.cpu().float() for a in out.attentions], out.logits.cpu().float()


def last_pos_attn(attentions):
    """Attention at last query position: list of (heads, seq_len) per layer."""
    return [a[0, :, -1, :].numpy() for a in attentions]


def compare_attn(attn_a, attn_b):
    """Compare attention distributions at last position across layers.

    For rolling, attn_b has fewer positions. We compare the last min_len
    positions (which correspond to the same tokens in both).
    """
    n_layers = len(attn_a)
    js_divs = []
    top5_overlaps = []
    cosine_sims = []

    for layer in range(n_layers):
        a = attn_a[layer]  # (heads, seq_a)
        b = attn_b[layer]  # (heads, seq_b)
        min_len = min(a.shape[1], b.shape[1])
        n_heads = a.shape[0]

        js_h, top5_h, cos_h = [], [], []
        for h in range(n_heads):
            # Align: take last min_len from a, all of b (or last min_len)
            ah = a[h, -min_len:]
            bh = b[h, -min_len:]
            # Renormalize
            ah = ah / (ah.sum() + 1e-10)
            bh = bh / (bh.sum() + 1e-10)

            # JS divergence
            m = 0.5 * (ah + bh)
            kl_am = np.sum(np.where(ah > 1e-10, ah * np.log(ah / (m + 1e-10)), 0))
            kl_bm = np.sum(np.where(bh > 1e-10, bh * np.log(bh / (m + 1e-10)), 0))
            js_h.append(0.5 * (kl_am + kl_bm))

            # Top-5 overlap
            top5_a = set(np.argsort(ah)[-5:])
            top5_b = set(np.argsort(bh)[-5:])
            top5_h.append(len(top5_a & top5_b) / 5.0)

            # Cosine sim
            cos = np.dot(ah, bh) / (np.linalg.norm(ah) * np.linalg.norm(bh) + 1e-10)
            cos_h.append(cos)

        js_divs.append(np.mean(js_h))
        top5_overlaps.append(np.mean(top5_h))
        cosine_sims.append(np.mean(cos_h))

    return {"js_div": js_divs, "top5_overlap": top5_overlaps,
            "cosine_sim": cosine_sims}


def run(model, tokenizer, device, out_dir):
    base_ids = torch.tensor(tokenizer.encode(BASELINE_TEXT))
    seq_len = len(base_ids)
    print(f"Baseline: {seq_len} tokens")

    print("Computing baseline...")
    attn_base, logits_base = forward_with_attention(model, base_ids, device)
    last_base = last_pos_attn(attn_base)
    n_layers = len(last_base)
    n_heads = last_base[0].shape[0]
    print(f"Architecture: {n_layers} layers, {n_heads} attn heads")

    roll_ns = [1, 2, 3, 5, 10, 20]
    attn_results = {}
    logit_cos = []
    top5_token_overlap = []
    token_predictions = {}

    # Baseline top tokens
    lo_base = logits_base[0, -1, :]
    probs_base = torch.softmax(lo_base, dim=-1)
    top10_base = lo_base.topk(10)
    base_tokens = tokenizer.convert_ids_to_tokens(top10_base.indices.tolist())
    base_probs = probs_base[top10_base.indices].tolist()
    token_predictions["original"] = {
        "tokens": base_tokens,
        "probs": base_probs,
        "ids": top10_base.indices.tolist(),
    }
    print(f"  Original top-10: {list(zip(base_tokens, [f'{p:.4f}' for p in base_probs]))}")

    for n in roll_ns:
        print(f"  Rolling by {n}...", end=" ")
        rolled_ids = base_ids[n:]
        attn_r, logits_r = forward_with_attention(model, rolled_ids, device)
        last_r = last_pos_attn(attn_r)

        comp = compare_attn(last_base, last_r)
        attn_results[f"roll_{n}"] = comp

        # Logit comparison at last position
        lo = logits_base[0, -1, :]
        lr = logits_r[0, -1, :]
        lcos = torch.nn.functional.cosine_similarity(
            lo.unsqueeze(0), lr.unsqueeze(0)).item()
        logit_cos.append(lcos)

        # Top-10 token predictions for rolled
        probs_r = torch.softmax(lr, dim=-1)
        top10_r = lr.topk(10)
        rolled_tokens = tokenizer.convert_ids_to_tokens(top10_r.indices.tolist())
        rolled_probs = probs_r[top10_r.indices].tolist()
        token_predictions[f"roll_{n}"] = {
            "tokens": rolled_tokens,
            "probs": rolled_probs,
            "ids": top10_r.indices.tolist(),
        }

        top5_o = set(lo.topk(5).indices.tolist())
        top5_r = set(lr.topk(5).indices.tolist())
        t5o = len(top5_o & top5_r) / 5.0
        top5_token_overlap.append(t5o)

        print(f"attn_cos={np.mean(comp['cosine_sim']):.4f}  "
              f"js={np.mean(comp['js_div']):.6f}  "
              f"logit_cos={lcos:.4f}  top5_tok={t5o:.2f}")
        print(f"    Top-5: {list(zip(rolled_tokens[:5], [f'{p:.4f}' for p in rolled_probs[:5]]))}")

    # ── Plot 1: Attention metrics summary ────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, metric, label in [
        (axes[0], "cosine_sim", "Attention Cos Sim"),
        (axes[1], "top5_overlap", "Top-5 Attn Overlap"),
        (axes[2], "js_div", "JS Divergence"),
    ]:
        vals = [np.mean(attn_results[f"roll_{n}"][metric]) for n in roll_ns]
        ax.bar(range(len(roll_ns)), vals, color="steelblue")
        ax.set_xticks(range(len(roll_ns)))
        ax.set_xticklabels([str(n) for n in roll_ns])
        ax.set_xlabel("Tokens rolled")
        ax.set_ylabel(label)
        ax.set_title(label)

    ax = axes[3]
    ax.bar(range(len(roll_ns)), logit_cos, color="darkorange")
    ax.set_xticks(range(len(roll_ns)))
    ax.set_xticklabels([str(n) for n in roll_ns])
    ax.set_xlabel("Tokens rolled")
    ax.set_ylabel("Logit Cos Sim")
    ax.set_title("Next-Token Logit Similarity")
    ax.set_ylim(0, 1.05)

    plt.suptitle("Attention & Output Similarity Under Rolling", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "attention_summary.png", dpi=150)
    plt.close()

    # ── Plot 2: Per-layer attention cosine similarity ────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, metric, label in [
        (axes[0], "cosine_sim", "Attention Cos Sim"),
        (axes[1], "top5_overlap", "Top-5 Overlap"),
        (axes[2], "js_div", "JS Divergence"),
    ]:
        for n in roll_ns:
            ax.plot(attn_results[f"roll_{n}"][metric],
                    label=f"roll_{n}", marker=".", ms=3)
        ax.set_xlabel("Layer")
        ax.set_ylabel(label)
        ax.set_title(f"Per-Layer {label}")
        ax.legend(fontsize=7)
    plt.suptitle("Per-Layer Attention Similarity (last position)", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "attention_per_layer.png", dpi=150)
    plt.close()

    # ── Plot 3: Attention heatmap at select layers ───────────────────────
    sample_layers = [0, n_layers // 4, n_layers // 2,
                     3 * n_layers // 4, n_layers - 1]
    attn_r5, _ = forward_with_attention(model, base_ids[5:], device)
    last_r5 = last_pos_attn(attn_r5)

    fig, axes = plt.subplots(2, len(sample_layers),
                              figsize=(4 * len(sample_layers), 6))
    for col, li in enumerate(sample_layers):
        orig = last_base[li].mean(axis=0)
        rolled = last_r5[li].mean(axis=0)

        axes[0][col].bar(range(len(orig)), orig, color="steelblue", alpha=0.7)
        axes[0][col].set_title(f"L{li} original", fontsize=9)
        if col == 0:
            axes[0][col].set_ylabel("Attn weight")

        axes[1][col].bar(range(len(rolled)), rolled, color="darkorange",
                          alpha=0.7)
        axes[1][col].set_title(f"L{li} rolled-5", fontsize=9)
        axes[1][col].set_xlabel("Position")
        if col == 0:
            axes[1][col].set_ylabel("Attn weight")

    plt.suptitle("Attention at Last Position (head-averaged)", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_dir / "attention_heatmap.png", dpi=150)
    plt.close()

    # ── Plot 4: Logit & token overlap ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(roll_ns, logit_cos, "o-", color="steelblue", lw=2)
    axes[0].set_xlabel("Tokens rolled")
    axes[0].set_ylabel("Cosine Similarity")
    axes[0].set_title("Next-Token Logit Similarity")
    axes[0].set_ylim(0, 1.05)

    axes[1].plot(roll_ns, top5_token_overlap, "s-", color="darkorange", lw=2)
    axes[1].set_xlabel("Tokens rolled")
    axes[1].set_ylabel("Overlap fraction")
    axes[1].set_title("Top-5 Next Token Overlap")
    axes[1].set_ylim(0, 1.05)

    plt.suptitle("Downstream Impact of Rolling on Predictions", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "logit_comparison.png", dpi=150)
    plt.close()

    # ── Plot 5: Token prediction table ──────────────────────────────────
    conditions = ["original"] + [f"roll_{n}" for n in roll_ns]
    n_show = 5  # top-N tokens to show

    fig, ax = plt.subplots(figsize=(16, 1.5 + 0.6 * len(conditions)))
    ax.axis("off")

    col_labels = [f"#{i+1}" for i in range(n_show)] + ["logit cos"]
    row_labels = conditions
    cell_text = []
    cell_colors = []

    base_top_ids = set(token_predictions["original"]["ids"][:n_show])

    for cond in conditions:
        tp = token_predictions[cond]
        row = []
        row_color = []
        for i in range(n_show):
            tok = tp["tokens"][i]
            prob = tp["probs"][i]
            row.append(f"{tok}\n{prob:.3f}")
            # Green if matches original, red if not
            if cond == "original":
                row_color.append("#e8f5e9")
            elif tp["ids"][i] in base_top_ids:
                row_color.append("#e8f5e9")
            else:
                row_color.append("#ffebee")
        # Add logit cos column
        if cond == "original":
            row.append("1.000")
            row_color.append("white")
        else:
            idx = roll_ns.index(int(cond.split("_")[1]))
            row.append(f"{logit_cos[idx]:.4f}")
            row_color.append("white")
        cell_text.append(row)
        cell_colors.append(row_color)

    table = ax.table(cellText=cell_text, rowLabels=row_labels,
                     colLabels=col_labels, cellColours=cell_colors,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)

    plt.title("Top-5 Next Token Predictions Under Rolling\n"
              "(green = in original top-5, red = new)", fontsize=12, pad=20)
    plt.tight_layout()
    plt.savefig(out_dir / "token_predictions.png", dpi=150,
                bbox_inches="tight")
    plt.close()

    # Save
    save = {
        "roll_ns": roll_ns,
        "attention": {k: v for k, v in attn_results.items()},
        "logit_cos": logit_cos,
        "top5_token_overlap": top5_token_overlap,
        "token_predictions": token_predictions,
    }
    def default_ser(o):
        if isinstance(o, (np.floating, np.float32, np.float64)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Not serializable: {type(o)}")
    with open(out_dir / "attention_results.json", "w") as f:
        json.dump(save, f, indent=2, default=default_ser)

    print(f"\nAll saved to {out_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--output", default="results_attention")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]

    out = Path(args.output)
    out.mkdir(exist_ok=True)
    model, tokenizer = load_model(args.model, device, dtype)
    run(model, tokenizer, device, out)


if __name__ == "__main__":
    main()
