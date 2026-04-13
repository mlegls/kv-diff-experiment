#!/usr/bin/env python3
"""
KV Cache Diff Experiment

Measures how KV cache vectors change under different prompt modifications
for causal language models.

Conditions tested:
  1. Roll by N:   Remove first N tokens, shifting the rest to earlier positions.
                  Compares same tokens at their new vs old positions.
  2. Scrambled:   Same tokens in random order. Position-by-position comparison.
  3. Unrelated:   Completely different text of similar length.
  4. Truncated:   Same prompt with last token removed — sanity check.
                  KV should be identical since causal attention means
                  earlier positions don't depend on later tokens.

Metrics:
  - Cosine similarity (per layer, head, position)
  - Normalized L2 distance (per layer, head, position)
"""

import argparse
import json
import random
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

UNRELATED_TEXT = (
    "Fresh pasta dough requires only flour, eggs, and a pinch of salt. "
    "Knead the mixture on a wooden board until smooth and elastic, then "
    "let it rest under a damp cloth for thirty minutes. Roll it thin with "
    "a long pin, dusting frequently to prevent sticking. Cut into ribbons "
    "for fettuccine or squares for ravioli. Boil in generously salted water "
    "for just two to three minutes until al dente."
)


# ── Model loading ────────────────────────────────────────────────────────────

def load_model(model_name, device, dtype):
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print(f"Loading model -> {device} ({dtype})...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model = model.to(device)
    model.eval()
    return model, tokenizer


# ── KV extraction ────────────────────────────────────────────────────────────

def extract_kv(model, input_ids, device):
    """Forward pass -> list of (K, V) float32 CPU tensors, one per layer.

    Each tensor has shape (1, num_kv_heads, seq_len, head_dim).
    """
    ids = input_ids.unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)

    kv = out.past_key_values
    layers = []

    # DynamicCache (transformers >= 4.36)
    if hasattr(kv, "key_cache"):
        for i in range(len(kv.key_cache)):
            layers.append((kv.key_cache[i].cpu().float(),
                           kv.value_cache[i].cpu().float()))
    else:
        for layer in kv:
            layers.append((layer[0].cpu().float(), layer[1].cpu().float()))

    return layers


# ── Metrics ──────────────────────────────────────────────────────────────────

def cosine_sim(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


def l2_dist(a, b):
    return (a - b).norm(dim=-1) / (a.norm(dim=-1) + 1e-10)


def compare(kv_a, kv_b, pos_a, pos_b):
    """Compare two KV caches at the given position slices.

    Returns dict of numpy arrays, each (num_layers, num_kv_heads, num_positions).
    """
    out = {m: [] for m in ("cos_k", "cos_v", "l2_k", "l2_v")}
    for i in range(len(kv_a)):
        ka, va = kv_a[i][0][0, :, pos_a, :], kv_a[i][1][0, :, pos_a, :]
        kb, vb = kv_b[i][0][0, :, pos_b, :], kv_b[i][1][0, :, pos_b, :]
        out["cos_k"].append(cosine_sim(ka, kb).numpy())
        out["cos_v"].append(cosine_sim(va, vb).numpy())
        out["l2_k"].append(l2_dist(ka, kb).numpy())
        out["l2_v"].append(l2_dist(va, vb).numpy())
    return {k: np.stack(v) for k, v in out.items()}


def summarize(metrics):
    s = {}
    for k, arr in metrics.items():
        s[k] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "per_layer": arr.mean(axis=(1, 2)).tolist(),
            "per_position": arr.mean(axis=(0, 1)).tolist(),
        }
    return s


# ── Experiment ───────────────────────────────────────────────────────────────

def run(model, tokenizer, device):
    base_ids = torch.tensor(tokenizer.encode(BASELINE_TEXT))
    seq_len = len(base_ids)
    tokens = tokenizer.convert_ids_to_tokens(base_ids.tolist())

    print(f"\nBaseline: {seq_len} tokens")
    print(f"First 20: {' '.join(tokens[:20])} ...\n")

    print("Computing baseline KV cache...")
    kv_base = extract_kv(model, base_ids, device)
    n_layers = len(kv_base)
    n_heads = kv_base[0][0].shape[1]
    head_dim = kv_base[0][0].shape[3]
    print(f"Architecture: {n_layers} layers, {n_heads} KV heads, dim {head_dim}\n")

    results = {}

    # ── 1. Roll by N ─────────────────────────────────────────────────────
    print("=== Roll by N tokens (remove first N, shift rest) ===")
    for n in [1, 2, 3, 5, 10, 20]:
        if n >= seq_len - 5:
            continue
        rolled_ids = base_ids[n:]
        kv_r = extract_kv(model, rolled_ids, device)
        # Same tokens compared: position i in base <-> position i-n in rolled
        m = compare(kv_base, kv_r,
                    pos_a=slice(n, seq_len),
                    pos_b=slice(0, seq_len - n))
        s = summarize(m)
        results[f"roll_{n}"] = s
        print(f"  roll {n:2d}:  cos_k={s['cos_k']['mean']:.6f}  "
              f"cos_v={s['cos_v']['mean']:.6f}  "
              f"l2_k={s['l2_k']['mean']:.6f}  "
              f"l2_v={s['l2_v']['mean']:.6f}")

    # ── 2. Scrambled ─────────────────────────────────────────────────────
    print("\n=== Scrambled (same tokens, random order) ===")
    perm = torch.randperm(seq_len)
    scrambled_ids = base_ids[perm]
    kv_s = extract_kv(model, scrambled_ids, device)
    m = compare(kv_base, kv_s, slice(None), slice(None))
    results["scrambled"] = summarize(m)
    s = results["scrambled"]
    print(f"  scrambled: cos_k={s['cos_k']['mean']:.6f}  "
          f"cos_v={s['cos_v']['mean']:.6f}  "
          f"l2_k={s['l2_k']['mean']:.6f}  "
          f"l2_v={s['l2_v']['mean']:.6f}")

    # ── 3. Unrelated text ────────────────────────────────────────────────
    print("\n=== Unrelated text ===")
    unrel_ids = torch.tensor(tokenizer.encode(UNRELATED_TEXT))
    min_len = min(seq_len, len(unrel_ids))
    unrel_ids = unrel_ids[:min_len]
    kv_u = extract_kv(model, unrel_ids, device)
    m = compare(kv_base, kv_u, slice(0, min_len), slice(0, min_len))
    results["unrelated"] = summarize(m)
    s = results["unrelated"]
    print(f"  unrelated: cos_k={s['cos_k']['mean']:.6f}  "
          f"cos_v={s['cos_v']['mean']:.6f}  "
          f"l2_k={s['l2_k']['mean']:.6f}  "
          f"l2_v={s['l2_v']['mean']:.6f}")

    # ── 4. Truncated (sanity check) ──────────────────────────────────────
    print("\n=== Truncated -- last token removed (sanity check) ===")
    trunc_ids = base_ids[:-1]
    kv_t = extract_kv(model, trunc_ids, device)
    m = compare(kv_base, kv_t,
                pos_a=slice(0, seq_len - 1),
                pos_b=slice(0, seq_len - 1))
    results["truncated"] = summarize(m)
    s = results["truncated"]
    print(f"  truncated: cos_k={s['cos_k']['mean']:.6f}  "
          f"cos_v={s['cos_v']['mean']:.6f}  "
          f"l2_k={s['l2_k']['mean']:.6f}  "
          f"l2_v={s['l2_v']['mean']:.6f}")

    return results, n_layers, seq_len


# ── Plotting ─────────────────────────────────────────────────────────────────

def cond_color(c):
    if c.startswith("roll"):
        return "steelblue"
    if c == "scrambled":
        return "darkorange"
    if c == "unrelated":
        return "crimson"
    return "seagreen"


def plot(results, n_layers, seq_len, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    conds = list(results.keys())

    # ── 1. Summary bar chart ─────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for row, (prefix, label) in enumerate([
        ("cos", "Cosine Similarity"), ("l2", "Norm. L2 Distance")
    ]):
        for col, kv in enumerate(["k", "v"]):
            ax = axes[row][col]
            metric = f"{prefix}_{kv}"
            vals = [results[c][metric]["mean"] for c in conds]
            colors = [cond_color(c) for c in conds]
            ax.bar(range(len(conds)), vals, color=colors)
            ax.set_xticks(range(len(conds)))
            ax.set_xticklabels(conds, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel(label)
            ax.set_title(f"{label} -- {kv.upper()} vectors")
            if prefix == "cos":
                ax.set_ylim(0, 1.05)
                ax.axhline(1.0, color="gray", ls="--", alpha=0.4)
    plt.suptitle("KV Cache Distance -- Summary", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "summary.png", dpi=150)
    plt.close()

    # ── 2. Per-layer curves ──────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for row, (prefix, label) in enumerate([
        ("cos", "Cosine Similarity"), ("l2", "Norm. L2 Distance")
    ]):
        for col, kv in enumerate(["k", "v"]):
            ax = axes[row][col]
            metric = f"{prefix}_{kv}"
            for c in conds:
                ax.plot(results[c][metric]["per_layer"], label=c,
                        color=cond_color(c), marker=".", ms=3)
            ax.set_xlabel("Layer")
            ax.set_ylabel(label)
            ax.set_title(f"Per-Layer {label} -- {kv.upper()}")
            ax.legend(fontsize=7)
    plt.suptitle("KV Cache Distance -- Per Layer", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "per_layer.png", dpi=150)
    plt.close()

    # ── 3. Per-position for rolling ──────────────────────────────────────
    rolls = [c for c in conds if c.startswith("roll")]
    if rolls:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for col, kv in enumerate(["k", "v"]):
            ax = axes[col]
            for c in rolls:
                per_pos = results[c][f"cos_{kv}"]["per_position"]
                ax.plot(per_pos, label=c, alpha=0.8)
            ax.set_xlabel("Position (in overlapping region)")
            ax.set_ylabel("Cosine Similarity")
            ax.set_title(f"Per-Position Cosine Sim -- {kv.upper()} (Rolling)")
            ax.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "per_position_rolling.png", dpi=150)
        plt.close()

    # ── 4. Roll amount vs similarity ─────────────────────────────────────
    if rolls:
        roll_ns = [int(c.split("_")[1]) for c in rolls]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(roll_ns,
                [results[c]["cos_k"]["mean"] for c in rolls],
                "o-", label="K vectors", color="steelblue")
        ax.plot(roll_ns,
                [results[c]["cos_v"]["mean"] for c in rolls],
                "s-", label="V vectors", color="darkorange")
        ax.set_xlabel("Tokens rolled (prefix removed)")
        ax.set_ylabel("Mean Cosine Similarity")
        ax.set_title("KV Similarity vs Roll Amount")
        ax.legend()
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(out_dir / "roll_vs_similarity.png", dpi=150)
        plt.close()

    print(f"Plots saved to {out_dir}/")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="KV Cache Diff Experiment")
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B",
                        help="HuggingFace model (default: Qwen/Qwen2.5-14B)")
    parser.add_argument("--device", default="auto",
                        help="auto | cpu | mps | cuda")
    parser.add_argument("--dtype", default="float16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--output", default="results",
                        help="Output directory (default: results)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Resolve device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    print(f"Config: device={device}  dtype={args.dtype}  model={args.model}")

    model, tokenizer = load_model(args.model, device, dtype)
    results, n_layers, seq_len = run(model, tokenizer, device)

    # Save raw results
    out = Path(args.output)
    out.mkdir(exist_ok=True)
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults -> {out / 'results.json'}")

    plot(results, n_layers, seq_len, out)


if __name__ == "__main__":
    main()
