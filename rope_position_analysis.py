#!/usr/bin/env python3
"""
RoPE correction: position-by-position similarity for rolled, scrambled, unrelated.

Shows raw K, de-rotated K, and V across all positions for:
  - Rolling (1, 5, 10, 20 tokens)
  - Scrambled (same tokens, random order)
  - Unrelated (different text entirely)

Uses the short 87-token AI history prompt.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

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
    "To make fresh pasta, combine two cups of all-purpose flour with three "
    "large eggs on a clean work surface. Knead the dough for about ten minutes "
    "until it becomes smooth and elastic. Let the dough rest for thirty minutes "
    "covered with a damp cloth. Roll out the dough as thin as possible and cut "
    "into desired shapes. Cook in boiling salted water for two to three minutes "
    "until the pasta floats to the surface."
)


def load_model(model_name, device, dtype):
    print(f"Loading {model_name} -> {device} ({dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def prefill(model, input_ids, device):
    ids = input_ids.unsqueeze(0).to(device) if input_ids.dim() == 1 else input_ids.to(device)
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
    return out.past_key_values, out.logits


def _get_kv(cache, i):
    if hasattr(cache, "key_cache"):
        return cache.key_cache[i], cache.value_cache[i]
    return cache.layers[i].keys, cache.layers[i].values


def _n_layers(cache):
    if hasattr(cache, "key_cache"):
        return len(cache.key_cache)
    return len(cache.layers)


def undo_rope(k_tensor, head_dim, base=1000000.0, pos_offset=0):
    """Remove RoPE using split-half convention (Qwen2/transformers rotate_half)."""
    seq_len = k_tensor.shape[2]
    dim = head_dim
    half = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    positions = torch.arange(seq_len, dtype=torch.float32) + float(pos_offset)
    angles = torch.outer(positions, inv_freq)
    cos_val = torch.cos(angles).to(k_tensor.device, dtype=k_tensor.dtype)
    sin_val = torch.sin(angles).to(k_tensor.device, dtype=k_tensor.dtype)

    k = k_tensor.float()
    k_first = k[..., :half]
    k_second = k[..., half:]
    k_derotated = torch.zeros_like(k)
    k_derotated[..., :half] = k_first * cos_val + k_second * sin_val
    k_derotated[..., half:] = k_second * cos_val - k_first * sin_val
    return k_derotated


def per_position_similarity(cache_a, cache_b, head_dim, overlap,
                            offset_a=0, offset_b=0, start_a=0, start_b=0):
    """Compute per-position (averaged over layers/heads) raw K, derotated K, and V cos sim."""
    n_layers = _n_layers(cache_a)
    raw_k_sum = np.zeros(overlap)
    derot_k_sum = np.zeros(overlap)
    v_sum = np.zeros(overlap)

    for layer in range(n_layers):
        ka, va = _get_kv(cache_a, layer)
        kb, vb = _get_kv(cache_b, layer)

        ka_s = ka[:, :, start_a:start_a + overlap, :]
        kb_s = kb[:, :, start_b:start_b + overlap, :]
        va_s = va[:, :, start_a:start_a + overlap, :]
        vb_s = vb[:, :, start_b:start_b + overlap, :]

        raw_k = torch.nn.functional.cosine_similarity(
            ka_s[0], kb_s[0], dim=-1).mean(dim=0).cpu().numpy()
        raw_k_sum += raw_k

        ka_derot = undo_rope(ka_s, head_dim, pos_offset=offset_a)
        kb_derot = undo_rope(kb_s, head_dim, pos_offset=offset_b)
        derot_k = torch.nn.functional.cosine_similarity(
            ka_derot[0], kb_derot[0], dim=-1).mean(dim=0).cpu().numpy()
        derot_k_sum += derot_k

        v_cos = torch.nn.functional.cosine_similarity(
            va_s[0], vb_s[0], dim=-1).mean(dim=0).cpu().numpy()
        v_sum += v_cos

    return (raw_k_sum / n_layers, derot_k_sum / n_layers, v_sum / n_layers)


def run(model, tokenizer, device, out_dir):
    base_ids = torch.tensor(tokenizer.encode(BASELINE_TEXT))
    seq_len = len(base_ids)
    print(f"Baseline: {seq_len} tokens")

    cache_orig, _ = prefill(model, base_ids, device)
    k0, _ = _get_kv(cache_orig, 0)
    head_dim = k0.shape[-1]
    print(f"  head_dim={head_dim}, layers={_n_layers(cache_orig)}")

    results = {}

    # --- Rolling ---
    for n in [1, 5, 10, 20]:
        rolled_ids = base_ids[n:]
        cache_roll, _ = prefill(model, rolled_ids, device)
        overlap = seq_len - n

        raw_k, derot_k, v = per_position_similarity(
            cache_orig, cache_roll, head_dim, overlap,
            offset_a=n, offset_b=0, start_a=n, start_b=0)

        key = f"roll_{n}"
        results[key] = {
            "raw_k": raw_k.tolist(),
            "derot_k": derot_k.tolist(),
            "v": v.tolist(),
        }
        print(f"  {key}: raw_k={raw_k.mean():.4f}  derot_k={derot_k.mean():.4f}  v={v.mean():.4f}")

    # --- Scrambled ---
    rng = np.random.default_rng(42)
    scrambled_ids = base_ids[rng.permutation(seq_len)].clone()
    cache_scram, _ = prefill(model, scrambled_ids, device)

    raw_k, derot_k, v = per_position_similarity(
        cache_orig, cache_scram, head_dim, seq_len,
        offset_a=0, offset_b=0, start_a=0, start_b=0)
    results["scrambled"] = {
        "raw_k": raw_k.tolist(),
        "derot_k": derot_k.tolist(),
        "v": v.tolist(),
    }
    print(f"  scrambled: raw_k={raw_k.mean():.4f}  derot_k={derot_k.mean():.4f}  v={v.mean():.4f}")

    # --- Unrelated ---
    unrel_ids = torch.tensor(tokenizer.encode(UNRELATED_TEXT))
    min_len = min(seq_len, len(unrel_ids))
    unrel_ids = unrel_ids[:min_len]
    cache_unrel, _ = prefill(model, unrel_ids, device)

    raw_k, derot_k, v = per_position_similarity(
        cache_orig, cache_unrel, head_dim, min_len,
        offset_a=0, offset_b=0, start_a=0, start_b=0)
    results["unrelated"] = {
        "raw_k": raw_k.tolist(),
        "derot_k": derot_k.tolist(),
        "v": v.tolist(),
        "overlap_len": min_len,
    }
    print(f"  unrelated: raw_k={raw_k.mean():.4f}  derot_k={derot_k.mean():.4f}  v={v.mean():.4f}")

    # --- Save ---
    save = {"seq_len": seq_len, "head_dim": head_dim, "conditions": results}
    with open(out_dir / "rope_position_analysis.json", "w") as f:
        json.dump(save, f, indent=2)

    # --- Plot ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Top row: rolling conditions
    for col, n in enumerate([1, 5, 20]):
        ax = axes[0][col]
        r = results[f"roll_{n}"]
        positions = np.arange(len(r["raw_k"]))
        ax.plot(positions, r["raw_k"], label="Raw K", color="steelblue",
                lw=0.8, alpha=0.7)
        ax.plot(positions, r["derot_k"], label="De-rotated K", color="darkorange",
                lw=0.8, alpha=0.7)
        ax.plot(positions, r["v"], label="V", color="seagreen",
                lw=0.8, alpha=0.7)
        ax.set_title(f"Roll {n}")
        ax.set_xlabel("Position")
        ax.set_ylabel("Cosine Similarity")
        ax.legend(fontsize=7)
        ax.set_ylim(-0.1, 1.1)

    # Bottom row: scrambled, unrelated, and summary bar chart
    for col, (key, title) in enumerate([("scrambled", "Scrambled"),
                                         ("unrelated", "Unrelated")]):
        ax = axes[1][col]
        r = results[key]
        positions = np.arange(len(r["raw_k"]))
        ax.plot(positions, r["raw_k"], label="Raw K", color="steelblue",
                lw=0.8, alpha=0.7)
        ax.plot(positions, r["derot_k"], label="De-rotated K", color="darkorange",
                lw=0.8, alpha=0.7)
        ax.plot(positions, r["v"], label="V", color="seagreen",
                lw=0.8, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("Position")
        ax.set_ylabel("Cosine Similarity")
        ax.legend(fontsize=7)
        ax.set_ylim(-0.1, 1.1)

    # Summary bar chart
    ax = axes[1][2]
    conditions = ["roll_1", "roll_5", "roll_20", "scrambled", "unrelated"]
    labels = ["Roll 1", "Roll 5", "Roll 20", "Scrambled", "Unrelated"]
    raw_means = [np.mean(results[c]["raw_k"]) for c in conditions]
    derot_means = [np.mean(results[c]["derot_k"]) for c in conditions]
    v_means = [np.mean(results[c]["v"]) for c in conditions]
    x = np.arange(len(conditions))
    w = 0.25
    ax.bar(x - w, raw_means, w, label="Raw K", color="steelblue")
    ax.bar(x, derot_means, w, label="De-rotated K", color="darkorange")
    ax.bar(x + w, v_means, w, label="V", color="seagreen")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=15)
    ax.set_ylabel("Mean Cosine Similarity")
    ax.set_title("Summary")
    ax.legend(fontsize=7)
    ax.set_ylim(0, 1.1)

    plt.suptitle(f"RoPE Correction: Per-Position Similarity ({seq_len} tokens)", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "rope_position_analysis.png", dpi=150)
    plt.close()
    print(f"\nSaved to {out_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--output", default="results_transplant")
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
