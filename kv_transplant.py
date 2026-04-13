#!/usr/bin/env python3
"""
Qualitative KV cache transplant experiments.

Exp 1: Cross-model -- prefill with 14B base, generate with 14B instruct
Exp 2: K/V swap -- use K from one context and V from another
Exp 3: Cross-text K/V swap -- AI text K with pasta text V and vice versa
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

PROMPTS = [
    (
        "The history of artificial intelligence began in antiquity, with myths, "
        "stories and rumors of artificial beings endowed with intelligence or "
        "consciousness by master craftsmen. The seeds of modern AI were planted "
        "by philosophers who attempted to describe the process of human thinking "
        "as the mechanical manipulation of symbols. This work culminated in the "
        "invention of the programmable digital computer in the 1940s, a machine "
        "based on the abstract essence of mathematical reasoning."
    ),
    (
        "User: What are the three laws of thermodynamics? Explain each one "
        "briefly.\nAssistant:"
    ),
]

ALT_TEXT = (
    "Fresh pasta dough requires only flour, eggs, and a pinch of salt. "
    "Knead the mixture on a wooden board until smooth and elastic, then "
    "let it rest under a damp cloth for thirty minutes. Roll it thin with "
    "a long pin, dusting frequently to prevent sticking."
)


def load_model(name, device, dtype):
    print(f"  Loading {name}...")
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True)
    model.to(device)
    model.eval()
    return model, tok


def prefill(model, input_ids, device):
    ids = input_ids.unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
    return out.past_key_values, out.logits[0, -1, :]


def gen(model, tokenizer, cache, first_logits, max_new=80):
    device = next(model.parameters()).device
    tokens = []
    next_id = first_logits.argmax().item()
    tokens.append(next_id)
    cur = cache
    for _ in range(max_new - 1):
        inp = torch.tensor([[next_id]], device=device)
        with torch.no_grad():
            out = model(input_ids=inp, past_key_values=cur, use_cache=True)
        cur = out.past_key_values
        next_id = out.logits[0, -1, :].argmax().item()
        tokens.append(next_id)
        if next_id == tokenizer.eos_token_id:
            break
    return tokenizer.decode(tokens)


def _get_kv(cache, i):
    """Get (key, value) tensors for layer i, handling API differences."""
    if hasattr(cache, "key_cache"):
        return cache.key_cache[i], cache.value_cache[i]
    return cache.layers[i].keys, cache.layers[i].values


def _n_layers(cache):
    if hasattr(cache, "key_cache"):
        return len(cache.key_cache)
    return len(cache.layers)


def _build_cache(kv_pairs):
    """Build a DynamicCache from list of (key, value) tensor pairs."""
    c = DynamicCache()
    for i, (k, v) in enumerate(kv_pairs):
        # First call to update initializes the layer
        c.update(k, v, i)
    return c


def _build_cache_direct(kv_pairs):
    """Build cache by directly setting layer tensors (avoids concat issues)."""
    c = DynamicCache()
    for i, (k, v) in enumerate(kv_pairs):
        # Initialize layer with a zero-length tensor, then overwrite
        dummy_k = k[:, :, :0, :]
        dummy_v = v[:, :, :0, :]
        c.update(dummy_k, dummy_v, i)
        c.layers[i].keys = k
        c.layers[i].values = v
    return c


def hybrid(k_cache, v_cache):
    pairs = []
    for i in range(_n_layers(k_cache)):
        k, _ = _get_kv(k_cache, i)
        _, v = _get_kv(v_cache, i)
        # Ensure same sequence length
        min_seq = min(k.shape[2], v.shape[2])
        k = k[:, :, :min_seq, :].clone()
        v = v[:, :, :min_seq, :].clone()
        pairs.append((k, v))
    return _build_cache_direct(pairs)


def to_cpu(cache):
    pairs = []
    for i in range(_n_layers(cache)):
        k, v = _get_kv(cache, i)
        pairs.append((k.cpu(), v.cpu()))
    return _build_cache_direct(pairs)


def to_dev(cache, device):
    pairs = []
    for i in range(_n_layers(cache)):
        k, v = _get_kv(cache, i)
        pairs.append((k.to(device), v.to(device)))
    return _build_cache_direct(pairs)


def trim(cache, start, end):
    pairs = []
    for i in range(_n_layers(cache)):
        k, v = _get_kv(cache, i)
        pairs.append((k[:, :, start:end, :].clone(), v[:, :, start:end, :].clone()))
    return _build_cache_direct(pairs)


def run_cross_model(device, dtype, out_dir):
    print("\n" + "=" * 60)
    print("Exp 1: Cross-Model KV Transplant")
    print("  14B Base cache -> 14B Instruct generation")
    print("=" * 60)

    results = []

    # Base model: compute caches
    model_b, tok = load_model("Qwen/Qwen2.5-14B", device, dtype)
    base_caches = []
    base_logits_list = []
    for prompt in PROMPTS:
        ids = torch.tensor(tok.encode(prompt))
        c, l = prefill(model_b, ids, device)
        base_caches.append(to_cpu(c))
        base_logits_list.append(l.cpu())

    print("\n--- Base model, own cache ---")
    for i, prompt in enumerate(PROMPTS):
        cg = to_dev(base_caches[i], device)
        text = gen(model_b, tok, cg, base_logits_list[i].to(device))
        print(f"\n  Prompt {i+1}: ...{prompt[-60:]}")
        print(f"  >> {text[:200]}")
        results.append({"prompt": prompt, "base_gen": text})

    del model_b
    torch.cuda.empty_cache()

    # Instruct model
    model_i, tok_i = load_model("Qwen/Qwen2.5-14B-Instruct", device, dtype)

    print("\n--- Instruct model, BASE cache (transplant) ---")
    for i, prompt in enumerate(PROMPTS):
        cg = to_dev(base_caches[i], device)
        text = gen(model_i, tok_i, cg, base_logits_list[i].to(device))
        print(f"\n  Prompt {i+1}: ...{prompt[-60:]}")
        print(f"  >> {text[:200]}")
        results[i]["transplant_gen"] = text

    # Instruct's own cache for baseline
    print("\n--- Instruct model, own cache ---")
    inst_caches = []
    inst_logits_list = []
    for i, prompt in enumerate(PROMPTS):
        ids = torch.tensor(tok_i.encode(prompt))
        c, l = prefill(model_i, ids, device)
        text = gen(model_i, tok_i, c, l)
        print(f"\n  Prompt {i+1}: ...{prompt[-60:]}")
        print(f"  >> {text[:200]}")
        results[i]["instruct_gen"] = text
        inst_caches.append(to_cpu(c))
        inst_logits_list.append(l.cpu())

    del model_i
    torch.cuda.empty_cache()

    # Reverse transplant: instruct cache -> base generation
    model_b2, tok2 = load_model("Qwen/Qwen2.5-14B", device, dtype)

    print("\n--- Base model, INSTRUCT cache (reverse transplant) ---")
    for i, prompt in enumerate(PROMPTS):
        cg = to_dev(inst_caches[i], device)
        text = gen(model_b2, tok2, cg, inst_logits_list[i].to(device))
        print(f"\n  Prompt {i+1}: ...{prompt[-60:]}")
        print(f"  >> {text[:200]}")
        results[i]["reverse_transplant_gen"] = text

    del model_b2
    torch.cuda.empty_cache()
    return results


def undo_rope(k_tensor, head_dim, base=1000000.0):
    """Remove RoPE rotation from K vectors.

    k_tensor: (batch, heads, seq_len, head_dim)
    Returns de-rotated K of same shape.
    """
    seq_len = k_tensor.shape[2]
    dim = head_dim
    # Compute inverse rotation angles
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    positions = torch.arange(seq_len, dtype=torch.float32)
    # angles: (seq_len, dim/2)
    angles = torch.outer(positions, inv_freq)
    # For inverse rotation, negate the angles
    cos_inv = torch.cos(-angles).to(k_tensor.device, dtype=k_tensor.dtype)
    sin_inv = torch.sin(-angles).to(k_tensor.device, dtype=k_tensor.dtype)

    # Apply inverse rotation pairwise to dimensions
    k = k_tensor.float()
    k1 = k[..., 0::2]  # even dims
    k2 = k[..., 1::2]  # odd dims
    k_derotated = torch.zeros_like(k)
    k_derotated[..., 0::2] = k1 * cos_inv - k2 * sin_inv
    k_derotated[..., 1::2] = k1 * sin_inv + k2 * cos_inv
    return k_derotated


def run_rope_correction(device, dtype, out_dir):
    """Compare K vectors after undoing RoPE rotation."""
    print("\n" + "=" * 60)
    print("Exp 4: RoPE Correction")
    print("  Undo positional rotation, compare residual K difference")
    print("=" * 60)

    model, tok = load_model("Qwen/Qwen2.5-14B", device, dtype)
    prompt = PROMPTS[0]
    ids = torch.tensor(tok.encode(prompt))
    seq_len = len(ids)

    cache_orig, _ = prefill(model, ids, device)
    k0, _ = _get_kv(cache_orig, 0)
    head_dim = k0.shape[-1]

    results = []

    for n in [1, 3, 5, 10, 20]:
        rolled_ids = ids[n:]
        cache_roll, _ = prefill(model, rolled_ids, device)
        overlap = seq_len - n

        cos_k_raw = []
        cos_k_derotated = []

        for layer in range(_n_layers(cache_orig)):
            # Raw K comparison (with RoPE)
            k_orig_l, _ = _get_kv(cache_orig, layer)
            k_roll_l, _ = _get_kv(cache_roll, layer)
            k_orig = k_orig_l[0, :, n:, :]  # (heads, overlap, dim)
            k_roll = k_roll_l[0, :, :overlap, :]

            raw_cos = torch.nn.functional.cosine_similarity(
                k_orig, k_roll, dim=-1).mean().item()
            cos_k_raw.append(raw_cos)

            # De-rotate both
            k_orig_full = k_orig_l[:, :, n:, :]
            k_roll_full = k_roll_l[:, :, :overlap, :]

            k_orig_derot = undo_rope(k_orig_full, head_dim)
            k_roll_derot = undo_rope(k_roll_full, head_dim)

            derot_cos = torch.nn.functional.cosine_similarity(
                k_orig_derot[0], k_roll_derot[0], dim=-1).mean().item()
            cos_k_derotated.append(derot_cos)

        mean_raw = sum(cos_k_raw) / len(cos_k_raw)
        mean_derot = sum(cos_k_derotated) / len(cos_k_derotated)
        print(f"  Roll {n:2d}: raw_K_cos={mean_raw:.6f}  "
              f"derotated_K_cos={mean_derot:.6f}  "
              f"diff={mean_derot - mean_raw:+.6f}")
        results.append({
            "roll_n": n,
            "raw_k_cos_per_layer": cos_k_raw,
            "derotated_k_cos_per_layer": cos_k_derotated,
            "raw_k_cos_mean": mean_raw,
            "derotated_k_cos_mean": mean_derot,
        })

    del model
    torch.cuda.empty_cache()

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart: raw vs derotated
    ns = [r["roll_n"] for r in results]
    raw = [r["raw_k_cos_mean"] for r in results]
    derot = [r["derotated_k_cos_mean"] for r in results]
    x = np.arange(len(ns))

    ax = axes[0]
    ax.bar(x - 0.15, raw, 0.3, label="Raw K (with RoPE)", color="steelblue")
    ax.bar(x + 0.15, derot, 0.3, label="De-rotated K (RoPE removed)",
           color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("Tokens rolled")
    ax.set_ylabel("Mean Cosine Similarity")
    ax.set_title("K Similarity: Raw vs RoPE-Corrected")
    ax.legend()
    ax.set_ylim(0, 1.05)

    # Per-layer for roll_5
    ax = axes[1]
    r5 = [r for r in results if r["roll_n"] == 5][0]
    ax.plot(r5["raw_k_cos_per_layer"], label="Raw K", color="steelblue", lw=2)
    ax.plot(r5["derotated_k_cos_per_layer"], label="De-rotated K",
            color="darkorange", lw=2)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Per-Layer K Similarity (roll=5)")
    ax.legend()

    plt.suptitle("RoPE Correction: Is K Difference Just Positional?", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "rope_correction.png", dpi=150)
    plt.close()

    return results


def run_kv_swap(device, dtype, out_dir):
    print("\n" + "=" * 60)
    print("Exp 2: K/V Swap -- Rolling")
    print("=" * 60)

    model, tok = load_model("Qwen/Qwen2.5-14B", device, dtype)
    prompt = PROMPTS[0]
    ids = torch.tensor(tok.encode(prompt))
    seq_len = len(ids)

    cache_orig, logits_orig = prefill(model, ids, device)
    text_orig = gen(model, tok, cache_orig, logits_orig)
    print(f"\n  Original: {text_orig[:200]}")
    results = [{"condition": "original", "text": text_orig}]

    for n in [5, 10, 20]:
        rolled_ids = ids[n:]
        cache_roll, logits_roll = prefill(model, rolled_ids, device)
        rolled_len = seq_len - n

        text_roll = gen(model, tok, cache_roll, logits_roll)
        print(f"\n  Rolled-{n}: {text_roll[:200]}")
        results.append({"condition": f"rolled_{n}", "text": text_roll})

        # Trim original to match rolled length
        cache_orig_t = trim(cache_orig, n, seq_len)

        # K_orig + V_rolled
        h1 = hybrid(cache_orig_t, cache_roll)
        t1 = gen(model, tok, h1, logits_roll)
        print(f"  K_orig + V_roll{n}: {t1[:200]}")
        results.append({"condition": f"K_orig_V_rolled{n}", "text": t1})

        # K_rolled + V_orig
        h2 = hybrid(cache_roll, cache_orig_t)
        t2 = gen(model, tok, h2, logits_orig)
        print(f"  K_roll{n} + V_orig: {t2[:200]}")
        results.append({"condition": f"K_rolled{n}_V_orig", "text": t2})

    # V swap for truncation (both directions)
    print("\n" + "=" * 60)
    print("Exp 2b: V Swap -- Truncation")
    print("  V from shorter context in longer, and vice versa")
    print("=" * 60)

    for n in [5, 10, 20]:
        trunc_ids = ids[:-n]
        cache_trunc, logits_trunc = prefill(model, trunc_ids, device)
        trunc_len = seq_len - n

        text_trunc = gen(model, tok, cache_trunc, logits_trunc)
        print(f"\n  Truncated-{n}: {text_trunc[:200]}")
        results.append({"condition": f"truncated_{n}", "text": text_trunc})

        # V from truncated -> full context (pad with orig K for last n positions)
        # Use first (seq_len-n) positions: V from truncated, K from original
        cache_orig_prefix = trim(cache_orig, 0, trunc_len)
        h_trunc_v = hybrid(cache_orig_prefix, cache_trunc)
        t_tv = gen(model, tok, h_trunc_v, logits_orig)
        print(f"  K_orig[:{trunc_len}] + V_trunc{n}: {t_tv[:200]}")
        results.append({
            "condition": f"K_orig_prefix_V_truncated{n}", "text": t_tv})

        # V from full context -> truncated (use first trunc_len of orig V)
        cache_orig_v_prefix = trim(cache_orig, 0, trunc_len)
        # K from truncated, V from original (first trunc_len positions)
        h_orig_v = hybrid(cache_trunc, cache_orig_v_prefix)
        t_ov = gen(model, tok, h_orig_v, logits_trunc)
        print(f"  K_trunc{n} + V_orig[:{trunc_len}]: {t_ov[:200]}")
        results.append({
            "condition": f"K_truncated{n}_V_orig_prefix", "text": t_ov})

    # Exp 3: Completely different texts
    print("\n" + "=" * 60)
    print("Exp 3: K/V Swap -- Different Texts (AI vs Pasta)")
    print("=" * 60)

    ids_a = torch.tensor(tok.encode(PROMPTS[0]))
    ids_b = torch.tensor(tok.encode(ALT_TEXT))
    ml = min(len(ids_a), len(ids_b))
    ids_a, ids_b = ids_a[:ml], ids_b[:ml]

    ca, la = prefill(model, ids_a, device)
    cb, lb = prefill(model, ids_b, device)

    ta = gen(model, tok, ca, la)
    tb = gen(model, tok, cb, lb)
    print(f"\n  AI text: {ta[:200]}")
    print(f"  Pasta text: {tb[:200]}")
    results.append({"condition": "AI_only", "text": ta})
    results.append({"condition": "pasta_only", "text": tb})

    # K_AI + V_pasta
    h_av = hybrid(ca, cb)
    t_av = gen(model, tok, h_av, la)
    print(f"\n  K_AI + V_pasta: {t_av[:200]}")
    results.append({"condition": "K_AI_V_pasta", "text": t_av})

    # K_pasta + V_AI
    h_ba = hybrid(cb, ca)
    t_ba = gen(model, tok, h_ba, lb)
    print(f"  K_pasta + V_AI: {t_ba[:200]}")
    results.append({"condition": "K_pasta_V_AI", "text": t_ba})

    del model
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--output", default="results_transplant")
    parser.add_argument("--skip-cross-model", action="store_true")
    parser.add_argument("--skip-kv-swap", action="store_true")
    parser.add_argument("--skip-rope", action="store_true")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]

    out = Path(args.output)
    out.mkdir(exist_ok=True)
    all_results = {}

    if not args.skip_rope:
        all_results["rope_correction"] = run_rope_correction(device, dtype, out)
    if not args.skip_cross_model:
        all_results["cross_model"] = run_cross_model(device, dtype, out)
    if not args.skip_kv_swap:
        all_results["kv_swap"] = run_kv_swap(device, dtype, out)

    with open(out / "transplant_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {out}/")


if __name__ == "__main__":
    main()
