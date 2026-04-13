#!/usr/bin/env python3
"""RoPE-corrected roll vs truncate (terminal state).

Replicates the terminal-state roll/truncate plot from kv_diff_extended.py,
but applies RoPE correction to K vectors. V vectors are unchanged.
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


def load_model(model_name, device, dtype):
    print(f"Loading {model_name} -> {device} ({dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def extract_kv(model, input_ids, device):
    ids = input_ids.unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
    kv = out.past_key_values
    layers = []
    if hasattr(kv, "key_cache"):
        for i in range(len(kv.key_cache)):
            layers.append((kv.key_cache[i].cpu().float(),
                           kv.value_cache[i].cpu().float()))
    else:
        for layer in kv:
            layers.append((layer[0].cpu().float(), layer[1].cpu().float()))
    return layers


def cos_sim(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)



def load_cached_conversations(cache_path):
    path = Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")
    with open(path) as f:
        convos = json.load(f)
    if not convos:
        raise RuntimeError(f"No conversations in cache: {cache_path}")
    return convos


def undo_rope(k_tensor, head_dim, base=1000000.0, pos_offset=0):
    """Remove RoPE rotation from K vectors.

    Uses the split-half convention matching Qwen2/transformers rotate_half:
    dimension pairs are (i, i + d/2), NOT interleaved (0,1),(2,3)...

    k_tensor: (batch, heads, seq_len, head_dim)
    Returns de-rotated K of same shape.
    """
    seq_len = k_tensor.shape[2]
    dim = head_dim
    half = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    positions = torch.arange(seq_len, dtype=torch.float32) + float(pos_offset)
    angles = torch.outer(positions, inv_freq)  # (seq_len, half)
    cos_val = torch.cos(angles).to(k_tensor.device, dtype=k_tensor.dtype)
    sin_val = torch.sin(angles).to(k_tensor.device, dtype=k_tensor.dtype)

    k = k_tensor.float()
    k_first = k[..., :half]   # dims 0..d/2-1
    k_second = k[..., half:]  # dims d/2..d-1
    k_derotated = torch.zeros_like(k)
    # Inverse rotation: R(-θ) applied to (k_first, k_second)
    k_derotated[..., :half] = k_first * cos_val + k_second * sin_val
    k_derotated[..., half:] = k_second * cos_val - k_first * sin_val
    return k_derotated


def apply_rope(k_tensor, head_dim, base=1000000.0, pos_offset=0):
    """Apply RoPE rotation to K vectors with a positional offset.

    Uses the split-half convention matching Qwen2/transformers rotate_half.
    """
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
    k_rot = torch.zeros_like(k)
    # Forward rotation: R(θ) applied to (k_first, k_second)
    k_rot[..., :half] = k_first * cos_val - k_second * sin_val
    k_rot[..., half:] = k_second * cos_val + k_first * sin_val
    return k_rot


def last_pos_k_rope_metrics(kv_a, kv_b, pos_a, pos_b):
    """Return (raw, derot, aligned) mean K cosine at a single position."""
    raw_vals = []
    derot_vals = []
    aligned_vals = []

    head_dim = kv_a[0][0].shape[-1]
    for i in range(len(kv_a)):
        ka = kv_a[i][0][0, :, pos_a, :]
        kb = kv_b[i][0][0, :, pos_b, :]
        raw_vals.append(cos_sim(ka, kb).mean().item())

        ka_der = undo_rope(ka[None, :, None, :], head_dim,
                           pos_offset=pos_a)[0, :, 0, :]
        kb_der = undo_rope(kb[None, :, None, :], head_dim,
                           pos_offset=pos_b)[0, :, 0, :]
        derot_vals.append(cos_sim(ka_der, kb_der).mean().item())

        kb_aligned = apply_rope(kb_der[None, :, None, :], head_dim,
                                pos_offset=pos_a)[0, :, 0, :]
        aligned_vals.append(cos_sim(ka, kb_aligned).mean().item())

    return (
        float(np.mean(raw_vals)),
        float(np.mean(derot_vals)),
        float(np.mean(aligned_vals)),
    )


def last_pos_v_cos(kv_a, kv_b, pos_a, pos_b):
    """Mean V cosine similarity at a single position."""
    vals = []
    for i in range(len(kv_a)):
        va = kv_a[i][1][0, :, pos_a, :]
        vb = kv_b[i][1][0, :, pos_b, :]
        vals.append(cos_sim(va, vb).mean().item())
    return float(np.mean(vals))


def experiment_roll_truncate_rope(model, tokenizer, device, out_dir,
                                  convos=None, target_tokens=2000, step=5,
                                  cache_path=None):
    if convos is None:
        if not cache_path:
            raise RuntimeError("cache_path required when convos is None")
        convos = load_cached_conversations(cache_path)
    if not convos:
        raise RuntimeError("No suitable conversations found")

    long_text = convos[0]["full"]
    ids = tokenizer.encode(long_text)[:target_tokens]
    long_text = tokenizer.decode(ids)
    base_ids = torch.tensor(tokenizer.encode(long_text))
    seq_len = len(base_ids)

    meta = {
        "source": f"{cache_path or 'convos'}[0].full",
        "available_tokens": convos[0].get("n_tokens", seq_len),
        "context_tokens": seq_len,
        "target_tokens": target_tokens,
    }

    print(f"\n{'='*60}")
    print(f"Experiment: Roll vs Truncate (RoPE-corrected K, step={step})")
    print(f"Sequence length: {seq_len} tokens")
    print(f"Source: {meta['source']}")
    print(f"{'='*60}")

    kv_base = extract_kv(model, base_ids, device)

    max_remove = seq_len - 10
    ns = list(range(1, max_remove + 1, step))

    roll_last_k_raw, roll_last_k_derot, roll_last_k_aligned = [], [], []
    trunc_last_k_raw, trunc_last_k_derot, trunc_last_k_aligned = [], [], []
    roll_last_v, trunc_last_v = [], []

    pos_base = seq_len - 1

    for n in ns:
        kv_r = extract_kv(model, base_ids[n:], device)
        kv_t = extract_kv(model, base_ids[:-n], device)

        pos_roll = seq_len - n - 1
        pos_trunc = seq_len - n - 1

        k_raw, k_derot, k_aligned = last_pos_k_rope_metrics(
            kv_base, kv_r, pos_base, pos_roll)
        roll_last_k_raw.append(k_raw)
        roll_last_k_derot.append(k_derot)
        roll_last_k_aligned.append(k_aligned)

        k_raw, k_derot, k_aligned = last_pos_k_rope_metrics(
            kv_base, kv_t, pos_base, pos_trunc)
        trunc_last_k_raw.append(k_raw)
        trunc_last_k_derot.append(k_derot)
        trunc_last_k_aligned.append(k_aligned)

        roll_last_v.append(last_pos_v_cos(kv_base, kv_r, pos_base, pos_roll))
        trunc_last_v.append(last_pos_v_cos(kv_base, kv_t, pos_base, pos_trunc))

        if n % 50 == 0 or n == ns[-1]:
            print(f"  n={n:3d}/{max_remove}: "
                  f"roll_k_derot={roll_last_k_derot[-1]:.4f} "
                  f"trunc_k_derot={trunc_last_k_derot[-1]:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.plot(ns, roll_last_k_derot, label="Roll (last pos)",
            color="steelblue", lw=1.2)
    ax.plot(ns, trunc_last_k_derot, label="Truncate (last pos)",
            color="seagreen", lw=1.2)
    ax.set_xlabel("Tokens removed")
    ax.set_ylabel("Cosine Similarity (last position only)")
    ax.set_title("K vectors -- terminal state (RoPE corrected)")
    ax.legend()
    ax.set_ylim(-0.05, 1.05)

    ax = axes[1]
    ax.plot(ns, roll_last_v, label="Roll (last pos)",
            color="steelblue", lw=1.2)
    ax.plot(ns, trunc_last_v, label="Truncate (last pos)",
            color="seagreen", lw=1.2)
    ax.set_xlabel("Tokens removed")
    ax.set_ylabel("Cosine Similarity (last position only)")
    ax.set_title("V vectors -- terminal state")
    ax.legend()
    ax.set_ylim(-0.05, 1.05)

    plt.suptitle(
        f"Terminal State: Rolling vs Truncating (RoPE-corrected K, {seq_len} tok)",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(out_dir / "roll_vs_truncate_last_rope.png", dpi=150)
    plt.close()

    return {
        "ns": ns,
        "seq_len": seq_len,
        "roll_last_k_raw": roll_last_k_raw,
        "roll_last_k_derot": roll_last_k_derot,
        "roll_last_k_aligned": roll_last_k_aligned,
        "trunc_last_k_raw": trunc_last_k_raw,
        "trunc_last_k_derot": trunc_last_k_derot,
        "trunc_last_k_aligned": trunc_last_k_aligned,
        "roll_last_v": roll_last_v,
        "trunc_last_v": trunc_last_v,
        "meta": meta,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--output", default="results_rope")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-tokens", type=int, default=2000)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--cache", default="data_cache.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]

    out = Path(args.output)
    out.mkdir(exist_ok=True)

    model, tokenizer = load_model(args.model, device, dtype)

    cache_path = str(out / args.cache)
    convos = load_cached_conversations(cache_path)

    res = experiment_roll_truncate_rope(
        model, tokenizer, device, out,
        convos=convos,
        target_tokens=args.target_tokens,
        step=args.step,
        cache_path=cache_path,
    )

    with open(out / "roll_truncate_rope.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"  Saved -> {out / 'roll_truncate_rope.json'}")

    print(f"\nAll results saved to {out}/")


if __name__ == "__main__":
    main()
