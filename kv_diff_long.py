#!/usr/bin/env python3
"""
Original KV cache diff experiment (roll/scramble/unrelated/truncate)
on a long text (from data_cache.json), storing full per-position
and per-layer data for FFT analysis.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
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


def cosine_sim(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


def compare(kv_a, kv_b, pos_a, pos_b):
    out = {m: [] for m in ("cos_k", "cos_v")}
    for i in range(len(kv_a)):
        ka, va = kv_a[i][0][0, :, pos_a, :], kv_a[i][1][0, :, pos_a, :]
        kb, vb = kv_b[i][0][0, :, pos_b, :], kv_b[i][1][0, :, pos_b, :]
        out["cos_k"].append(cosine_sim(ka, kb).numpy())
        out["cos_v"].append(cosine_sim(va, vb).numpy())
    return {k: np.stack(v) for k, v in out.items()}


def summarize(metrics):
    s = {}
    for k, arr in metrics.items():
        s[k] = {
            "mean": float(arr.mean()),
            "per_layer": arr.mean(axis=(1, 2)).tolist(),
            "per_position": arr.mean(axis=(0, 1)).tolist(),
        }
    return s


def run(model, tokenizer, device, text, target_tokens):
    base_ids = torch.tensor(tokenizer.encode(text)[:target_tokens])
    seq_len = len(base_ids)
    print(f"Baseline: {seq_len} tokens")

    print("Computing baseline KV cache...")
    kv_base = extract_kv(model, base_ids, device)
    n_layers = len(kv_base)
    n_heads = kv_base[0][0].shape[1]
    print(f"Architecture: {n_layers} layers, {n_heads} KV heads")

    results = {}

    # Roll
    print("\n=== Rolling ===")
    for n in [1, 2, 3, 5, 10, 20, 50, 100, 200]:
        if n >= seq_len - 10:
            continue
        rolled_ids = base_ids[n:]
        kv_r = extract_kv(model, rolled_ids, device)
        m = compare(kv_base, kv_r,
                    pos_a=slice(n, seq_len),
                    pos_b=slice(0, seq_len - n))
        s = summarize(m)
        results[f"roll_{n}"] = s
        print(f"  roll {n:3d}: cos_k={s['cos_k']['mean']:.6f}  "
              f"cos_v={s['cos_v']['mean']:.6f}")

    # Scrambled
    print("\n=== Scrambled ===")
    perm = torch.randperm(seq_len)
    scrambled_ids = base_ids[perm]
    kv_s = extract_kv(model, scrambled_ids, device)
    m = compare(kv_base, kv_s, slice(None), slice(None))
    results["scrambled"] = summarize(m)
    s = results["scrambled"]
    print(f"  scrambled: cos_k={s['cos_k']['mean']:.6f}  "
          f"cos_v={s['cos_v']['mean']:.6f}")

    # Unrelated (reversed tokens)
    print("\n=== Reversed ===")
    rev_ids = base_ids.flip(0)
    kv_u = extract_kv(model, rev_ids, device)
    m = compare(kv_base, kv_u, slice(None), slice(None))
    results["reversed"] = summarize(m)
    s = results["reversed"]
    print(f"  reversed: cos_k={s['cos_k']['mean']:.6f}  "
          f"cos_v={s['cos_v']['mean']:.6f}")

    # Truncated
    print("\n=== Truncated ===")
    trunc_ids = base_ids[:-1]
    kv_t = extract_kv(model, trunc_ids, device)
    m = compare(kv_base, kv_t,
                pos_a=slice(0, seq_len - 1),
                pos_b=slice(0, seq_len - 1))
    results["truncated"] = summarize(m)
    s = results["truncated"]
    print(f"  truncated: cos_k={s['cos_k']['mean']:.6f}  "
          f"cos_v={s['cos_v']['mean']:.6f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--output", default="results_long_orig")
    parser.add_argument("--cache", default="data_cache.json")
    parser.add_argument("--target-tokens", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]

    out = Path(args.output)
    out.mkdir(exist_ok=True)

    with open(args.cache) as f:
        convos = json.load(f)
    text = convos[0]["full"]
    print(f"Using conversation ({convos[0]['n_tokens']} tok, "
          f"truncating to {args.target_tokens})")

    model, tokenizer = load_model(args.model, device, dtype)
    results = run(model, tokenizer, device, text, args.target_tokens)

    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults -> {out / 'results.json'}")


if __name__ == "__main__":
    main()
