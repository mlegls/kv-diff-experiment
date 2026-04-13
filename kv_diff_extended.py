#!/usr/bin/env python3
"""
Extended KV Cache Diff Experiments

Experiment 1: Roll vs Truncate, 1 token at a time, on a long text from a dataset.
  Shows how KV similarity degrades as prefix is removed (rolling) vs suffix (truncating).
  Truncation should always give cos_sim=1.0 (causal attention sanity check).

Experiment 2: Long text vs its summary vs unrelated short text (multiple examples).
  Uses CNN/DailyMail dataset for real article-summary pairs.
  Tests whether summaries produce KV representations more similar to the full text
  than random text of the same length. Uses both position-by-position and centroid
  (mean KV vector) comparisons.
"""

import argparse
import json
import os
from pathlib import Path

import anthropic
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# -- Shared utilities --

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


def mean_cos(kv_a, kv_b, pos_a, pos_b):
    """Scalar mean cosine similarity across all layers, heads, and positions."""
    ck, cv = [], []
    for i in range(len(kv_a)):
        ka, va = kv_a[i][0][0, :, pos_a, :], kv_a[i][1][0, :, pos_a, :]
        kb, vb = kv_b[i][0][0, :, pos_b, :], kv_b[i][1][0, :, pos_b, :]
        ck.append(cos_sim(ka, kb).mean().item())
        cv.append(cos_sim(va, vb).mean().item())
    return float(np.mean(ck)), float(np.mean(cv))


def last_pos_cos(kv_a, kv_b, pos_a=-1, pos_b=-1):
    """Cosine similarity at a single position (default: last), averaged over layers and heads."""
    ck, cv = [], []
    for i in range(len(kv_a)):
        ka = kv_a[i][0][0, :, pos_a, :]  # (heads, dim)
        va = kv_a[i][1][0, :, pos_a, :]
        kb = kv_b[i][0][0, :, pos_b, :]
        vb = kv_b[i][1][0, :, pos_b, :]
        ck.append(cos_sim(ka, kb).mean().item())
        cv.append(cos_sim(va, vb).mean().item())
    return float(np.mean(ck)), float(np.mean(cv))


def per_layer_cos(kv_a, kv_b, pos_a, pos_b):
    """Per-layer mean cosine similarity (averaged over heads and positions)."""
    ck, cv = [], []
    for i in range(len(kv_a)):
        ka, va = kv_a[i][0][0, :, pos_a, :], kv_a[i][1][0, :, pos_a, :]
        kb, vb = kv_b[i][0][0, :, pos_b, :], kv_b[i][1][0, :, pos_b, :]
        ck.append(cos_sim(ka, kb).mean().item())
        cv.append(cos_sim(va, vb).mean().item())
    return ck, cv


def centroid_cos(kv_a, kv_b):
    """Compare mean KV vectors (centroids across all positions) per layer."""
    ck, cv = [], []
    for i in range(len(kv_a)):
        ka_c = kv_a[i][0][0].mean(dim=1)  # (heads, dim)
        va_c = kv_a[i][1][0].mean(dim=1)
        kb_c = kv_b[i][0][0].mean(dim=1)
        vb_c = kv_b[i][1][0].mean(dim=1)
        ck.append(cos_sim(ka_c, kb_c).mean().item())
        cv.append(cos_sim(va_c, vb_c).mean().item())
    return ck, cv


# -- Dataset loading --

def format_conversation(turns):
    """Format a WildChat conversation as flat text."""
    parts = []
    for t in turns:
        role = t.get("role", "user").capitalize()
        content = t.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n\n".join(parts)


COMPACTION_SYSTEM = """\
You are a context compaction system. Do NOT respond to any questions or \
requests in the conversation -- only output the structured summary.

Create a structured handoff summary for a different assistant that will \
continue this conversation after earlier turns are compacted.

Use this EXACT format:

## Goal
[What the user is trying to accomplish]

## Constraints & Preferences
[Any mentioned requirements, preferences, or style requests]

## Progress
- Done: [completed tasks]
- In Progress: [current work]
- Blocked: [impediments if any]

## Key Decisions
[Important choices made, with brief rationale]

## Relevant Files & Code
[Exact file paths, function names, error messages, code snippets if any]

## Remaining Work
[What still needs to happen]

## Critical Context
[Anything the next assistant MUST know to continue correctly]

IMPORTANT: Preserve exact file paths, function names, and error messages. \
Be concise but complete. Do not omit technical details."""

COMPACTION_USER = """\
Compact the following conversation history into a structured handoff summary.

<conversation>
{conversation}
</conversation>"""


def summarize_conversation(turns, client):
    """Use Claude to generate a compaction summary (excluding last 2 turns)."""
    to_summarize = turns[:-2] if len(turns) > 2 else turns
    text = format_conversation(to_summarize)
    resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=COMPACTION_SYSTEM,
        messages=[{
            "role": "user",
            "content": COMPACTION_USER.format(conversation=text),
        }],
    )
    return resp.content[0].text


def compact_conversation(turns, summary):
    """Create compacted version: structured summary + last 2 turns verbatim."""
    last_turns = format_conversation(turns[-2:]) if len(turns) >= 2 else ""
    return (
        f"[Context from compacted conversation history]\n\n"
        f"{summary}\n\n"
        f"[Recent conversation]\n\n"
        f"{last_turns}"
    )


def load_wildchat_conversations(tokenizer, n_convos=20, min_tokens=500,
                                max_tokens=4000, cache_path=None):
    """Load long conversations from WildChat, generate compaction summaries.

    Caches results to avoid re-downloading and re-summarizing.
    Returns list of dicts with 'full', 'compacted', 'n_tokens', etc.
    """
    # Check cache first
    if cache_path and Path(cache_path).exists():
        print(f"Loading cached conversations from {cache_path}")
        with open(cache_path) as f:
            convos = json.load(f)
        print(f"  Loaded {len(convos)} cached conversations")
        return convos

    print(f"Loading UltraChat (looking for {n_convos} conversations "
          f"with {min_tokens}-{max_tokens} tokens)...")
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft",
                      streaming=True)

    # First pass: collect candidate conversations
    candidates = []
    checked = 0
    for row in ds:
        turns = row.get("messages", [])
        if len(turns) < 6:  # at least 3 exchanges
            continue
        full_text = format_conversation(turns)
        n_tok = len(tokenizer.encode(full_text))
        if n_tok < min_tokens or n_tok > max_tokens:
            continue
        candidates.append({"turns": turns, "full": full_text,
                           "n_tokens": n_tok, "n_turns": len(turns)})
        if len(candidates) >= n_convos:
            break
        checked += 1
        if checked >= 20000:
            break

    candidates.sort(key=lambda x: -x["n_tokens"])
    print(f"  Found {len(candidates)} conversations, generating summaries...")

    # Second pass: generate compaction summaries via Claude API
    client = anthropic.Anthropic()
    convos = []
    for i, c in enumerate(candidates):
        print(f"  Summarizing {i+1}/{len(candidates)} "
              f"({c['n_tokens']} tok, {c['n_turns']} turns)...")
        try:
            summary = summarize_conversation(c["turns"], client)
        except Exception as e:
            print(f"    FAILED: {e}, skipping")
            continue
        comp_text = compact_conversation(c["turns"], summary)
        comp_tok = len(tokenizer.encode(comp_text))
        convos.append({
            "full": c["full"],
            "compacted": comp_text,
            "summary": summary,
            "n_tokens": c["n_tokens"],
            "comp_tokens": comp_tok,
            "n_turns": c["n_turns"],
        })

    # Cache results
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(convos, f, indent=2)
        print(f"  Cached to {cache_path}")

    print(f"  Ready: {len(convos)} conversations with summaries")
    if convos:
        print(f"  Token range: {convos[-1]['n_tokens']}-{convos[0]['n_tokens']}")
    return convos


# -- Experiment 1: Roll vs Truncate 1-by-1 --

def experiment_roll_truncate(model, tokenizer, device, out_dir,
                             convos=None, target_tokens=2000, step=1):
    if convos is None:
        convos = load_wildchat_conversations(
            tokenizer, n_convos=5, min_tokens=target_tokens,
            max_tokens=target_tokens * 2)
    if not convos:
        raise RuntimeError("No suitable conversations found")
    # Use the longest conversation, truncated to target
    long_text = convos[0]["full"]
    ids = tokenizer.encode(long_text)[:target_tokens]
    long_text = tokenizer.decode(ids)
    base_ids = torch.tensor(tokenizer.encode(long_text))
    seq_len = len(base_ids)
    print(f"\n{'='*60}")
    print(f"Experiment 1: Roll vs Truncate (step={step})")
    print(f"Sequence length: {seq_len} tokens")
    print(f"{'='*60}")

    kv_base = extract_kv(model, base_ids, device)

    max_remove = seq_len - 10
    ns = list(range(1, max_remove + 1, step))

    # Average over all overlapping positions
    roll_k, roll_v = [], []
    trunc_k, trunc_v = [], []
    # Last-position only: compare generating state
    roll_last_k, roll_last_v = [], []
    trunc_last_k, trunc_last_v = [], []

    for n in ns:
        # Roll: remove first n tokens -> [tN, ..., tM]
        kv_r = extract_kv(model, base_ids[n:], device)
        # Average over overlapping positions (same tokens, shifted)
        ck, cv = mean_cos(kv_base, kv_r,
                          slice(n, seq_len), slice(0, seq_len - n))
        roll_k.append(ck)
        roll_v.append(cv)
        # Last position: base pos (seq_len-1) vs rolled pos (seq_len-1-n)
        # Both have the same token (tM) but different pos encoding and context
        ck, cv = last_pos_cos(kv_base, kv_r, pos_a=-1, pos_b=-1)
        roll_last_k.append(ck)
        roll_last_v.append(cv)

        # Truncate: remove last n tokens -> [t0, ..., tM-n]
        kv_t = extract_kv(model, base_ids[:-n], device)
        # Average over overlapping positions (trivially 1.0)
        ck, cv = mean_cos(kv_base, kv_t,
                          slice(0, seq_len - n), slice(0, seq_len - n))
        trunc_k.append(ck)
        trunc_v.append(cv)
        # Last position: base pos (seq_len-1) vs truncated pos (seq_len-1-n)
        # Different tokens at different positions — how much does terminal state differ?
        ck, cv = last_pos_cos(kv_base, kv_t, pos_a=-1, pos_b=-1)
        trunc_last_k.append(ck)
        trunc_last_v.append(cv)

        if n % 50 == 0 or n == ns[-1]:
            print(f"  n={n:3d}/{max_remove}: "
                  f"roll_avg_k={roll_k[-1]:.4f} roll_last_k={roll_last_k[-1]:.4f}  "
                  f"trunc_last_k={trunc_last_k[-1]:.4f}")

    # -- Plot 1: All-position average --
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, label, rd, td in [
        (axes[0], "K", roll_k, trunc_k),
        (axes[1], "V", roll_v, trunc_v),
    ]:
        ax.plot(ns, rd, label="Roll (remove prefix)", color="steelblue", lw=1.2)
        ax.plot(ns, td, label="Truncate (remove suffix)", color="seagreen", lw=1.2)
        ax.set_xlabel("Tokens removed")
        ax.set_ylabel("Mean Cosine Similarity (all positions)")
        ax.set_title(f"{label} vectors -- all-position average")
        ax.legend()
        ax.set_ylim(-0.05, 1.05)
    plt.suptitle(f"Rolling vs Truncating -- avg over positions ({seq_len} tok)",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(out_dir / "roll_vs_truncate_avg.png", dpi=150)
    plt.close()

    # -- Plot 2: Last-position only (terminal generating state) --
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, label, rd, td in [
        (axes[0], "K", roll_last_k, trunc_last_k),
        (axes[1], "V", roll_last_v, trunc_last_v),
    ]:
        ax.plot(ns, rd, label="Roll (last pos)", color="steelblue", lw=1.2)
        ax.plot(ns, td, label="Truncate (last pos)", color="seagreen", lw=1.2)
        ax.set_xlabel("Tokens removed")
        ax.set_ylabel("Cosine Similarity (last position only)")
        ax.set_title(f"{label} vectors -- terminal state")
        ax.legend()
        ax.set_ylim(-0.05, 1.05)
    plt.suptitle(f"Terminal State: Rolling vs Truncating ({seq_len} tok)",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(out_dir / "roll_vs_truncate_last.png", dpi=150)
    plt.close()

    # -- Plot 3: Normalized by fraction, both metrics --
    fracs = [n / seq_len for n in ns]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for col, (label, rl, tl, rla, tla) in enumerate([
        ("K", roll_k, trunc_k, roll_last_k, trunc_last_k),
        ("V", roll_v, trunc_v, roll_last_v, trunc_last_v),
    ]):
        # All-position average
        ax = axes[0][col]
        ax.plot(fracs, rl, label="Roll", color="steelblue", lw=1.2)
        ax.plot(fracs, tl, label="Truncate", color="seagreen", lw=1.2)
        ax.set_ylabel("Cosine Sim (all positions)")
        ax.set_title(f"{label} -- all-position average")
        ax.legend()
        ax.set_ylim(-0.05, 1.05)
        # Last-position
        ax = axes[1][col]
        ax.plot(fracs, rla, label="Roll", color="steelblue", lw=1.2)
        ax.plot(fracs, tla, label="Truncate", color="seagreen", lw=1.2)
        ax.set_xlabel("Fraction removed")
        ax.set_ylabel("Cosine Sim (last position)")
        ax.set_title(f"{label} -- terminal state")
        ax.legend()
        ax.set_ylim(-0.05, 1.05)
    plt.suptitle(f"Rolling vs Truncating ({seq_len} tokens)", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_dir / "roll_vs_truncate_combined.png", dpi=150)
    plt.close()

    return {"ns": ns, "seq_len": seq_len,
            "roll_k": roll_k, "roll_v": roll_v,
            "trunc_k": trunc_k, "trunc_v": trunc_v,
            "roll_last_k": roll_last_k, "roll_last_v": roll_last_v,
            "trunc_last_k": trunc_last_k, "trunc_last_v": trunc_last_v}


# -- Experiment 2: Summary vs Unrelated --

def experiment_summary(model, tokenizer, device, out_dir, convos=None):
    print(f"\n{'='*60}")
    print(f"Experiment 2: Full chat vs Compacted vs Unrelated")
    print(f"{'='*60}")

    if convos is None:
        convos = load_wildchat_conversations(
            tokenizer, n_convos=10, min_tokens=400, max_tokens=4000)
    print(f"Using {len(convos)} conversations")

    results = []

    for i, item in enumerate(convos):
        art_ids = torch.tensor(tokenizer.encode(item["full"]))
        sum_ids = torch.tensor(tokenizer.encode(item["compacted"]))
        art_len, sum_len = len(art_ids), len(sum_ids)

        print(f"\n  Chat {i+1}/{len(convos)}: "
              f"{art_len} tok full, {sum_len} tok compacted "
              f"({item['n_turns']} turns)")

        kv_art = extract_kv(model, art_ids, device)

        # -- vs summary --
        kv_sum = extract_kv(model, sum_ids, device)
        cmp = min(art_len, sum_len)
        ck_sum, cv_sum = mean_cos(kv_art, kv_sum, slice(0, cmp), slice(0, cmp))
        ck_sum_l, cv_sum_l = per_layer_cos(
            kv_art, kv_sum, slice(0, cmp), slice(0, cmp))
        cent_k_sum, cent_v_sum = centroid_cos(kv_art, kv_sum)

        # -- vs unrelated (other articles, truncated to summary length) --
        unrel_pos_k, unrel_pos_v = [], []
        unrel_cent_k, unrel_cent_v = [], []
        unrel_per_layer_k, unrel_per_layer_v = [], []
        for j, other in enumerate(convos):
            if j == i:
                continue
            other_ids = torch.tensor(
                tokenizer.encode(other["full"]))[:sum_len]
            if len(other_ids) < 10:
                continue
            kv_o = extract_kv(model, other_ids, device)
            ol = min(art_len, len(other_ids))
            ck, cv = mean_cos(kv_art, kv_o, slice(0, ol), slice(0, ol))
            unrel_pos_k.append(ck)
            unrel_pos_v.append(cv)
            ck_l, cv_l = per_layer_cos(
                kv_art, kv_o, slice(0, ol), slice(0, ol))
            unrel_per_layer_k.append(ck_l)
            unrel_per_layer_v.append(cv_l)
            ck_c, cv_c = centroid_cos(kv_art, kv_o)
            unrel_cent_k.append(float(np.mean(ck_c)))
            unrel_cent_v.append(float(np.mean(cv_c)))

        r = {
            "art_len": art_len, "sum_len": sum_len,
            "summary_pos_k": ck_sum, "summary_pos_v": cv_sum,
            "summary_pos_k_layers": ck_sum_l,
            "summary_pos_v_layers": cv_sum_l,
            "summary_cent_k": float(np.mean(cent_k_sum)),
            "summary_cent_v": float(np.mean(cent_v_sum)),
            "summary_cent_k_layers": cent_k_sum,
            "summary_cent_v_layers": cent_v_sum,
            "unrel_pos_k": unrel_pos_k, "unrel_pos_v": unrel_pos_v,
            "unrel_cent_k": unrel_cent_k, "unrel_cent_v": unrel_cent_v,
            "unrel_per_layer_k": unrel_per_layer_k,
            "unrel_per_layer_v": unrel_per_layer_v,
        }
        results.append(r)

        print(f"    Compacted: pos_k={ck_sum:.4f}  pos_v={cv_sum:.4f}  "
              f"cent_k={r['summary_cent_k']:.4f}  "
              f"cent_v={r['summary_cent_v']:.4f}")
        if unrel_pos_k:
            print(f"    Unrelated: pos_k={np.mean(unrel_pos_k):.4f}  "
                  f"pos_v={np.mean(unrel_pos_v):.4f}  "
                  f"cent_k={np.mean(unrel_cent_k):.4f}  "
                  f"cent_v={np.mean(unrel_cent_v):.4f}")

    # -- Plot: grouped bar chart --
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    x = np.arange(len(results))

    for col, kv in enumerate(["k", "v"]):
        # Position-by-position
        ax = axes[0][col]
        s_vals = [r[f"summary_pos_{kv}"] for r in results]  # "summary" = compacted
        u_vals = [np.mean(r[f"unrel_pos_{kv}"]) for r in results]
        ax.bar(x - 0.15, s_vals, 0.3, label="Compacted", color="darkorange")
        ax.bar(x + 0.15, u_vals, 0.3,
               label="Unrelated (mean)", color="crimson")
        for xi, r in zip(x, results):
            for v in r[f"unrel_pos_{kv}"]:
                ax.scatter(xi + 0.15, v, color="crimson",
                           s=15, alpha=0.5, zorder=5)
        ax.set_ylabel("Cosine Similarity")
        ax.set_title(f"Position-by-position -- {kv.upper()}")
        ax.legend()
        ax.set_xticks(x)
        ax.set_xticklabels([f"Art {i+1}" for i in x])

        # Centroid
        ax = axes[1][col]
        s_vals = [r[f"summary_cent_{kv}"] for r in results]
        u_vals = [np.mean(r[f"unrel_cent_{kv}"]) for r in results]
        ax.bar(x - 0.15, s_vals, 0.3, label="Compacted", color="darkorange")
        ax.bar(x + 0.15, u_vals, 0.3,
               label="Unrelated (mean)", color="crimson")
        for xi, r in zip(x, results):
            for v in r[f"unrel_cent_{kv}"]:
                ax.scatter(xi + 0.15, v, color="crimson",
                           s=15, alpha=0.5, zorder=5)
        ax.set_ylabel("Cosine Similarity")
        ax.set_title(f"Centroid -- {kv.upper()}")
        ax.legend()
        ax.set_xticks(x)
        ax.set_xticklabels([f"Art {i+1}" for i in x])

    plt.suptitle("Full Chat vs Compacted vs Unrelated Chat", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "summary_vs_unrelated.png", dpi=150)
    plt.close()

    # -- Plot: per-layer for article 1 --
    r0 = results[0]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    for col, kv in enumerate(["k", "v"]):
        # Position-by-position per layer
        ax = axes[0][col]
        ax.plot(r0[f"summary_pos_{kv}_layers"], label="Compacted",
                color="darkorange", lw=2)
        for j, ul in enumerate(r0[f"unrel_per_layer_{kv}"]):
            lbl = "Unrelated" if j == 0 else None
            ax.plot(ul, label=lbl, color="crimson", alpha=0.4)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Cosine Similarity")
        ax.set_title(f"Position-by-position per layer -- {kv.upper()}")
        ax.legend()

        # Centroid per layer
        ax = axes[1][col]
        ax.plot(r0[f"summary_cent_{kv}_layers"], label="Summary",
                color="darkorange", lw=2)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Cosine Similarity")
        ax.set_title(f"Centroid per layer -- {kv.upper()}")
        ax.legend()

    plt.suptitle("Per-Layer Comparison (Article 1)", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir / "summary_per_layer.png", dpi=150)
    plt.close()

    return results


# -- Main --

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--output", default="results_extended")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-tokens", type=int, default=2000,
                        help="Target token count for long text (exp 1)")
    parser.add_argument("--step", type=int, default=5,
                        help="Step size for rolling/truncating (default: 5)")
    parser.add_argument("--n-convos", type=int, default=8,
                        help="Number of conversations for exp 2")
    parser.add_argument("--cache", default="data_cache.json",
                        help="Cache file for preprocessed conversations")
    parser.add_argument("--skip-exp1", action="store_true")
    parser.add_argument("--skip-exp2", action="store_true")
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

    # Load conversations once, share between experiments
    cache_path = str(out / args.cache)
    convos = load_wildchat_conversations(
        tokenizer,
        n_convos=max(args.n_convos, 5),  # need at least 5 for unrelated
        min_tokens=400,
        max_tokens=args.target_tokens * 2,
        cache_path=cache_path,
    )

    if not args.skip_exp1:
        res1 = experiment_roll_truncate(
            model, tokenizer, device, out,
            convos=convos, target_tokens=args.target_tokens,
            step=args.step)
        with open(out / "roll_truncate.json", "w") as f:
            json.dump(res1, f)
        print(f"  Saved -> {out / 'roll_truncate.json'}")

    if not args.skip_exp2:
        res2 = experiment_summary(
            model, tokenizer, device, out, convos=convos)
        def default_ser(o):
            if hasattr(o, "item"):
                return o.item()
            raise TypeError
        with open(out / "summary.json", "w") as f:
            json.dump(res2, f, indent=2, default=default_ser)
        print(f"  Saved -> {out / 'summary.json'}")

    print(f"\nAll results saved to {out}/")


if __name__ == "__main__":
    main()
