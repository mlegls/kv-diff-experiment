#!/usr/bin/env python3
"""Preprocess WildChat conversations and generate compaction summaries.

Run locally to create data_cache.json, then upload to the pod.
Requires ANTHROPIC_API_KEY in environment.
"""
import json
import os
import sys
from pathlib import Path

from kv_diff_extended import (
    format_conversation,
    summarize_conversation,
    compact_conversation,
)

import anthropic
from datasets import load_dataset
from transformers import AutoTokenizer


def main():
    n_convos = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    min_tokens = 500
    max_tokens = 4000
    out_path = sys.argv[2] if len(sys.argv) > 2 else "data_cache.json"

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-14B", trust_remote_code=True)

    print(f"Loading UltraChat (need {n_convos} convos, "
          f"{min_tokens}-{max_tokens} tokens)...")
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft",
                      streaming=True)

    candidates = []
    checked = 0
    for row in ds:
        turns = row.get("messages", [])
        if len(turns) < 6:
            continue
        full_text = format_conversation(turns)
        n_tok = len(tokenizer.encode(full_text))
        if n_tok < min_tokens or n_tok > max_tokens:
            continue
        candidates.append({
            "turns": turns, "full": full_text,
            "n_tokens": n_tok, "n_turns": len(turns),
        })
        if len(candidates) >= n_convos:
            break
        checked += 1
        if checked >= 20000:
            break

    candidates.sort(key=lambda x: -x["n_tokens"])
    print(f"Found {len(candidates)} conversations")

    client = anthropic.Anthropic()
    convos = []
    for i, c in enumerate(candidates):
        print(f"Summarizing {i+1}/{len(candidates)} "
              f"({c['n_tokens']} tok, {c['n_turns']} turns)...", end=" ")
        try:
            summary = summarize_conversation(c["turns"], client)
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
            print(f"OK ({comp_tok} tok compacted)")
        except Exception as e:
            print(f"FAILED: {e}")

    with open(out_path, "w") as f:
        json.dump(convos, f, indent=2)
    print(f"\nSaved {len(convos)} conversations to {out_path}")


if __name__ == "__main__":
    main()
