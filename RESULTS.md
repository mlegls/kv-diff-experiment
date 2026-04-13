# KV Cache Diff Experiments — Results Summary

Experiments measuring how KV cache vectors change under different prompt modifications in Qwen2.5 models (7B, 14B base, 14B instruct), motivated by questions about what happens to a model's "thought process" during context window compaction.

## Experiment Overview

| Experiment | Script | Data |
|---|---|---|
| Original conditions (roll, scramble, unrelated, truncate) | `kv_diff.py` | `orig_*/results.json` |
| Roll vs truncate 1-by-1, compaction comparison | `kv_diff_extended.py` | `results_*/roll_truncate.json`, `results_*/summary.json` |
| Attention patterns & downstream output | `kv_diff_attention.py` | `results_attention/` |
| FFT & regression analysis | `plot_fft_comprehensive.py` | `results_comparison/` |
| KV transplant (cross-model, K/V swap) | `kv_transplant.py` | `results_transplant/transplant_results.json` |
| RoPE correction (per-roll-amount) | `kv_transplant.py` | `results_transplant/rope_correction.json` |
| RoPE correction (full curve, 2000 tok) | `kv_diff_extended_rope.py` | `results_rope_2000/` |
| Cross-model comparison plots | `plot_comparison.py`, `plot_original_comparison.py` | `results_comparison/` |

---

## 1. Rolling Context

When the context window slides by N tokens (removing the first N, shifting the rest to earlier positions):

- **K vectors** degrade significantly: cos_sim drops from 0.97 (roll 1) to 0.82 (roll 20) on an 87-token prompt, and to ~0.49 when rolling through 99% of a 2000-token prompt.
- **V vectors** are remarkably stable: cos_sim stays above 0.97 until ~80% of context is removed, then drops sharply.
- **K degrades faster than V** because RoPE positional encoding is applied to K (and Q) but not V. V only changes due to altered attention patterns propagating through layers.
- The degradation is **sublinear** — the first few tokens matter most, then it plateaus.
- Results are **nearly identical across 7B, 14B base, and 14B instruct** — neither model size nor instruct tuning affects KV cache stability.

### Terminal State (Last Position)

Comparing just the generating position (most relevant for actual output):

- **Rolling**: K stays ~0.6-0.8, V stays ~0.95+ until very late. The model's "generating state" is well preserved.
- **Truncation**: K drops to ~0.3-0.5 (totally different terminal token), V drops to near-random immediately. Removing the end of the context changes the generating state drastically.
- The roll-vs-truncate K difference is a **near-constant vertical offset** (~0.15-0.19) with essentially zero slope (confirmed by regression, R^2 near 0 for the difference curve).

## 2. Scrambled & Unrelated Text

- **Scrambled** (same tokens, random order): cos_k ~0.66, cos_v ~0.10
- **Unrelated** (completely different text): cos_k ~0.63, cos_v ~0.07
- Scrambled is barely distinguishable from unrelated — **token order (context) is everything for V representations**. Just having the same vocabulary present is not enough.

## 3. Truncation (Sanity Check)

Removing the last token: cos_sim = **1.000000** for both K and V at all overlapping positions. Confirms causal attention — earlier positions are completely unaffected by later tokens.

## 4. Compaction: Full Chat vs Summary vs Unrelated

Using UltraChat conversations with Claude-generated compaction summaries (structured handoff format + last 2 turns verbatim):

### Position-by-position comparison
- K: Compacted (~0.67) vs Unrelated (~0.67) — virtually identical. Can't distinguish summary from random text.
- V: Compacted (~0.09) vs Unrelated (~0.07) — both near zero. Different token sequences have unrelated V representations position-by-position.

### Centroid comparison (mean KV vector across all positions)
- K centroids: Compacted (~0.95) vs Unrelated (~0.92) — small but consistent gap.
- **V centroids: Compacted (~0.85) vs Unrelated (~0.59)** — the big finding. The compacted conversation's average V representation is much closer to the original than random text.

This suggests **V vectors encode semantic content** — a good compaction summary preserves ~85% of the V centroid vs ~59% for unrelated text. The gap is consistent across all three models.

## 5. FFT & Frequency Analysis

### Rolling K curves show clear RoPE periodicity
- Sharp periodic peaks in the FFT, consistent across all models
- Spectrogram shows **perfectly stable horizontal frequency bands** — the same frequencies persist uniformly regardless of how much has been rolled. These are fixed properties of the RoPE architecture, not data-dependent.
- Dominant autocorrelation period: ~45 tokens

### Rolling V curves have no periodic structure
- FFT is essentially flat — V changes are aperiodic/smooth, purely content-driven

### Truncation K (last-pos) also shows periodic structure
- Similar RoPE-related peaks but noisier and weaker than rolling

### Truncation V (last-pos) is white noise
- No frequency structure — changes are purely from different token content

### Power ratio (roll/truncate)
- Oscillates periodically, confirming both share RoPE frequencies but with different amplitudes

## 6. Attention Patterns & Downstream Output

| Roll amount | K cos_sim | Attention cos_sim | Logit cos_sim | Top-5 token overlap |
|---|---|---|---|---|
| 1 | 0.972 | 0.413 | 0.986 | 100% |
| 5 | 0.889 | 0.455 | 0.980 | 80% |
| 10 | 0.847 | 0.413 | 0.971 | 80% |
| 20 | 0.825 | 0.396 | 0.958 | 60% |

- **Attention patterns change substantially** (cos_sim ~0.41) even at roll 1. The K changes DO affect attention routing.
- **But the output barely changes** — logit cos_sim stays >0.95, top-1 prediction is the same ("This") across all conditions.
- The model is **remarkably robust** to attention pattern changes — different routing, same result.
- Per-layer: attention is most disrupted in early layers (~0.2 cos_sim), more preserved in later layers (~0.4-0.6).

### Actual Token Predictions

The top-1 next token prediction ("This", probability 0.78-0.88) is identical across all roll amounts. Top-5 tokens largely overlap with minor reorderings. Even rolling 20 tokens (~23% of context), the model's prediction is qualitatively unchanged.

## 7. RoPE Correction

**[Updated]** The original experiment used incorrect dimension pairing (interleaved instead of split-half), producing a spurious null result. After fixing `undo_rope` to match Qwen2's actual `rotate_half` convention (pairs `(i, i+d/2)` not `(2i, 2i+1)`), the result reverses completely:

| Roll | Raw K cos | De-rotated K cos | RoPE deflation |
|------|-----------|------------------|----------------|
| 1 | 0.987 | **0.999** | −0.011 |
| 3 | 0.935 | **0.999** | −0.064 |
| 5 | 0.925 | **0.998** | −0.073 |
| 10 | 0.895 | **0.998** | −0.102 |
| 20 | 0.879 | **0.998** | −0.119 |

**The K difference from rolling IS almost entirely RoPE positional encoding.** Once the position rotation is removed, the content-only K vectors are ~0.998 similar — essentially identical — even at roll 20 on a 2000-token context. The raw cosine similarity was being *deflated* by the RoPE position mismatch.

Per-layer analysis confirms this: the raw K curve oscillates wildly across layers (RoPE interference), while the de-rotated curve is a flat line near 1.0. All roll amounts stay >0.997 de-rotated at every layer, with only a tiny content divergence in early-mid layers.

This explains why the model's output is so robust to rolling (§6): the K *content* barely changes, so attention routing is essentially preserved. The raw K cosine similarity drop was a measurement artifact of comparing vectors at mismatched RoPE positions.

## 8. KV Cache Transplant Experiments

### Cross-Model (14B Base <-> 14B Instruct)

| Condition | AI history prompt | Thermodynamics prompt |
|---|---|---|
| Base, own cache | Coherent continuation | Coherent explanation |
| **Instruct, BASE cache** | **Coherent** — nearly indistinguishable | Same opening |
| Instruct, own cache | Slightly different style | Same opening |
| **Base, INSTRUCT cache** | Meta-response: "This is a summary...can you provide more details?" | Different framing |

The instruct model generates coherently from the base model's KV cache. The reverse is more interesting: the base model reading the instruct cache produces a meta-response, interpreting the instruct model's "be helpful" representations as a request for interaction.

### K/V Swap — Rolling

| Condition | Output |
|---|---|
| Original | Coherent AI history continuation |
| K_orig + V_roll5 | Semi-coherent, fragments of original content ("Turing machine", "abstract essence") |
| **K_roll5 + V_orig** | **Degenerate** — "1 in 1 in 1 in 1..." |
| K_orig + V_roll20 | Repetitive fragments about "computer" |
| **K_roll20 + V_orig** | **Degenerate** — just periods |

**V determines content, K determines coherence.** With correct K but wrong V, the output contains topic-related fragments. With wrong K but correct V, the model degenerates — it cannot route attention properly.

### K/V Swap — Truncation

| Condition | Output |
|---|---|
| K_trunc5 + V_orig[:82] | Identical to truncated baseline |
| K_orig[:82] + V_trunc5 | Coherent but different ("This article is about the history of AI...") |

For truncation, K_trunc + V_orig = K_trunc + V_trunc (because V values are identical at overlapping positions due to causal attention). Swapping V from the original has no effect — confirming the V identity result.

### Cross-Text (AI vs Pasta)

| Condition | Output |
|---|---|
| **K_AI + V_pasta** | **Follows pasta topic** ("The dough is ready when smooth and elastic...") |
| K_pasta + V_AI | Degenerate, but fragments leak AI content ("the first computer") |

**V carries the topic.** K_AI + V_pasta talks about pasta. The content follows V, not K.

---

## Key Takeaways

1. **V vectors carry semantic meaning, K vectors carry positional/routing information.** This is demonstrated by centroid similarity (compaction preserves V meaning), K/V swap experiments (V determines topic), and the differential sensitivity to rolling vs truncation.

2. **Rolling the context window is a soft reset.** The model's V representations (and therefore its "thoughts") are substantially preserved when the window slides. K vectors change more, but the downstream output is robust to these changes.

3. **Compaction summaries genuinely preserve semantic information** in V-space (85% centroid similarity vs 59% for random text), even though position-by-position alignment is completely lost.

4. **K changes from rolling ARE almost entirely RoPE position shifts.** After correctly removing RoPE (split-half convention), K content similarity is ~0.998 even at roll 20. The apparent K degradation in raw cosine similarity was a measurement artifact of comparing vectors at mismatched positions. This explains the model's output robustness — the actual attention routing information is nearly unchanged.

5. **Attention patterns change substantially under rolling, but output doesn't.** The attention *weight distributions* shift (cos_sim ~0.41), but since the underlying K content is preserved, the model reaches the same conclusions through slightly different routing.

6. **Model size and instruct tuning don't meaningfully affect any of these properties.** The KV cache behavior is a fundamental property of the transformer architecture, not a learned characteristic that varies with training.

7. **Cross-model KV transplant works** between base and instruct variants of the same architecture. The representations are compatible enough for coherent generation.
