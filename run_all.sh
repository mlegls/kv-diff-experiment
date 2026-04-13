#!/bin/bash
set -e

MODELS=("Qwen/Qwen2.5-14B" "Qwen/Qwen2.5-14B-Instruct" "Qwen/Qwen2.5-7B")
OUTDIRS=("results_14b_base" "results_14b_instruct" "results_7b_base")
CACHE="/workspace/data_cache.json"

for i in "${!MODELS[@]}"; do
    model="${MODELS[$i]}"
    outdir="/workspace/${OUTDIRS[$i]}"
    echo ""
    echo "========================================================"
    echo "Running: $model -> $outdir"
    echo "========================================================"
    python -u /workspace/kv_diff_extended.py \
        --model "$model" \
        --dtype float16 \
        --output "$outdir" \
        --cache "$CACHE" \
        --target-tokens 2000 \
        --step 5 \
        --n-convos 8
done

echo ""
echo "All models complete!"
