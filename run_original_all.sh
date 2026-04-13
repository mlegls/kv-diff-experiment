#!/bin/bash
set -e

pip install --break-system-packages transformers accelerate matplotlib numpy 2>&1 | tail -1

MODELS=("Qwen/Qwen2.5-14B" "Qwen/Qwen2.5-14B-Instruct" "Qwen/Qwen2.5-7B")
OUTDIRS=("orig_14b_base" "orig_14b_instruct" "orig_7b_base")

for i in "${!MODELS[@]}"; do
    model="${MODELS[$i]}"
    outdir="/workspace/${OUTDIRS[$i]}"
    echo ""
    echo "======== $model ========"
    python -u /workspace/kv_diff.py --model "$model" --dtype float16 --output "$outdir"
    # Free cache between 14B models
    if [[ "$model" == *"14B"* ]]; then
        rm -rf /root/.cache/huggingface/hub/models--Qwen--Qwen2.5-14B*
        echo "Freed cache"
    fi
done

echo "All done!"
