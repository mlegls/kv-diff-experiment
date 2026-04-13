#!/bin/bash
set -e
CACHE="/workspace/data_cache.json"
ARGS="--dtype float16 --cache $CACHE --target-tokens 2000 --step 5 --n-convos 8"

echo "======== Qwen2.5-14B-Instruct ========"
python -u /workspace/kv_diff_extended.py --model Qwen/Qwen2.5-14B-Instruct --output /workspace/results_14b_instruct $ARGS
# Free disk for next model
rm -rf /root/.cache/huggingface/hub/models--Qwen--Qwen2.5-14B-Instruct
echo "Freed instruct cache"

echo "======== Qwen2.5-7B ========"
python -u /workspace/kv_diff_extended.py --model Qwen/Qwen2.5-7B --output /workspace/results_7b_base $ARGS

echo "All done!"
