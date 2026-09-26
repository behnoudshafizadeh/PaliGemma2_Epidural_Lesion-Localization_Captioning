#!/bin/bash
# text -> box: build prompt dataset, fine-tune, evaluate on the test split (2 GPUs).   Usage: scripts/run_detection.sh <adapter_epoch>   e.g. 10
set -e
cd "$(dirname "$0")/../detection"
python build_prompt_dataset.py --coco_dir ../data/annotations --images_dir ../data/images
GPUS=${GPUS:-0,1} EPOCHS=${EPOCHS:-25} ./train_detection.sh
ADAPTER=$(ls -d "$(pwd)"/training_output/*/checkpoints/epoch_$(printf %03d ${1:-10}) | tail -1)
for s in 0 1; do
  CUDA_VISIBLE_DEVICES=$s python evaluate_prompts.py --adapter_dir "$ADAPTER" --split test --shard $s --num_shards 2 \
    --conditions matched,baseline,mismatched --out_prefix eval &
done
wait
python evaluate_prompts.py --aggregate --split test --out_prefix eval      # -> eval_test_summary.txt
