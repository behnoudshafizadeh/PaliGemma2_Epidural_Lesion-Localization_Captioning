#!/bin/bash
# image -> caption: build the datasets, fine-tune the bundle, evaluate on the test split (2 GPUs).   Usage: scripts/run_captioning.sh <adapter_epoch>  e.g. 15
set -e
cd "$(dirname "$0")/../captioning"
python build_captioning_dataset.py --coco_dir ../data/annotations --images_dir ../data/images --masks_dir ../data/masks
python build_lesion_only_variant.py
python build_bundle_dataset.py
CUDA_VISIBLE_DEVICES=${GPUS:-0,1} python training/train_paligemma_captioning.py --config training/config_bundle.yaml
ADAPTER=$(ls -d "$(pwd)"/training/checkpoints/lesion_bundle/*/checkpoints/epoch_$(printf %03d ${1:-15}) | head -1)
for s in 0 1; do
  CUDA_VISIBLE_DEVICES=$s python training/evaluate_captioning.py --adapter_dir "$ADAPTER" --split test --shard $s --num_shards 2 --out_prefix evalcap &
done
wait
python training/evaluate_captioning.py --aggregate --split test --out_prefix evalcap     # -> training/evalcap_test_summary.txt
python training/compare_captions.py --split test                                          # -> caption_comparison/ (contains lesion images: keep private)
