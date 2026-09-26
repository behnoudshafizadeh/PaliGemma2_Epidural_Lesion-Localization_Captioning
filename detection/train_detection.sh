#!/bin/bash
# Fine-tune PaliGemma 2 (448 px) with LoRA to answer a text prompt with a bounding box (text -> box).
# Prompts describe the lesion by size / shape / region, e.g.
#   "detect the small epidural hemorrhage with a compact bounding box in the middle-right region of the image"
# target: "<locY1><locX1><locY2><locX2> hemorrhage<eos>"
#
# Recipe: peak lr 1e-4 with 200 warmup steps + cosine decay, LoRA dropout 0.1 (rank 8), mild geometric/intensity augmentation
# (the size/shape/region prompt is regenerated from the augmented box, so the hint stays true), the FULL validation split after
# every epoch (metrics_per_epoch.csv), a permanent adapter snapshot every 5 epochs, full-state checkpoint every epoch (resume).
#
# Prepare the data first:   python build_prompt_dataset.py --coco_dir ../data/annotations --images_dir ../data/images
# Usage:  GPUS=0,1 EPOCHS=25 ./train_detection.sh
#         RESUME=<run>/checkpoints/full_state_latest.ckpt ./train_detection.sh      # resume a stopped run
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RESUME_ARGS=()
if [ -n "$RESUME" ]; then RESUME_ARGS=(--resume_from "$RESUME"); fi
CUDA_VISIBLE_DEVICES=${GPUS:-0,1} python train_paligemma_python.py \
  --dataset "$(pwd)/prompt_dataset/datasets/size_shape_region" \
  --model_id google/paligemma2-3b-pt-448 \
  --optimization_strategy lora \
  --epochs ${EPOCHS:-25} \
  --lr 1e-4 \
  --lr_schedule cosine \
  --warmup_steps 200 \
  --lora_dropout 0.1 \
  --augment \
  --val_every 1 \
  --limit_val_batches 1.0 \
  --batch_size ${BATCH:-1} \
  --accumulate_grad_batches ${ACCUM:-8} \
  --val_batch_size ${VAL_BATCH:-1} \
  --num_workers 2 \
  --val_num_workers 1 \
  --max_new_tokens 48 \
  --random_seed 42 \
  --snapshot_every 5 \
  --output_dir "$(pwd)/training_output" \
  "${RESUME_ARGS[@]}"
