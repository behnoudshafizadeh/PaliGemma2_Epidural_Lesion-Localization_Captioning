"""
train_paligemma_python.py -- same training as `maestro paligemma_2 train`, but
launched through maestro's Python building blocks instead of its CLI/`train()`
function, so we can:
  1. pass a *fixed* mAP metric object (paligemma_map_metric.py) instead of the
     string "mean_average_precision", which routes through maestro's own broken
     metric glue and crashes on the first validation step; and
  2. support resuming a stopped run from a full training-state checkpoint
     (paligemma_resume.py), which maestro's own train() has no path for at all.

This reimplements the ~30 lines of maestro.trainer.models.paligemma_2.core.train()
inline (see that function for the original) so both of the above can be wired into
the Lightning Trainer's callbacks/fit() call, which maestro's train() doesn't
expose.

Run in the `paligemma2` env, same flags as the CLI, plus --resume_from:
    python train_paligemma_python.py \
        --dataset .../dataset --model_id google/paligemma2-3b-pt-448 \
        --epochs 100 --lr 2e-4 --batch_size 2 --accumulate_grad_batches 4 \
        --val_batch_size 2 --num_workers 2 --val_num_workers 1 \
        --output_dir .../training_output

    # to resume a stopped run (epoch-boundary granularity -- see paligemma_resume.py):
        ... --resume_from .../training_output/1/checkpoints/full_state_latest.ckpt
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))   # shared training helpers
import argparse
import os
from dataclasses import replace
from functools import partial
from pathlib import Path

import lightning
from maestro.trainer.common.datasets import create_data_loaders
from maestro.trainer.common.utils.path import create_new_run_directory
from maestro.trainer.common.utils.seed import ensure_reproducibility
from maestro.trainer.models.paligemma_2.checkpoints import OptimizationStrategy, load_model, save_model
from maestro.trainer.models.paligemma_2.core import PaliGemma2Configuration, PaliGemma2Trainer
from maestro.trainer.models.paligemma_2.loaders import evaluation_collate_fn, train_collate_fn

from paligemma_map_metric import PaliGemmaMeanAveragePrecisionMetric
from paligemma_resume import FullStateCheckpoint, PeriodicAdapterSnapshot, RankZeroSaveCheckpoint
from paligemma_training_extras import (AugConfig, EpochMetricsCSV, ScheduledTrainer, apply_lora_dropout,
                                       augmented_train_collate)

# Single-class box dataset (dataset/): prefix "detect hemorrhage", suffix
# "<loc>...<loc> hemorrhage[ ; ...]". Update if pointing this at dataset_points/
# or dataset_posneg/, which also use a "background" class.
CLASSES = ["hemorrhage"]
IMAGE_RESOLUTION_WH = (512, 512)  # matches build_paligemma_dataset.py's W,H
FULL_CHECKPOINT_NAME = "full_state_latest.ckpt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--model_id", default="google/paligemma2-3b-pt-224")
    p.add_argument("--revision", default="refs/heads/main")
    p.add_argument("--device", default="auto")
    p.add_argument("--optimization_strategy", default="lora")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--accumulate_grad_batches", type=int, default=2)
    p.add_argument("--val_batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--val_num_workers", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--classes", nargs="+", default=CLASSES,
                    help="Class names present in the suffix, in the order used for class_id.")
    p.add_argument("--resume_from", default=None,
                    help="Path to a full_state_latest.ckpt saved by a previous run of this script. "
                         "Resumes model+optimizer+epoch state and keeps writing into that same run "
                         "directory, instead of starting a new numbered run.")
    p.add_argument("--snapshot_every", type=int, default=0,
                    help="keep a permanent adapter snapshot every N epochs (0 = off). Needed to pick the "
                         "best epoch afterwards, since 'latest' is overwritten every epoch.")
    p.add_argument("--lora_dropout", type=float, default=None,
                    help="override maestro's hard-coded LoRA dropout (0.05) after the model is loaded")
    p.add_argument("--lr_schedule", choices=["constant", "cosine"], default="constant",
                    help="constant = maestro's behaviour; cosine = linear warmup then cosine decay to 1%% of --lr")
    p.add_argument("--warmup_steps", type=int, default=200, help="optimizer steps of linear warmup (cosine only)")
    p.add_argument("--augment", action="store_true",
                    help="mild intensity + small shift/scale augmentation; the size/shape/region prompt is "
                         "regenerated from the augmented box (size_shape_region dataset only)")
    p.add_argument("--aug_thresholds", default=str(Path(__file__).resolve().parent / "prompt_dataset" / "thresholds.json"))
    p.add_argument("--val_every", type=int, default=1, help="run validation every N epochs")
    p.add_argument("--limit_val_batches", type=lambda v: int(v) if v.isdigit() else float(v), default=1,
                    help="int = number of validation batches (default 1, the old behaviour), float = fraction "
                         "(1.0 = the whole validation split)")
    p.add_argument("--no_gradient_checkpointing", action="store_true",
                    help="Disable gradient checkpointing (on by default -- see the comment where it's "
                         "enabled). Only useful for debugging; it exists purely as an escape hatch.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    metric = PaliGemmaMeanAveragePrecisionMetric(classes=args.classes, resolution_wh=IMAGE_RESOLUTION_WH)
    config = PaliGemma2Configuration(
        dataset=args.dataset,
        model_id=args.model_id,
        revision=args.revision,
        device=args.device,
        optimization_strategy=args.optimization_strategy,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        accumulate_grad_batches=args.accumulate_grad_batches,
        val_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        val_num_workers=args.val_num_workers,
        output_dir=args.output_dir,
        metrics=[metric],  # pre-built object -> skips maestro's broken string lookup path
        max_new_tokens=args.max_new_tokens,
        random_seed=args.random_seed,
    )

    ensure_reproducibility(seed=config.random_seed, avoid_non_deterministic_algorithms=False)

    if args.resume_from:
        # .../<output_dir>/<run_number>/checkpoints/full_state_latest.ckpt -> reuse <run_number>
        run_dir = str(Path(args.resume_from).resolve().parent.parent)
        print(f"Resuming from {args.resume_from} -- reusing run directory {run_dir}")
    else:
        run_dir = create_new_run_directory(base_output_dir=config.output_dir)
    config = replace(config, output_dir=run_dir)

    processor, model = load_model(
        model_id_or_path=config.model_id,
        revision=config.revision,
        device=config.device,
        optimization_strategy=OptimizationStrategy(config.optimization_strategy),
        cache_dir=config.cache_dir,
    )

    if args.lora_dropout is not None:
        n_do = apply_lora_dropout(model, args.lora_dropout)
        print(f"LoRA dropout set to {args.lora_dropout} on {n_do} adapter modules")
        assert n_do > 0, "no LoRA dropout modules found -- refusing to continue with an unverified setting"

    if not args.no_gradient_checkpointing:
        # Real sequence length here is ~1033 tokens (1024 image tokens at 448x448 +
        # a handful of text tokens -- confirmed by running the processor directly,
        # not the max_length=512 training-collation setting, which never actually
        # binds: max/typical suffix length in this dataset is 5-12 tokens). That
        # long a sequence through every decoder layer's activations is the real
        # memory cost at training time. Gradient checkpointing recomputes those
        # activations during backward instead of storing them -- trades ~20-30%
        # more compute for a large activation-memory cut, with NO change to the
        # model's numerics (unlike switching to sdpa attention, which HF disables
        # by default for Gemma2 specifically because sdpa silently drops logit
        # softcapping -- not used here for that reason).
        # enable_input_require_grads() is needed alongside it for a LoRA/PEFT
        # model: the base model's embeddings are frozen, so without this the
        # gradient chain into the trainable LoRA adapters breaks under checkpointing.
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        print("Gradient checkpointing enabled (trades compute for activation memory, no numeric change)")

    # max_length in maestro's train collate never changes the labels (verified at 512 and 48), so it is
    # fixed here and NOT tied to max_new_tokens, which we lower to bound validation-generation time.
    if args.augment:
        aug_cfg = AugConfig(thresholds_json=args.aug_thresholds)
        train_collect_fn = partial(augmented_train_collate, processor=processor, max_length=512, cfg=aug_cfg)
        print(f"Augmentation ON: p_geometric={aug_cfg.p_geometric} scale={aug_cfg.scale} shift=+-{aug_cfg.shift_frac:.0%} "
              f"p_intensity={aug_cfg.p_intensity}; prompt regenerated from the augmented box; no flips")
    else:
        train_collect_fn = partial(train_collate_fn, processor=processor, max_length=512)

    train_loader, valid_loader, _test_loader = create_data_loaders(
        dataset_location=config.dataset,
        train_batch_size=config.batch_size,
        train_collect_fn=train_collect_fn,
        train_num_workers=config.num_workers,
        test_batch_size=config.val_batch_size,
        test_collect_fn=partial(evaluation_collate_fn, processor=processor),
        test_num_workers=config.val_num_workers,
    )

    trainer_kwargs = dict(processor=processor, model=model, train_loader=train_loader,
                          valid_loader=valid_loader, config=config)
    if args.lr_schedule == "cosine":
        pl_module = ScheduledTrainer(**trainer_kwargs, warmup_steps=args.warmup_steps)
    else:
        pl_module = PaliGemma2Trainer(**trainer_kwargs)

    checkpoints_dir = os.path.join(config.output_dir, "checkpoints")
    adapter_checkpoint_cb = RankZeroSaveCheckpoint(result_path=checkpoints_dir, save_model_callback=save_model)
    full_checkpoint_path = os.path.join(checkpoints_dir, FULL_CHECKPOINT_NAME)
    resume_checkpoint_cb = FullStateCheckpoint(path=full_checkpoint_path)

    trainer = lightning.Trainer(
        max_epochs=config.epochs,
        accumulate_grad_batches=config.accumulate_grad_batches,
        check_val_every_n_epoch=args.val_every,
        limit_val_batches=args.limit_val_batches,
        log_every_n_steps=10,
        callbacks=[adapter_checkpoint_cb, resume_checkpoint_cb,
                   EpochMetricsCSV(os.path.join(config.output_dir, "metrics_per_epoch.csv"))]
        + ([PeriodicAdapterSnapshot(checkpoints_dir, args.snapshot_every, save_model)] if args.snapshot_every > 0 else []),
    )
    trainer.fit(pl_module, ckpt_path=args.resume_from)


if __name__ == "__main__":
    main()
