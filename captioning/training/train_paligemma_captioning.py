"""
train_paligemma_captioning.py -- fine-tune PaliGemma 2 as a lesion CAPTIONER: crop + prompt -> caption.
(Phase 6 of a captioning project specification (not part of this repository).) Separate from the detection model.

Built on the pieces already proven in the detection project: maestro's PaliGemma-2 trainer + our fixes
(rank-zero-safe checkpoints, full-state resume, periodic adapter snapshots, warmup+cosine lr, LoRA dropout).
Captions have no <loc> tokens, so maestro's loss reduces to plain token cross-entropy on the caption
(prompt tokens are not supervised) and ends with <eos> so the model learns to stop.

Deviation from the spec: a teacher-forced VALIDATION LOSS is not computed (maestro's validation step only
generates). Validation instead reports generation metrics every epoch: feat_acc (share of the fields stated
in the reference caption that the generated caption gets right) and ROUGE-L, on every val_stride-th
validation lesion. Full validation/test evaluation: evaluate_captioning.py.

Usage (config.yaml holds the defaults):
    CUDA_VISIBLE_DEVICES=0,1 python train_paligemma_captioning.py --variant padded
    CUDA_VISIBLE_DEVICES=0,1 python train_paligemma_captioning.py --variant padded --resume_from <run>/checkpoints/full_state_latest.ckpt
"""
import argparse
import json
import os
import sys
from dataclasses import replace
from functools import partial
from pathlib import Path

import lightning
import yaml
from torch.utils.data import DataLoader, Subset

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
PALI = PROJ.parent / "common"        # shared training helpers
sys.path[:0] = [str(HERE), str(PROJ), str(PALI)]

from maestro.trainer.common.datasets import create_data_loaders                       # noqa: E402
from maestro.trainer.common.utils.path import create_new_run_directory                # noqa: E402
from maestro.trainer.common.utils.seed import ensure_reproducibility                  # noqa: E402
from maestro.trainer.models.paligemma_2.checkpoints import OptimizationStrategy, load_model, save_model  # noqa: E402
from maestro.trainer.models.paligemma_2.core import PaliGemma2Configuration, PaliGemma2Trainer  # noqa: E402
from maestro.trainer.models.paligemma_2.loaders import evaluation_collate_fn, train_collate_fn  # noqa: E402

from caption_metrics import CaptionMetric, letterbox                                 # noqa: E402
import caption_bundle as CB                                                           # noqa: E402
from paligemma_resume import FullStateCheckpoint, PeriodicAdapterSnapshot, RankZeroSaveCheckpoint  # noqa: E402
from paligemma_training_extras import EpochMetricsCSV, ScheduledTrainer, apply_lora_dropout      # noqa: E402


def train_collate(batch, processor):
    return train_collate_fn([(letterbox(img), e) for img, e in batch], processor=processor, max_length=512)


def eval_collate(batch, processor):
    return evaluation_collate_fn([(letterbox(img), e) for img, e in batch], processor=processor)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--variant", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--resume_from", default=None)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--dataset", default=None, help="override the dataset folder (default: <project>/training/datasets/<variant>)")
    ap.add_argument("--dry_run", action="store_true", help="build everything up to the trainer and print, without training")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    for k in ("variant", "epochs", "output_dir"):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)

    dataset = Path(args.dataset) if args.dataset else PROJ / "training" / "datasets" / cfg["variant"]
    out_base = Path(cfg["output_dir"]) if os.path.isabs(cfg["output_dir"]) else PROJ / cfg["output_dir"]
    out_base = out_base / cfg["variant"]

    config = PaliGemma2Configuration(
        dataset=str(dataset), model_id=cfg["model_id"], device="auto", optimization_strategy=cfg["optimization_strategy"],
        epochs=cfg["epochs"], lr=float(cfg["lr"]), batch_size=cfg["batch_size"],
        accumulate_grad_batches=cfg["accumulate_grad_batches"], val_batch_size=1, num_workers=cfg["num_workers"],
        val_num_workers=cfg["val_num_workers"], output_dir=str(out_base), metrics=[CaptionMetric()],
        max_new_tokens=cfg["max_new_tokens"], random_seed=cfg["seed"])
    ensure_reproducibility(seed=config.random_seed, avoid_non_deterministic_algorithms=False)

    if args.resume_from:
        run_dir = str(Path(args.resume_from).resolve().parent.parent)
        print(f"Resuming from {args.resume_from} -- reusing run directory {run_dir}")
    else:
        run_dir = create_new_run_directory(base_output_dir=config.output_dir)
    config = replace(config, output_dir=run_dir)
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    (Path(run_dir) / "config_used.json").write_text(json.dumps(cfg, indent=2))           # save configuration

    processor, model = load_model(model_id_or_path=config.model_id, revision=config.revision, device=config.device,
                                  optimization_strategy=OptimizationStrategy(config.optimization_strategy),
                                  cache_dir=config.cache_dir)
    if cfg.get("lora_dropout") is not None:
        n = apply_lora_dropout(model, float(cfg["lora_dropout"]))
        assert n > 0, "no LoRA dropout modules found"
        print(f"LoRA dropout set to {cfg['lora_dropout']} on {n} adapter modules")
    if cfg.get("gradient_checkpointing", True):
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        print("Gradient checkpointing enabled")

    if cfg.get("bundle"):
        # "learn each fact separately": random fact-subset question/answer per draw + label-true augmentation (caption_bundle.py)
        thr = CB.load_thresholds(PROJ / "features" / "feature_thresholds_bundle.json")
        train_fn = partial(CB.bundle_train_collate, processor=processor, thr=thr, max_length=512,
                           p_generic=float(cfg.get("p_generic", 0.25)), augment_on=bool(cfg.get("augment", True)))
        print(f"BUNDLE mode: random fact-subset prompts (p_generic={cfg.get('p_generic', 0.25)}), augmentation={cfg.get('augment', True)}, "
              "facts re-measured after every augmentation")
    else:
        train_fn = partial(train_collate, processor=processor)
    train_loader, valid_loader, _ = create_data_loaders(
        dataset_location=config.dataset, train_batch_size=config.batch_size,
        train_collect_fn=train_fn, train_num_workers=config.num_workers,
        test_batch_size=1, test_collect_fn=partial(eval_collate, processor=processor), test_num_workers=config.val_num_workers)
    stride = int(cfg.get("val_stride", 1))
    if stride > 1:
        sub = Subset(valid_loader.dataset, list(range(0, len(valid_loader.dataset), stride)))
        valid_loader = DataLoader(sub, batch_size=1, shuffle=False, collate_fn=valid_loader.collate_fn,
                                  num_workers=config.val_num_workers)
    print(f"variant={cfg['variant']}  train samples={len(train_loader.dataset)}  in-training validation samples={len(valid_loader.dataset)}")

    kw = dict(processor=processor, model=model, train_loader=train_loader, valid_loader=valid_loader, config=config)
    module = ScheduledTrainer(**kw, warmup_steps=cfg["warmup_steps"]) if cfg["lr_schedule"] == "cosine" else PaliGemma2Trainer(**kw)
    if args.dry_run:
        b = next(iter(train_loader))
        print("dry run OK: one training batch built; input_ids", tuple(b[0].shape), "pixels", tuple(b[3].shape))
        sup = b[4][0][b[4][0] != -100].tolist()
        print("supervised tokens:", len(sup), "| last is <eos>:", sup[-1] == 1)
        return

    ckpt_dir = os.path.join(config.output_dir, "checkpoints")
    callbacks = [RankZeroSaveCheckpoint(result_path=ckpt_dir, save_model_callback=save_model),
                 FullStateCheckpoint(path=os.path.join(ckpt_dir, "full_state_latest.ckpt")),
                 EpochMetricsCSV(os.path.join(config.output_dir, "metrics_per_epoch.csv"), keys=("feat_acc", "rougeL"))]
    if cfg.get("snapshot_every", 0) > 0:
        callbacks.append(PeriodicAdapterSnapshot(ckpt_dir, cfg["snapshot_every"], save_model))
    trainer = lightning.Trainer(max_epochs=config.epochs, accumulate_grad_batches=config.accumulate_grad_batches,
                                check_val_every_n_epoch=cfg["val_every"], limit_val_batches=1.0, log_every_n_steps=10,
                                callbacks=callbacks)
    trainer.fit(module, ckpt_path=args.resume_from)


if __name__ == "__main__":
    main()
