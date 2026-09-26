"""
paligemma_training_extras.py -- training options maestro's PaliGemma-2 recipe doesn't offer,
added in OUR code (nothing in the shared maestro install is edited):

  * apply_lora_dropout      maestro hard-codes LoRA dropout 0.05; this sets it after model load.
  * ScheduledTrainer        AdamW + linear warmup then cosine decay (maestro: constant lr, no schedule).
  * augmented_train_collate mild on-the-fly augmentation (intensity jitter, small shift/scale) for the
                            size_shape_region dataset. Geometric augmentation moves/rescales the target
                            box, so the size/shape/region WORDS of the prompt could become false; the
                            prompt is therefore REGENERATED from the augmented box with the exact same
                            rules and train-only thresholds as the dataset builder. No flips (a flip would
                            swap left/right and change the meaning of region words).
  * EpochMetricsCSV         appends per-epoch validation metrics + lr to metrics_per_epoch.csv (rank 0),
                            so the curve survives an early stop.

Only used when the matching flags are passed to train_paligemma_python.py; without them behaviour is
identical to the earlier runs.
"""
import csv
import json
import math
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "detection"))   # build_prompt_dataset (prompt rules)
import build_prompt_dataset as B  # noqa: E402
from maestro.trainer.models.paligemma_2.core import PaliGemma2Trainer
from maestro.trainer.models.paligemma_2.loaders import train_collate_fn


# ------------------------------------------------------------------ LoRA dropout
def apply_lora_dropout(model, p: float) -> int:
    """Set the dropout probability of every LoRA adapter (peft names them '...lora_dropout.default')."""
    n = 0
    for name, m in model.named_modules():
        if "lora_dropout" in name and isinstance(m, torch.nn.Dropout):
            m.p = p
            n += 1
    return n


# ---------------------------------------------------------------------- schedule
def lr_factor(step: int, warmup: int, total: int, floor: float = 0.01) -> float:
    """Multiplier on the peak lr: linear warmup, then cosine decay to `floor` x peak."""
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


class ScheduledTrainer(PaliGemma2Trainer):
    def __init__(self, *args, warmup_steps: int = 200, **kwargs):
        super().__init__(*args, **kwargs)
        self.warmup_steps = warmup_steps

    def configure_optimizers(self):
        opt = AdamW(self.model.parameters(), lr=self.config.lr)  # same optimizer as maestro's
        total = int(self.trainer.estimated_stepping_batches)      # optimizer steps (accum + DDP aware)
        sched = LambdaLR(opt, lambda s: lr_factor(s, self.warmup_steps, total))
        print(f"LR schedule: warmup {self.warmup_steps} steps, cosine to 1% of peak {self.config.lr:g} "
              f"over {total} optimizer steps")
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1}}


# ------------------------------------------------------------------- augmentation
LOC_RE = re.compile(r"<loc(\d{4})><loc(\d{4})><loc(\d{4})><loc(\d{4})>")


@dataclass
class AugConfig:
    thresholds_json: str
    p_geometric: float = 0.5      # small shift + scale
    p_intensity: float = 0.5      # brightness/contrast jitter
    scale: tuple = (0.90, 1.10)
    shift_frac: float = 0.05      # +-5% of the image side
    contrast: tuple = (0.90, 1.10)
    brightness_off: float = 12.8  # +-10% of 128 grey levels
    min_box_kept: float = 0.80    # skip the geometric change if it would push >20% of the box out of frame
    label: str = "hemorrhage"
    class_phrase: str = "epidural hemorrhage"

    def __post_init__(self):
        t = json.loads(Path(self.thresholds_json).read_text())
        self.size_thr = t["size_thresholds"]
        self.shape_thr = t["shape_thresholds"]


def decode_boxes(suffix, W, H):
    return [[int(x0) / 1024 * W, int(y0) / 1024 * H, int(x1) / 1024 * W, int(y1) / 1024 * H]
            for y0, x0, y1, x1 in LOC_RE.findall(suffix)]


def prompt_for_box(box, W, H, cfg):
    """Same rules/thresholds as build_prompt_dataset (single source of truth)."""
    w, h = box[2] - box[0], box[3] - box[1]
    hor, ver, region = B.infer_image_region((box[0] + box[2]) / 2 / W, (box[1] + box[3]) / 2 / H)
    lesion = {"relative_size": B.assign_relative_size(w * h / (W * H), cfg.size_thr),
              "bbox_shape": B.assign_shape(w / h, cfg.shape_thr["wide_if_aspect_gt"], cfg.shape_thr["tall_if_aspect_lt"]),
              "image_horizontal_position": hor, "image_vertical_position": ver, "image_region": region}
    return B.lesion_prompt("size_shape_region", lesion, cfg.class_phrase)


def augment_sample(image: Image.Image, entry: dict, cfg: AugConfig, rnd=random):
    W, H = image.size
    boxes = decode_boxes(entry["suffix"], W, H)
    arr = np.asarray(image, dtype=np.float32)
    new_prefix, new_suffix, changed = entry["prefix"], entry["suffix"], False

    if len(boxes) == 1 and rnd.random() < cfg.p_geometric:
        s = rnd.uniform(*cfg.scale)
        tx, ty = rnd.uniform(-cfg.shift_frac, cfg.shift_frac) * W, rnd.uniform(-cfg.shift_frac, cfg.shift_frac) * H
        ox, oy = W / 2 * (1 - s) + tx, H / 2 * (1 - s) + ty          # x' = s*x + ox
        x0, y0, x1, y1 = boxes[0]
        nb = [s * x0 + ox, s * y0 + oy, s * x1 + ox, s * y1 + oy]
        full_area = (nb[2] - nb[0]) * (nb[3] - nb[1])
        cb = [max(0.0, nb[0]), max(0.0, nb[1]), min(float(W), nb[2]), min(float(H), nb[3])]
        if cb[2] - cb[0] >= 4 and cb[3] - cb[1] >= 4 and (cb[2] - cb[0]) * (cb[3] - cb[1]) >= cfg.min_box_kept * full_area:
            M = np.array([[s, 0, ox], [0, s, oy]], dtype=np.float32)
            arr = cv2.warpAffine(arr, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            new_suffix = B.loc_string(cb, W, H, cfg.label) + "<eos>"
            # Describe the box AS ENCODED (1024-step grid), not the exact float box: a box within a
            # fraction of a pixel of a size/shape/region boundary can land on the other side after
            # quantization, and the target the model learns -- and that evaluation decodes -- is the
            # quantized one. (Caught by test_training_extras.py: 12 of 1486 prompts were off otherwise.)
            prompt_box = decode_boxes(new_suffix, W, H)[0]
            new_prefix = prompt_for_box(prompt_box, W, H, cfg)
            changed = True

    if rnd.random() < cfg.p_intensity:
        c = rnd.uniform(*cfg.contrast)
        off = rnd.uniform(-cfg.brightness_off, cfg.brightness_off)
        arr = np.clip((arr - 128.0) * c + 128.0 + off, 0, 255)
        changed = True

    if not changed:
        return image, entry
    return Image.fromarray(arr.astype(np.uint8)), {**entry, "prefix": new_prefix, "suffix": new_suffix}


def augmented_train_collate(batch, processor, max_length, cfg):
    return train_collate_fn([augment_sample(img, entry, cfg) for img, entry in batch],
                            processor=processor, max_length=max_length)


# ---------------------------------------------------------------- per-epoch metrics
class EpochMetricsCSV(Callback):
    """After each validation pass (rank 0): append epoch, lr, validation metrics to a CSV and print them.
    Metrics are what maestro's validation_step logs (per-image mAP averaged over this rank's
    validation shard; with 1 box per image, map50 ~ share of images with IoU>=0.5)."""

    KEYS = ("map50:95", "map50", "map75")

    def __init__(self, path: str, keys=None):
        self.path = path
        self.KEYS = tuple(keys) if keys else self.KEYS   # captioning passes its own metric names
        self.t0 = time.time()

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        m = {k: float(trainer.callback_metrics[k]) for k in self.KEYS if k in trainer.callback_metrics}
        lr = trainer.optimizers[0].param_groups[0]["lr"] if trainer.optimizers else float("nan")
        row = {"epoch": trainer.current_epoch + 1, "lr": f"{lr:.3e}", **{k: f"{m.get(k, float('nan')):.4f}" for k in self.KEYS},
               "minutes_since_start": f"{(time.time() - self.t0) / 60:.1f}"}
        new = not Path(self.path).exists()
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        shown = " ".join(f"{k}={row[k]}" for k in self.KEYS)
        print(f"[epoch {row['epoch']}] validation: {shown} | lr={row['lr']} | {row['minutes_since_start']} min elapsed")
