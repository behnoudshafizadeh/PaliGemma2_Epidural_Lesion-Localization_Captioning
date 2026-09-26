"""
evaluate_validation.py -- runs a LoRA adapter over the WHOLE validation (or test)
split and reports real numbers, replacing the single-image check the training loop
does (limit_val_batches=1) and the 8-image visual spot check.

Two things are measured separately on purpose, because they are different problems:
  * first-box quality: IoU of the FIRST generated box vs ground truth (localization)
  * whole-output quality: mAP over ALL generated boxes. PaliGemma has no stop
    behaviour here, so it emits many extra boxes; every extra box is a false
    positive (all get confidence 1.0), which drags mAP down even when the first
    box is right. Comparing "all boxes" vs "first box only" isolates that effect.

Usage (one shard per GPU, then aggregate):
    CUDA_VISIBLE_DEVICES=0 python evaluate_validation.py --adapter_dir ... --shard 0 --num_shards 2
    CUDA_VISIBLE_DEVICES=1 python evaluate_validation.py --adapter_dir ... --shard 1 --num_shards 2
    python evaluate_validation.py --aggregate --out_prefix eval_final
"""
import argparse
import json
from pathlib import Path

import numpy as np
import supervision as sv
from supervision.metrics.mean_average_precision import MeanAveragePrecision

HERE = Path(__file__).resolve().parent
BASE_MODEL_ID = "google/paligemma2-3b-pt-448"
CLASSES = ["hemorrhage"]
RESOLUTION_WH = (512, 512)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter_dir")
    p.add_argument("--split_dir", default="dataset/valid")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out_prefix", default="eval_final")
    p.add_argument("--aggregate", action="store_true")
    return p.parse_args()


def to_det(text):
    try:
        return sv.Detections.from_lmm(sv.LMM.PALIGEMMA, text, resolution_wh=RESOLUTION_WH, classes=CLASSES)
    except Exception:
        return sv.Detections.empty()


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = max(0, a[2] - a[0]) * max(0, a[3] - a[1]) + max(0, b[2] - b[0]) * max(0, b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def run_shard(args):
    import torch
    from peft import PeftModel
    from PIL import Image
    from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

    split_dir = HERE / args.split_dir
    rows = [json.loads(l) for l in open(split_dir / "annotations.jsonl")]
    if args.limit:
        rows = rows[: args.limit]
    rows = rows[args.shard :: args.num_shards]

    device = "cuda"
    processor = PaliGemmaProcessor.from_pretrained(BASE_MODEL_ID)
    base = PaliGemmaForConditionalGeneration.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.bfloat16).to(device)
    model = PeftModel.from_pretrained(base, str(HERE / args.adapter_dir)).to(device).eval()

    out_path = HERE / f"{args.out_prefix}_shard{args.shard}.jsonl"
    with open(out_path, "w") as fout:
        for i, row in enumerate(rows):
            image = Image.open(split_dir / row["image"]).convert("RGB")
            inputs = processor(text="<image>" + row["prefix"], images=image, return_tensors="pt").to(device, torch.bfloat16)
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            text = processor.decode(gen[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            fout.write(json.dumps({"image": row["image"], "gt": row["suffix"], "pred": text}) + "\n")
            fout.flush()
            if i % 20 == 0:
                print(f"shard {args.shard}: {i}/{len(rows)}", flush=True)


def aggregate(args):
    recs = []
    for f in sorted(HERE.glob(f"{args.out_prefix}_shard*.jsonl")):
        recs += [json.loads(l) for l in open(f)]
    print(f"images evaluated: {len(recs)}")

    first_ious, n_boxes, malformed = [], [], 0
    all_t, all_p, first_p = [], [], []
    per_patient = {}
    for r in recs:
        gt, pd_all = to_det(r["gt"]), to_det(r["pred"])
        n_boxes.append(len(pd_all))
        if len(pd_all) == 0:
            malformed += 1
        # best-overlap GT for the FIRST predicted box
        v = 0.0
        if len(pd_all) > 0 and len(gt) > 0:
            v = max(iou(pd_all.xyxy[0], g) for g in gt.xyxy)
        first_ious.append(v)
        per_patient.setdefault(r["image"].split("_")[0], []).append(v)

        p_all = to_det(r["pred"])
        p_all.confidence = np.ones(len(p_all))
        p_first = p_all[:1] if len(p_all) else p_all
        if len(p_first):
            p_first.confidence = np.ones(len(p_first))
        all_t.append(gt)
        all_p.append(p_all)
        first_p.append(p_first)

    fi = np.array(first_ious)
    print("\n== first-box localization (what a 'keep only first box' post-process gives) ==")
    print(f"mean IoU:            {fi.mean():.3f}")
    print(f"median IoU:          {np.median(fi):.3f}")
    print(f"IoU >= 0.5:          {(fi >= 0.5).mean()*100:.1f}%")
    print(f"IoU >= 0.3:          {(fi >= 0.3).mean()*100:.1f}%")
    print(f"IoU  > 0 (any overlap): {(fi > 0).mean()*100:.1f}%")
    print(f"IoU == 0 (complete miss): {(fi == 0).mean()*100:.1f}%")
    print(f"no parseable box at all:  {malformed} images ({malformed/len(recs)*100:.1f}%)")

    print("\n== repetition problem ==")
    nb = np.array(n_boxes)
    print(f"boxes generated per image: mean {nb.mean():.1f}, median {np.median(nb):.0f}, "
          f"max {nb.max()}  (ground truth is ~1)")

    def m(preds):
        r = MeanAveragePrecision().update(targets=all_t, predictions=preds).compute()
        return r.map50_95, r.map50, r.map75

    a, b, c = m(all_p)
    d, e, f = m(first_p)
    print("\n== mAP: all generated boxes vs first box only ==")
    print(f"all boxes  : mAP50:95={a:.3f}  mAP50={b:.3f}  mAP75={c:.3f}")
    print(f"first only : mAP50:95={d:.3f}  mAP50={e:.3f}  mAP75={f:.3f}")

    print("\n== per-patient mean first-box IoU ==")
    for k in sorted(per_patient):
        v = per_patient[k]
        print(f"{k:>5}: n={len(v):3d}  mean IoU={np.mean(v):.3f}  IoU>=0.5: {np.mean(np.array(v)>=0.5)*100:.0f}%")


if __name__ == "__main__":
    a = parse_args()
    aggregate(a) if a.aggregate else run_shard(a)
