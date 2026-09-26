"""
evaluate_prompts.py -- test protocol for the size_shape_region model.

Question it answers: on held-out patients, does the model use the IMAGE, the PROMPT, or both?
Every row of datasets/size_shape_region/<split> is a (image, lesion) pair whose prompt was built from
that lesion's own box, so the "right prompt" for each test image is known: it is that row's prompt.

Conditions (same images, same targets, only the prompt changes):
  matched     the row's own size+shape+region prompt              (what the model was trained on)
  baseline    "detect epidural hemorrhage"  (no hint)             (a hint-free, deployable prompt)
  size_only   only the size word        region_only   only the region words
  mismatched  the prompt of ANOTHER row with a different size AND region (a deliberately WRONG hint)

Plus a model-free reference: prior_box = the mean TRAIN box for the prompt's (size,shape,region).
If the fine-tuned model is not clearly better than prior_box under `matched`, its score is mostly the
prompt talking, not the model looking at the image.

Prompt obedience (mismatched condition): does the predicted box's region/size follow the WRONG prompt
(obedience) or the IMAGE (true lesion)? Both are reported.

Protocol: pick the snapshot by VALIDATION (matched) only, then run the TEST split once with it.

Usage (one shard per GPU, then aggregate):
  CUDA_VISIBLE_DEVICES=0 python evaluate_prompts.py --adapter_dir <snapshot> --split test --shard 0 --num_shards 2
  CUDA_VISIBLE_DEVICES=1 python evaluate_prompts.py --adapter_dir <snapshot> --split test --shard 1 --num_shards 2
  python evaluate_prompts.py --aggregate --split test --out_prefix eval_test
  python evaluate_prompts.py --prior_only            # CPU only, no model
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import supervision as sv
from supervision.metrics.mean_average_precision import MeanAveragePrecision

import audit_prompts as A
import build_prompt_dataset as B

HERE = Path(__file__).resolve().parent
BASE_MODEL_ID = "google/paligemma2-3b-pt-448"
CLS = "epidural hemorrhage"
CONDITIONS = ["matched", "baseline", "size_only", "region_only", "mismatched"]
DATA = A.DS / "size_shape_region"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter_dir")
    p.add_argument("--split", default="test", choices=["valid", "test"])
    p.add_argument("--conditions", default=",".join(CONDITIONS))
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=48)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out_prefix", default="eval_prompts")
    p.add_argument("--aggregate", action="store_true")
    p.add_argument("--prior_only", action="store_true")
    p.add_argument("--fake_oracle", action="store_true", help="test the pipeline without a model: answer = the target")
    return p.parse_args()


# ------------------------------------------------------------------------- helpers
def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def to_det(text, wh=(512, 512)):
    s = text.replace("<eos>", "")
    if "<loc" not in s:
        return sv.Detections.empty()
    try:
        return sv.Detections.from_lmm(sv.LMM.PALIGEMMA, s, resolution_wh=wh, classes=["hemorrhage"])
    except Exception:
        return sv.Detections.empty()


def lesion_from_box(box, W, H):
    d = A.derive(box, W, H)
    return {"relative_size": d["size"], "bbox_shape": d["shape"], "image_horizontal_position": d["h"],
            "image_vertical_position": d["v"], "image_region": d["region"]}


def load_rows(split):
    out = []
    for r in A.load_rows("size_shape_region", split):
        W, H = A.img_size(split, "size_shape_region", r["image"])
        box = A.decode(r["suffix"], W, H)[0]
        d = A.derive(box, W, H)
        out.append({**r, "W": W, "H": H, "box": box, "size": d["size"], "shape": d["shape"], "region": d["region"]})
    return out


def build_prompts(rows, condition, seed=13):
    """-> list of prompt strings aligned with rows."""
    if condition == "matched":
        return [r["prefix"] for r in rows]
    if condition == "baseline":
        return [B.PROMPT_TEMPLATES["baseline"].format(cls=CLS)] * len(rows)
    if condition in ("size_only", "region_only"):
        g = "size" if condition == "size_only" else "region"
        return [B.lesion_prompt(g, lesion_from_box(r["box"], r["W"], r["H"]), CLS) for r in rows]
    if condition == "mismatched":
        rng, out = random.Random(seed), []
        for r in rows:
            cands = [o for o in rows if o["region"] != r["region"] and o["size"] != r["size"]] or \
                    [o for o in rows if o["region"] != r["region"]]
            out.append(rng.choice(cands)["prefix"])
        return out
    raise ValueError(condition)


def prior_boxes():
    """Mean TRAIN box per (size,shape,region); back-off to (shape,region) then region then global."""
    tr = load_rows("train")
    groups = defaultdict(list)
    for r in tr:
        for key in ((r["size"], r["shape"], r["region"]), (r["shape"], r["region"]), (r["region"],), ()):
            groups[key].append(r["box"])
    mean = {k: np.mean(v, axis=0) for k, v in groups.items()}

    def get(size, shape, region):
        for key in ((size, shape, region), (shape, region), (region,), ()):
            if key in mean:
                return mean[key].tolist()
    return get


def prior_only(split):
    get = prior_boxes()
    rows = load_rows(split)
    vals = np.array([iou(get(r["size"], r["shape"], r["region"]), r["box"]) for r in rows])
    return vals


def summarize(v):
    v = np.array(v)
    return (f"n={len(v):<5} mean IoU {v.mean():.3f}  median {np.median(v):.3f}  IoU>=0.5 {np.mean(v >= .5)*100:5.1f}%  "
            f"IoU>=0.3 {np.mean(v >= .3)*100:5.1f}%  miss(0) {np.mean(v == 0)*100:4.1f}%")


# ------------------------------------------------------------------------ generation
def run_shard(a):
    rows = load_rows(a.split)
    if a.limit:
        rows = rows[: a.limit]
    conds = a.conditions.split(",")
    prompts = {c: build_prompts(rows, c) for c in conds}
    idx = list(range(a.shard, len(rows), a.num_shards))
    model = processor = None
    if not a.fake_oracle:
        import torch
        from peft import PeftModel
        from PIL import Image
        from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
        processor = PaliGemmaProcessor.from_pretrained(BASE_MODEL_ID)
        base = PaliGemmaForConditionalGeneration.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.bfloat16).to("cuda")
        model = PeftModel.from_pretrained(base, str(HERE / a.adapter_dir)).to("cuda").eval()
    out_path = HERE / f"{a.out_prefix}_{a.split}_shard{a.shard}.jsonl"
    with open(out_path, "w") as f:
        for n, i in enumerate(idx):
            r = rows[i]
            for c in conds:
                if a.fake_oracle:
                    pred = r["suffix"]
                else:
                    image = Image.open(DATA / a.split / r["image"]).convert("RGB")
                    inputs = processor(text="<image>" + prompts[c][i], images=image, return_tensors="pt").to("cuda", torch.bfloat16)
                    with torch.no_grad():
                        gen = model.generate(**inputs, max_new_tokens=a.max_new_tokens, do_sample=False)
                    pred = processor.decode(gen[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
                f.write(json.dumps({"row": i, "image": r["image"], "condition": c, "prompt": prompts[c][i],
                                    "gt": r["suffix"], "pred": pred}) + "\n")
            f.flush()
            if n % 20 == 0:
                print(f"shard {a.shard}: {n}/{len(idx)} rows x {len(conds)} conditions", flush=True)


# ------------------------------------------------------------------------ aggregation
def aggregate(a):
    rows = load_rows(a.split)
    recs = []
    for f in sorted(HERE.glob(f"{a.out_prefix}_{a.split}_shard*.jsonl")):
        recs += [json.loads(l) for l in open(f)]
    by_cond = defaultdict(dict)
    for r in recs:
        by_cond[r["condition"]][r["row"]] = r
    get = prior_boxes()
    lines = [f"PROMPT EVALUATION on the {a.split.upper()} split ({len(rows)} rows = image+lesion pairs)", ""]
    per_row = defaultdict(dict)
    boxes_all = {}
    for c in [c for c in CONDITIONS if c in by_cond]:
        ious, nbox, nobox, first, allb, tgt = [], [], 0, [], [], []
        for i, rec in sorted(by_cond[c].items()):
            r = rows[i]
            det = to_det(rec["pred"], (r["W"], r["H"]))   # decode with the TRUE image size (some images are not square)
            nbox.append(len(det))
            v = iou(det.xyxy[0], r["box"]) if len(det) else 0.0
            nobox += len(det) == 0
            ious.append(v)
            per_row[i][c] = v
            boxes_all[(c, i)] = det.xyxy[0].tolist() if len(det) else None
            t = to_det(r["suffix"], (r["W"], r["H"]))
            tgt.append(t)
            d = to_det(rec["pred"], (r["W"], r["H"]))
            d.confidence = np.ones(len(d))
            allb.append(d)
            f1 = d[:1] if len(d) else d
            if len(f1):
                f1.confidence = np.ones(len(f1))
            first.append(f1)
        m_all = MeanAveragePrecision().update(targets=tgt, predictions=allb).compute()
        m_first = MeanAveragePrecision().update(targets=tgt, predictions=first).compute()
        lines.append(f"[{c}]  {summarize(ious)}")
        lines.append(f"    boxes per answer: mean {np.mean(nbox):.2f} (target 1); answers with no box: {nobox}")
        lines.append(f"    mAP50 all boxes {m_all.map50:.3f} | first box only {m_first.map50:.3f}    "
                     f"mAP50:95 all {m_all.map50_95:.3f} | first only {m_first.map50_95:.3f}")
    pri = np.array([iou(get(r["size"], r["shape"], r["region"]), r["box"]) for r in rows])
    lines += ["", "[prior_box  (NO model, NO image: mean TRAIN box for the prompt's size/shape/region)]", "    " + summarize(pri)]

    if "matched" in by_cond:
        lines += ["", "== How much does the model add beyond the prompt alone? =="]
        mi = np.array([per_row[i]["matched"] for i in sorted(by_cond["matched"])])
        pi = pri[sorted(by_cond["matched"])]
        lines.append(f"matched model mean IoU {mi.mean():.3f}  vs  prior_box {pi.mean():.3f}   "
                     f"(model better on {np.mean(mi > pi)*100:.1f}% of rows, worse on {np.mean(mi < pi)*100:.1f}%)")
    if "matched" in by_cond and "baseline" in by_cond:
        ks = sorted(set(by_cond["matched"]) & set(by_cond["baseline"]))
        m = np.array([per_row[i]["matched"] for i in ks]); b = np.array([per_row[i]["baseline"] for i in ks])
        lines.append(f"hint helps: matched {m.mean():.3f} vs baseline(no hint) {b.mean():.3f}; "
                     f"matched > baseline on {np.mean(m > b)*100:.1f}% of rows, < on {np.mean(m < b)*100:.1f}%")
    if "mismatched" in by_cond:
        lines += ["", "== Does the model obey the prompt or look at the image? (mismatched = deliberately wrong hint) =="]
        follow_p = follow_i = size_p = size_i = n = 0
        for i, rec in by_cond["mismatched"].items():
            b = boxes_all.get(("mismatched", i))
            if b is None:
                continue
            claims = A.claims(rec["prompt"])
            d = A.derive(b, rows[i]["W"], rows[i]["H"])
            n += 1
            follow_p += d["region"] == claims["region"]
            follow_i += d["region"] == rows[i]["region"]
            size_p += d["size"] == claims["size"]
            size_i += d["size"] == rows[i]["size"]
        lines.append(f"predicted region = the WRONG prompt's region: {follow_p/n*100:.1f}%  | = the image's true region: {follow_i/n*100:.1f}%  (n={n})")
        lines.append(f"predicted size   = the WRONG prompt's size:   {size_p/n*100:.1f}%  | = the image's true size:   {size_i/n*100:.1f}%")
        lines.append("  (mostly 'true region' => the model reads the image; mostly 'prompt region' => it obeys the hint)")

    if "matched" in by_cond:
        lines += ["", "== matched vs baseline by lesion size / shape (mean IoU) =="]
        for key, vals in (("size", ["small", "medium", "large"]), ("shape", ["wide", "compact", "tall"])):
            for v in vals:
                ks = [i for i in sorted(by_cond["matched"]) if rows[i][key] == v]
                if ks:
                    cs = "  ".join(f"{c} {np.mean([per_row[i][c] for i in ks if c in per_row[i]]):.3f}"
                                   for c in ("matched", "baseline") if c in by_cond)
                    lines.append(f"  {key}={v:<8} n={len(ks):<4} {cs}   prior_box {pri[ks].mean():.3f}")
        train_keys = {(t["size"], t["shape"], t["region"]) for t in load_rows("train")}
        lines += ["", "== prompt combination seen in training vs never seen (size+shape+region sentence) =="]
        for name, keep in (("SEEN in training   ", True), ("NEVER seen in train", False)):
            ks = [i for i in sorted(by_cond["matched"]) if ((rows[i]["size"], rows[i]["shape"], rows[i]["region"]) in train_keys) == keep]
            if ks:
                cs = "  ".join(f"{c} {np.mean([per_row[i][c] for i in ks if c in per_row[i]]):.3f}"
                               for c in ("matched", "baseline", "mismatched") if c in by_cond)
                lines.append(f"  {name} n={len(ks):<5} mean IoU: {cs}   prior_box {pri[ks].mean():.3f}   matched IoU>=0.5: "
                             f"{np.mean([per_row[i]['matched'] >= 0.5 for i in ks])*100:.1f}%")
        lines += ["", "== per patient (matched mean IoU) =="]
        pp = defaultdict(list)
        for i in sorted(by_cond["matched"]):
            pp[rows[i]["image"].split("_")[0]].append(per_row[i]["matched"])
        for pat in sorted(pp):
            lines.append(f"  {pat:>4}: n={len(pp[pat]):<4} mean IoU {np.mean(pp[pat]):.3f}  IoU>=0.5 {np.mean(np.array(pp[pat]) >= .5)*100:.0f}%")

    with open(HERE / f"{a.out_prefix}_{a.split}_per_row.csv", "w") as f:
        f.write("row,image,prompt_matched,size,shape,region," + ",".join(CONDITIONS) + ",prior_box\n")
        for i in sorted(per_row):
            r = rows[i]
            f.write(f"{i},{r['image']},\"{r['prefix']}\",{r['size']},{r['shape']},{r['region']}," +
                    ",".join(f"{per_row[i].get(c, float('nan')):.3f}" for c in CONDITIONS) + f",{pri[i]:.3f}\n")
    text = "\n".join(lines)
    (HERE / f"{a.out_prefix}_{a.split}_summary.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    a = parse_args()
    if a.prior_only:
        for s in ("valid", "test"):
            print(f"prior_box on {s}: {summarize(prior_only(s))}")
    elif a.aggregate:
        aggregate(a)
    else:
        run_shard(a)
