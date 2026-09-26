"""
evaluate_captioning.py -- fair, per-fact evaluation of the lesion captioner (Phase 7 of the spec, bundle version).

For every lesion of the chosen split it asks the model
    generic         "Describe the visible characteristics of this epidural hemorrhage."      (all five facts)
    explicit_full   "Describe the shape, main-axis direction, ... of this epidural hemorrhage." (all five facts)
    single_<fact>   "Describe the <fact> of this epidural hemorrhage."                        (one fact, x5)
parses the generated caption (controlled vocabulary -> exact parse) and reports, PER FACT:
    accuracy, balanced accuracy (mean recall over classes), the "always answer the most common training class" baseline,
    uniform chance, fraction of answers that state the fact at all, and a 95% CI from resampling PATIENTS (the real unit of
    independence -- consecutive slices of one patient are near-duplicates).
Also: confusion matrices, accuracy on label combinations seen vs never seen in training, ROUGE-L / BLEU-4 of the generic
answer against the composed reference caption, and per-patient scores.

NOTE on validation vs test: the validation split contains NO lesion with a horizontal main axis (test has 182), so the
direction fact can only be judged on the test split.

Usage (one shard per GPU, then aggregate):
  CUDA_VISIBLE_DEVICES=0 python evaluate_captioning.py --adapter_dir <snapshot> --split test --shard 0 --num_shards 2
  CUDA_VISIBLE_DEVICES=1 python evaluate_captioning.py --adapter_dir <snapshot> --split test --shard 1 --num_shards 2
  python evaluate_captioning.py --aggregate --split test
  python evaluate_captioning.py --fake_oracle --split val --limit 200      # pipeline test without a model
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path[:0] = [str(HERE), str(PROJ)]
import caption_bundle as CB                                                   # noqa: E402
from build_captioning_dataset import parse_caption                            # noqa: E402
from caption_metrics import bleu4, clean, letterbox, rouge_l                  # noqa: E402

BASE = "google/paligemma2-3b-pt-448"
SPLIT_DIR = {"val": "valid", "test": "test"}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_dir")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--dataset", default="lesion_bundle")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--out_prefix", default="evalcap")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--fake_oracle", action="store_true", help="pipeline test: answer with the correct facts")
    ap.add_argument("--fake_majority", action="store_true", help="pipeline test: always answer the most common training class")
    return ap.parse_args()


def load_split(a):
    p = PROJ / "training" / "datasets" / a.dataset / SPLIT_DIR[a.split] / "annotations.jsonl"
    rows = [json.loads(l) for l in open(p)]
    return rows[:a.limit] if a.limit else rows


def train_labels():
    return [json.loads(l) for l in open(PROJ / "features" / "lesion_bundle_labels.jsonl")]


def prompt_types():
    t = {"generic": (CB.GENERIC_PROMPT, CB.FACTS), "explicit_full": (CB.explicit_full_prompt(), CB.FACTS)}
    for f in CB.FACTS:
        t[f"single_{f}"] = (CB.compose({"shape": "low", "orientation": "vertical", "outline": "low", "intensity": "low", "variation": "low"}, [f])[0], [f])
    return t


def majority(rows_train):
    return {f: Counter(r["labels"][f] for r in rows_train).most_common(1)[0][0] for f in CB.FACTS}


def run_shard(a):
    rows = load_split(a)
    idx = list(range(a.shard, len(rows), a.num_shards))
    types = prompt_types()
    maj = majority([r for r in train_labels() if r["split"] == "train"])
    model = proc = None
    if not (a.fake_oracle or a.fake_majority):
        import torch
        from peft import PeftModel
        from PIL import Image
        from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
        proc = PaliGemmaProcessor.from_pretrained(BASE)
        base = PaliGemmaForConditionalGeneration.from_pretrained(BASE, torch_dtype=torch.bfloat16).to("cuda")
        model = PeftModel.from_pretrained(base, a.adapter_dir).to("cuda").eval()
    ds_dir = PROJ / "training" / "datasets" / a.dataset / SPLIT_DIR[a.split]
    with open(HERE / f"{a.out_prefix}_{a.split}_shard{a.shard}.jsonl", "w") as f:
        for n, i in enumerate(idx):
            r = rows[i]
            for name, (prompt, facts) in types.items():
                if a.fake_oracle:
                    pred = CB.compose(r["labels"], facts)[1]
                elif a.fake_majority:
                    pred = CB.compose(maj, facts)[1]
                else:
                    img = letterbox(Image.open(ds_dir / r["image"]).convert("RGB"))
                    inp = proc(text="<image>" + prompt, images=img, return_tensors="pt").to("cuda", torch.bfloat16)
                    with torch.no_grad():
                        g = model.generate(**inp, max_new_tokens=a.max_new_tokens, do_sample=False)
                    pred = proc.decode(g[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True)
                f.write(json.dumps({"row": i, "sample_id": r["sample_id"], "patient": r["patient"], "prompt_type": name,
                                    "prompt": prompt, "pred": pred, "labels": r["labels"]}) + "\n")
            f.flush()
            if n % 20 == 0:
                print(f"shard {a.shard}: {n}/{len(idx)} lesions x {len(types)} prompts", flush=True)


ORDINAL = {"shape": {"low": 0, "mid": 1, "high": 2}, "outline": {"low": 0, "mid": 1, "high": 2}, "intensity": {"low": 0, "mid": 1, "high": 2},
           "variation": {"low": 0, "mid": 1, "high": 2}, "orientation": {"vertical": 0, "diagonal": 1, "horizontal": 2}}


def ordinal_stats(pairs, f):
    """For ordered facts an off-by-one answer is less wrong than off-by-two: (share within one level, mean level error)."""
    mp = ORDINAL.get(f)
    errs = [abs(mp[y] - mp[p]) for y, p in pairs if mp and y in mp and p in mp]
    return (f"{np.mean([e <= 1 for e in errs]) * 100:5.1f}%", f"{np.mean(errs):.2f}") if errs else ("   n/a", "  n/a")


def balanced_acc(pairs):
    per = defaultdict(list)
    for y, p in pairs:
        per[y].append(p == y)
    return float(np.mean([np.mean(v) for v in per.values()])) if per else float("nan")


def aggregate(a):
    recs = []
    for f in sorted(HERE.glob(f"{a.out_prefix}_{a.split}_shard*.jsonl")):
        recs += [json.loads(l) for l in open(f)]
    tr = [r for r in train_labels() if r["split"] == "train"]
    maj = majority(tr)
    classes = {f: sorted({r["labels"][f] for r in tr}) for f in CB.FACTS}
    train_combos = {tuple(r["labels"][f] for f in CB.FACTS) for r in tr}
    by_type = defaultdict(list)
    for r in recs:
        by_type[r["prompt_type"]].append(r)
    n_l = len({r["sample_id"] for r in recs})
    L = [f"CAPTION EVALUATION on the {a.split.upper()} split: {n_l} lesions, {len({r['patient'] for r in recs})} patients", ""]
    rng = random.Random(0)
    summary = {}
    for name in ["generic", "explicit_full"] + [f"single_{f}" for f in CB.FACTS]:
        if name not in by_type:
            continue
        rs = by_type[name]
        facts = CB.FACTS if name in ("generic", "explicit_full") else [name.split("_", 1)[1]]
        L.append(f"[{name}]  prompt: \"{rs[0]['prompt']}\"")
        L.append(f"   {'fact':12s} {'accuracy':>8s} {'95% CI (patients)':>19s} {'balanced':>9s} {'majority-class baseline':>24s} {'chance':>7s} {'stated':>7s} {'within 1':>9s} {'level err':>9s}")
        pats = sorted({r["patient"] for r in rs})
        for f in facts:
            pairs, stated = [], 0
            per_patient = defaultdict(list)
            for r in rs:
                got = parse_caption(clean(r["pred"])).get(f)
                stated += got is not None
                ok = got == r["labels"][f]
                pairs.append((r["labels"][f], got))
                per_patient[r["patient"]].append(ok)
            acc = float(np.mean([p == y for y, p in pairs]))
            base = float(np.mean([r["labels"][f] == maj[f] for r in rs]))
            boots = []
            for _ in range(500):
                pick = [rng.choice(pats) for _ in pats]
                boots.append(np.mean([x for p in pick for x in per_patient[p]]))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            L.append(f"   {f:12s} {acc*100:7.1f}% {'[%.1f, %.1f]' % (lo*100, hi*100):>19s} {balanced_acc(pairs)*100:8.1f}% "
                     f"{base*100:15.1f}% ({maj[f]:>8s}) {100/len(classes[f]):6.1f}% {stated/len(rs)*100:6.1f}% "
                     f"{ordinal_stats(pairs, f)[0]:>9s} {ordinal_stats(pairs, f)[1]:>9s}")
            summary[(name, f)] = (acc, base)
        if name == "generic":
            ref = CB.compose(rs[0]["labels"], CB.FACTS)[1]
            r_l = float(np.mean([rouge_l(CB.compose(r["labels"], CB.FACTS)[1], r["pred"]) for r in rs]))
            b4 = bleu4([CB.compose(r["labels"], CB.FACTS)[1] for r in rs], [r["pred"] for r in rs])
            exact = float(np.mean([parse_caption(clean(r["pred"])) == {f: r["labels"][f] for f in CB.FACTS} for r in rs]))
            L.append(f"   text similarity vs composed reference caption: ROUGE-L {r_l:.3f}   BLEU-4 {b4:.3f}   "
                     f"all five facts right at once: {exact*100:.1f}%")
            seen = [r for r in rs if tuple(r["labels"][f] for f in CB.FACTS) in train_combos]
            unseen = [r for r in rs if tuple(r["labels"][f] for f in CB.FACTS) not in train_combos]
            for nm, sub in (("label combination SEEN in training", seen), ("label combination NEVER seen in training", unseen)):
                if sub:
                    m = np.mean([np.mean([parse_caption(clean(r["pred"])).get(f) == r["labels"][f] for f in CB.FACTS]) for r in sub])
                    L.append(f"   {nm}: n={len(sub)}, mean per-fact accuracy {m*100:.1f}%")
        L.append("")
    if "single_orientation" in by_type or "generic" in by_type:
        L.append("== confusion (rows = true, columns = answered) for the single-fact prompts ==")
        for f in CB.FACTS:
            key = f"single_{f}" if f"single_{f}" in by_type else "generic"
            cm = Counter((r["labels"][f], parse_caption(clean(r["pred"])).get(f, "none stated")) for r in by_type[key])
            cols = classes[f] + ["none stated"]
            L.append(f"  {f} ({key}):")
            L.append("    " + " " * 10 + "".join(f"{c:>12s}" for c in cols))
            for t in classes[f]:
                L.append("    " + f"{t:>10s}" + "".join(f"{cm[(t, c)]:12d}" for c in cols))
        L.append("")
    if "generic" in by_type:
        L.append("== per patient (generic prompt, mean accuracy over the five facts) ==")
        pp = defaultdict(list)
        for r in by_type["generic"]:
            g = parse_caption(clean(r["pred"]))
            pp[r["patient"]].append(np.mean([g.get(f) == r["labels"][f] for f in CB.FACTS]))
        for p in sorted(pp):
            L.append(f"  {p:>4s}: n={len(pp[p]):4d}  {np.mean(pp[p])*100:5.1f}%")
    text = "\n".join(L)
    (HERE / f"{a.out_prefix}_{a.split}_summary.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    a = parse_args()
    aggregate(a) if a.aggregate else run_shard(a)
