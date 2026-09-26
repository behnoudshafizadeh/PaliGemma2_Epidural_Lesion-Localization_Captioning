"""
build_bundle_dataset.py -- dataset for the "learn each fact separately" captioning bundle (training/caption_bundle.py).

Reads the 448x448 lesion-only images (dataset/lesion_only/, made by build_lesion_only_variant.py) and

  1. MEASURES the five fact quantities from each image itself with caption_bundle.measure()
     (axis ratio, main-axis angle, solidity, mean grey, grey std) -- exactly the function that re-measures the facts
     after every augmentation during training, so offline labels and online labels can never disagree;
  2. fits train-only terciles -> features/feature_thresholds_bundle.json;
  3. writes training/datasets/lesion_bundle/{train,valid,test}/annotations.jsonl (+ image symlinks):
       valid/test rows : generic spec prompt + full five-fact caption (the standard evaluation target)
       train rows      : BALANCED RESAMPLE of the training lesions (weights = mean over the 5 facts of the inverse
                         class frequency, times an inverse-sqrt patient factor so no patient dominates); prompt/caption
                         are only placeholders -- the training collate composes a fresh random-subset question and
                         answer for every draw (see caption_bundle.py), so duplicates are never identical;
  4. reports how the balancing changed each fact's class distribution.

    python build_bundle_dataset.py
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "training"))
import caption_bundle as CB                    # noqa: E402

SPLIT_NAME = {"train": "train", "val": "valid", "test": "test"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_dir", default=str(HERE))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--draws", type=int, default=None, help="size of the resampled train set (default: number of train lesions)")
    ap.add_argument("--patient_power", type=float, default=0.5, help="weight ~ patient_count^-power (0 = ignore patients)")
    ap.add_argument("--scheme", choices=["class", "combo", "patient"], default="patient",
                    help="patient (default): balance across patients only. Compared on the real labels: class/combo weighting skews the "
                         "tercile balance (shape 39/34/27) and repeats single lesions up to 18x while barely reducing rare combinations, "
                         "so it is not used; rare combinations are handled by the fact-subset prompts instead.")
    ap.add_argument("--combo_power", type=float, default=0.5, help="weight ~ combination_count^-power (scheme=combo)")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    P = Path(a.project_dir)
    rows = [json.loads(l) for l in open(P / "features" / "lesion_features.jsonl")]
    if a.limit:
        rows = rows[:a.limit]

    meas = {}
    for r in rows:
        img = cv2.imread(str(P / "dataset" / "lesion_only" / f"{r['sample_id']}.png"), cv2.IMREAD_GRAYSCALE)
        meas[r["sample_id"]] = CB.measure(img)
        r["patient"] = r["sample_id"].split("_")[0]

    tr = [r for r in rows if r["split"] == "train"]
    q = lambda key: [float(np.quantile([meas[r["sample_id"]][key] for r in tr], x)) for x in (0.33, 0.67)]
    thr = {"axis_ratio": q("axis_ratio"), "solidity": q("solidity"), "mean": q("mean"), "std": q("std"),
           "orientation_bins_deg": "horizontal |angle|<=22.5, diagonal <=67.5, vertical >67.5, none if axis ratio<1.5",
           "n_train": len(tr), "note": "fitted on TRAIN lesions, measured on the 448x448 lesion-only image; PNG grey values, not HU"}
    (P / "features" / "feature_thresholds_bundle.json").write_text(json.dumps(thr, indent=2))
    for r in rows:
        r["labels"] = CB.label_from_measures(meas[r["sample_id"]], thr)
        r["measures"] = meas[r["sample_id"]]

    with open(P / "features" / "lesion_bundle_labels.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({"sample_id": r["sample_id"], "split": r["split"], "patient": r["patient"], "labels": r["labels"],
                                "measures": r["measures"]}) + "\n")

    # ---- balanced resample of the training lesions
    pat = Counter(r["patient"] for r in tr)
    combo = Counter(tuple(r["labels"][f] for f in CB.FACTS) for r in tr)
    freq = {f: Counter(r["labels"][f] for r in tr) for f in CB.FACTS}
    K = {f: len(freq[f]) for f in CB.FACTS}
    if a.scheme == "class":       # mean inverse class frequency of the five facts (first attempt: distorts the marginals)
        w = np.array([np.mean([1.0 / (K[f] * freq[f][r["labels"][f]] / len(tr)) for f in CB.FACTS]) for r in tr])
    elif a.scheme == "combo":     # damped inverse frequency of the whole label combination
        w = np.array([combo[tuple(r["labels"][f] for f in CB.FACTS)] ** (-a.combo_power) for r in tr])
    else:                         # "patient": only patient balancing
        w = np.ones(len(tr))
    w = w * np.array([pat[r["patient"]] ** (-a.patient_power) for r in tr])
    w /= w.sum()
    rng = np.random.default_rng(a.seed)
    n = a.draws or len(tr)
    idx = rng.choice(len(tr), size=n, replace=True, p=w)
    resampled = [tr[i] for i in idx]

    def write(split, lst, dynamic):
        d = P / "training" / "datasets" / "lesion_bundle" / SPLIT_NAME[split]
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "annotations.jsonl", "w") as f:
            for r in lst:
                prompt, cap = CB.compose(r["labels"], CB.FACTS)
                f.write(json.dumps({"image": f"{r['sample_id']}.png", "prefix": prompt, "suffix": cap + "<eos>",
                                    "sample_id": r["sample_id"], "patient": r["patient"], "labels": r["labels"],
                                    "dynamic_caption": dynamic}) + "\n")
                dst = d / f"{r['sample_id']}.png"
                if not dst.is_symlink() and not dst.exists():
                    dst.symlink_to((P / "dataset" / "lesion_only" / f"{r['sample_id']}.png").resolve())
    write("train", resampled, True)
    write("val", [r for r in rows if r["split"] == "val"], False)
    write("test", [r for r in rows if r["split"] == "test"], False)

    # ---- report
    def dist(lst, f):
        c = Counter(r["labels"][f] for r in lst)
        t = sum(c.values())
        return {k: f"{v} ({v / t * 100:.0f}%)" for k, v in sorted(c.items())}
    st = ["# lesion_bundle statistics", "",
          f"- lesions: {len(rows)}  train {len(tr)}  val {sum(r['split'] == 'val' for r in rows)}  test {sum(r['split'] == 'test' for r in rows)}",
          f"- resampled train set: {n} draws from {len(tr)} lesions; distinct lesions drawn: {len(set(idx.tolist()))}; "
          f"most repeated lesion drawn {Counter(idx.tolist()).most_common(1)[0][1]}x; patients in train: {len(pat)}",
          f"- thresholds (train terciles, 448 image): axis ratio {thr['axis_ratio']}, solidity {thr['solidity']}, mean {thr['mean']}, std {thr['std']}", "",
          "## class distribution of each fact: original train -> balanced resample -> val -> test", ""]
    for f in CB.FACTS:
        st.append(f"- {f}: {dist(tr, f)}  ->  {dist(resampled, f)}  | val {dist([r for r in rows if r['split'] == 'val'], f)}  | test {dist([r for r in rows if r['split'] == 'test'], f)}")
    combos = Counter(tuple(r["labels"][f] for f in CB.FACTS) for r in tr)
    combos_rs = Counter(tuple(r["labels"][f] for f in CB.FACTS) for r in resampled)
    st += ["", f"- label combinations in train: {len(combos)} of 324 possible; combinations with <10 train lesions: {sum(1 for v in combos.values() if v < 10)}",
           f"- after resampling: combinations with <10 draws: {sum(1 for v in combos_rs.values() if v < 10)} (most common combination: "
           f"{combos.most_common(1)[0][1]} -> {max(combos_rs.values())} draws)"]
    (P / "reports" / "lesion_bundle_statistics.md").write_text("\n".join(st) + "\n")
    print("\n".join(st))
    print("\nexample (prompt -> caption) for one lesion, every subset size:")
    r0 = rows[0]
    rnd = random.Random(1)
    for _ in range(4):
        p, c = CB.compose(r0["labels"], CB.sample_facts(rnd))
        print(f"  {p}\n    -> {c}")


if __name__ == "__main__":
    main()
