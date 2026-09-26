"""
build_lesion_only_variant.py -- variant "lesion_only" of the captioning dataset (added on the user's request).

Problem with the first crops: they are tiny (median lesion ~76 px on its longer side, 670 tight crops < 16 px
wide) and contain bone / other tissue that is not the lesion. Since every lesion has a segmentation mask:

  1. take the lesion's OWN blob of the mask (the 8-connected component the provided box was made from)
  2. keep ONLY those pixels; everything else (bone, brain, background) becomes black
  3. pad to a square with black (aspect ratio preserved, so "elongated" stays visible)
  4. resize EVERY lesion to the same size (default 448x448 = PaliGemma 2-448's input)

Resizing detail: pixels outside the lesion are first filled with the lesion's mean grey value, the image is
enlarged with bicubic interpolation, and only then is the (bilinear-enlarged, thresholded) mask applied, so bright
bone next to the lesion can't bleed into its edge.

What this variant can NOT show (so its captions never state it): where the lesion sits in the slice, how big it is
compared with the slice (all lesions are rescaled to fill the frame), how bright it is compared with its surroundings
(removed), and texture/edge complexity (measured on 300 lesions: gradient/edge statistics of the enlarged crop correlate
only 0.20/0.17 with the original ones and track the enlargement factor, r=-0.59/-0.43 -- they describe blur).

Captions state FIVE visible, scale-safe facts:
  shape (elongation: axis ratio terciles)     -> less / moderately / highly elongated
  main-axis direction (fixed geometric bins)  -> vertical / diagonal / horizontal, or "no dominant axis" if axis ratio < 1.5
  outline regularity (solidity terciles)      -> low / moderate / high
  overall intensity (mean grey, 448 image)    -> low / moderate / high
  internal variation (grey std, 448 image)    -> low / moderate / high
All thresholds are train-only terciles except the direction bins. Directions are IMAGE directions, not anatomical.
PNG grey values, not Hounsfield units.

In addition ~14 further features are measured from the lesion-only image (skewness, kurtosis, p10/p90, coefficient of
variation, core-vs-rim intensity, intensity trend along the axis, mirror symmetry, taper, ...). Each is also measured at
original resolution; reports/lesion_only_extra_features_reliability.md says which survive the enlargement.

Reads features/lesion_features.jsonl (from build_captioning_dataset.py); nothing else is modified.
    python build_lesion_only_variant.py
"""
import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
import sys                                    # noqa: E402
sys.path.insert(0, str(HERE))
from build_captioning_dataset import BANNED, LEVEL_W, MAESTRO_SPLIT, PROMPT, parse_caption  # noqa: E402

SHAPE_LO = {"low": "less elongated", "mid": "moderately elongated", "high": "highly elongated"}
ORIENT_PHRASE = {"vertical": "a vertical main axis", "diagonal": "a diagonal main axis", "horizontal": "a horizontal main axis",
                 "none": "no dominant axis"}
TEMPLATES = {
    "F1": "The lesion region is {shape}, with {orient} and {outline} outline regularity. It has {intensity} overall intensity "
          "and {variation} internal intensity variation.",
    "F2": "This epidural hemorrhage region is {shape}, has {orient}, and shows {outline} outline regularity, {intensity} overall "
          "intensity and {variation} internal intensity variation.",
    "F3": "{a} {shape} epidural hemorrhage region with {orient}, {outline} outline regularity, {intensity} overall intensity "
          "and {variation} internal intensity variation.",
    "F4": "The segmented epidural hemorrhage is {shape}, with {orient}. It has {outline} outline regularity, {intensity} overall "
          "intensity and {variation} intensity variation.",
}
FIELDS = {k: ["shape", "orientation", "outline", "intensity", "variation"] for k in TEMPLATES}
MIN_WORDS = 20


def article(word):
    return "An" if word[0] in "aeiou" else "A"


def orientation_label(theta_deg, axis_ratio):
    if axis_ratio < 1.5:
        return "none"
    t = abs(theta_deg)
    return "horizontal" if t <= 22.5 else ("diagonal" if t <= 67.5 else "vertical")


def make_caption(sample_id, lab):
    k = ["F1", "F2", "F3", "F4"][int(hashlib.sha1(("lo5:" + sample_id).encode()).hexdigest()[:8], 16) % 4]
    shape = SHAPE_LO[lab["shape"]]
    return k, TEMPLATES[k].format(a=article(shape), shape=shape, orient=ORIENT_PHRASE[lab["orientation"]],
                                  outline=LEVEL_W[lab["outline"]], intensity=LEVEL_W[lab["intensity"]],
                                  variation=LEVEL_W[lab["variation"]])


# ------------------------------------------------------------------ the lesion-only crop
def make_lesion_only(gray, lm, bbox, S, margin):
    x0, y0, x1, y1 = bbox
    sub_orig = gray[y0:y1, x0:x1].astype(np.float32)
    m_orig = lm[y0:y1, x0:x1]
    fill = float(sub_orig[m_orig].mean())
    sub = np.where(m_orig, sub_orig, fill)
    sub = np.pad(sub, margin, constant_values=fill)
    m = np.pad(m_orig, margin, constant_values=False)
    h, w = m.shape
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    sq = np.full((side, side), fill, np.float32)
    sq[top:top + h, left:left + w] = sub
    mm = np.zeros((side, side), np.float32)
    mm[top:top + h, left:left + w] = m
    sq = cv2.resize(sq, (S, S), interpolation=cv2.INTER_CUBIC)
    mm = cv2.resize(mm, (S, S), interpolation=cv2.INTER_LINEAR) > 0.5
    img = np.where(mm, np.clip(sq, 0, 255), 0).astype(np.uint8)
    return img, S / side, mm, sub_orig, m_orig


# ------------------------------------------------------------ extra scale-safe features
EXTRA_KEYS = ["intensity_skewness", "intensity_kurtosis", "intensity_p10", "intensity_p90", "intensity_cv", "bright_fraction",
              "core_minus_rim_intensity", "core_rim_ratio", "axis_intensity_trend", "cross_axis_intensity_trend",
              "mirror_symmetry_long_axis", "mirror_symmetry_short_axis", "taper_ratio", "width_variability"]


def extra_features(vals, mask):
    """Scale-safe statistics of the lesion pixels. vals: 2-D grey image, mask: bool lesion mask (same shape)."""
    pad = max(mask.shape)
    vals = np.pad(vals.astype(np.float64), pad)
    mask = np.pad(mask, pad)
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    v = vals[mask]
    mean, std = v.mean(), v.std()
    f = {"intensity_skewness": float(((v - mean) ** 3).mean() / (std ** 3 + 1e-8)),
         "intensity_kurtosis": float(((v - mean) ** 4).mean() / (std ** 4 + 1e-8) - 3.0),
         "intensity_p10": float(np.percentile(v, 10)), "intensity_p90": float(np.percentile(v, 90)),
         "intensity_cv": float(std / (mean + 1e-8)), "bright_fraction": float((v > mean + std).mean())}

    dt = ndimage.distance_transform_edt(mask)
    dmax = dt.max()
    core, rim = dt > 0.5 * dmax, (dt > 0) & (dt <= 0.25 * dmax)
    cm = float(vals[core].mean()) if core.any() else mean
    rm = float(vals[rim].mean()) if rim.any() else mean
    f["core_minus_rim_intensity"], f["core_rim_ratio"] = cm - rm, cm / (rm + 1e-8)

    pts = np.stack([xs, ys], 1).astype(np.float64)
    c = pts.mean(0)
    q = pts - c
    _, V = np.linalg.eigh(np.cov(q.T) + np.eye(2) / 12.0)
    major, minor = V[:, 1], V[:, 0]
    u, t = q @ major, q @ minor
    f["axis_intensity_trend"] = float(abs(np.corrcoef(u, v)[0, 1])) if v.std() > 0 and u.std() > 0 else 0.0
    f["cross_axis_intensity_trend"] = float(abs(np.corrcoef(t, v)[0, 1])) if v.std() > 0 and t.std() > 0 else 0.0

    def mirror(su, st):
        r = c + np.outer(u * su, major) + np.outer(t * st, minor)
        xi, yi = np.rint(r[:, 0]).astype(int), np.rint(r[:, 1]).astype(int)
        ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        m2 = np.zeros((H, W), np.uint8)
        m2[yi[ok], xi[ok]] = 1
        m2 = cv2.morphologyEx(m2, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
        return float((m2 & mask).sum() / max(1, (m2 | mask).sum()))
    f["mirror_symmetry_long_axis"] = mirror(1, -1)     # reflect across the long axis (one side vs the other)
    f["mirror_symmetry_short_axis"] = mirror(-1, 1)    # reflect across the short axis (end-to-end)

    edges = np.linspace(u.min(), u.max() + 1e-9, 11)
    idx = np.clip(np.digitize(u, edges) - 1, 0, 9)
    widths = np.array([t[idx == b].max() - t[idx == b].min() if (idx == b).sum() > 1 else 0.0 for b in range(10)])
    f["taper_ratio"] = float((widths[:2].mean() + widths[-2:].mean()) / 2 / (widths[4:6].mean() + 1e-8))   # end width / middle width
    f["width_variability"] = float(widths.std() / (widths.mean() + 1e-8))
    return f


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_dir", default=str(HERE))
    ap.add_argument("--out_size", type=int, default=448)
    ap.add_argument("--margin", type=int, default=2)
    ap.add_argument("--qc_samples", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    P = Path(a.project_dir)
    for d in ("dataset/lesion_only", "qc/lesion_only_visualizations"):
        (P / d).mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(P / "features" / "lesion_features.jsonl")]
    if a.limit:
        rows = rows[:a.limit]

    by_img = defaultdict(list)
    for r in rows:
        by_img[(r["image_path"], r["mask_path"])].append(r)
    scale, not_found = {}, []
    for (ip, mp), lst in by_img.items():
        gray = cv2.imread(ip, cv2.IMREAD_UNCHANGED)
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray[..., :3], cv2.COLOR_BGR2GRAY)
        mask = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
        if mask.ndim == 3:                                     # some masks are stored as 3-channel PNGs
            mask = mask[..., :3].max(axis=2)
        lab, _ = ndimage.label(mask > 0, structure=np.ones((3, 3)))
        boxes = {tuple([sl[1].start, sl[0].start, sl[1].stop, sl[0].stop]): k for k, sl in enumerate(ndimage.find_objects(lab), 1)}
        for r in lst:
            k = boxes.get(tuple(r["mask_bbox"]))
            if k is None:
                not_found.append(r["sample_id"])
                continue
            img, sc, mm, sub_o, m_o = make_lesion_only(gray, lab == k, r["mask_bbox"], a.out_size, a.margin)
            cv2.imwrite(str(P / "dataset" / "lesion_only" / f"{r['sample_id']}.png"), img)
            scale[r["sample_id"]] = sc
            r["mean448"], r["std448"] = float(img[mm].mean()), float(img[mm].std())
            r["x448"] = extra_features(img.astype(np.float64), mm)
            r["xorig"] = extra_features(sub_o, m_o)
    rows = [r for r in rows if r["sample_id"] in scale]

    # ---- train-only terciles (measured on what the model receives, or scale-invariant original features)
    tr = [r for r in rows if r["split"] == "train"]
    tq = lambda vals: [float(np.quantile(vals, x)) for x in (0.33, 0.67)]
    q_int, q_var, q_sol = tq([r["mean448"] for r in tr]), tq([r["std448"] for r in tr]), tq([r["solidity"] for r in tr])
    tri = lambda x, t, names=("low", "mid", "high"): names[0] if x <= t[0] else (names[1] if x <= t[1] else names[2])
    base_thr = json.loads((P / "features" / "feature_thresholds.json").read_text())
    (P / "features" / "feature_thresholds_lesion_only.json").write_text(json.dumps({
        "quantiles": [0.33, 0.67], "n_train_lesions": len(tr),
        "shape_by_axis_ratio (reused from feature_thresholds.json)": base_thr["elongation_by_axis_ratio"],
        "outline_regularity_by_solidity": q_sol, "overall_intensity_by_mean_grey_448": q_int, "internal_variation_by_grey_std_448": q_var,
        "orientation_bins_deg": {"horizontal": "|angle| <= 22.5", "diagonal": "22.5 < |angle| <= 67.5", "vertical": "|angle| > 67.5",
                                 "none": "axis ratio < 1.5 (no dominant axis)"},
        "dropped": "texture / edge complexity (dominated by enlargement blur)", "note": "PNG grey values, not Hounsfield units; directions are image directions"}, indent=2))
    for r in rows:
        r["intensity_label"] = tri(r["mean448"], q_int)
        r["variation_label_lo"] = tri(r["std448"], q_var)
        r["outline_label"] = tri(r["solidity"], q_sol)
        r["orientation_label"] = orientation_label(r["orientation_deg"], r["axis_ratio"])

    # ---- captions + consistency check
    bad = []
    for r in rows:
        lab = {"shape": r["elongation_label"], "orientation": r["orientation_label"], "outline": r["outline_label"],
               "intensity": r["intensity_label"], "variation": r["variation_label_lo"]}
        r["template_lo"], r["caption_lo"] = make_caption(r["sample_id"], lab)
        r["fields_lo"] = FIELDS[r["template_lo"]]
        got = parse_caption(r["caption_lo"])
        if set(got) != set(r["fields_lo"]):
            bad.append((r["sample_id"], f"states {sorted(got)} != template {r['fields_lo']}"))
        for k, v in got.items():
            if lab[k] != v:
                bad.append((r["sample_id"], f"{k}: caption {v} != label {lab[k]}"))
        n = len(r["caption_lo"].split())
        if not MIN_WORDS <= n <= 60:
            bad.append((r["sample_id"], f"{n} words"))
        low = " " + r["caption_lo"].lower() + " "
        for w in BANNED + ["nearby", "surrounding", "located", "region of the image"]:
            if w in low:
                bad.append((r["sample_id"], f"unobservable/banned wording '{w}'"))

    # ---- features table (base + extras at 448 and original resolution)
    with open(P / "features" / "lesion_only_features.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "split", "original_box_xyxy", "upscale_factor", "axis_ratio", "orientation_deg", "solidity",
                    "mean_grey_448", "std_grey_448", "shape_label", "orientation_label", "outline_label", "intensity_label",
                    "variation_label"] + [f"{k}_448" for k in EXTRA_KEYS] + [f"{k}_orig" for k in EXTRA_KEYS])
        for r in rows:
            w.writerow([r["sample_id"], r["split"], r["mask_bbox"], f"{scale[r['sample_id']]:.2f}", f"{r['axis_ratio']:.3f}",
                        f"{r['orientation_deg']:.1f}", f"{r['solidity']:.4f}", f"{r['mean448']:.2f}", f"{r['std448']:.2f}",
                        r["elongation_label"], r["orientation_label"], r["outline_label"], r["intensity_label"], r["variation_label_lo"]]
                       + [f"{r['x448'][k]:.5g}" for k in EXTRA_KEYS] + [f"{r['xorig'][k]:.5g}" for k in EXTRA_KEYS])

    # ---- reliability of the extra features: does the 448 measurement match the original-resolution one?
    rel = ["# lesion_only: which extra features survive the enlargement?", "",
           "Each feature is measured on the enlarged 448x448 lesion-only image and on the original-resolution lesion; "
           "correlation across all lesions (1.0 = identical ranking).", "",
           "| feature | correlation 448 vs original | verdict |", "|---|---|---|"]
    reliab = {}
    for k in EXTRA_KEYS + ["mean_grey", "std_grey"]:
        if k == "mean_grey":
            x, y = [r["mean448"] for r in rows], [r["mean_intensity"] for r in rows]
        elif k == "std_grey":
            x, y = [r["std448"] for r in rows], [r["std_intensity"] for r in rows]
        else:
            x, y = [r["x448"][k] for r in rows], [r["xorig"][k] for r in rows]
        c = float(np.corrcoef(x, y)[0, 1])
        reliab[k] = c
        rel.append(f"| {k} | {c:.3f} | {'RELIABLE' if c >= 0.9 else ('use with caution' if c >= 0.7 else 'NOT reliable (scale-dependent)')} |")
    (P / "reports" / "lesion_only_extra_features_reliability.md").write_text("\n".join(rel) + "\n")

    # ---- captions + maestro-ready datasets
    files = defaultdict(list)
    for r in rows:
        files[r["split"]].append({
            "sample_id": r["sample_id"], "image": f"dataset/lesion_only/{r['sample_id']}.png", "prompt": PROMPT,
            "caption": r["caption_lo"], "split": r["split"], "template": r["template_lo"], "caption_fields": r["fields_lo"],
            "labels": {"shape": r["elongation_label"], "orientation": r["orientation_label"], "outline": r["outline_label"],
                       "intensity": r["intensity_label"], "variation": r["variation_label_lo"]},
            "original_box_xyxy": r["mask_bbox"], "upscale_factor": round(scale[r["sample_id"]], 2)})
    with open(P / "captions" / "captions_lesion_only_all.jsonl", "w") as f:
        for s in files:
            for rec in files[s]:
                f.write(json.dumps(rec) + "\n")
    for s in files:
        with open(P / "captions" / f"captions_lesion_only_{s}.jsonl", "w") as f:
            for rec in files[s]:
                f.write(json.dumps(rec) + "\n")
    for s, ms in MAESTRO_SPLIT.items():
        d = P / "training" / "datasets" / "lesion_only" / ms
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "annotations.jsonl", "w") as f:
            for rec in files[s]:
                f.write(json.dumps({"image": f"{rec['sample_id']}.png", "prefix": PROMPT, "suffix": rec["caption"] + "<eos>"}) + "\n")
                dst = d / f"{rec['sample_id']}.png"
                if not dst.is_symlink() and not dst.exists():
                    dst.symlink_to((P / "dataset" / "lesion_only" / f"{rec['sample_id']}.png").resolve())

    # ---- QC panels
    rng = random.Random(a.seed)
    for r in rng.sample(rows, min(a.qc_samples, len(rows))):
        old = cv2.imread(str(P / "dataset" / "padded_crops" / f"{r['sample_id']}.png"), cv2.IMREAD_GRAYSCALE)
        s = a.out_size / max(old.shape)
        old = cv2.resize(old, (max(1, int(old.shape[1] * s)), max(1, int(old.shape[0] * s))), interpolation=cv2.INTER_NEAREST)
        left = np.zeros((a.out_size, a.out_size), np.uint8)
        left[:old.shape[0], :old.shape[1]] = old
        new = cv2.imread(str(P / "dataset" / "lesion_only" / f"{r['sample_id']}.png"), cv2.IMREAD_GRAYSCALE)
        top = cv2.cvtColor(np.hstack([left, new]), cv2.COLOR_GRAY2BGR)
        strip = np.full((130, top.shape[1], 3), 255, np.uint8)
        lines = [f"{r['sample_id']} ({r['split']})   LEFT: old padded crop (enlarged)   RIGHT: lesion_only {a.out_size}x{a.out_size}   "
                 f"x{scale[r['sample_id']]:.1f} from {r['mask_bbox'][2]-r['mask_bbox'][0]}x{r['mask_bbox'][3]-r['mask_bbox'][1]} px",
                 f"labels: shape={r['elongation_label']} direction={r['orientation_label']} outline={r['outline_label']} "
                 f"intensity={r['intensity_label']} variation={r['variation_label_lo']}   (axis ratio {r['axis_ratio']:.1f}, angle {r['orientation_deg']:.0f} deg, solidity {r['solidity']:.2f})"]
        cap = r["caption_lo"]
        while cap:
            cut = cap[:118].rfind(" ") if len(cap) > 118 else len(cap)
            lines.append("  " + cap[:cut]); cap = cap[cut:].strip()
        for j, ln in enumerate(lines[:6]):
            cv2.putText(strip, ln, (6, 20 + 20 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imwrite(str(P / "qc" / "lesion_only_visualizations" / f"lo_{r['sample_id']}.png"), np.vstack([top, strip]))

    # ---- report
    sc = np.array(list(scale.values()))
    dist = lambda key, s: dict(Counter(r[key] for r in rows if r["split"] == s))
    st = ["# lesion_only variant statistics", "",
          f"- samples: {len(rows)} (per split {dict(Counter(r['split'] for r in rows))}); lesion blob not found for: {len(not_found)}",
          f"- output {a.out_size}x{a.out_size}, black margin {a.margin}px, square padding, bicubic enlargement, mask applied after enlargement",
          f"- enlargement factor: median x{np.median(sc):.1f}, p90 x{np.percentile(sc, 90):.1f}, max x{sc.max():.0f}; more than 20x: {(sc > 20).sum()}; more than 40x: {(sc > 40).sum()}",
          f"- caption consistency failures: {len(bad)}   templates: {dict(Counter(r['template_lo'] for r in rows))}",
          f"- caption length (words): min {min(len(r['caption_lo'].split()) for r in rows)}, max {max(len(r['caption_lo'].split()) for r in rows)}",
          f"- thresholds (train terciles): outline regularity by solidity {q_sol[0]:.3f}/{q_sol[1]:.3f}; overall intensity {q_int[0]:.1f}/{q_int[1]:.1f}; variation std {q_var[0]:.1f}/{q_var[1]:.1f}",
          f"- extra-feature reliability (corr 448 vs original): " + ", ".join(f"{k} {v:.2f}" for k, v in reliab.items()), ""]
    for key, name in (("elongation_label", "shape"), ("orientation_label", "direction"), ("outline_label", "outline"), ("intensity_label", "intensity"), ("variation_label_lo", "variation")):
        st.append(f"- {name}: " + " | ".join(f"{s}: {dist(key, s)}" for s in ("train", "val", "test")))
    (P / "reports" / "lesion_only_statistics.md").write_text("\n".join(st) + "\n")
    print("\n".join(st))
    if bad:
        print("first problems:", bad[:5])
    print("\nexample captions:")
    for r in rows[:3]:
        print(f"  {r['sample_id']} [{r['template_lo']}] {r['caption_lo']}")


if __name__ == "__main__":
    main()
