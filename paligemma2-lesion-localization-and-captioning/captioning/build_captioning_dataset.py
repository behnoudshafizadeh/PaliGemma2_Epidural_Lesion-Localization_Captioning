"""
build_captioning_dataset.py -- Phases 1-5 of a captioning project specification (not part of this repository)

Whole-slice captioning is very hard, so each annotated hemorrhage is CUT OUT and the crop is described
by a caption generated programmatically from measurements of that lesion (no LLM, no clinical claims).

Phase 1  audit: match PNG + mask + bbox, validate, report problems (nothing silently dropped)
Phase 2  per-lesion mask (8-connected blob of the mask == how the provided boxes were made), exact /
         padded / highlighted / fixed-window crops, overlays
Phase 3  ~40 objective features (size, shape, position, intensity, local contrast, texture)
Phase 4  train-only thresholds -> safe labels -> template captions -> caption consistency check
Phase 5  QC panels, statistics, error reports

Originals are never modified (images/masks are symlinked). Split = the existing patient-level
train/val/test of the COCO annotation files (55/7/17 patients) instead of a new 70/15/15.

Documented choices (spec asks for them):
  * skimage is not installed -> shape features are computed with OpenCV/NumPy (formulas in FEATURE_NOTES).
  * local-contrast ring = dilate(lesion, ring_px) - ALL mask pixels; ring pixels that are background (<=5)
    or saturated bone (>=250) are excluded because they are not tissue. (--ring_keep_extremes turns this off;
    the all-pixels version is saved as surrounding_mean_intensity_all.)
  * brightness label: effect size z = (lesion mean - surrounding mean) / surrounding std.
    "similar" if |z| <= 1 (lesion mean within one standard deviation of its surroundings), otherwise brighter/darker
    by sign. Train-quantile bands were tried first and rejected: every lesion is brighter than its surroundings, so
    quantile bands would call clearly brighter lesions "similar" -- a false statement.
  * PNG intensities are NOT Hounsfield units.

Run in the `paligemma2` env (CPU only):
    python build_captioning_dataset.py
"""
import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "data"                                 # data/{annotations,images,masks}
SPLIT_MAP = {"train": "train", "val": "val", "test": "test"}    # annotation-file name -> spec name
MAESTRO_SPLIT = {"train": "train", "val": "valid", "test": "test"}
PROMPT = "Describe the visible characteristics of this epidural hemorrhage."
MIN_AREA_PX = 20   # smaller blobs were never annotated in the source annotations

FEATURE_NOTES = {
    "perimeter": "cv2.arcLength of the lesion's external contour (closed)",
    "major/minor_axis_length": "4*sqrt(eigenvalues) of the pixel-coordinate covariance (+1/12 pixel variance, as for pixel squares)",
    "axis_ratio": "major_axis_length / minor_axis_length",
    "eccentricity": "sqrt(1 - (minor/major)^2)",
    "orientation_deg": "angle of the major axis vs the image x-axis, degrees in (-90, 90], image coordinates (y down)",
    "convex_area": "pixel count of the rasterised convex hull of the lesion contour",
    "solidity": "mask_area / convex_area",
    "extent": "mask_area / bbox_area (pixel bbox of the lesion)",
    "circularity": "4*pi*area / perimeter^2 (can exceed 1 for tiny pixelated blobs)",
    "compactness": "perimeter^2 / (4*pi*area); 1 for a circle, larger = less compact (= 1/circularity)",
    "entropy": "Shannon entropy (base 2) of the lesion's grey-value histogram",
    "edge_density": "share of lesion pixels that are Canny edges (thresholds 50/150) on the whole slice",
    "mean/std_gradient_magnitude": "Sobel gradient magnitude on the whole slice, statistics over lesion pixels",
    "surrounding_*": "ring of ring_px around the lesion excluding every mask pixel (and background/bone pixels unless --ring_keep_extremes)",
}

# ---------------------------------------------------------------- vocabulary / templates
SIZE_W = {"small": "small", "medium": "medium-sized", "large": "large"}
ELONG_W = {"low": "compact", "mid": "moderately elongated", "high": "elongated"}
BRIGHT_W = {"darker": "relatively darker than nearby pixels", "similar": "similar in brightness to nearby pixels",
            "brighter": "relatively brighter than nearby pixels"}
LEVEL_W = {"low": "low", "mid": "moderate", "high": "high"}
BANNED = ["hyperdense", "hypodense", "acute", "chronic", "biconvex", "lentiform", "extra-axial", "mass effect",
          "midline shift", "edema", "skull fracture", "brain compression", "severe", "mild ", "stable", "active bleeding",
          "hounsfield", " hu "]

TEMPLATES = {
    "A": "The annotated epidural hemorrhage is {size} and {shape}. It is located in the {region} region of the image. "
         "The region is {bright} and shows {variation} internal intensity variation.",
    "B": "This epidural hemorrhage occupies a {size} region in the {region} part of the image. Its shape is {shape}, "
         "with {variation} intensity variation and {texture} texture complexity.",
    "C": "A {size}, {shape} epidural hemorrhage is present in the {region} region. The lesion is {bright} "
         "and has {texture} texture complexity.",
    "D": "The annotated epidural hemorrhage is {size} and {shape}. It is located in the {region} region of the image, "
         "is {bright}, and shows {variation} internal intensity variation with {texture} texture complexity.",
}
TEMPLATE_FIELDS = {"A": ["size", "shape", "region", "brightness", "variation"],
                   "B": ["size", "region", "shape", "variation", "texture"],
                   "C": ["size", "shape", "region", "brightness", "texture"],
                   "D": ["size", "shape", "region", "brightness", "variation", "texture"]}

REGION_RE = re.compile(r"\b(upper|middle|lower)-(left|center|right)\b")


def parse_caption(text):
    """Recover the label words a caption states (controlled vocabulary => exact parsing)."""
    out = {}
    m = re.search(r"\b(small|medium-sized|large)\b", text)
    if m:
        out["size"] = {"medium-sized": "medium"}.get(m.group(1), m.group(1))
    m = re.search(r"\b(moderately elongated|less elongated|highly elongated|elongated|compact)\b", text)
    if m:
        out["shape"] = {"compact": "low", "less elongated": "low", "moderately elongated": "mid",
                        "elongated": "high", "highly elongated": "high"}[m.group(1)]
    m = re.search(r"\b(vertical|diagonal|horizontal) main axis|\b(no dominant axis)", text)          # lesion_only captions
    if m:
        out["orientation"] = m.group(1) or "none"
    m = re.search(r"\b(low|moderate|high) outline regularity", text)                              # lesion_only captions
    if m:
        out["outline"] = {"low": "low", "moderate": "mid", "high": "high"}[m.group(1)]
    m = REGION_RE.search(text)
    if m:
        out["region"] = m.group(0)
    m = re.search(r"(relatively brighter|relatively darker|similar in brightness)", text)
    if m:
        out["brightness"] = {"relatively brighter": "brighter", "relatively darker": "darker", "similar in brightness": "similar"}[m.group(1)]
    m = re.search(r"\b(low|moderate|high) (?:internal )?intensity variation", text)
    if m:
        out["variation"] = {"low": "low", "moderate": "mid", "high": "high"}[m.group(1)]
    m = re.search(r"\b(low|moderate|high) texture complexity", text)
    if m:
        out["texture"] = {"low": "low", "moderate": "mid", "high": "high"}[m.group(1)]
    m = re.search(r"\b(low|moderate|high) overall intensity", text)          # lesion_only captions
    if m:
        out["intensity"] = {"low": "low", "moderate": "mid", "high": "high"}[m.group(1)]
    m = re.search(r"\b(low|moderate|high) edge complexity", text)            # lesion_only captions
    if m:
        out["edge"] = {"low": "low", "moderate": "mid", "high": "high"}[m.group(1)]
    return out


def make_caption(sample_id, lab):
    t = "ABCD"[int(hashlib.sha1(sample_id.encode()).hexdigest()[:8], 16) % 4]     # deterministic per sample
    text = TEMPLATES[t].format(size=SIZE_W[lab["size"]], shape=ELONG_W[lab["shape"]], region=lab["region"],
                               bright=BRIGHT_W[lab["brightness"]], variation=LEVEL_W[lab["variation"]],
                               texture=LEVEL_W[lab["texture"]])
    return t, text


# ------------------------------------------------------------------------------ args
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project_dir", default=str(HERE))
    p.add_argument("--coco_dir", default=str(ROOT / "annotations"))
    p.add_argument("--images_dir", default=str(ROOT / "images"))
    p.add_argument("--masks_dir", default=str(ROOT / "masks"))
    p.add_argument("--pad_frac", type=float, default=0.20, help="context around the box for padded crops (spec: 0.10/0.20/0.25)")
    p.add_argument("--fixed_window", type=int, default=256, help="side of the scale-preserving window (px)")
    p.add_argument("--ring_px", type=int, default=10)
    p.add_argument("--ring_keep_extremes", action="store_true")
    p.add_argument("--bbox_iou_flag", type=float, default=0.9, help="flag samples whose provided vs mask-derived box IoU is below this")
    p.add_argument("--quantiles", type=float, nargs=2, default=[0.33, 0.67])
    p.add_argument("--bright_z", type=float, default=1.0, help="|effect size| below this => 'similar in brightness'")
    p.add_argument("--qc_samples", type=int, default=60)
    p.add_argument("--no_overlays", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None, help="images per split (smoke test)")
    return p.parse_args()


# ------------------------------------------------------------------------- geometry
def box_iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def read_gray(path):
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        return None, None
    info = {"dtype": str(im.dtype), "channels": 1 if im.ndim == 2 else im.shape[2]}
    if im.ndim == 3:
        im = cv2.cvtColor(im[..., :3], cv2.COLOR_BGR2GRAY)      # one consistent grey conversion
    return im, info


# --------------------------------------------------------------------------- features
def lesion_features(gray, lm, all_mask, bbox, a):
    """gray uint8 HxW, lm = this lesion's bool mask, all_mask = every mask pixel, bbox xyxy (provided)."""
    H, W = gray.shape
    ys, xs = np.nonzero(lm)
    area = int(lm.sum())
    bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    f = {"mask_area_pixels": area, "mask_area_ratio": area / (H * W), "bbox_area_pixels": bw * bh,
         "bbox_area_ratio": bw * bh / (H * W), "equivalent_diameter": math.sqrt(4 * area / math.pi)}

    cnts, _ = cv2.findContours(lm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)
    per = float(cv2.arcLength(cnt, True))
    hull = cv2.convexHull(cnt)
    hm = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(hm, hull, 1)
    convex_area = int(max(hm.sum(), area))
    cov = np.cov(np.vstack([xs, ys]), bias=True) + np.eye(2) / 12.0
    ev, evec = np.linalg.eigh(cov)
    minor, major = 4 * math.sqrt(ev[0]), 4 * math.sqrt(ev[1])
    theta = 0.5 * math.atan2(2 * cov[0, 1], cov[0, 0] - cov[1, 1])
    f.update({"perimeter": per, "major_axis_length": major, "minor_axis_length": minor, "axis_ratio": major / minor,
              "eccentricity": math.sqrt(max(0.0, 1 - (minor / major) ** 2)), "solidity": area / convex_area,
              "extent": area / max(1, (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)),
              "orientation_deg": math.degrees(theta), "convex_area": convex_area,
              "circularity": 4 * math.pi * area / per ** 2 if per > 0 else float("nan"),
              "compactness": per ** 2 / (4 * math.pi * area) if area > 0 else float("nan")})

    cx, cy = float(xs.mean()), float(ys.mean())
    hpos = "left" if cx / W < 1 / 3 else ("center" if cx / W < 2 / 3 else "right")
    vpos = "upper" if cy / H < 1 / 3 else ("middle" if cy / H < 2 / 3 else "lower")
    f.update({"centroid_x": cx, "centroid_y": cy, "centroid_x_norm": cx / W, "centroid_y_norm": cy / H,
              "horizontal_image_position": hpos, "vertical_image_position": vpos, "image_region": f"{vpos}-{hpos}"})

    v = gray[lm].astype(np.float64)
    q25, q75 = np.percentile(v, 25), np.percentile(v, 75)
    f.update({"mean_intensity": v.mean(), "median_intensity": float(np.median(v)), "std_intensity": v.std(),
              "min_intensity": v.min(), "max_intensity": v.max(), "q25_intensity": q25, "q75_intensity": q75,
              "intensity_iqr": q75 - q25})

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * a.ring_px + 1, 2 * a.ring_px + 1))
    ring_all = (cv2.dilate(lm.astype(np.uint8), k) > 0) & ~all_mask
    ring = ring_all if a.ring_keep_extremes else ring_all & (gray > 5) & (gray < 250)
    fallback = ring.sum() < MIN_AREA_PX
    if fallback:
        ring = ring_all
    sm = float(gray[ring].mean()) if ring.any() else float("nan")
    f.update({"surrounding_mean_intensity": sm, "surrounding_mean_intensity_all": float(gray[ring_all].mean()) if ring_all.any() else float("nan"),
              "lesion_surrounding_difference": f["mean_intensity"] - sm, "lesion_surrounding_ratio": f["mean_intensity"] / (sm + 1e-8),
              "surrounding_std_intensity": float(gray[ring].std()) if ring.any() else float("nan"),
              "ring_pixels": int(ring.sum()), "ring_fallback_used": bool(fallback)})

    f["lesion_surrounding_effect_size"] = f["lesion_surrounding_difference"] / (f["surrounding_std_intensity"] + 1e-8)
    _, counts = np.unique(gray[lm], return_counts=True)
    p = counts / counts.sum()
    gf = gray.astype(np.float32)
    mag = np.hypot(cv2.Sobel(gf, cv2.CV_32F, 1, 0), cv2.Sobel(gf, cv2.CV_32F, 0, 1))
    edges = cv2.Canny(gray, 50, 150)
    f.update({"entropy": float(-(p * np.log2(p)).sum()), "intensity_variance": float(v.var()),
              "mean_gradient_magnitude": float(mag[lm].mean()), "std_gradient_magnitude": float(mag[lm].std()),
              "edge_density": float((edges[lm] > 0).mean())})
    return f


# ------------------------------------------------------------------------------ crops
def clip_box(b, W, H):
    return [max(0, int(math.floor(b[0]))), max(0, int(math.floor(b[1]))), min(W, int(math.ceil(b[2]))), min(H, int(math.ceil(b[3])))]


def make_crops(gray, lm, bbox, a):
    H, W = gray.shape
    x0, y0, x1, y1 = bbox
    exact = gray[y0:y1, x0:x1]
    bw, bh = x1 - x0, y1 - y0
    px, py = bw * a.pad_frac, bh * a.pad_frac
    pb = clip_box([x0 - px, y0 - py, x1 + px, y1 + py], W, H)
    padded = gray[pb[1]:pb[3], pb[0]:pb[2]]
    hl = (padded.astype(np.float32) * 0.35)
    m = lm[pb[1]:pb[3], pb[0]:pb[2]]
    hl[m] = padded[m]
    cx, cy, s = (x0 + x1) // 2, (y0 + y1) // 2, a.fixed_window
    wx0, wy0 = cx - s // 2, cy - s // 2
    win = np.zeros((s, s), np.uint8)
    sx0, sy0, sx1, sy1 = max(0, wx0), max(0, wy0), min(W, wx0 + s), min(H, wy0 + s)
    win[sy0 - wy0:sy1 - wy0, sx0 - wx0:sx1 - wx0] = gray[sy0:sy1, sx0:sx1]
    return {"exact": exact, "padded": padded, "highlighted": hl.astype(np.uint8), "fixed_window": win}, pb


# ---------------------------------------------------------------------------- main
def main():
    a = parse_args()
    P = Path(a.project_dir)
    random.seed(a.seed)
    np.random.seed(a.seed)
    for d in ("dataset/crops", "dataset/padded_crops", "dataset/highlighted_crops", "dataset/fixed_window_crops",
              "dataset/overlays", "dataset/train", "dataset/val", "dataset/test", "features", "captions",
              "qc/caption_visualizations", "reports", "training"):
        (P / d).mkdir(parents=True, exist_ok=True)
    for name, src in (("images", a.images_dir), ("masks", a.masks_dir)):      # originals are only linked, never copied/modified
        link = P / "dataset" / name
        if not link.exists() and not link.is_symlink():
            link.symlink_to(src)

    unmatched, invalid, info_rows = [], [], []
    img_files = {p.name for p in Path(a.images_dir).glob("*.png")}
    msk_files = {p.name for p in Path(a.masks_dir).glob("*.png")}
    for n in sorted(img_files - msk_files):
        unmatched.append(["png_without_mask", n, ""])
    for n in sorted(msk_files - img_files):
        unmatched.append(["mask_without_png", n, ""])

    lesions, seen_png = [], set()
    n_neg = Counter()
    unannotated_small = 0
    for src, split in SPLIT_MAP.items():
        d = json.loads((Path(a.coco_dir) / f"{src}.json").read_text())
        anns = defaultdict(list)
        for an in d["annotations"]:
            anns[an["image_id"]].append(an)
        entries = sorted(d["images"], key=lambda x: x["file_name"])
        if a.limit:
            entries = entries[:a.limit]
        for e in entries:
            ip = Path(e["file_name"])
            seen_png.add(ip.name)
            mp = Path(a.masks_dir) / ip.name
            if not ip.exists() or not mp.exists():
                unmatched.append(["coco_entry_without_file", ip.name, f"png={ip.exists()} mask={mp.exists()}"])
                continue
            gray, gi = read_gray(ip)
            mask, _ = read_gray(mp)
            if gray is None or mask is None:
                invalid.append([split, ip.name, "", "unreadable_png_or_mask", ""])
                continue
            if gray.shape != mask.shape:
                invalid.append([split, ip.name, "", "image_mask_size_mismatch", f"{gray.shape} vs {mask.shape}"])
                continue
            H, W = gray.shape
            mask_bin = mask > 0
            file_anns = sorted(anns.get(e["id"], []), key=lambda x: x["id"])
            if not file_anns:
                n_neg[split] += 1
                info_rows.append([split, ip.name, "", "no_annotation_negative_image", f"mask_pixels={int(mask_bin.sum())}"])
                continue
            if not mask_bin.any():
                invalid.append([split, ip.name, "", "annotation_but_empty_mask", ""])
                continue
            lab, n = ndimage.label(mask_bin, structure=np.ones((3, 3)))          # 8-connected, like the box maker
            comps = {}
            for k, sl in enumerate(ndimage.find_objects(lab), 1):
                comps[k] = {"bbox": [sl[1].start, sl[0].start, sl[1].stop, sl[0].stop], "area": int((lab[sl] == k).sum())}
            unannotated_small += sum(1 for c in comps.values() if c["area"] < MIN_AREA_PX)
            used = set()
            for j, an in enumerate(file_anns, 1):
                bx, by, bw, bh = an["bbox"]
                bbox = [int(round(bx)), int(round(by)), int(round(bx + bw)), int(round(by + bh))]
                sid = f"{ip.stem}_lesion{j:02d}"
                if not (bbox[0] < bbox[2] and bbox[1] < bbox[3] and 0 <= bbox[0] and 0 <= bbox[1] and bbox[2] <= W and bbox[3] <= H):
                    invalid.append([split, ip.name, sid, "invalid_bbox", str(bbox)])
                    continue
                best = max(comps, key=lambda k: box_iou(bbox, comps[k]["bbox"]))
                iou = box_iou(bbox, comps[best]["bbox"])
                if iou == 0:
                    invalid.append([split, ip.name, sid, "bbox_does_not_overlap_mask", str(bbox)])
                    continue
                if best in used:
                    invalid.append([split, ip.name, sid, "two_annotations_map_to_the_same_mask_blob", str(bbox)])
                    continue
                used.add(best)
                lm = lab == best
                whole = [int(np.nonzero(mask_bin)[1].min()), int(np.nonzero(mask_bin)[0].min()),
                         int(np.nonzero(mask_bin)[1].max()) + 1, int(np.nonzero(mask_bin)[0].max()) + 1]
                feats = lesion_features(gray, lm, mask_bin, bbox, a)
                crops, pb = make_crops(gray, lm, bbox, a)
                for key, folder in (("exact", "crops"), ("padded", "padded_crops"), ("highlighted", "highlighted_crops"),
                                    ("fixed_window", "fixed_window_crops")):
                    cv2.imwrite(str(P / "dataset" / folder / f"{sid}.png"), crops[key])
                if not a.no_overlays:
                    ov = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    cnts, _ = cv2.findContours(lm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    cv2.drawContours(ov, cnts, -1, (0, 255, 0), 1)
                    cv2.rectangle(ov, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 255), 1)
                    cv2.putText(ov, sid, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.imwrite(str(P / "dataset" / "overlays" / f"{sid}.jpg"), ov, [cv2.IMWRITE_JPEG_QUALITY, 85])
                row = {"sample_id": sid, "split": split, "image_path": str(ip), "mask_path": str(mp),
                       "provided_bbox": bbox, "mask_bbox": comps[best]["bbox"], "whole_mask_bbox": whole,
                       "bbox_iou": iou, "image_width": W, "image_height": H,
                       "qc_flag_bbox_disagreement": iou < a.bbox_iou_flag, "png_dtype": gi["dtype"], "png_channels": gi["channels"],
                       "padded_box": pb, "exact_crop_hw": list(crops["exact"].shape), "padded_crop_hw": list(crops["padded"].shape),
                       "lesion_larger_than_fixed_window": (bbox[2] - bbox[0] > a.fixed_window) or (bbox[3] - bbox[1] > a.fixed_window)}
                row.update(feats)
                lesions.append(row)
                if iou < a.bbox_iou_flag:
                    invalid.append([split, ip.name, sid, "QC_flag_bbox_vs_mask_disagreement", f"IoU {iou:.3f}"])
    for n in sorted(img_files - seen_png):
        unmatched.append(["png_not_in_any_coco_split", n, ""])

    # ---- Phase 4: thresholds from TRAIN lesions only
    tr = [r for r in lesions if r["split"] == "train"]
    q = lambda key: [float(np.quantile([r[key] for r in tr], x)) for x in a.quantiles]
    thr = {"quantiles": a.quantiles, "n_train_lesions": len(tr), "size_by_mask_area_ratio": q("mask_area_ratio"),
           "elongation_by_axis_ratio": q("axis_ratio"), "brightness_similar_band_effect_size": a.bright_z,
           "variation_by_std_intensity": q("std_intensity"), "texture_by_entropy": q("entropy"), "edge_by_edge_density": q("edge_density"),
           "notes": {"brightness": "effect size z=(lesion mean - surrounding mean)/surrounding std: |z|<=bright_z 'similar', z>bright_z 'brighter', z<-bright_z 'darker' (a fixed, interpretable rule, NOT train-quantile bands: all lesions are brighter than their surroundings)",
                     "all_thresholds": "fitted on TRAIN lesions only and applied unchanged to val/test"}, "feature_formulas": FEATURE_NOTES}

    def band(x, t, names):
        return names[0] if x <= t[0] else (names[1] if x <= t[1] else names[2])
    for r in lesions:
        r["size_label"] = band(r["mask_area_ratio"], thr["size_by_mask_area_ratio"], ["small", "medium", "large"])
        r["elongation_label"] = band(r["axis_ratio"], thr["elongation_by_axis_ratio"], ["low", "mid", "high"])
        z = r["lesion_surrounding_effect_size"]
        r["brightness_label"] = "similar" if not math.isfinite(z) or abs(z) <= a.bright_z else ("brighter" if z > 0 else "darker")
        r["variation_label"] = band(r["std_intensity"], thr["variation_by_std_intensity"], ["low", "mid", "high"])
        r["texture_label"] = band(r["entropy"], thr["texture_by_entropy"], ["low", "mid", "high"])
        r["edge_label"] = band(r["edge_density"], thr["edge_by_edge_density"], ["low", "mid", "high"])
        lab = {"size": r["size_label"], "shape": r["elongation_label"], "region": r["image_region"],
               "brightness": r["brightness_label"], "variation": r["variation_label"], "texture": r["texture_label"]}
        r["template"], r["generated_caption"] = make_caption(r["sample_id"], lab)
        r["caption_fields"] = TEMPLATE_FIELDS[r["template"]]

    # ---- caption consistency check: every phrase must map back to a stored label; no banned terms; length limits
    bad = []
    for r in lesions:
        c = r["generated_caption"]
        got = parse_caption(c)
        want = {"size": r["size_label"], "shape": r["elongation_label"], "region": r["image_region"],
                "brightness": r["brightness_label"], "variation": r["variation_label"], "texture": r["texture_label"]}
        if set(got) != set(r["caption_fields"]):
            bad.append((r["sample_id"], f"states {sorted(got)} but template lists {r['caption_fields']}"))
        for k, v in got.items():
            if want[k] != v:
                bad.append((r["sample_id"], f"{k}: caption '{v}' != label '{want[k]}'"))
        words = len(c.split())
        if not 20 <= words <= 60:
            bad.append((r["sample_id"], f"{words} words"))
        low = " " + c.lower() + " "
        for w in BANNED:
            if w in low:
                bad.append((r["sample_id"], f"banned term '{w}'"))
        if r["template"] == "A" and False:
            pass
    for sid, why in bad:
        invalid.append(["", "", sid, "CAPTION_CONSISTENCY_FAILURE", why])

    # ---- write features + captions
    cols = ["sample_id", "image_path", "mask_path", "provided_bbox", "mask_bbox", "bbox_iou", "image_width", "image_height",
            "mask_area_pixels", "mask_area_ratio", "bbox_area_ratio", "equivalent_diameter", "perimeter", "major_axis_length",
            "minor_axis_length", "axis_ratio", "eccentricity", "solidity", "extent", "orientation_deg", "convex_area", "circularity",
            "compactness", "centroid_x", "centroid_y", "centroid_x_norm", "centroid_y_norm", "horizontal_image_position",
            "vertical_image_position", "image_region", "mean_intensity", "median_intensity", "std_intensity", "min_intensity",
            "max_intensity", "q25_intensity", "q75_intensity", "intensity_iqr", "surrounding_mean_intensity",
            "surrounding_mean_intensity_all", "lesion_surrounding_difference", "lesion_surrounding_ratio", "surrounding_std_intensity",
            "lesion_surrounding_effect_size", "ring_pixels",
            "ring_fallback_used", "entropy", "intensity_variance", "mean_gradient_magnitude", "std_gradient_magnitude", "edge_density",
            "size_label", "elongation_label", "brightness_label", "variation_label", "texture_label",
            "edge_label", "template", "generated_caption", "split", "qc_flag_bbox_disagreement", "bbox_area_pixels",
            "whole_mask_bbox", "exact_crop_hw", "padded_crop_hw", "lesion_larger_than_fixed_window", "png_dtype", "png_channels"]
    with open(P / "features" / "lesion_features.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in lesions:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items() if k in cols})
    with open(P / "features" / "lesion_features.jsonl", "w") as f:
        for r in lesions:
            f.write(json.dumps({k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in r.items()}) + "\n")
    (P / "features" / "feature_thresholds.json").write_text(json.dumps(thr, indent=2))
    (P / "features" / "build_config.json").write_text(json.dumps(vars(a), indent=2))

    key_feats = ["mask_area_ratio", "axis_ratio", "eccentricity", "solidity", "mean_intensity", "std_intensity", "entropy", "edge_density", "image_region"]
    variants = {"exact": "crops", "padded": "padded_crops", "highlighted": "highlighted_crops", "fixed_window": "fixed_window_crops"}
    files = {s: [] for s in SPLIT_MAP.values()}
    for r in lesions:
        rec = {"sample_id": r["sample_id"], "image": f"dataset/padded_crops/{r['sample_id']}.png", "prompt": PROMPT,
               "caption": r["generated_caption"], "split": r["split"], "template": r["template"], "caption_fields": r["caption_fields"],
               "labels": {"size": r["size_label"], "shape": r["elongation_label"], "region": r["image_region"], "brightness": r["brightness_label"],
                          "variation": r["variation_label"], "texture": r["texture_label"], "edge": r["edge_label"]},
               "features": {k: (float(r[k]) if isinstance(r[k], (float, np.floating)) else r[k]) for k in key_feats},
               "images": {v: f"dataset/{fold}/{r['sample_id']}.png" for v, fold in variants.items()}}
        files[r["split"]].append(rec)
    with open(P / "captions" / "captions_all.jsonl", "w") as f:
        for s in files:
            for rec in files[s]:
                f.write(json.dumps(rec) + "\n")
    for s in files:
        with open(P / "captions" / f"captions_{s}.jsonl", "w") as f:
            for rec in files[s]:
                f.write(json.dumps(rec) + "\n")
        (P / "dataset" / s / "sample_ids.txt").write_text("\n".join(x["sample_id"] for x in files[s]) + "\n")

    # maestro-ready training datasets (one per crop variant); "<eos>" teaches the model to stop
    for v, fold in variants.items():
        for s, ms in MAESTRO_SPLIT.items():
            d = P / "training" / "datasets" / v / ms
            d.mkdir(parents=True, exist_ok=True)
            with open(d / "annotations.jsonl", "w") as f:
                for rec in files[s]:
                    f.write(json.dumps({"image": f"{rec['sample_id']}.png", "prefix": PROMPT, "suffix": rec["caption"] + "<eos>"}) + "\n")
                    dst = d / f"{rec['sample_id']}.png"
                    if not dst.is_symlink() and not dst.exists():
                        dst.symlink_to((P / "dataset" / fold / f"{rec['sample_id']}.png").resolve())

    with open(P / "reports" / "unmatched_files.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["type", "file", "detail"]); w.writerows(unmatched)
    with open(P / "reports" / "invalid_samples.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["split", "image", "sample_id", "problem", "detail"]); w.writerows(invalid + info_rows)

    # ---- Phase 5: QC panels + statistics
    rng = random.Random(a.seed)
    pick = rng.sample(lesions, min(a.qc_samples, len(lesions)))
    for r in pick:
        sid = r["sample_id"]
        gray, _ = read_gray(r["image_path"])
        big = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        m, _ = read_gray(r["mask_path"])
        ov = big.copy()
        ov[m > 0] = (0.5 * ov[m > 0] + 0.5 * np.array([0, 200, 0])).astype(np.uint8)
        b = r["provided_bbox"]
        cv2.rectangle(ov, (b[0], b[1]), (b[2], b[3]), (0, 255, 255), 1)
        tiles = [cv2.resize(ov, (384, 384))]
        for fold in ("crops", "padded_crops", "highlighted_crops", "fixed_window_crops"):
            c = cv2.imread(str(P / "dataset" / fold / f"{sid}.png"), cv2.IMREAD_COLOR)
            s = 384 / max(c.shape[:2])
            c = cv2.resize(c, (max(1, int(c.shape[1] * s)), max(1, int(c.shape[0] * s))), interpolation=cv2.INTER_NEAREST)
            canvas = np.zeros((384, 384, 3), np.uint8)
            canvas[:c.shape[0], :c.shape[1]] = c
            tiles.append(canvas)
        top = np.hstack(tiles)
        strip = np.full((150, top.shape[1], 3), 255, np.uint8)
        lines = [f"{sid} ({r['split']})  region={r['image_region']}  area_ratio={r['mask_area_ratio']:.4f}  axis_ratio={r['axis_ratio']:.2f}  "
                 f"solidity={r['solidity']:.2f}  mean_int={r['mean_intensity']:.0f}  vs ring={r['surrounding_mean_intensity']:.0f}  entropy={r['entropy']:.2f}",
                 f"labels: size={r['size_label']} | shape={r['elongation_label']} | brightness={r['brightness_label']} | variation={r['variation_label']} | texture={r['texture_label']}",
                 "panels: slice+mask+box | exact crop | padded crop | highlighted | fixed 256px window", "CAPTION (template " + r["template"] + "):"]
        cap = r["generated_caption"]
        while cap:
            cut = cap[:150].rfind(" ") if len(cap) > 150 else len(cap)
            lines.append("  " + cap[:cut]); cap = cap[cut:].strip()
        for j, ln in enumerate(lines[:7]):
            cv2.putText(strip, ln, (6, 18 + 19 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imwrite(str(P / "qc" / "caption_visualizations" / f"qc_{sid}.png"), np.vstack([top, strip]))

    n_by = Counter(r["split"] for r in lesions)
    def dist(lbl, split):
        return dict(Counter(r[lbl] for r in lesions if r["split"] == split))
    st = ["# Captioning dataset statistics", "",
          f"- lesions (samples): {len(lesions)}  per split: {dict(n_by)}   negative images (no lesion, skipped): {dict(n_neg)}",
          f"- invalid/flagged items: {len(invalid)}  unmatched files: {len(unmatched)}  (see reports/*.csv)",
          f"- provided-vs-mask box IoU: min {min(r['bbox_iou'] for r in lesions):.3f}, mean {np.mean([r['bbox_iou'] for r in lesions]):.3f}; flagged (<{a.bbox_iou_flag}): "
          f"{sum(r['qc_flag_bbox_disagreement'] for r in lesions)}",
          f"- mask blobs smaller than {MIN_AREA_PX}px that were never annotated (ignored): {unannotated_small}",
          f"- caption consistency failures: {len(bad)}   templates: {dict(Counter(r['template'] for r in lesions))}",
          f"- caption length (words): min {min(len(r['generated_caption'].split()) for r in lesions)}, max {max(len(r['generated_caption'].split()) for r in lesions)}",
          f"- exact crops smaller than 16px on a side: {sum(1 for r in lesions if min(r['exact_crop_hw']) < 16)}; padded crops smaller than 32px: "
          f"{sum(1 for r in lesions if min(r['padded_crop_hw']) < 32)}; lesions larger than the {a.fixed_window}px window: "
          f"{sum(1 for r in lesions if r['lesion_larger_than_fixed_window'])}",
          f"- ring fallback used (too few tissue pixels around lesion): {sum(1 for r in lesions if r['ring_fallback_used'])}",
          f"- surrounding-relative brightness: lesion brighter than its ring in {np.mean([r['lesion_surrounding_difference'] > 0 for r in lesions]) * 100:.1f}% of lesions; "
          f"median difference {np.median([r['lesion_surrounding_difference'] for r in lesions]):.1f} grey levels, median effect size {np.nanmedian([r['lesion_surrounding_effect_size'] for r in lesions]):.2f}",
          f"- train-only thresholds: {json.dumps({k: [round(x, 5) for x in v] for k, v in thr.items() if isinstance(v, list) and k != 'quantiles'})}", ""]
    for lbl in ("size_label", "elongation_label", "brightness_label", "variation_label", "texture_label", "edge_label", "image_region"):
        st.append(f"- {lbl}: " + " | ".join(f"{s}: {dist(lbl, s)}" for s in ("train", "val", "test")))
    (P / "reports" / "dataset_statistics.md").write_text("\n".join(st) + "\n")
    print("\n".join(st))
    print("\nexample captions:")
    for r in lesions[:3]:
        print(f"  {r['sample_id']} [{r['template']}] {r['generated_caption']}")


if __name__ == "__main__":
    main()
