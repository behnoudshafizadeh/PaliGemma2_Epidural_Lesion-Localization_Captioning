"""
build_prompt_dataset.py -- feature-based prompt generation for PaliGemma 2 box
detection, implementing a feature-based prompt-generation specification (not part of this repository; "spec s.N" below refers to its sections).

Instead of one fixed prompt ("detect hemorrhage") for every image, each lesion gets
prompts derived ONLY from what can be measured from the PNG and its bounding box
(relative size, image position/region, optional box shape). Nothing clinical is
invented (no HU, hyperdense, anatomical left/right, ...). Intensity/texture
features are computed and saved numerically but never put into prompts (spec s.13-15, 18).

Inputs : existing COCO annotations data/annotations/{train,val,test}.json (already
         patient-split, so the split is reused as-is, not redone).
Outputs (under --out_dir):
  lesion_features.csv     one row per lesion: all geometry/intensity/texture features
  prompts_all.jsonl       one row per image+prompt+target (spec s.23), all groups
  thresholds.json         size/shape/region thresholds (TRAIN split only) + config
  statistics.json/.csv/.md, errors.csv, qc/*.png
  datasets/<group>/{train,valid,test}/annotations.jsonl (+ symlinked images),
      maestro-ready: {"image","prefix"=prompt,"suffix"=loc tokens}.
      groups: baseline, size, horizontal, vertical, region, size_region,
      [shape if --enable_shape], and "mixed" = ONE randomly chosen prompt per image
      (seeded), i.e. genuinely different prompts across images in a single dataset.

IMPORTANT interpretation note (also in statistics.md): every feature prompt is built
from the GROUND-TRUTH box, so it tells the model roughly where/how big the lesion
is. Only the "baseline" group is a deployable detector; the other groups measure
how much a correct hint helps (or a human-provided hint in an interactive tool).

Decisions that differ from / extend the spec, on purpose:
  * loc tokens use round(v / size * 1024) clamped to 1023, NOT *1023. This is the
    convention already used for all training here and is the exact inverse of how
    supervision decodes PaliGemma output (loc/1024*size). Verified round-trip below
    (spec s.31 asks to verify against the installed code).
  * --append_eos (default ON) adds <eos> to every target. Verified with maestro's own
    collate function that the existing datasets have NO <eos> in the labels, so the
    model was never taught to stop (it emitted ~9 repeated boxes per image), and
    negative images (empty target) would have no supervised token at all.
  * Split: reuses the existing 55/7/17-patient train/valid/test split instead of
    re-splitting 70/15/15; no patient appears in two splits.

Run in the `paligemma2` env (CPU only: numpy, opencv, supervision):
    python build_prompt_dataset.py
"""
import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_COCO_DIR = HERE.parent / "data" / "annotations"     # train.json / val.json / test.json (COCO format)
SPLIT_MAP = {"train": "train", "val": "valid", "test": "test"}  # annotation-file name -> maestro name
SPLITS = ["train", "valid", "test"]
DEFAULT_GROUPS = ["baseline", "size", "horizontal", "vertical", "region", "size_region"]
# Uniform-description dataset: EVERY lesion is described with the SAME three features
# in the SAME sentence (size + box shape + image region). Deterministic, no random
# choice. Deliberately NOT part of the "mixed" draw, so adding it never changes `mixed`.
UNIFORM_GROUPS = ["size_shape_region"]
MIXED_EXCLUDE = set(UNIFORM_GROUPS)
SIZE_WORD = {"small": "small", "medium": "medium-sized", "large": "large"}

# spec s.19 (with {cls} substituted by --class_phrase)
PROMPT_TEMPLATES = {
    "baseline": "detect {cls}",
    "size": "detect the {size} {cls}",
    "horizontal": "detect the {cls} on the {horizontal} side of the image",
    "horizontal_center": "detect the {cls} in the center of the image",
    "vertical": "detect the {cls} in the {vertical} part of the image",
    "region": "detect the {cls} in the {region} region of the image",
    "size_region": "detect the {size} {cls} in the {region} region of the image",
    "shape": "detect the {shape} {cls} bounding region",
    # same 3 features for every lesion; wording says the SHAPE describes the BOX, not the lesion
    "size_shape_region": "detect the {size} {cls} with a {shape} bounding box in the {region} region of the image",
}
CLUE_FLAGS = {
    "baseline": {},
    "size": {"size": True},
    "horizontal": {"horizontal": True},
    "vertical": {"vertical": True},
    "region": {"horizontal": True, "vertical": True},
    "size_region": {"size": True, "horizontal": True, "vertical": True},
    "shape": {"shape": True},
    "size_shape_region": {"size": True, "horizontal": True, "vertical": True, "shape": True},
}

FEATURE_COLUMNS = [
    "split", "image", "lesion_id", "class",
    "x_min", "y_min", "x_max", "y_max", "image_width", "image_height",
    "bbox_width", "bbox_height", "bbox_area", "bbox_area_ratio",
    "bbox_width_norm", "bbox_height_norm", "aspect_ratio",
    "center_x", "center_y", "center_x_norm", "center_y_norm",
    "image_horizontal_position", "image_vertical_position", "image_region",
    "relative_size", "bbox_shape",
    "mean_intensity", "median_intensity", "std_intensity", "min_intensity", "max_intensity",
    "q25_intensity", "q75_intensity", "surrounding_mean_intensity", "intensity_difference",
    "intensity_ratio", "entropy", "edge_density", "mean_gradient", "std_gradient", "roi_variance",
    "n_lesions_in_image", "ambiguous_groups",
    "loc_tokens_yxyx", "png_dtype", "png_channels", "png_channels_identical",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--coco_dir", default=str(DEFAULT_COCO_DIR), help="folder with train.json, val.json, test.json (COCO)")
    p.add_argument("--images_dir", default=None, help="if the COCO file_name entries are not valid paths: folder that contains the images")
    p.add_argument("--out_dir", default=str(HERE / "prompt_dataset"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--class_phrase", default="epidural hemorrhage",
                   help="wording used inside prompts (spec uses 'epidural hemorrhage')")
    p.add_argument("--target_label", default="hemorrhage",
                   help="label word after the loc tokens in the TARGET; kept identical to the existing "
                        "datasets/evaluation scripts (one word keeps supervision's parser happy)")
    p.add_argument("--groups", default=",".join(DEFAULT_GROUPS + UNIFORM_GROUPS))
    p.add_argument("--enable_shape", action="store_true", help="spec group 6 (optional, off by default)")
    p.add_argument("--no_eos", action="store_true", help="do NOT append <eos> to targets (not recommended)")
    p.add_argument("--size_quantiles", type=float, nargs=2, default=[0.33, 0.67])
    p.add_argument("--shape_wide", type=float, default=1.5)
    p.add_argument("--shape_tall", type=float, default=0.67)
    p.add_argument("--surround_margin_px", type=int, default=16)
    p.add_argument("--bbox_tolerance_px", type=float, default=0.0,
                   help="allowed overshoot outside the image before a box is called invalid")
    p.add_argument("--qc_samples", type=int, default=24)
    p.add_argument("--limit", type=int, default=None, help="images per split (smoke test)")
    p.add_argument("--no_symlinks", action="store_true", help="write JSONL only, skip image symlinks")
    return p.parse_args()


# --------------------------------------------------------------------------- loading
def load_annotations(coco_dir, limit=None, images_dir=None):
    """COCO json per split -> {split: [ {image_id, path, width, height, anns:[xywh...]} ]}"""
    out = {}
    for src, dst in SPLIT_MAP.items():
        d = json.loads((Path(coco_dir) / f"{src}.json").read_text())
        anns = defaultdict(list)
        for a in d["annotations"]:
            anns[a["image_id"]].append(a)
        image_ids = {im["id"] for im in d["images"]}
        entries = []
        for im in sorted(d["images"], key=lambda x: x["file_name"]):
            entries.append({"image_id": im["id"], "path": (Path(images_dir) / Path(im["file_name"]).name) if images_dir else Path(im["file_name"]), "width": im["width"],
                            "height": im["height"], "anns": sorted(anns.get(im["id"], []), key=lambda a: a["id"])})
        orphan = [a for a in d["annotations"] if a["image_id"] not in image_ids]
        out[dst] = {"entries": entries[:limit] if limit else entries, "orphan_annotations": orphan}
    return out


def read_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, None
    info = {"png_dtype": str(img.dtype), "png_channels": 1 if img.ndim == 2 else img.shape[2],
            "png_channels_identical": ""}
    if img.ndim == 3:
        ch = img[..., :3]
        info["png_channels_identical"] = bool(np.array_equal(ch[..., 0], ch[..., 1]) and
                                              np.array_equal(ch[..., 1], ch[..., 2]))
        gray = cv2.cvtColor(ch, cv2.COLOR_BGR2GRAY)  # one consistent grayscale conversion (spec s.13)
    else:
        gray = img
    return gray, info


# ----------------------------------------------------------------- box validation/features
def validate_bbox(b, W, H, tol):
    x0, y0, x1, y1 = b
    if not (x0 < x1 and y0 < y1):
        return "zero_area_or_inverted_bbox"
    if x0 < -tol or y0 < -tol or x1 > W + tol or y1 > H + tol or x0 >= W or y0 >= H:
        return "bbox_outside_image"
    return None


def extract_bbox_features(b, W, H):
    x0, y0, x1, y1 = b
    w, h = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return {"x_min": x0, "y_min": y0, "x_max": x1, "y_max": y1, "image_width": W, "image_height": H,
            "bbox_width": w, "bbox_height": h, "bbox_area": w * h, "bbox_area_ratio": (w * h) / (W * H),
            "bbox_width_norm": w / W, "bbox_height_norm": h / H, "aspect_ratio": w / h,
            "center_x": cx, "center_y": cy, "center_x_norm": cx / W, "center_y_norm": cy / H}


def infer_image_region(cx_norm, cy_norm):
    horizontal = "left" if cx_norm < 1 / 3 else ("center" if cx_norm < 2 / 3 else "right")
    vertical = "upper" if cy_norm < 1 / 3 else ("middle" if cy_norm < 2 / 3 else "lower")
    return horizontal, vertical, f"{vertical}-{horizontal}"


def roi_box(b, W, H):
    x0, y0 = max(0, int(math.floor(b[0]))), max(0, int(math.floor(b[1])))
    x1, y1 = min(W, int(math.ceil(b[2]))), min(H, int(math.ceil(b[3])))
    x1, y1 = max(x1, min(W, x0 + 1)), max(y1, min(H, y0 + 1))
    return x0, y0, x1, y1


def extract_png_intensity_features(gray, b, margin):
    """Numerical PNG intensities only. NOT Hounsfield units, and never turned into
    'hyperdense'/'hypodense' words (spec s.13, 14, 18)."""
    H, W = gray.shape
    x0, y0, x1, y1 = roi_box(b, W, H)
    roi = gray[y0:y1, x0:x1].astype(np.float64)
    f = {"mean_intensity": float(roi.mean()), "median_intensity": float(np.median(roi)),
         "std_intensity": float(roi.std()), "min_intensity": float(roi.min()), "max_intensity": float(roi.max()),
         "q25_intensity": float(np.percentile(roi, 25)), "q75_intensity": float(np.percentile(roi, 75))}
    ex0, ey0, ex1, ey1 = max(0, x0 - margin), max(0, y0 - margin), min(W, x1 + margin), min(H, y1 + margin)
    region = gray[ey0:ey1, ex0:ex1].astype(np.float64)
    mask = np.ones(region.shape, bool)
    mask[y0 - ey0:y1 - ey0, x0 - ex0:x1 - ex0] = False
    sur = region[mask]
    if sur.size:
        f["surrounding_mean_intensity"] = float(sur.mean())
        f["intensity_difference"] = f["mean_intensity"] - f["surrounding_mean_intensity"]
        f["intensity_ratio"] = f["mean_intensity"] / f["surrounding_mean_intensity"] if sur.mean() > 0 else float("nan")
    else:
        f["surrounding_mean_intensity"] = f["intensity_difference"] = f["intensity_ratio"] = float("nan")
    return f


def image_texture_maps(gray):
    g8 = gray if gray.dtype == np.uint8 else np.clip(gray / max(float(gray.max()), 1.0) * 255, 0, 255).astype(np.uint8)
    edges = cv2.Canny(g8, 50, 150)
    gf = gray.astype(np.float32)
    grad = np.hypot(cv2.Sobel(gf, cv2.CV_32F, 1, 0), cv2.Sobel(gf, cv2.CV_32F, 0, 1))
    return edges, grad


def extract_texture_features(gray, maps, b):
    H, W = gray.shape
    x0, y0, x1, y1 = roi_box(b, W, H)
    roi = gray[y0:y1, x0:x1]
    _, counts = np.unique(roi, return_counts=True)
    p = counts / counts.sum()
    edges, grad = maps
    g = grad[y0:y1, x0:x1]
    return {"entropy": float(-(p * np.log2(p)).sum()),  # Shannon entropy, base 2 (= skimage shannon_entropy)
            "edge_density": float((edges[y0:y1, x0:x1] > 0).mean()),
            "mean_gradient": float(g.mean()), "std_gradient": float(g.std()),
            "roi_variance": float(roi.astype(np.float64).var())}


# ------------------------------------------------------------------ thresholds / labels
def calculate_size_thresholds(train_lesions, q_lo, q_hi):
    r = np.array([l["bbox_area_ratio"] for l in train_lesions])
    return {"small_max": float(np.quantile(r, q_lo)), "medium_max": float(np.quantile(r, q_hi)),
            "quantiles": [q_lo, q_hi], "n_train_lesions": int(len(r)),
            "note": "dataset-relative geometric categories from TRAIN lesions only; not clinical severity"}


def assign_relative_size(ratio, thr):
    return "small" if ratio <= thr["small_max"] else ("medium" if ratio <= thr["medium_max"] else "large")


def assign_shape(aspect, wide, tall):
    return "wide" if aspect > wide else ("tall" if aspect < tall else "compact")


# ---------------------------------------------------------------------- paligemma coords
def to_loc_yxyx(b, W, H):
    """[ymin, xmin, ymax, xmax] bins in 0..1023, same convention as training/decoding."""
    x0, y0, x1, y1 = b
    q = lambda v, s: round(max(0, min(1023, v / s * 1024)))
    return [q(y0, H), q(x0, W), q(y1, H), q(x1, W)]


def loc_string(b, W, H, label):
    y0, x0, y1, x1 = to_loc_yxyx(b, W, H)
    return f"<loc{y0:04d}><loc{x0:04d}><loc{y1:04d}><loc{x1:04d}> {label}"


def build_suffix(lesions, label, eos):
    s = " ; ".join(loc_string(l["bbox"], l["W"], l["H"], label) for l in lesions)
    return s + ("<eos>" if eos else "")


# ------------------------------------------------------------------------- prompts
def lesion_prompt(group, l, cls):
    t = PROMPT_TEMPLATES
    if group == "baseline":
        return t["baseline"].format(cls=cls)
    if group == "size":
        return t["size"].format(cls=cls, size=SIZE_WORD[l["relative_size"]])
    if group == "horizontal":
        if l["image_horizontal_position"] == "center":
            return t["horizontal_center"].format(cls=cls)
        return t["horizontal"].format(cls=cls, horizontal=l["image_horizontal_position"])
    if group == "vertical":
        return t["vertical"].format(cls=cls, vertical=l["image_vertical_position"])
    if group == "region":
        return t["region"].format(cls=cls, region=l["image_region"])
    if group == "size_region":
        return t["size_region"].format(cls=cls, size=SIZE_WORD[l["relative_size"]], region=l["image_region"])
    if group == "shape":
        return t["shape"].format(cls=cls, shape=l["bbox_shape"])
    if group == "size_shape_region":
        return t["size_shape_region"].format(cls=cls, size=SIZE_WORD[l["relative_size"]],
                                             shape=l["bbox_shape"], region=l["image_region"])
    raise ValueError(group)


def clue_metadata(group):
    c = CLUE_FLAGS[group]
    return {"prompt_group": group, "contains_size_clue": c.get("size", False),
            "contains_horizontal_clue": c.get("horizontal", False), "contains_vertical_clue": c.get("vertical", False),
            "contains_shape_clue": c.get("shape", False), "contains_intensity_clue": False,
            "contains_texture_clue": False}


def generate_prompts(img, groups, cls, label, eos):
    """All prompt rows for one image + ambiguous lesion list.
    baseline  -> one row, ALL boxes as target (spec mode A; negatives -> empty target)
    feature groups -> one row per lesion, single-box target (mode B), skipped if the
                      same prompt text would also describe another lesion (ambiguous)."""
    rows, ambiguous = [], []
    lesions = img["lesions"]
    for g in groups:
        if g == "baseline":
            rows.append({**clue_metadata(g), "image": img["name"], "split": img["split"],
                         "lesion_ids": [l["lesion_id"] for l in lesions], "prompt": lesion_prompt(g, {}, cls),
                         "target_bboxes_xyxy": [l["bbox"] for l in lesions],
                         "target_loc_yxyx": [to_loc_yxyx(l["bbox"], l["W"], l["H"]) for l in lesions],
                         "suffix": build_suffix(lesions, label, eos) if lesions else ("<eos>" if eos else ""),
                         "is_negative": not lesions, "is_ambiguous": False})
            continue
        if not lesions:
            continue  # spec s.25: negatives only get the baseline prompt
        texts = [lesion_prompt(g, l, cls) for l in lesions]
        counts = Counter(texts)
        for l, text in zip(lesions, texts):
            if counts[text] > 1:
                ambiguous.append((g, l["lesion_id"], text))
                l.setdefault("ambiguous_groups", []).append(g)
                continue
            rows.append({**clue_metadata(g), "image": img["name"], "split": img["split"],
                         "lesion_ids": [l["lesion_id"]], "prompt": text, "target_bboxes_xyxy": [l["bbox"]],
                         "target_loc_yxyx": [to_loc_yxyx(l["bbox"], l["W"], l["H"])],
                         "suffix": build_suffix([l], label, eos), "is_negative": False, "is_ambiguous": False,
                         "features": {k: l[k] for k in ("relative_size", "image_horizontal_position",
                                                        "image_vertical_position", "image_region",
                                                        "bbox_shape", "bbox_area_ratio")}})
    return rows, ambiguous


def pick_mixed(rows_by_image, seed):
    """ONE prompt per image: pick a group uniformly among the groups that have a valid
    row for that image, then a row inside it. Deterministic per (seed, split, image)."""
    mixed = []
    for (split, name), rows in sorted(rows_by_image.items()):
        by_group = defaultdict(list)
        for r in rows:
            if r["prompt_group"] not in MIXED_EXCLUDE:
                by_group[r["prompt_group"]].append(r)
        rng = random.Random(f"{seed}:{split}:{name}")
        g = rng.choice(sorted(by_group))
        mixed.append({**rng.choice(by_group[g]), "mixed_from_group": g})
    return mixed


# --------------------------------------------------------------------------- outputs
def write_group_dataset(out_root, group, rows, image_paths, label_eos, symlink):
    stats = {}
    for split in SPLITS:
        srows = [r for r in rows if r["split"] == split]
        skipped = [r for r in srows if not r["suffix"]]
        srows = [r for r in srows if r["suffix"]]
        d = out_root / "datasets" / group / split
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "annotations.jsonl", "w") as f:
            for r in srows:
                f.write(json.dumps({"image": r["image"], "prefix": r["prompt"], "suffix": r["suffix"]}) + "\n")
        if symlink:
            for name in sorted({r["image"] for r in srows}):
                dst = d / name
                if not dst.is_symlink() and not dst.exists():
                    dst.symlink_to(image_paths[(split, name)])
        stats[split] = (len(srows), len(skipped))
    return stats


def draw_qc(out_dir, samples, image_paths, thresholds):
    qc = out_dir / "qc"
    qc.mkdir(parents=True, exist_ok=True)
    colors = [(0, 255, 0), (0, 200, 255), (255, 0, 255), (255, 128, 0)]
    for i, (img, rows) in enumerate(samples):
        base = cv2.imread(str(image_paths[(img["split"], img["name"])]), cv2.IMREAD_COLOR)
        if base is None:
            continue
        for k, l in enumerate(img["lesions"]):
            x0, y0, x1, y1 = [int(round(v)) for v in l["bbox"]]
            cv2.rectangle(base, (x0, y0), (x1, y1), colors[k % 4], 2)
            cv2.putText(base, str(k + 1), (x0 + 3, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[k % 4], 1)
        lines = [f"Image: {img['name']}  ({img['split']})"]
        for k, l in enumerate(img["lesions"]):
            lines += [f"[{k + 1}] bbox=[{l['bbox'][0]:.0f},{l['bbox'][1]:.0f},{l['bbox'][2]:.0f},{l['bbox'][3]:.0f}]",
                      f"    size={l['relative_size']} (area ratio {l['bbox_area_ratio']:.4f})  region={l['image_region']}",
                      f"    shape={l['bbox_shape']}  mean intensity={l['mean_intensity']:.1f}"]
        if not img["lesions"]:
            lines.append("NEGATIVE image (no boxes)")
        lines += ["", f"size thresholds: small<={thresholds['small_max']:.4f}, medium<={thresholds['medium_max']:.4f}",
                  "", "Prompts:"]
        for r in sorted(rows, key=lambda r: (r["prompt_group"], r["lesion_ids"])):
            lines.append(f"- [{r['prompt_group']}] {r['prompt']}")
        panel = np.full((max(base.shape[0], 22 * len(lines) + 10), 760, 3), 255, np.uint8)
        for j, ln in enumerate(lines):
            cv2.putText(panel, ln, (8, 22 + 22 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        canvas = np.full((panel.shape[0], base.shape[1] + 760, 3), 255, np.uint8)
        canvas[:base.shape[0], :base.shape[1]] = base
        canvas[:, base.shape[1]:] = panel
        cv2.imwrite(str(qc / f"qc_{i:02d}_{img['name']}"), canvas)


def verify_roundtrip(lesions, label):
    """Encode -> decode with supervision (the same parser evaluation uses) and compare."""
    import supervision as sv
    worst, n_bad = 0.0, 0
    for l in lesions:
        det = sv.Detections.from_lmm(sv.LMM.PALIGEMMA, loc_string(l["bbox"], l["W"], l["H"], label),
                                     resolution_wh=(l["W"], l["H"]), classes=[label])
        if len(det) != 1:
            n_bad += 1
            continue
        err = float(np.abs(det.xyxy[0] - np.array(l["bbox"])).max())
        worst = max(worst, err)
        n_bad += err > 0.51 * max(l["W"], l["H"]) / 512 + 0.01
    return worst, n_bad


# ------------------------------------------------------------------------------ main
def main():
    a = parse_args()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eos = not a.no_eos
    groups = [g for g in a.groups.split(",") if g]
    if a.enable_shape and "shape" not in groups:
        groups.append("shape")
    assert "baseline" in groups, "baseline group is required (spec group 0)"
    random.seed(a.seed)
    np.random.seed(a.seed)

    errors, image_paths, images = [], {}, {s: [] for s in SPLITS}
    png_info = Counter()
    coco = load_annotations(a.coco_dir, a.limit, a.images_dir)

    # ---- stage 1: images, validation, geometry + intensity + texture features
    for split in SPLITS:
        for a_ in coco[split]["orphan_annotations"]:
            errors.append([split, "", "", "annotation_without_image", f"ann id {a_['id']}"])
        for e in coco[split]["entries"]:
            name = e["path"].name
            if not e["path"].exists():
                errors.append([split, name, "", "missing_png", str(e["path"])])
                continue
            gray, info = read_gray(e["path"])
            if gray is None:
                errors.append([split, name, "", "unreadable_image", str(e["path"])])
                continue
            png_info[(info["png_dtype"], info["png_channels"], info["png_channels_identical"])] += 1
            H, W = gray.shape
            if (W, H) != (e["width"], e["height"]):
                errors.append([split, name, "", "image_size_mismatch",
                               f"json {e['width']}x{e['height']} vs png {W}x{H}; using png size"])
            image_paths[(split, name)] = e["path"]
            maps = image_texture_maps(gray)
            img = {"name": name, "split": split, "W": W, "H": H, "lesions": []}
            for k, ann in enumerate(e["anns"], 1):
                bx, by, bw, bh = ann["bbox"]
                b = [float(bx), float(by), float(bx + bw), float(by + bh)]
                lid = f"{Path(name).stem}_box{k:02d}"
                err = validate_bbox(b, W, H, a.bbox_tolerance_px)
                if err:
                    errors.append([split, name, lid, err, f"bbox_xyxy={b} image={W}x{H}"])
                    continue
                l = {"lesion_id": lid, "split": split, "image": name, "class": a.class_phrase, "bbox": b, "W": W, "H": H}
                l.update(extract_bbox_features(b, W, H))
                l["image_horizontal_position"], l["image_vertical_position"], l["image_region"] = \
                    infer_image_region(l["center_x_norm"], l["center_y_norm"])
                l.update(extract_png_intensity_features(gray, b, a.surround_margin_px))
                l.update(extract_texture_features(gray, maps, b))
                l.update(info)
                l["loc_tokens_yxyx"] = ",".join(map(str, to_loc_yxyx(b, W, H)))
                img["lesions"].append(l)
            if e["anns"] and not img["lesions"]:
                errors.append([split, name, "", "image_has_only_invalid_annotations", "dropped from all groups"])
                continue
            images[split].append(img)

    # ---- stage 2: TRAIN-only thresholds, applied identically to every split (spec s.27)
    train_l = [l for im in images["train"] for l in im["lesions"]]
    thr = calculate_size_thresholds(train_l, *a.size_quantiles)
    for split in SPLITS:
        for im in images[split]:
            for l in im["lesions"]:
                l["relative_size"] = assign_relative_size(l["bbox_area_ratio"], thr)
                l["bbox_shape"] = assign_shape(l["aspect_ratio"], a.shape_wide, a.shape_tall)
                l["n_lesions_in_image"] = len(im["lesions"])

    # ---- stage 3: prompts
    all_rows, rows_by_image, n_amb = [], defaultdict(list), Counter()
    for split in SPLITS:
        for im in images[split]:
            rows, amb = generate_prompts(im, groups, a.class_phrase, a.target_label, eos)
            for g, lid, text in amb:
                errors.append([split, im["name"], lid, "ambiguous_multiple_lesion_prompt", f"[{g}] {text}"])
                n_amb[g] += 1
            for r in rows:
                rows_by_image[(split, im["name"])].append(r)
            all_rows += rows
    mixed_rows = pick_mixed(rows_by_image, a.seed)

    # ---- outputs
    lesions_all = [l for s in SPLITS for im in images[s] for l in im["lesions"]]
    with open(out_dir / "lesion_features.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FEATURE_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for l in lesions_all:
            row = dict(l)
            row["ambiguous_groups"] = "|".join(l.get("ambiguous_groups", []))
            w.writerow(row)
    with open(out_dir / "prompts_all.jsonl", "w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")
    with open(out_dir / "errors.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "image", "lesion_id", "error_type", "detail"])
        w.writerows(errors)
    thr_out = {"size_thresholds": thr, "shape_thresholds": {"wide_if_aspect_gt": a.shape_wide, "tall_if_aspect_lt": a.shape_tall},
               "region_rule": "center_x_norm / center_y_norm split at 1/3 and 2/3 of the PNG (image position, NOT anatomical left/right)",
               "config": {k: v for k, v in vars(a).items()}, "groups": groups, "append_eos": eos}
    (out_dir / "thresholds.json").write_text(json.dumps(thr_out, indent=2))

    ds_stats = {}
    for g in groups + ["mixed"]:
        rows = mixed_rows if g == "mixed" else [r for r in all_rows if r["prompt_group"] == g]
        ds_stats[g] = write_group_dataset(out_dir, g, rows, image_paths, eos, not a.no_symlinks)

    # ---- QC
    rng = random.Random(a.seed)
    pos = [im for s in SPLITS for im in images[s] if im["lesions"]]
    multi = [im for im in pos if len(im["lesions"]) > 1]
    neg = [im for s in SPLITS for im in images[s] if not im["lesions"]]
    pick = rng.sample(pos, min(a.qc_samples, len(pos))) + rng.sample(multi, min(4, len(multi))) + neg[:2]
    draw_qc(out_dir, [(im, rows_by_image[(im["split"], im["name"])]) for im in pick], image_paths, thr)

    # ---- round-trip verification of the coordinate encoding
    worst, n_bad = verify_roundtrip(lesions_all, a.target_label)

    # ---- statistics
    def dist(vals):
        v = np.array(vals, float)
        return {"n": int(len(v)), "min": float(v.min()), "p25": float(np.percentile(v, 25)), "median": float(np.median(v)),
                "p75": float(np.percentile(v, 75)), "max": float(v.max()), "mean": float(v.mean())} if len(v) else {}
    n_img = {s: len(images[s]) for s in SPLITS}
    stats = {
        "images_per_split": n_img, "positive_images": sum(1 for s in SPLITS for im in images[s] if im["lesions"]),
        "negative_images": len(neg), "total_boxes": len(lesions_all),
        "images_with_multiple_boxes": len(multi),
        "bbox_area_ratio": dist([l["bbox_area_ratio"] for l in lesions_all]),
        "aspect_ratio": dist([l["aspect_ratio"] for l in lesions_all]),
        "horizontal_counts": dict(Counter(l["image_horizontal_position"] for l in lesions_all)),
        "vertical_counts": dict(Counter(l["image_vertical_position"] for l in lesions_all)),
        "region_counts": dict(Counter(l["image_region"] for l in lesions_all)),
        "size_counts": dict(Counter(l["relative_size"] for l in lesions_all)),
        "shape_counts": dict(Counter(l["bbox_shape"] for l in lesions_all)),
        "size_counts_by_split": {s: dict(Counter(l["relative_size"] for im in images[s] for l in im["lesions"])) for s in SPLITS},
        "prompt_count_total": len(all_rows),
        "prompt_count_per_group": dict(Counter(r["prompt_group"] for r in all_rows)),
        "prompt_count_per_group_and_split": {g: dict(Counter(r["split"] for r in all_rows if r["prompt_group"] == g)) for g in groups},
        "distinct_prompts_per_group": {g: len({r["prompt"] for r in all_rows if r["prompt_group"] == g}) for g in groups},
        "mixed_dataset_group_share": dict(Counter(r["mixed_from_group"] for r in mixed_rows)),
        "invalid_annotations": sum(1 for e in errors if e[3] in ("zero_area_or_inverted_bbox", "bbox_outside_image")),
        "ambiguous_prompts_per_group": dict(n_amb),
        "errors_by_type": dict(Counter(e[3] for e in errors)),
        "png_formats": {f"dtype={k[0]} channels={k[1]} identical_rgb={k[2]}": v for k, v in png_info.items()},
        "loc_roundtrip_max_abs_error_px": worst, "loc_roundtrip_boxes_over_tolerance": int(n_bad),
        "dataset_rows_written(train,valid,test | skipped_empty_target)": ds_stats,
        "append_eos": eos,
    }
    (out_dir / "statistics.json").write_text(json.dumps(stats, indent=2))
    with open(out_dir / "statistics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "split", "rows"])
        for g, d in stats["prompt_count_per_group_and_split"].items():
            for s, n in d.items():
                w.writerow([g, s, n])
    md = [f"# Prompt dataset statistics", "",
          f"- images per split: {n_img}", f"- positive / negative images: {stats['positive_images']} / {stats['negative_images']}",
          f"- total boxes: {stats['total_boxes']}  (images with >1 box: {len(multi)})",
          f"- size thresholds (train only): small <= {thr['small_max']:.4f}, medium <= {thr['medium_max']:.4f}",
          f"- region counts: {stats['region_counts']}", f"- size counts: {stats['size_counts']}",
          f"- prompts per group: {stats['prompt_count_per_group']}",
          f"- distinct prompt strings per group: {stats['distinct_prompts_per_group']}",
          f"- ambiguous (skipped) prompts: {stats['ambiguous_prompts_per_group']}",
          f"- invalid annotations: {stats['invalid_annotations']}  errors by type: {stats['errors_by_type']}",
          f"- PNG formats: {stats['png_formats']}",
          f"- loc-token round-trip: max abs error {worst:.3f} px, boxes over tolerance: {n_bad}",
          f"- append_eos: {eos}", "",
          "## How to read results", "",
          "Feature prompts (size/horizontal/vertical/region/size_region) are generated from the GROUND-TRUTH box, so",
          "they leak roughly where/how big the lesion is. Only `baseline` is a deployable detector; the other groups",
          "measure how much a correct hint helps (e.g. a reviewer typing a hint in an interactive tool). Compare groups",
          "on the SAME validation images and report baseline separately.", "",
          "## Datasets (maestro format)", ""] + [f"- `datasets/{g}/` rows (train,valid,test | skipped-empty): {v}" for g, v in ds_stats.items()]
    (out_dir / "statistics.md").write_text("\n".join(md) + "\n")

    print("\n".join(md))
    print(f"\nSample rows from the mixed dataset (one different prompt per image):")
    for r in mixed_rows[:6]:
        print(f"  {r['image']:<14} [{r['mixed_from_group']:<11}] {r['prompt']}   ->   {r['suffix']}")
    print(f"\nOutputs in {out_dir}")


if __name__ == "__main__":
    main()
