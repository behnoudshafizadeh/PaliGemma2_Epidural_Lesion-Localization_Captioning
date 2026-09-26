"""
audit_prompts.py -- independent check that the generated prompts are TRUE statements
about their target boxes, plus sample images to eyeball.

Independence: nothing here reuses the generator's stored features. It re-reads the
FINAL maestro dataset files (datasets/<group>/<split>/annotations.jsonl), decodes the
<loc> tokens back to pixel boxes with supervision (the same parser evaluation uses),
re-derives position / size from those pixels, and compares against the words in the
prompt. Region/size claims are checked with the train-only thresholds in thresholds.json.

Run (CPU):  python audit_prompts.py
Writes:     prompt_dataset/audit_report.txt, prompt_dataset/prompt_samples_1.png, _2.png
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from PIL import Image

ROOT = Path(__file__).resolve().parent / "prompt_dataset"
DS = ROOT / "datasets"
SPLITS = ["train", "valid", "test"]
FEATURE_GROUPS = ["size", "horizontal", "vertical", "region", "size_region", "size_shape_region"]
_T_FILE = ROOT / "thresholds.json"            # written by build_prompt_dataset.py (train-only size / shape thresholds)
if _T_FILE.exists():
    _T = json.loads(_T_FILE.read_text())
    THR = _T["size_thresholds"]
    SHAPE_THR = _T["shape_thresholds"]
else:
    THR = SHAPE_THR = None                      # run build_prompt_dataset.py first (this module still imports, --help works)
LABEL = "hemorrhage"

_size_cache = {}


def img_size(split, group, name):
    key = (split, name)
    if key not in _size_cache:
        _size_cache[key] = Image.open(DS / group / split / name).size  # (W, H), header only
    return _size_cache[key]


def decode(suffix, W, H):
    s = suffix.replace("<eos>", "")
    if "<loc" not in s:
        return []
    det = sv.Detections.from_lmm(sv.LMM.PALIGEMMA, s, resolution_wh=(W, H), classes=[LABEL])
    return [list(map(float, b)) for b in det.xyxy]


def derive(box, W, H):
    cx, cy = (box[0] + box[2]) / 2 / W, (box[1] + box[3]) / 2 / H
    h = "left" if cx < 1 / 3 else ("center" if cx < 2 / 3 else "right")
    v = "upper" if cy < 1 / 3 else ("middle" if cy < 2 / 3 else "lower")
    ratio = (box[2] - box[0]) * (box[3] - box[1]) / (W * H)
    size = "small" if ratio <= THR["small_max"] else ("medium" if ratio <= THR["medium_max"] else "large")
    aspect = (box[2] - box[0]) / (box[3] - box[1])
    shape = "wide" if aspect > SHAPE_THR["wide_if_aspect_gt"] else ("tall" if aspect < SHAPE_THR["tall_if_aspect_lt"] else "compact")
    return {"h": h, "v": v, "region": f"{v}-{h}", "size": size, "ratio": ratio, "cx": cx, "cy": cy,
            "shape": shape, "aspect": aspect}


def claims(prompt):
    c = {}
    m = re.search(r"the (small|medium-sized|large) ", prompt)
    if m:
        c["size"] = {"medium-sized": "medium"}.get(m.group(1), m.group(1))
    m = re.search(r"on the (left|right) side of the image", prompt)
    if m:
        c["h"] = m.group(1)
    if "in the center of the image" in prompt:
        c["h"] = "center"
    m = re.search(r"in the (upper|middle|lower) part of the image", prompt)
    if m:
        c["v"] = m.group(1)
    m = re.search(r"with a (wide|tall|compact) bounding box", prompt)
    if m:
        c["shape"] = m.group(1)
    m = re.search(r"in the ([a-z]+-[a-z]+) region of the image", prompt)
    if m:
        c["region"] = m.group(1)
    return c


def load_rows(group, split):
    return [json.loads(l) for l in open(DS / group / split / "annotations.jsonl")]


def audit():
    out, total_claims, bad = [], Counter(), []
    base_locs = {}
    for split in SPLITS:
        for r in load_rows("baseline", split):
            base_locs.setdefault((split, r["image"]), set()).update(re.findall(r"(?:<loc\d{4}>){4}", r["suffix"]))
    n_rows = Counter()
    for g in FEATURE_GROUPS:
        for split in SPLITS:
            for r in load_rows(g, split):
                n_rows[g] += 1
                W, H = img_size(split, g, r["image"])
                boxes = decode(r["suffix"], W, H)
                if len(boxes) != 1:
                    bad.append((g, split, r["image"], "target is not exactly 1 box", r["suffix"]))
                    continue
                if not r["suffix"].endswith("<eos>"):
                    bad.append((g, split, r["image"], "target has no <eos>", r["suffix"]))
                loc = re.findall(r"(?:<loc\d{4}>){4}", r["suffix"])[0]
                if loc not in base_locs.get((split, r["image"]), set()):
                    bad.append((g, split, r["image"], "target box is not one of the image's baseline boxes", loc))
                d = derive(boxes[0], W, H)
                for k, claimed in claims(r["prefix"]).items():
                    total_claims[(g, k)] += 1
                    if d[k] != claimed:
                        if k in ("h", "v", "region"):
                            ax = d["cx"] if k in ("h", "region") else d["cy"]
                            dim = W if k in ("h", "region") else H
                            edge = min(abs(ax * dim - t * dim) for t in (1 / 3, 2 / 3))
                            extra = f"centre {edge:.2f}px from a 1/3 boundary"
                        else:
                            extra = f"ratio {d['ratio']:.5f}, aspect {d['aspect']:.3f}"
                        bad.append((g, split, r["image"], f"claim {k}='{claimed}' but recomputed '{d[k]}' ({extra})", r["prefix"]))
    # uniform-group check: every prompt of size_shape_region must make all of size/h?/region/shape claims
    for split in SPLITS:
        for r in load_rows("size_shape_region", split):
            c = claims(r["prefix"])
            if set(c) != {"size", "shape", "region"}:
                bad.append(("size_shape_region", split, r["image"], f"not the same feature set: {sorted(c)}", r["prefix"]))
    # baseline-group checks
    neg_ok = Counter()
    for split in SPLITS:
        for r in load_rows("baseline", split):
            has_loc = "<loc" in r["suffix"]
            neg_ok["negatives"] += (not has_loc)
            if not has_loc and r["suffix"] != "<eos>":
                bad.append(("baseline", split, r["image"], "negative target is not exactly '<eos>'", r["suffix"]))
            if r["prefix"] != "detect epidural hemorrhage":
                bad.append(("baseline", split, r["image"], "baseline prompt text changed", r["prefix"]))
            if not r["suffix"].endswith("<eos>"):
                bad.append(("baseline", split, r["image"], "baseline target has no <eos>", r["suffix"]))
    for g in FEATURE_GROUPS:
        for split in SPLITS:
            for r in load_rows(g, split):
                if "<loc" not in r["suffix"]:
                    bad.append((g, split, r["image"], "feature prompt on a negative image", r["suffix"]))

    lines = ["PROMPT AUDIT (independent re-derivation from decoded loc tokens)", "",
             f"feature-group rows checked: {dict(n_rows)}",
             f"negative images found in baseline group: {neg_ok['negatives']}", "", "claims checked per group/kind:"]
    for (g, k), n in sorted(total_claims.items()):
        wrong = sum(1 for b in bad if b[0] == g and f"claim {k}=" in b[3])
        lines.append(f"  {g:<12} {k:<7} checked {n:>5}  wrong {wrong}")
    lines += ["", f"TOTAL problems: {len(bad)}"]
    kinds = Counter(re.sub(r"'.*", "", b[3])[:60] for b in bad)
    lines += [f"  {k}: {n}" for k, n in kinds.items()]
    lines += ["", "first 15 problems:"] + [f"  {b[0]}/{b[1]}/{b[2]}: {b[3]}" for b in bad[:15]]
    return "\n".join(lines), bad


# ---------------------------------------------------------------- sample images
def wrap(t, n):
    words, lines, cur = t.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > n:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    return lines + [cur]


def tile(split, group, row, others, note=""):
    W, H = img_size(split, group, row["image"])
    im = cv2.imread(str(DS / group / split / row["image"]), cv2.IMREAD_COLOR)
    for b in others:
        cv2.rectangle(im, tuple(int(round(v)) for v in b[:2]), tuple(int(round(v)) for v in b[2:]), (0, 165, 255), 1)
    boxes = decode(row["suffix"], W, H)
    lines = [f"{row['image']} ({split}) | group: {group}", ""]
    for b in boxes:
        cv2.rectangle(im, tuple(int(round(v)) for v in b[:2]), tuple(int(round(v)) for v in b[2:]), (0, 255, 0), 3)
        d = derive(b, W, H)
        facts = f"TRUE box: region={d['region']} size={d['size']} (ratio {d['ratio']:.4f}) shape={d['shape']} (w/h {d['aspect']:.2f})"
    if not boxes:
        facts = "TRUE target: no lesion (<eos> only)"
    im = cv2.resize(im, (480, 480))
    cap = np.full((150, 480, 3), 255, np.uint8)
    y = 18
    for ln in wrap("PROMPT: " + row["prefix"], 56) + wrap(facts, 56) + (wrap(note, 56) if note else []):
        cv2.putText(cap, ln, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 1, cv2.LINE_AA)
        y += 19
    return np.vstack([im, cap])


def grid(tiles, path, cols=3):
    blank = np.full_like(tiles[0], 255)
    tiles = tiles + [blank] * (-len(tiles) % cols)  # pad the last row so every row has the same width
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(str(path), np.vstack(rows))


def samples():
    import random
    rng = random.Random(7)
    data = {(g, s): load_rows(g, s) for g in ["baseline"] + FEATURE_GROUPS for s in SPLITS}
    base_by = {(s, r["image"]): r for s in SPLITS for r in data[("baseline", s)]}

    def all_boxes(split, name):
        W, H = img_size(split, "baseline", name)
        return decode(base_by[(split, name)]["suffix"], W, H)

    # grid 1: one validation image per group (patients differ)
    t1, used = [], set()
    for g in ["baseline"] + FEATURE_GROUPS:
        cand = [r for r in data[(g, "valid")] if r["image"].split("_")[0] not in used]
        r = rng.choice(cand)
        used.add(r["image"].split("_")[0])
        t1.append(tile("valid", g, r, [b for b in all_boxes("valid", r["image"]) if b not in decode(r["suffix"], *img_size("valid", g, r["image"]))]))
    grid(t1, ROOT / "prompt_samples_1.png")

    # grid 2: hard cases -- multi-lesion disambiguation, ambiguity skip, negative, more hints
    t2 = []
    multi = [(s, n) for (s, n), r in base_by.items() if r["suffix"].count("<loc") > 4]
    size_rows = defaultdict(list)
    for s in SPLITS:
        for r in data[("size", s)]:
            size_rows[(s, r["image"])].append(r)
    both = [k for k in multi if len(size_rows.get(k, [])) >= 2]
    if both:
        s, n = rng.choice(both)
        for r in size_rows[(s, n)][:2]:
            t2.append(tile(s, "size", r, [b for b in all_boxes(s, n)
                                          if b not in decode(r["suffix"], *img_size(s, "size", n))],
                           "orange = the OTHER lesion in this image"))
    hor_imgs = {(s, r["image"]) for s in SPLITS for r in data[("horizontal", s)]}
    amb = [k for k in multi if k not in hor_imgs]
    if amb:
        s, n = rng.choice(amb)
        r = dict(base_by[(s, n)])
        t2.append(tile(s, "baseline", r, [], "horizontal-group prompt SKIPPED for this image: both lesions on the same side -> ambiguous"))
    neg = [(s, n) for (s, n), r in base_by.items() if "<loc" not in r["suffix"]]
    if neg:
        s, n = rng.choice(neg)
        t2.append(tile(s, "baseline", base_by[(s, n)], []))
    for g in ("region", "size_region"):
        r = rng.choice(data[(g, "test")])
        t2.append(tile("test", g, r, []))
    grid(t2[:6], ROOT / "prompt_samples_2.png")

    # grid 3: the uniform dataset -- every prompt states the same 3 features
    t3, used = [], set()
    for _ in range(6):
        cand = [r for r in data[("size_shape_region", "valid")] if r["image"].split("_")[0] not in used]
        r = rng.choice(cand)
        used.add(r["image"].split("_")[0])
        W, H = img_size("valid", "size_shape_region", r["image"])
        t3.append(tile("valid", "size_shape_region", r, [b for b in all_boxes("valid", r["image"]) if b not in decode(r["suffix"], W, H)]))
    grid(t3, ROOT / "prompt_samples_uniform.png")


if __name__ == "__main__":
    report, bad = audit()
    (ROOT / "audit_report.txt").write_text(report + "\n")
    print(report)
    samples()
    print("\nwrote prompt_samples_1.png, prompt_samples_2.png, prompt_samples_uniform.png")
