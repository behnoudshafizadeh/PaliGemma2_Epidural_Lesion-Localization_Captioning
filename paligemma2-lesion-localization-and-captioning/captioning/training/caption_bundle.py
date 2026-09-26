"""
caption_bundle.py -- the "learn each fact separately" training bundle for the lesion_only captioning task.

Why: five facts (shape, direction, outline, intensity, variation) make up to 324 label combinations; many are rare and
the model would have to memorise combinations. Instead of one fixed caption per lesion, every training sample is turned
into a small question about a RANDOM SUBSET of the facts, and the caption states exactly those facts:

    prompt : "Describe the overall intensity and main-axis direction of this epidural hemorrhage."
    caption: "The lesion region has moderate overall intensity and has a vertical main axis."

25% of samples use the generic spec prompt ("Describe the visible characteristics ...") with all five facts.
Each fact is therefore learned as its own skill; unseen combinations stop being a problem.

Augmentation that keeps the captions TRUE: the lesion-only image is flipped / rotated by 90-degree steps (exact) /
rotated a few degrees / intensity-jittered, then EVERY fact is RE-MEASURED from the augmented image with the same
function and the same train-only thresholds used for the dataset labels. So a vertical lesion rotated 90 degrees becomes
a horizontal one *and its caption says so* (this also fixes the direction imbalance: 50% vertical, 8% horizontal).

Used by: build_bundle_dataset.py (labels, thresholds, resampling), train_paligemma_captioning.py (bundle mode),
evaluate_captioning.py, test_bundle.py.
"""
import json
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from build_captioning_dataset import parse_caption                    # noqa: E402

FACTS = ["shape", "orientation", "outline", "intensity", "variation"]      # canonical order
GENERIC_PROMPT = "Describe the visible characteristics of this epidural hemorrhage."
ASK = {"shape": "shape", "orientation": "main-axis direction", "outline": "outline regularity",
       "intensity": "overall intensity", "variation": "internal intensity variation"}
SHAPE_W = {"low": "less elongated", "mid": "moderately elongated", "high": "highly elongated"}
LEVEL_W = {"low": "low", "mid": "moderate", "high": "high"}
ORIENT_W = {"vertical": "a vertical main axis", "diagonal": "a diagonal main axis", "horizontal": "a horizontal main axis",
            "none": "no dominant axis"}
CLAUSE = {"shape": lambda v: f"is {SHAPE_W[v]}", "orientation": lambda v: f"has {ORIENT_W[v]}",
          "outline": lambda v: f"shows {LEVEL_W[v]} outline regularity", "intensity": lambda v: f"has {LEVEL_W[v]} overall intensity",
          "variation": lambda v: f"shows {LEVEL_W[v]} internal intensity variation"}


# ----------------------------------------------------------------------------- measuring
def measure(img):
    """Measure the five fact quantities from a lesion-only uint8 image (lesion pixels > 0, background 0)."""
    mask = img > 0
    ys, xs = np.nonzero(mask)
    area = int(mask.sum())
    cov = np.cov(np.vstack([xs, ys]), bias=True) + np.eye(2) / 12.0
    ev = np.linalg.eigvalsh(cov)
    axis_ratio = math.sqrt(ev[1] / ev[0])
    angle = math.degrees(0.5 * math.atan2(2 * cov[0, 1], cov[0, 0] - cov[1, 1]))
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)
    hm = np.zeros(mask.shape, np.uint8)
    cv2.fillConvexPoly(hm, cv2.convexHull(cnt), 1)
    solidity = area / max(int(hm.sum()), area)
    v = img[mask].astype(np.float64)
    return {"axis_ratio": axis_ratio, "angle": angle, "solidity": solidity, "mean": float(v.mean()), "std": float(v.std())}


def orientation_label(angle, axis_ratio):
    if axis_ratio < 1.5:
        return "none"
    t = abs(angle)
    return "horizontal" if t <= 22.5 else ("diagonal" if t <= 67.5 else "vertical")


def tercile(x, t):
    return "low" if x <= t[0] else ("mid" if x <= t[1] else "high")


def label_from_measures(m, thr):
    return {"shape": tercile(m["axis_ratio"], thr["axis_ratio"]), "orientation": orientation_label(m["angle"], m["axis_ratio"]),
            "outline": tercile(m["solidity"], thr["solidity"]), "intensity": tercile(m["mean"], thr["mean"]),
            "variation": tercile(m["std"], thr["std"])}


# -------------------------------------------------------------------- prompts / captions
def compose(labels, facts):
    """-> (prompt, caption) about exactly `facts` (canonical order)."""
    facts = [f for f in FACTS if f in facts]
    if len(facts) == len(FACTS):
        prompt = GENERIC_PROMPT
    else:
        names = [ASK[f] for f in facts]
        prompt = "Describe the " + (names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]) + " of this epidural hemorrhage."
    cl = [CLAUSE[f](labels[f]) for f in facts]
    body = cl[0] if len(cl) == 1 else ", ".join(cl[:-1]) + " and " + cl[-1]
    return prompt, f"The lesion region {body}."


def explicit_full_prompt():
    names = [ASK[f] for f in FACTS]
    return "Describe the " + ", ".join(names[:-1]) + " and " + names[-1] + " of this epidural hemorrhage."


def sample_facts(rnd=random, p_generic=0.25):
    if rnd.random() < p_generic:
        return list(FACTS)
    k = rnd.choice([1, 2, 3, 4])
    return sorted(rnd.sample(FACTS, k), key=FACTS.index)


# ------------------------------------------------------------------------- augmentation
def augment(img, rnd=random, p_small_rot=0.5, small_rot_deg=15.0, jitter=True):
    """img: uint8 lesion-only image. Returns an augmented uint8 image (lesion pixels stay > 0)."""
    mask = img > 0
    if rnd.random() < 0.5:
        img = np.ascontiguousarray(img[:, ::-1])
    if rnd.random() < 0.5:
        img = np.ascontiguousarray(img[::-1, :])
    k = rnd.choice([0, 1, 2, 3])
    if k:
        img = np.ascontiguousarray(np.rot90(img, k))
    if rnd.random() < p_small_rot:
        a = rnd.uniform(-small_rot_deg, small_rot_deg)
        h, w = img.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
        rot = cv2.warpAffine(img.astype(np.float32), M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
        rm = cv2.warpAffine((img > 0).astype(np.float32), M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0) > 0.5
        out = np.where(rm, np.clip(rot, 1, 255), 0).astype(np.uint8)
        if rm.sum() >= 0.97 * (img > 0).sum():          # skip if the rotation would cut the lesion
            img = out
    if jitter and rnd.random() < 0.5:
        m = img > 0
        c, off = rnd.uniform(0.9, 1.1), rnd.uniform(-10, 10)
        img = np.where(m, np.clip((img.astype(np.float32) - 128.0) * c + 128.0 + off, 1, 255), 0).astype(np.uint8)
    return img


def to_gray(pil):
    a = np.asarray(pil)
    return a if a.ndim == 2 else a[..., 0]


# --------------------------------------------------------------------------- collate
def load_thresholds(path):
    return json.loads(Path(path).read_text())


def bundle_train_collate(batch, processor, thr, max_length=512, p_generic=0.25, augment_on=True):
    from maestro.trainer.models.paligemma_2.loaders import train_collate_fn
    out = []
    for pil, entry in batch:
        img = to_gray(pil)
        if augment_on:
            img = augment(img)
        labels = label_from_measures(measure(img), thr)          # re-measured AFTER augmentation: the caption stays true
        prompt, caption = compose(labels, sample_facts(p_generic=p_generic))
        out.append((Image.fromarray(img).convert("RGB"), {"prefix": prompt, "suffix": caption + "<eos>"}))
    return train_collate_fn(out, processor=processor, max_length=max_length)


def consistency(prompt, caption, labels):
    """Does the caption state exactly the facts the prompt asks about, with the right labels?"""
    got = parse_caption(caption)
    asked = [f for f in FACTS if ASK[f] in prompt] if prompt != GENERIC_PROMPT else list(FACTS)
    conv = {"shape": lambda x: x, "orientation": lambda x: x, "outline": lambda x: x, "intensity": lambda x: x, "variation": lambda x: x}
    return set(got) == set(asked) and all(conv[k](got[k]) == labels[k] for k in got)
