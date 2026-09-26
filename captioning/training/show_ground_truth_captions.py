"""
show_ground_truth_captions.py -- pictures of lesion-only images next to the caption training will teach for them.
No model needed: captions are composed from the measured labels (caption_bundle.compose). Picks lesions with different
label combinations. Writes training/ground_truth_caption_examples.{png,md}.
    python show_ground_truth_captions.py [--split test] [--num_examples 12] [--seed 3]
"""
import argparse, json, random, sys
from pathlib import Path
import cv2, numpy as np
HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path[:0] = [str(HERE), str(PROJ)]
import caption_bundle as CB   # noqa: E402


def wrap(t, n):
    out, cur = [], ""
    for w in t.split():
        if len(cur) + len(w) + 1 > n:
            out.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    return out + [cur]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--num_examples", type=int, default=12)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(PROJ / "features" / "lesion_bundle_labels.jsonl")]
    rows = [r for r in rows if r["split"] == a.split]
    rnd = random.Random(a.seed)
    rnd.shuffle(rows)
    picked, seen = [], set()
    for r in rows:                                   # diverse: distinct (shape, orientation) pairs first
        k = (r["labels"]["shape"], r["labels"]["orientation"], r["labels"]["intensity"])
        if k not in seen:
            seen.add(k); picked.append(r)
        if len(picked) == a.num_examples:
            break
    md = [f"# Ground-truth captions for lesion-only images ({a.split} split)", "",
          "Caption = what the model is trained to say. Second line = one random fact-subset prompt used in training.", ""]
    tiles = []
    for r in picked:
        img = cv2.imread(str(PROJ / "dataset" / "lesion_only" / f"{r['sample_id']}.png"), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (360, 360))
        p1, c1 = CB.compose(r["labels"], CB.FACTS)
        p2, c2 = CB.compose(r["labels"], CB.sample_facts(rnd, p_generic=0.0))
        strip = np.full((230, 360, 3), 255, np.uint8)
        cv2.putText(strip, f"{r['sample_id']} (patient {r['patient']})", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        y = 36
        for ln in wrap("CAPTION: " + c1, 52):
            cv2.putText(strip, ln, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 90, 0), 1, cv2.LINE_AA); y += 16
        y += 8
        for ln in wrap("Q: " + p2, 52) + wrap("A: " + c2, 52):
            cv2.putText(strip, ln, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (90, 60, 0), 1, cv2.LINE_AA); y += 14
        tiles.append(np.vstack([img, strip]))
        md += [f"- `{r['sample_id']}`", f"  - generic prompt: {p1}", f"  - caption: {c1}", f"  - subset example: {p2} -> {c2}", ""]
    cols = 4
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(str(HERE / "ground_truth_caption_examples.png"), grid)
    (HERE / "ground_truth_caption_examples.md").write_text("\n".join(md))
    print("\n".join(md[:14])); print("wrote", HERE / "ground_truth_caption_examples.png")


main()
