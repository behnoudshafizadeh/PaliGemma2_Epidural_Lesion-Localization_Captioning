"""
show_caption_examples.py -- read the generated captions next to the ground truth, with every wrong fact marked.

Reads training/evalcap_<split>_shard*.jsonl (written by evaluate_captioning.py), takes the GENERIC-prompt answers and writes
    caption_examples_<split>.md    a table of examples: fully right / partly right / mostly wrong (seeded random picks)
    caption_examples_<split>.png   the same as pictures (lesion image + reference + generated caption, wrong facts in red)
so "how similar is the generated text to the ground truth?" can be judged by eye, not only by a number.

    python show_caption_examples.py --split test [--num_examples 4]
"""
import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path[:0] = [str(HERE), str(PROJ)]
import caption_bundle as CB                                        # noqa: E402
from build_captioning_dataset import parse_caption                 # noqa: E402
from caption_metrics import clean                                  # noqa: E402

SPLIT_DIR = {"val": "valid", "test": "test"}


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
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--out_prefix", default="evalcap")
    ap.add_argument("--dataset", default="lesion_bundle")
    ap.add_argument("--num_examples", type=int, default=4, help="examples per group (fully right / partly right / mostly wrong)")
    a = ap.parse_args()
    recs = []
    for f in sorted(HERE.glob(f"{a.out_prefix}_{a.split}_shard*.jsonl")):
        recs += [json.loads(l) for l in open(f) if '"prompt_type": "generic"' in l]
    rows = []
    for r in recs:
        got = parse_caption(clean(r["pred"]))
        ok = {f: got.get(f) == r["labels"][f] for f in CB.FACTS}
        rows.append({**r, "got": got, "ok": ok, "score": sum(ok.values()), "ref": CB.compose(r["labels"], CB.FACTS)[1]})
    groups = {"fully right (5/5 facts)": [r for r in rows if r["score"] == 5],
              "partly right (2-4/5)": [r for r in rows if 2 <= r["score"] <= 4],
              "mostly wrong (0-1/5)": [r for r in rows if r["score"] <= 1]}
    rng = random.Random(0)
    picks = {g: rng.sample(v, min(a.num_examples, len(v))) for g, v in groups.items()}
    md = [f"# Generated vs ground-truth captions ({a.split} split, generic prompt)", "",
          "Facts: " + ", ".join(CB.FACTS), "",
          f"lesions: {len(rows)}   fully right: {len(groups['fully right (5/5 facts)'])} ({len(groups['fully right (5/5 facts)'])/len(rows)*100:.1f}%)   "
          f"partly right: {len(groups['partly right (2-4/5)'])}   mostly wrong: {len(groups['mostly wrong (0-1/5)'])}", ""]
    tiles = []
    ds = PROJ / "training" / "datasets" / a.dataset / SPLIT_DIR[a.split]
    for g, lst in picks.items():
        md += [f"## {g}", ""]
        for r in lst:
            wrong = [f"{f}: true **{r['labels'][f]}**, answered *{r['got'].get(f, 'nothing stated')}*" for f in CB.FACTS if not r["ok"][f]]
            md += [f"- `{r['sample_id']}` (patient {r['patient']}), {r['score']}/5 facts right",
                   f"  - ground truth: {r['ref']}", f"  - generated:    {clean(r['pred'])}",
                   f"  - wrong: {'; '.join(wrong) if wrong else 'none'}", ""]
            img = cv2.imread(str(ds / f"{r['sample_id']}.png"), cv2.IMREAD_COLOR)
            img = cv2.resize(img, (300, 300))
            strip = np.full((330, 300, 3), 255, np.uint8)
            cv2.putText(strip, f"{r['sample_id']}  {r['score']}/5", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            y = 36
            for f in CB.FACTS:
                col = (0, 130, 0) if r["ok"][f] else (0, 0, 200)
                txt = f"{f}: {r['labels'][f]}" + ("" if r["ok"][f] else f" -> {r['got'].get(f, '-')}")
                cv2.putText(strip, ("OK  " if r["ok"][f] else "X   ") + txt, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
                y += 18
            y += 8
            for ln in wrap("generated: " + clean(r["pred"]), 44)[:9]:
                cv2.putText(strip, ln, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (60, 60, 60), 1, cv2.LINE_AA)
                y += 15
            tiles.append(np.vstack([img, strip]))
    (HERE / f"caption_examples_{a.split}.md").write_text("\n".join(md) + "\n")
    if tiles:
        cols = 4
        while len(tiles) % cols:
            tiles.append(np.full_like(tiles[0], 255))
        grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
        cv2.imwrite(str(HERE / f"caption_examples_{a.split}.png"), grid)
    print("\n".join(md[:8]))
    print(f"wrote caption_examples_{a.split}.md and .png in {HERE}")


if __name__ == "__main__":
    main()
