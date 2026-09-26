"""
compare_captions.py -- ground-truth caption vs model-generated caption, lesion by lesion, for the whole test split.

Reads training/evalcap_<split>_shard*.jsonl (generic-prompt answers) and writes ../caption_comparison/
    comparison_all.csv          one row per lesion: GT caption, generated caption, per-fact true/answered/match, facts right, similarity
    differences_summary.md      how the generated captions differ: identical-string rate, facts-right histogram, per-fact confusions
                                (true -> answered, with counts), which facts are wrong most, per-patient facts right
    comparison_examples.md      side-by-side text of examples in four groups (identical / 1 fact wrong / 2-3 wrong / 4-5 wrong),
                                the wrong clause marked  [[ ... ]]
    examples_<group>/  the lesion image + <id>_comparison.png (image, ground truth vs generated per fact) for each example
    python compare_captions.py --split test [--per_group 8]
"""
import argparse, csv, difflib, json, random, shutil, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
sys.path[:0] = [str(HERE), str(PROJ)]
import caption_bundle as CB                            # noqa: E402
from build_captioning_dataset import parse_caption     # noqa: E402
from caption_metrics import clean                      # noqa: E402

DIR = {"val": "valid", "test": "test"}


def clause(f, v):
    return CB.CLAUSE[f](v) if v in CB.CLAUSE_VALUES[f] else "(not stated)"


def render(r, path):
    """lesion image (left) + per-fact table: ground truth vs generated, wrong facts in red (right)."""
    try:
        f13, f15 = ImageFont.truetype("DejaVuSans.ttf", 14), ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
    except Exception:
        f13 = f15 = ImageFont.load_default()
    im = Image.open(PROJ / "dataset" / "lesion_only" / f"{r['sample_id']}.png").convert("RGB").resize((420, 420))
    W = 420 + 760
    cv = Image.new("RGB", (W, 470), "white"); cv.paste(im, (0, 40))
    d = ImageDraw.Draw(cv)
    d.text((8, 10), f"{r['sample_id']}  (patient {r['patient']})   {r['right']}/5 facts right", fill=(0, 0, 0), font=f15)
    x0, y = 436, 44
    d.text((x0, y), "GROUND TRUTH (from measurements)", fill=(0, 90, 0), font=f15); d.text((x0 + 380, y), "MODEL GENERATED", fill=(0, 60, 160), font=f15)
    y += 32
    for k in CB.FACTS:
        ok = r["ok"][k]
        d.text((x0, y), k, fill=(110, 110, 110), font=f13)
        d.text((x0, y + 18), clause(k, r["labels"][k]), fill=(0, 0, 0), font=f13)
        d.text((x0 + 380, y + 18), clause(k, r["got"].get(k)), fill=(0, 130, 0) if ok else (210, 0, 0), font=f13)
        d.text((x0 + 340, y + 18), "OK" if ok else "X", fill=(0, 130, 0) if ok else (210, 0, 0), font=f15)
        y += 52
    d.text((x0, y + 8), "full generated caption:", fill=(110, 110, 110), font=f13)
    words, line, yy = r["gen"].split(), "", y + 28
    for w in words:
        if len(line) + len(w) > 78:
            d.text((x0, yy), line, fill=(40, 40, 40), font=f13); yy += 17; line = w
        else:
            line = (line + " " + w).strip()
    d.text((x0, yy), line, fill=(40, 40, 40), font=f13)
    cv.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--prefix", default="evalcap")
    ap.add_argument("--per_group", type=int, default=10)
    ap.add_argument("--out", default=str(PROJ / "caption_comparison"))
    a = ap.parse_args()
    CB.CLAUSE_VALUES = {"shape": CB.SHAPE_W, "orientation": CB.ORIENT_W, "outline": CB.LEVEL_W, "intensity": CB.LEVEL_W, "variation": CB.LEVEL_W}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    recs = {}
    for f in sorted(HERE.glob(f"{a.prefix}_{a.split}_shard*.jsonl")):
        for l in open(f):
            if '"prompt_type": "generic"' in l:
                r = json.loads(l); recs[r["sample_id"]] = r
    rows = []
    for sid, r in sorted(recs.items()):
        got = parse_caption(clean(r["pred"]))
        gt = CB.compose(r["labels"], CB.FACTS)[1]
        ok = {f: got.get(f) == r["labels"][f] for f in CB.FACTS}
        sim = difflib.SequenceMatcher(None, gt.split(), clean(r["pred"]).split()).ratio()
        rows.append({"sample_id": sid, "patient": r["patient"], "gt": gt, "gen": clean(r["pred"]), "labels": r["labels"], "got": got,
                     "ok": ok, "right": sum(ok.values()), "identical": gt == clean(r["pred"]), "sim": sim})
    n = len(rows)
    with open(out / "comparison_all.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "patient", "ground_truth_caption", "generated_caption", "facts_right_of_5", "identical_text", "word_similarity"]
                   + [f"{k}_{s}" for k in CB.FACTS for s in ("true", "answered", "match")])
        for r in rows:
            w.writerow([r["sample_id"], r["patient"], r["gt"], r["gen"], r["right"], int(r["identical"]), f"{r['sim']:.3f}"]
                       + [x for k in CB.FACTS for x in (r["labels"][k], r["got"].get(k, "-"), int(r["ok"][k]))])
    # ---- summary
    hist = Counter(r["right"] for r in rows)
    L = [f"# Generated vs ground-truth captions ({a.split} split, generic prompt, {n} lesions)", "",
         f"- captions IDENTICAL to the ground truth, word for word: **{sum(r['identical'] for r in rows)}** ({sum(r['identical'] for r in rows)/n*100:.1f}%)",
         f"- mean word-sequence similarity (difflib ratio, 1 = identical): {np.mean([r['sim'] for r in rows]):.3f}   "
         f"(high even when facts are wrong, because the sentences share the template words)",
         f"- mean facts right per caption: {np.mean([r['right'] for r in rows]):.2f} of 5", "", "## number of facts right (of 5)", "",
         "| facts right | lesions | share |", "|---|---|---|"] + [f"| {k} | {hist.get(k, 0)} | {hist.get(k, 0)/n*100:.1f}% |" for k in range(5, -1, -1)]
    L += ["", "## per fact: how often the generated caption states the true value", "", "| fact | correct | most frequent mistakes (true -> answered: count) |", "|---|---|---|"]
    for k in CB.FACTS:
        c = Counter((r["labels"][k], r["got"].get(k, "not stated")) for r in rows)
        wrong = [(t, g, v) for (t, g), v in c.items() if t != g]
        wrong.sort(key=lambda x: -x[2])
        L.append(f"| {k} | {sum(r['ok'][k] for r in rows)/n*100:.1f}% | " + "; ".join(f"{t} -> {g}: {v}" for t, g, v in wrong[:4]) + " |")
    pp = defaultdict(list)
    for r in rows:
        pp[r["patient"]].append(r["right"])
    L += ["", "## per patient (mean facts right of 5)", "", "| patient | lesions | mean facts right |", "|---|---|---|"]
    L += [f"| {p} | {len(v)} | {np.mean(v):.2f} |" for p, v in sorted(pp.items(), key=lambda kv: -np.mean(kv[1]))]
    (out / "differences_summary.md").write_text("\n".join(L) + "\n")
    # ---- side-by-side examples
    groups = {"1_identical": [r for r in rows if r["right"] == 5], "2_one_fact_wrong": [r for r in rows if r["right"] == 4],
              "3_two_or_three_wrong": [r for r in rows if r["right"] in (2, 3)], "4_four_or_five_wrong": [r for r in rows if r["right"] <= 1]}
    rng = random.Random(0)
    M = ["# Side by side: ground truth vs generated (random examples per group; wrong clause marked [[like this]])", ""]
    for g, lst in groups.items():
        M += [f"## {g.split('_', 1)[1].replace('_', ' ')}  -- {len(lst)} of {n} lesions ({len(lst)/n*100:.1f}%)", ""]
        d = out / f"examples_{g}"
        d.mkdir(exist_ok=True)
        for r in rng.sample(lst, min(a.per_group, len(lst))):
            shutil.copyfile(PROJ / "dataset" / "lesion_only" / f"{r['sample_id']}.png", d / f"{r['sample_id']}.png")
            render(r, d / f"{r['sample_id']}_comparison.png")
            gtc = [clause(k, r["labels"][k]) for k in CB.FACTS]
            gnc = [("[[" + clause(k, r["got"].get(k)) + "]]") if not r["ok"][k] else clause(k, r["got"].get(k)) for k in CB.FACTS]
            M += [f"### `{r['sample_id']}`  (patient {r['patient']}, image: `examples_{g}/{r['sample_id']}.png`) -- {r['right']}/5 facts right", "",
                  f"- ground truth: The lesion region {', '.join(gtc[:-1])} and {gtc[-1]}.",
                  f"- generated:    The lesion region {', '.join(gnc[:-1])} and {gnc[-1]}.", ""]
    (out / "comparison_examples.md").write_text("\n".join(M) + "\n")
    print("\n".join(L[:22]))
    print("wrote", out)


main()
