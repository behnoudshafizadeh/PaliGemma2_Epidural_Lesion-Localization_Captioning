"""
make_result_figures.py -- turn the raw evaluation outputs into SHAREABLE results (no patient images, no real identifiers):
  * charts (detection IoU by prompt type, per-fact caption accuracy vs baseline, confusion matrices, training curves)
  * box drawings on a SCHEMATIC head outline (not a scan): ground-truth box (green) vs model box (red) with the prompt
  * anonymised text tables: generated vs ground-truth captions (patients -> P01.., lesions -> case_001..)

Inputs (all produced by the evaluation scripts of this repo):
  --det_jsonl   detection/evaluate_prompts.py output   evalfinal_test_shard*.jsonl   (glob)
  --det_summary detection/evaluate_prompts.py --aggregate summary text
  --det_metrics detection validation csv (metrics_per_epoch.csv)
  --cap_jsonl   captioning/training/evaluate_captioning.py output   evalcap_test_shard*.jsonl   (glob)
  --cap_summary captioning summary text
  --cap_metrics captioning validation csv
  --out         results/ folder to write
    python scripts/make_result_figures.py --det_jsonl 'evalfinal_test_shard*.jsonl' ... --out results
"""
import argparse, csv, glob, json, random, re, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import supervision as sv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "captioning")); sys.path.insert(0, str(REPO / "captioning" / "training"))
import caption_bundle as CB                          # noqa: E402
from build_captioning_dataset import parse_caption   # noqa: E402

FACTS = CB.FACTS


def nat(x):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", x)]


def read(pattern):
    out = []
    for f in sorted(glob.glob(pattern)):
        out += [json.loads(l) for l in open(f)]
    return out


def boxes(text):
    """'<loc y1><loc x1><loc y2><loc x2> label' -> FIRST box, normalised [x1,y1,x2,y2]; decoded with supervision exactly as in
    detection/evaluate_prompts.py (images are 512x512), so the numbers agree with the evaluation summary."""
    s = text.replace("<eos>", "")
    if "<loc" not in s:
        return None
    try:
        d = sv.Detections.from_lmm(sv.LMM.PALIGEMMA, s, resolution_wh=(512, 512), classes=["hemorrhage"])
    except Exception:
        return None
    return [float(v) / 512 for v in d.xyxy[0]] if len(d.xyxy) else None


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def anonymise_text(text, patients):
    """replace real patient ids by P01.. inside a summary text"""
    mp = {p: f"P{i + 1:02d}" for i, p in enumerate(sorted(patients, key=nat))}
    for p in sorted(mp, key=len, reverse=True):
        text = re.sub(rf"(?<![\w.]){re.escape(p)}(?![\w.])", mp[p], text)
    return text, mp


def head_canvas(ax):
    ax.set_xlim(0, 1); ax.set_ylim(1, 0); ax.set_aspect("equal"); ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 1, 1, color="#1b1b1b"))
    ax.add_patch(Ellipse((0.5, 0.52), 0.62, 0.78, fill=False, ec="#777", lw=2))
    ax.add_patch(Ellipse((0.5, 0.52), 0.55, 0.71, fill=False, ec="#444", lw=1))


# ------------------------------------------------------------------------------ detection
def detection(a, out):
    recs = read(a.det_jsonl)
    by = defaultdict(dict)
    for r in recs:
        by[r["condition"]][r["row"]] = r
    conds = [c for c in ("matched", "baseline", "mismatched") if c in by]
    names = {"matched": "correct hint", "baseline": "no hint", "mismatched": "wrong hint"}
    stats = {}
    for c in conds:
        v = []
        for i, r in by[c].items():
            g, p = boxes(r["gt"]), boxes(r["pred"])
            v.append(iou(g, p) if g and p else 0.0)
        stats[c] = (np.mean(v), np.mean(np.array(v) >= 0.5) * 100)
    prior = None
    if a.det_summary and Path(a.det_summary).exists():
        m = re.search(r"prior_box[^\n]*\n\s*n=\d+\s+mean IoU ([0-9.]+)", Path(a.det_summary).read_text())
        prior = float(m.group(1)) if m else None
    labels = [names[c] for c in conds] + (["prior box\n(no model, no image)"] if prior is not None else [])
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    vals = [stats[c][0] for c in conds] + ([prior] if prior is not None else [])
    ax[0].bar(labels, vals, color=["#2e8b57", "#4a78c2", "#c25a4a", "#999"][:len(vals)]); ax[0].set_ylabel("mean IoU (test)"); ax[0].set_ylim(0, 0.8)
    for k, v in enumerate(vals):
        ax[0].text(k, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    hit = [stats[c][1] for c in conds]
    ax[1].bar([names[c] for c in conds], hit, color=["#2e8b57", "#4a78c2", "#c25a4a"][:len(conds)]); ax[1].set_ylabel("boxes with IoU >= 0.5 (%)"); ax[1].set_ylim(0, 100)
    for k, v in enumerate(hit):
        ax[1].text(k, v + 1, f"{v:.1f}", ha="center", fontsize=9)
    for x in ax:
        x.tick_params(axis="x", labelsize=8)
    fig.suptitle("Text -> box: effect of the prompt hint (test split)"); fig.tight_layout(); fig.savefig(out / "detection_iou_by_prompt.png", dpi=140); plt.close(fig)
    # ---- schematic box drawings, 6 random cases x 3 prompts
    rows = sorted(set(by["matched"]) & set(by["baseline"]) & set(by["mismatched"]))
    rnd = random.Random(3); rnd.shuffle(rows)
    pick = rows[:6]
    fig, axs = plt.subplots(len(pick), 3, figsize=(10.5, 3.9 * len(pick)))
    for k, i in enumerate(pick):
        for j, c in enumerate(("matched", "baseline", "mismatched")):
            ax = axs[k][j]; head_canvas(ax)
            g, p = boxes(by[c][i]["gt"]), boxes(by[c][i]["pred"])
            ax.add_patch(Rectangle((g[0], g[1]), g[2] - g[0], g[3] - g[1], fill=False, ec="#00e000", lw=2))
            if p:
                ax.add_patch(Rectangle((p[0], p[1]), p[2] - p[0], p[3] - p[1], fill=False, ec="#ff4040", lw=2))
            u = iou(g, p) if p else 0.0
            ax.set_title(f"case {k + 1} - {names[c]} - IoU {u:.2f}", fontsize=9)
            ax.text(0.5, -0.02, "\n".join(re.findall(r".{1,52}(?:\s|$)", by[c][i]["prompt"])), transform=ax.transAxes, va="top", ha="center", fontsize=7, color="#222")
    fig.suptitle("Ground truth (green) vs model (red) on a SCHEMATIC head outline - not patient data", y=0.995, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.985)); fig.savefig(out / "detection_boxes_by_prompt.png", dpi=110); plt.close(fig)
    # ---- validation curve
    if a.det_metrics:
        r = list(csv.DictReader(open(a.det_metrics))); e = [int(x["epoch"]) for x in r]
        plt.figure(figsize=(6.4, 3.8))
        for k, c in (("map50", "tab:blue"), ("map50:95", "tab:red"), ("map75", "tab:green")):
            plt.plot(e, [float(x[k]) for x in r], marker="o", ms=3, label=k, color=c)
        plt.xlabel("epoch"); plt.ylabel("validation mAP"); plt.title("Text -> box: validation after every epoch"); plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
        plt.savefig(out / "detection_validation_curve.png", dpi=140); plt.close()
    return {c: stats[c] for c in conds}, prior


# ------------------------------------------------------------------------------ captioning
def captioning(a, out):
    recs = [r for r in read(a.cap_jsonl) if r["prompt_type"] == "generic"]
    patients = sorted({r["patient"] for r in recs}, key=nat)
    txt = Path(a.cap_summary).read_text()
    blk = txt.split("[generic]")[1].split("[explicit_full]")[0]
    rows = []
    for m in re.finditer(r"^\s+(shape|orientation|outline|intensity|variation)\s+([0-9.]+)%\s+\[([0-9.]+), ([0-9.]+)\]\s+([0-9.]+)%\s+([0-9.]+)%", blk, re.M):
        rows.append((m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5)), float(m.group(6))))
    fig, ax = plt.subplots(figsize=(7.5, 3.8)); x = np.arange(len(rows)); w = 0.38
    ax.bar(x - w / 2, [r[5] for r in rows], w, label="always answer the most common class", color="#bbb")
    ax.bar(x + w / 2, [r[1] for r in rows], w, label="model", color="#4a78c2",
           yerr=[[r[1] - r[2] for r in rows], [r[3] - r[1] for r in rows]], capsize=3)
    for k, r in enumerate(rows):
        ax.text(k + w / 2, r[1] + 1.5, f"{r[1]:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows]); ax.set_ylabel("accuracy on test lesions (%)"); ax.set_ylim(0, 105); ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Image -> caption: per-fact accuracy (95% CI over patients)"); fig.tight_layout(); fig.savefig(out / "captioning_per_fact_accuracy.png", dpi=140); plt.close(fig)
    # ---- confusion matrices
    order = {"shape": ["low", "mid", "high"], "outline": ["low", "mid", "high"], "intensity": ["low", "mid", "high"], "variation": ["low", "mid", "high"],
             "orientation": ["vertical", "diagonal", "horizontal", "none"]}
    fig, axs = plt.subplots(1, 5, figsize=(17, 3.6))
    for ax, f in zip(axs, FACTS):
        cl = order[f]; M = np.zeros((len(cl), len(cl)), int)
        for r in recs:
            g = parse_caption(r["pred"].replace("<eos>", "").strip()).get(f)
            if g in cl:
                M[cl.index(r["labels"][f]), cl.index(g)] += 1
        ax.imshow(M / M.sum(1, keepdims=True).clip(1), cmap="Blues", vmin=0, vmax=1)
        for i in range(len(cl)):
            for j in range(len(cl)):
                ax.text(j, i, M[i, j], ha="center", va="center", fontsize=8, color="white" if M[i, j] > 0.5 * M[i].sum() else "black")
        ax.set_xticks(range(len(cl))); ax.set_xticklabels(cl, fontsize=7); ax.set_yticks(range(len(cl))); ax.set_yticklabels(cl, fontsize=7)
        ax.set_title(f, fontsize=9); ax.set_xlabel("generated"); ax.set_ylabel("ground truth")
    fig.suptitle("Image -> caption: confusion matrices, generic prompt (counts, colour = row share)"); fig.tight_layout(); fig.savefig(out / "captioning_confusion_matrices.png", dpi=130); plt.close(fig)
    # ---- training curve
    if a.cap_metrics:
        r = list(csv.DictReader(open(a.cap_metrics))); e = [int(x["epoch"]) for x in r]
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        ax.plot(e, [float(x["feat_acc"]) for x in r], marker="o", ms=3, label="facts correct (feat_acc)", color="tab:orange")
        ax.plot(e, [float(x["rougeL"]) for x in r], marker="o", ms=3, label="ROUGE-L vs reference", color="tab:purple")
        ax.set_xlabel("epoch"); ax.set_ylabel("validation score"); ax.set_title("Image -> caption: validation after every epoch"); ax.grid(alpha=.3); ax.legend(); fig.tight_layout()
        fig.savefig(out / "captioning_validation_curve.png", dpi=140); plt.close(fig)
    # ---- anonymised text comparison (no images, no ids)
    mp = {p: f"P{i + 1:02d}" for i, p in enumerate(patients)}
    rows2 = []
    for r in sorted(recs, key=lambda r: r["sample_id"]):
        got = parse_caption(r["pred"].replace("<eos>", "").strip())
        rows2.append((r, got, {f: got.get(f) == r["labels"][f] for f in FACTS}))
    cl = lambda f, v: CB.CLAUSE[f](v) if v in {"shape": CB.SHAPE_W, "orientation": CB.ORIENT_W}.get(f, CB.LEVEL_W) else "(not stated)"
    groups = {"all five facts right": [x for x in rows2 if all(x[2].values())], "one fact wrong": [x for x in rows2 if sum(x[2].values()) == 4],
              "two or three facts wrong": [x for x in rows2 if sum(x[2].values()) <= 3]}
    rnd = random.Random(0); n = 0
    md = ["# Generated vs ground-truth captions (test split, generic prompt)", "",
          "Ground-truth captions are composed from measurements of the lesion (not written by a person or a model). The wrong clause is marked [[like this]]. "
          "Identifiers are anonymised; no images are included.", ""]
    for g, lst in groups.items():
        md += [f"## {g}: {len(lst)} of {len(rows2)} lesions ({len(lst) / len(rows2) * 100:.1f}%)", ""]
        for r, got, ok in rnd.sample(lst, min(6, len(lst))):
            n += 1
            gt = [cl(f, r["labels"][f]) for f in FACTS]
            ge = [("[[" + cl(f, got.get(f)) + "]]") if not ok[f] else cl(f, got.get(f)) for f in FACTS]
            md += [f"**case_{n:03d}** ({mp[r['patient']]}) - {sum(ok.values())}/5 facts right", "",
                   f"- ground truth: The lesion region {', '.join(gt[:-1])} and {gt[-1]}.", f"- generated:    The lesion region {', '.join(ge[:-1])} and {ge[-1]}.", ""]
    (out / "captioning_examples.md").write_text("\n".join(md) + "\n")
    return rows, mp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det_jsonl"); ap.add_argument("--det_summary"); ap.add_argument("--det_metrics")
    ap.add_argument("--cap_jsonl"); ap.add_argument("--cap_summary"); ap.add_argument("--cap_metrics")
    ap.add_argument("--out", default=str(REPO / "results"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.det_jsonl:
        print("detection:", detection(a, out))
        if a.det_summary:
            t = Path(a.det_summary).read_text()
            ids = set(re.findall(r"^\s+(\d+A):", t, re.M))
            t2, _ = anonymise_text(t, ids)
            (out / "detection_test_summary.txt").write_text(t2)
    if a.cap_jsonl:
        rows, mp = captioning(a, out)
        t2, _ = anonymise_text(Path(a.cap_summary).read_text(), set(mp))
        (out / "captioning_test_summary.txt").write_text(t2)
    print("written to", out)


main()
