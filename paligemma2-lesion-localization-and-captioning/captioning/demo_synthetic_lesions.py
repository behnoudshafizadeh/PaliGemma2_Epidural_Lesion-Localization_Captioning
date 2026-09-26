"""
demo_synthetic_lesions.py -- shows the captioning label pipeline WITHOUT any patient data.

It draws synthetic lesion-like shapes (random elongation, direction, edge irregularity, brightness, texture) as 448x448
"lesion-only" images (lesion pixels > 0, background 0), measures the five facts with caption_bundle.measure() and writes the
ground-truth caption with caption_bundle.compose() -- exactly the code that builds the real training captions.
These are NOT model outputs and NOT medical images. The tercile thresholds below are example numbers (the real ones are
fitted on the training split by build_bundle_dataset.py).

    python demo_synthetic_lesions.py --out ../results/synthetic_caption_demo.png --num_images 12
"""
import argparse, sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "training"))
import caption_bundle as CB  # noqa: E402

EXAMPLE_THRESHOLDS = {"axis_ratio": [4.05, 6.04], "solidity": [0.834, 0.925], "mean": [182.7, 210.1], "std": [31.2, 38.7]}
S = 448


def synth(rng):
    a = rng.uniform(60, 190)                           # half-length of the main axis
    b = a / rng.choice([rng.uniform(1.2, 3.0), rng.uniform(3.0, 6.0), rng.uniform(6.0, 10.0)])
    ang = rng.uniform(-90, 90)
    th = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    amp = rng.choice([0.0, 0.05, 0.12])                # edge irregularity
    ripple = sum(amp * rng.uniform(0.3, 1) * np.sin(k * th + rng.uniform(0, 6.28)) for k in (3, 5, 8, 13))
    r = 1 + ripple
    pts = np.stack([a * r * np.cos(th), b * r * np.sin(th)], 1)
    c, s = np.cos(np.radians(ang)), np.sin(np.radians(ang))
    pts = pts @ np.array([[c, s], [-s, c]]) + S / 2
    mask = np.zeros((S, S), np.uint8)
    cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
    mean, sd = rng.uniform(150, 235), rng.choice([15, 35, 55])
    noise = cv2.GaussianBlur(rng.normal(0, 1, (S, S)).astype(np.float32), (0, 0), 6)
    noise = noise / (noise.std() + 1e-6)
    img = np.clip(mean + sd * noise, 1, 255) * mask
    return img.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_images", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=str(HERE.parent / "results" / "synthetic_caption_demo.png"))
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    cols = 4; rows = (a.num_images + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(3.6 * cols, 4.6 * rows))
    for ax, i in zip(np.ravel(axs), range(a.num_images)):
        img = synth(rng)
        lab = CB.label_from_measures(CB.measure(img), EXAMPLE_THRESHOLDS)
        prompt, cap = CB.compose(lab, CB.FACTS)
        ax.imshow(img, cmap="gray", vmin=0, vmax=255); ax.axis("off")
        ax.set_title("\n".join(__import__("textwrap").wrap(cap.replace("The lesion region ", ""), 42)), fontsize=7.5, loc="left", y=-0.02, va="top")
    for ax in list(np.ravel(axs))[a.num_images:]:
        ax.axis("off")
    fig.suptitle("Synthetic lesion-like shapes with their ground-truth captions (\"The lesion region ...\") - not patient data, not model output", fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=120)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
