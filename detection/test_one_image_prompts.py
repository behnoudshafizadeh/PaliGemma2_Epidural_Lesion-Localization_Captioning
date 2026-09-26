"""
test_one_image_prompts.py -- run ONE test image through the trained detection model with many different prompts and
compare every answer with the ground-truth box (IoU). Answers "which prompt works best for this image?".

Prompt groups (all built from the image's own true size / shape / region unless marked WRONG):
  no hint | one fact (size, shape, region) | two facts | all three (= training prompt) | WRONG size | WRONG shape |
  WRONG region | all three WRONG | the full prompt with each of the 9 regions (region sweep: does the model follow the text?)

    python test_one_image_prompts.py --adapter_dir training_output/1/checkpoints/epoch_010 \
        --images case_a.png case_b.png --out_prefix oneimage
Writes <out_prefix>_<image>.png (boxes drawn) and <out_prefix>_<image>.md (table sorted by IoU).
"""
import argparse, json, random, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import audit_prompts as A
import build_prompt_dataset as B

HERE = Path(__file__).resolve().parent
BASE = "google/paligemma2-3b-pt-448"
CLS = "epidural hemorrhage"
DATA = A.DS / "size_shape_region"
REGIONS = [f"{v}-{h}" for v in ("upper", "middle", "lower") for h in ("left", "center", "right")]
SIZES, SHAPES = ["small", "medium", "large"], ["wide", "compact", "tall"]
T = B.PROMPT_TEMPLATES


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def full(size, shape, region):
    return T["size_shape_region"].format(cls=CLS, size=B.SIZE_WORD[size], shape=shape, region=region)


def prompts_for(d):
    s, sh, rg = d["size"], d["shape"], d["region"]
    w_s = next(x for x in SIZES if x != s and abs(SIZES.index(x) - SIZES.index(s)) == max(abs(SIZES.index(y) - SIZES.index(s)) for y in SIZES))
    w_sh = "tall" if sh != "tall" else "wide"
    far = lambda r: max(REGIONS, key=lambda o: abs(REGIONS.index(o) // 3 - REGIONS.index(r) // 3) + abs(REGIONS.index(o) % 3 - REGIONS.index(r) % 3))
    w_rg = far(rg)
    P = [("no hint", T["baseline"].format(cls=CLS)),
         ("size only", T["size"].format(cls=CLS, size=B.SIZE_WORD[s])),
         ("shape only", T["shape"].format(cls=CLS, shape=sh)),
         ("region only", T["region"].format(cls=CLS, region=rg)),
         ("size + region", T["size_region"].format(cls=CLS, size=B.SIZE_WORD[s], region=rg)),
         ("ALL THREE true (training prompt)", full(s, sh, rg)),
         (f"WRONG size ({w_s})", full(w_s, sh, rg)),
         (f"WRONG shape ({w_sh})", full(s, w_sh, rg)),
         (f"WRONG region ({w_rg})", full(s, sh, w_rg)),
         ("ALL THREE wrong", full(w_s, w_sh, w_rg))]
    P += [(f"region sweep: {r}" + ("  <- true" if r == rg else ""), full(s, sh, r)) for r in REGIONS if r != w_rg or True]
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--images", nargs="*", default=None, help="image file names in the split (default: random ones, see --num_images)")
    ap.add_argument("--num_images", type=int, default=3)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out_prefix", default="oneimage")
    a = ap.parse_args()
    rows = A.load_rows("size_shape_region", a.split)
    byname = {r["image"]: r for r in rows}
    if a.images:
        chosen = a.images
    else:
        rnd = random.Random(a.seed)
        pats, chosen = set(), []
        for r in rnd.sample(rows, len(rows)):
            p = r["image"].split("_")[0]
            if p not in pats:
                pats.add(p); chosen.append(r["image"])
            if len(chosen) == a.num_images:
                break
    import torch
    from peft import PeftModel
    from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
    processor = PaliGemmaProcessor.from_pretrained(BASE)
    base = PaliGemmaForConditionalGeneration.from_pretrained(BASE, torch_dtype=torch.bfloat16).to("cuda")
    model = PeftModel.from_pretrained(base, str(Path(a.adapter_dir).resolve())).to("cuda").eval()
    for name in chosen:
        r = byname[name]
        W, H = A.img_size(a.split, "size_shape_region", name)
        gt = A.decode(r["suffix"], W, H)[0]
        d = A.derive(gt, W, H)
        image = Image.open(DATA / a.split / name).convert("RGB")
        res = []
        for label, prompt in prompts_for(d):
            inputs = processor(text="<image>" + prompt, images=image, return_tensors="pt").to("cuda", torch.bfloat16)
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=48, do_sample=False)
            pred = processor.decode(gen[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            try:
                bx = A.decode(pred, W, H)
            except Exception:
                bx = []
            box = bx[0] if bx else None
            res.append({"label": label, "prompt": prompt, "pred": pred, "box": box, "iou": iou(gt, box) if box else 0.0})
        # ---- markdown table
        md = [f"# {name} ({a.split})  -- true lesion: size **{d['size']}**, shape **{d['shape']}**, region **{d['region']}**", "",
              "| prompt type | IoU with ground truth | prompt text |", "|---|---|---|"]
        for x in sorted(res, key=lambda x: -x["iou"]):
            md.append(f"| {x['label']} | {x['iou']:.3f}{'' if x['box'] else ' (no box)'} | {x['prompt']} |")
        (HERE / f"{a.out_prefix}_{name.replace('.png', '')}.md").write_text("\n".join(md) + "\n")
        # ---- picture: one tile per prompt, GT green, prediction red
        tiles, sc = [], 300 / max(W, H)
        for x in sorted(res, key=lambda x: -x["iou"]):
            im = image.resize((int(W * sc), int(H * sc))).copy()
            dr = ImageDraw.Draw(im)
            dr.rectangle([c * sc for c in gt], outline=(0, 220, 0), width=2)
            if x["box"]:
                dr.rectangle([c * sc for c in x["box"]], outline=(255, 40, 40), width=2)
            canvas = Image.new("RGB", (300, 300 + 44), "white")
            canvas.paste(im, (0, 0))
            dc = ImageDraw.Draw(canvas)
            dc.text((3, 302), f"IoU {x['iou']:.2f}  {x['label'][:40]}", fill=(0, 0, 0))
            dc.text((3, 318), "green=truth  red=model", fill=(90, 90, 90))
            tiles.append(canvas)
        cols = 5
        rws = (len(tiles) + cols - 1) // cols
        grid = Image.new("RGB", (cols * 300, rws * 344), "white")
        for i, t in enumerate(tiles):
            grid.paste(t, ((i % cols) * 300, (i // cols) * 344))
        grid.save(HERE / f"{a.out_prefix}_{name.replace('.png', '')}.png")
        print("\n".join(md[:2] + md[2:8]), flush=True)


main()
