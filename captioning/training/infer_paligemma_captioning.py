"""
infer_paligemma_captioning.py -- caption new lesion-only crops with a fine-tuned captioner (Phase 7 of the spec).

    CUDA_VISIBLE_DEVICES=0 python infer_paligemma_captioning.py --adapter_dir <run>/checkpoints/epoch_015 \
        --images crop1.png crop2.png [--prompt "Describe the shape and main-axis direction of this epidural hemorrhage."] --out captions.jsonl

Input crops must be built like the training images (lesion-only, black background, see build_lesion_only_variant.py).
Non-square images are padded to a square with black (no stretching), then resized by the processor to 448x448.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import torch                                                        # noqa: E402
from peft import PeftModel                                          # noqa: E402
from PIL import Image                                               # noqa: E402
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor  # noqa: E402

from caption_metrics import clean, letterbox                        # noqa: E402
from caption_bundle import GENERIC_PROMPT                           # noqa: E402

BASE = "google/paligemma2-3b-pt-448"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--prompt", default=GENERIC_PROMPT)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max_new_tokens", type=int, default=96)
    a = ap.parse_args()
    proc = PaliGemmaProcessor.from_pretrained(BASE)
    base = PaliGemmaForConditionalGeneration.from_pretrained(BASE, torch_dtype=torch.bfloat16).to("cuda")
    model = PeftModel.from_pretrained(base, a.adapter_dir).to("cuda").eval()
    out = []
    for path in a.images:
        img = letterbox(Image.open(path).convert("RGB"))
        inp = proc(text="<image>" + a.prompt, images=img, return_tensors="pt").to("cuda", torch.bfloat16)
        with torch.no_grad():
            g = model.generate(**inp, max_new_tokens=a.max_new_tokens, do_sample=False)
        text = clean(proc.decode(g[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True))
        out.append({"image": path, "prompt": a.prompt, "caption": text})
        print(f"{path}: {text}")
    if a.out:
        Path(a.out).write_text("\n".join(json.dumps(o) for o in out) + "\n")


if __name__ == "__main__":
    main()
