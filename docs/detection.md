# Text → box

**Target** `<locY1><locX1><locY2><locX2> hemorrhage<eos>` – bins of 1/1024 of the image side, clamped to 1023.

**Prompts** (`detection/build_prompt_dataset.py`): every lesion is described from its own ground-truth box.
* size: small / medium / large – terciles of box-area / image-area fitted on the *training* split only
* shape: wide (w/h > 1.5), tall (w/h < 0.67), otherwise compact – describes the *box*
* region: 3 × 3 grid (upper|middle|lower – left|center|right)
* template: `detect the {size} epidural hemorrhage with a {shape} bounding box in the {region} region of the image` (81 combinations); `detect epidural hemorrhage` is the no-hint prompt.
Ambiguous cases (several lesions in one image that would receive the same description) are skipped; `audit_prompts.py` re-derives the words from the box for every row and reports failures (0 on the data used here).

**Training** (`train_detection.sh`): LoRA r = 8 / alpha 16, peak lr 1e-4 with 200 warm-up steps then cosine decay, LoRA dropout 0.1, mild augmentation (intensity jitter + small shift/scale, no flips; the prompt is regenerated from the augmented box), full validation split after every epoch, adapter snapshot every 5 epochs, full-state checkpoint every epoch (`RESUME=...` resumes). Loss: cross-entropy on the text tokens plus, on `<loc>` tokens, cross-entropy + 0.5 × smooth-L1 between the soft-argmax of the token distribution and the target bin.

**Test protocol** (`evaluate_prompts.py`): pick the snapshot on *validation* only, run the test split once with three prompt conditions – `matched` (own hint), `baseline` (no hint), `mismatched` (another lesion's hint with different size and region) – plus a model-free `prior_box` reference; it also reports whether the answer follows the prompt or the image, results by lesion size/shape, prompt combinations seen vs never seen in training, and per patient. `test_one_image_prompts.py` runs one image through ~20 prompt variants (single facts, wrong facts, all nine regions) and draws every answer.
