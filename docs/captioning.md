# Image → caption

**Input** the lesion only (mask pixels, black elsewhere), letter-boxed to a square and resized to 448 × 448; a prompt.
**Facts** (measured on that 448 × 448 image, tercile thresholds fitted on the training split): shape (axis ratio), main-axis direction (vertical / diagonal / horizontal / none), outline regularity (solidity), overall intensity (mean grey), internal intensity variation (grey std). Size, position and brightness-versus-surroundings are *not* used because they are not visible after masking and resizing.
**Caption** = `The lesion region ` + one clause per fact, e.g. `is less elongated, has a vertical main axis, shows high outline regularity, has high overall intensity and shows low internal intensity variation.`

**Training bundle** (`captioning/training/caption_bundle.py`, `config_bundle.yaml`)
1. *Fact-subset prompts*: each draw asks about a random subset of the facts ("Describe the overall intensity and main-axis direction of this epidural hemorrhage.") and the caption answers exactly those; 25% use the generic all-five prompt.
2. *Label-true augmentation*: flips, 90° rotations, small rotations, intensity jitter; every fact is re-measured on the augmented image with the same function used for the dataset labels.
3. *Patient-balanced resampling* of the training lesions (class/combination re-weighting was tried and rejected: it skewed the level balance and repeated single lesions up to 18×).
4. LoRA r = 8, lr 1e-4 warm-up + cosine, dropout 0.1, 15 epochs; validation generates the caption for every 4th validation lesion each epoch (`feat_acc` = share of facts right).

**Evaluation** (`evaluate_captioning.py`): per-fact accuracy vs the majority-class baseline and chance, balanced accuracy, 95% CI by resampling patients, within-one-level rate, "all five right", confusion matrices, seen vs never-seen label combinations, single-fact prompts, ROUGE-L / BLEU-4 (with the caveat above). `compare_captions.py` writes, for every test lesion, ground truth vs generated caption with the wrong clauses marked.
Note: the validation split has no horizontal lesion, so judge direction on the test split. METEOR / BERTScore are not implemented.
