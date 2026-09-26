# Data you have to provide

```
data/
  annotations/{train,val,test}.json      COCO: images[{id, file_name, width, height}], annotations[{id, image_id, bbox:[x,y,w,h], category_id}]
  images/<patient>_<slice>.png           8-bit slices, e.g. P07_123.png
  masks/<patient>_<slice>.png            binary lesion masks with the same file names (captioning only; one connected blob per annotated lesion)
```
* **The patient id is the part of the file name before the first `_`.** Splits must be made per patient; several scripts group and report by that id.
* `file_name` in the COCO files may be an absolute path, or just a name if you pass `--images_dir`.
* Box `bbox` must agree with the mask blob (the caption builder flags samples whose provided box and mask-derived box have IoU < 0.9).

## What the builders write (all git-ignored)
**Detection** – `detection/prompt_dataset/datasets/<group>/{train,valid,test}/annotations.jsonl` (+ image symlinks), maestro format, one line per image-lesion pair:
```json
{"image": "P07_123.png", "prefix": "detect the small epidural hemorrhage with a compact bounding box in the middle-right region of the image", "suffix": "<loc0646><loc0680><loc0698><loc0722> hemorrhage<eos>"}
```
plus `thresholds.json` (train-only size terciles; shape and region rules) and audit reports. Groups: `baseline` (no hint), `size`, `horizontal`, `vertical`, `region`, `size_region`, `size_shape_region` (used for the results above), `mixed`.

**Captioning** – `captioning/training/datasets/lesion_bundle/{train,valid,test}/annotations.jsonl`:
```json
{"image": "P07_123_lesion01.png", "prefix": "Describe the visible characteristics of this epidural hemorrhage.",
 "suffix": "The lesion region is less elongated, has a diagonal main axis, shows moderate outline regularity, has low overall intensity and shows high internal intensity variation.<eos>",
 "sample_id": "P07_123_lesion01", "patient": "P07", "labels": {"shape": "low", "orientation": "diagonal", "outline": "mid", "intensity": "low", "variation": "high"}}
```
The images are lesion-only 448 × 448 crops: only the mask's pixels, black elsewhere, letter-boxed to a square.
