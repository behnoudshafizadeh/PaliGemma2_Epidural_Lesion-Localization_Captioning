"""
Tests that need NO data and NO GPU (synthetic boxes / synthetic lesion shapes).   python tests/test_no_data.py   (or pytest)
  detection : box <-> <loc> token round trip (incl. NON-square images), prompt words agree with the box they describe
  captioning: caption text parses back to the labels it was composed from, fact-subset prompts are consistent,
              label-true augmentation (a 90-degree rotation turns "vertical" into "horizontal" and the re-measured caption says so)
"""
import random, sys
from pathlib import Path
import numpy as np
import supervision as sv

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "detection"), str(REPO / "common"), str(REPO / "captioning"), str(REPO / "captioning" / "training")]
import build_prompt_dataset as B            # noqa: E402
import caption_bundle as CB                 # noqa: E402
from build_captioning_dataset import parse_caption   # noqa: E402
sys.path.insert(0, str(REPO / "captioning"))
from demo_synthetic_lesions import synth, EXAMPLE_THRESHOLDS   # noqa: E402


def test_box_token_round_trip_non_square():
    rnd = random.Random(0)
    for W, H in [(512, 512), (512, 555), (512, 588)]:
        for _ in range(200):
            x0, y0 = rnd.uniform(0, W - 30), rnd.uniform(0, H - 30)
            box = [x0, y0, x0 + rnd.uniform(5, W - x0 - 1), y0 + rnd.uniform(5, H - y0 - 1)]
            text = B.loc_string(box, W, H, "hemorrhage")
            got = sv.Detections.from_lmm(sv.LMM.PALIGEMMA, text, resolution_wh=(W, H), classes=["hemorrhage"]).xyxy[0]
            assert np.abs(got - np.array(box)).max() <= max(W, H) / 1024 + 1.0, (W, H, box, got)
    # decoding with the WRONG height (512 instead of the real 555) must NOT reproduce the box: this is why evaluation decodes with the true size
    text = B.loc_string([100, 300, 200, 500], 512, 555, "hemorrhage")
    bad = sv.Detections.from_lmm(sv.LMM.PALIGEMMA, text, resolution_wh=(512, 512), classes=["hemorrhage"]).xyxy[0]
    assert abs(bad[3] - 500) > 20


def test_prompt_words_match_box():
    thr = {"small_max": 0.02, "medium_max": 0.08}
    rnd = random.Random(1)
    for _ in range(300):
        W, H = 512, 512
        x0, y0 = rnd.uniform(0, 400), rnd.uniform(0, 400)
        b = [x0, y0, x0 + rnd.uniform(10, 110), y0 + rnd.uniform(10, 110)]
        w, h = b[2] - b[0], b[3] - b[1]
        _, _, region = B.infer_image_region((b[0] + b[2]) / 2 / W, (b[1] + b[3]) / 2 / H)
        lesion = {"relative_size": B.assign_relative_size(w * h / (W * H), thr), "bbox_shape": B.assign_shape(w / h, 1.5, 0.67), "image_region": region}
        p = B.lesion_prompt("size_shape_region", lesion, "epidural hemorrhage")
        assert B.SIZE_WORD[lesion["relative_size"]] in p and lesion["bbox_shape"] in p and region in p, p


def test_caption_round_trip_and_subset_consistency():
    rng = np.random.default_rng(0); rnd = random.Random(0)
    for _ in range(60):
        img = synth(rng)
        lab = CB.label_from_measures(CB.measure(img), EXAMPLE_THRESHOLDS)
        prompt, cap = CB.compose(lab, CB.FACTS)
        assert parse_caption(cap) == lab, (cap, lab)
        facts = CB.sample_facts(rnd)
        p2, c2 = CB.compose(lab, facts)
        assert CB.consistency(p2, c2, lab), (p2, c2)


def test_label_true_augmentation_rotation():
    rng = np.random.default_rng(3)
    checked = 0
    for _ in range(200):
        img = synth(rng)
        m = CB.measure(img)
        lab = CB.label_from_measures(m, EXAMPLE_THRESHOLDS)
        if lab["orientation"] not in ("vertical", "horizontal"):
            continue
        rot = np.ascontiguousarray(np.rot90(img))
        lab2 = CB.label_from_measures(CB.measure(rot), EXAMPLE_THRESHOLDS)
        assert lab2["orientation"] == ("horizontal" if lab["orientation"] == "vertical" else "vertical")
        assert lab2["shape"] == lab["shape"] and lab2["intensity"] == lab["intensity"]      # rotation must not change the other facts
        checked += 1
    assert checked >= 20, checked


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok  ", name)
