"""
paligemma_map_metric.py -- fixes a bug in maestro's built-in
MeanAveragePrecisionMetric for the PaliGemma-2 recipe.

Bug: maestro.trainer.models.paligemma_2.core.PaliGemma2TrainerModule.validation_step
calls `metric.compute(predictions=generated_suffixes, targets=suffixes)` with RAW
TEXT (the PaliGemma "<locY><locX>... label" strings), but
maestro.trainer.common.metrics.MeanAveragePrecisionMetric.compute expects
`predictions`/`targets` to already be `sv.Detections` objects and calls
`detections.xyxy` directly -- crashing with:
    AttributeError: 'str' object has no attribute 'xyxy'
(confirmed against maestro installed in the `paligemma2` conda env, supervision 0.25.1)

Fix: parse each raw suffix string into `sv.Detections` with
`sv.Detections.from_lmm(sv.LMM.PALIGEMMA, text, resolution_wh=..., classes=...)`
before handing it to supervision's MeanAveragePrecision -- same box-space (pixels
on the original image, e.g. 512x512) is used for both predictions and targets, so
IoU is computed consistently regardless of the model's own input resolution
(224 or 448). Malformed/garbage generations (repeated tokens, stray non-loc text)
parse to an empty Detections object instead of raising -- verified with
`from_lmm` on real garbage output from this project's earlier runs.

Second bug found while verifying the fix above (not present in maestro's code,
this one's in how from_lmm's output is used): `from_lmm` never sets `.confidence`
(PaliGemma is generative, it has no per-box score), leaving it `None`. supervision's
MeanAveragePrecision._compute concatenates `predictions.confidence` across images
and crashes with `ValueError: zero-dimensional arrays cannot be concatenated` when
it's None. Fix: assign `confidence = np.ones(len(detections))` on every parsed
PREDICTION Detections object (every generated box is "fully confident" -- there's
no score to rank by). Confirmed unnecessary on targets (ground truth has no
confidence field in the mAP computation) and confirmed the full compute() runs end
to end afterward, both on a matching prediction (map50=1.0) and on garbage text
(0 detections, treated as a miss).

Use from a training launcher script (see train_paligemma_python.py), NOT via the
`maestro paligemma_2 train --metrics mean_average_precision` CLI flag, which always
goes through the broken maestro code path above.
"""
from typing import Any

import numpy as np
import supervision as sv
from maestro.trainer.common.metrics import BaseMetric
from supervision.metrics.mean_average_precision import MeanAveragePrecision


class PaliGemmaMeanAveragePrecisionMetric(BaseMetric):
    """Drop-in replacement for maestro's MeanAveragePrecisionMetric that correctly
    parses PaliGemma-2 "<loc> label" text into boxes before computing mAP.
    """

    name = "mean_average_precision"

    def __init__(self, classes: list[str], resolution_wh: tuple[int, int] = (512, 512)):
        self.classes = classes
        self.resolution_wh = resolution_wh

    def describe(self) -> list[str]:
        return ["map50:95", "map50", "map75"]

    def _parse(self, text: str) -> sv.Detections:
        try:
            return sv.Detections.from_lmm(
                sv.LMM.PALIGEMMA, text, resolution_wh=self.resolution_wh, classes=self.classes
            )
        except Exception:
            # Any unparseable/garbage generation -> treat as "no detections",
            # matching from_lmm's own behavior on non-matching text.
            return sv.Detections.empty()

    def compute(self, targets: list[Any], predictions: list[Any]) -> dict[str, float]:
        target_detections = [self._parse(t) for t in targets]
        prediction_detections = []
        for p in predictions:
            det = self._parse(p)
            # from_lmm never sets confidence (PaliGemma has no per-box score);
            # supervision's mAP needs a numeric array here, not None.
            det.confidence = np.ones(len(det), dtype=float)
            prediction_detections.append(det)
        result = MeanAveragePrecision().update(targets=target_detections, predictions=prediction_detections).compute()
        return {"map50:95": result.map50_95, "map50": result.map50, "map75": result.map75}
