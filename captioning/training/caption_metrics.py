"""
caption_metrics.py -- shared helpers for the captioning project (training, inference, evaluation).

* letterbox()        pad a crop to a SQUARE with black, no scaling. PaliGemma's processor resizes every image to
                     448x448; without this a tall 27x80 lesion would be stretched into a square and lose the very
                     shape ("elongated") the caption talks about.
* rouge_l(), bleu4() text similarity (own LCS / nltk, no downloads needed)
* CaptionMetric      maestro-compatible validation metric: feature accuracy + ROUGE-L, computed per sample.
* feature_accuracy() what fraction of the fields stated in the REFERENCE caption the generated caption gets right.
                     Captions come from a controlled vocabulary, so this parse is exact.
"""
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                       # build_captioning_dataset.py (parse_caption)
from build_captioning_dataset import parse_caption          # noqa: E402

from maestro.trainer.common.metrics import BaseMetric        # noqa: E402

FIELDS = ["size", "shape", "region", "brightness", "variation", "texture"]


def letterbox(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = max(w, h)
    out = Image.new(img.mode, (s, s), 0)
    out.paste(img, ((s - w) // 2, (s - h) // 2))
    return out


def clean(text: str) -> str:
    return text.replace("<eos>", "").strip()


def _lcs(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l(ref: str, hyp: str) -> float:
    r, h = clean(ref).lower().split(), clean(hyp).lower().split()
    if not r or not h:
        return 0.0
    l = _lcs(r, h)
    if l == 0:
        return 0.0
    p, rec = l / len(h), l / len(r)
    return 2 * p * rec / (p + rec)


def bleu4(refs, hyps) -> float:
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
    return corpus_bleu([[clean(r).lower().split()] for r in refs], [clean(h).lower().split() for h in hyps],
                       smoothing_function=SmoothingFunction().method1)


def feature_accuracy(ref: str, hyp: str):
    """-> (n_correct, n_stated_in_reference, {field: bool})"""
    want, got = parse_caption(clean(ref)), parse_caption(clean(hyp))
    per = {k: (got.get(k) == v) for k, v in want.items()}
    return sum(per.values()), len(per), per


class CaptionMetric(BaseMetric):
    """maestro validation metric (called once per validation batch; logs the epoch mean)."""
    name = "caption"

    def describe(self):
        return ["feat_acc", "rougeL"]

    def compute(self, targets, predictions):
        accs, rouges = [], []
        for t, p in zip(targets, predictions):
            c, n, _ = feature_accuracy(t, p)
            accs.append(c / n if n else 0.0)
            rouges.append(rouge_l(t, p))
        return {"feat_acc": sum(accs) / len(accs), "rougeL": sum(rouges) / len(rouges)}
