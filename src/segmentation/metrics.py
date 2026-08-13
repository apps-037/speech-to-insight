"""
Segmentation metrics: Pk and WindowDiff (lower is better for both).

A segmentation is a 0/1 per unit, with 1 marking the start of a new segment. We
turn that into the boundary strings nltk wants and set the window size `k` to
half the average reference segment length, which is the usual convention. Both
reference and hypothesis use the same representation, so the scores compare.

We skip plain accuracy because boundaries are only ~2-3% of turns, so a model
that never splits scores ~97% while being useless. Pk and WindowDiff don't fall
for that.
"""
import numpy as np
from nltk.metrics.segmentation import pk as _pk, windowdiff as _windowdiff


def _mask(labels):
    return "".join("1" if x else "0" for x in labels)


def _window_k(ref_mask):
    n_bound = max(1, ref_mask.count("1"))
    return max(2, round(len(ref_mask) / (2 * n_bound)))


def pk_windowdiff(ref_labels, hyp_labels):
    """Return (Pk, WindowDiff) for one document."""
    ref, hyp = _mask(ref_labels), _mask(hyp_labels)
    k = _window_k(ref)
    return _pk(ref, hyp, k), _windowdiff(ref, hyp, k)


def mean_scores(pairs):
    """pairs: iterable of (ref_labels, hyp_labels). Return (mean_Pk, mean_WD)."""
    P, W = [], []
    for ref, hyp in pairs:
        a, b = pk_windowdiff(ref, hyp)
        P.append(a)
        W.append(b)
    return float(np.mean(P)), float(np.mean(W))
