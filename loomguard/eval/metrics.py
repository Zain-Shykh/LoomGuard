"""Metrics (docs/EVALUATION.md section 2), our own code: AUROC, AP, pixel AUROC/AP, AUPRO.

Everything funnels through ScoreCounts (per-bin counts of defect and normal pixels). The exact functions give every
distinct score its own bin; the streaming versions quantise scores into a fixed number of bins for large test sets.
A threshold t flags a pixel when its score is >= t.
"""
import numpy as np
import torch


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """8-connected labelling of a bool H x W mask: (int32 labels, 0 = background, count). Run-length union-find."""
    mask = np.asarray(mask, bool)
    h, w = mask.shape
    pad = np.zeros((h, w + 2), np.int8)
    pad[:, 1:-1] = mask
    d = np.diff(pad, axis=1)
    ys, xs = np.nonzero(d == 1)  # run starts, row-major
    _, xe = np.nonzero(d == -1)  # exclusive run ends, same order
    n = len(ys)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first = np.searchsorted(ys, np.arange(h + 1))
    for y in range(1, h):
        i, ie, j, je = first[y - 1], first[y], first[y], first[y + 1]
        while i < ie and j < je:
            if xs[i] <= xe[j] and xs[j] <= xe[i]:  # touch or overlap, diagonals included
                parent[find(i)] = find(j)
            if xe[i] <= xe[j]:
                i += 1
            else:
                j += 1
    labels = np.zeros((h, w), np.int32)
    ids: dict[int, int] = {}
    for k in range(n):
        labels[ys[k], xs[k]:xe[k]] = ids.setdefault(find(k), len(ids) + 1)
    return labels, len(ids)


class ScoreCounts:
    """Streaming counts per score bin, for pixel AUROC, AP and AUPRO on large test sets."""

    def __init__(self, bins: int = 10_000):
        self.bins = bins
        self.pos = np.zeros(bins, np.int64)  # defect pixels per bin
        self.neg = np.zeros(bins, np.int64)  # normal pixels per bin
        self.pro_sum = np.zeros(bins, np.float64)  # sum over regions of (region pixels in bins >= k) / region size
        self.regions = 0

    def add(self, idx: np.ndarray, mask: np.ndarray, regions: bool = False) -> None:
        idx, mask = np.asarray(idx), np.asarray(mask, bool)
        self.pos += np.bincount(idx[mask], minlength=self.bins)
        self.neg += np.bincount(idx[~mask], minlength=self.bins)
        if regions:
            lab, n = connected_components(mask)
            if n:
                m = lab > 0
                h = np.bincount((lab[m] - 1) * self.bins + idx[m], minlength=n * self.bins).reshape(n, self.bins)
                at_or_above = h[:, ::-1].cumsum(1)[:, ::-1]
                self.pro_sum += (at_or_above / h.sum(1, keepdims=True)).sum(0)
                self.regions += n

    def _descending(self):
        """tp, fp: cumulative counts flagged at thresholds from the highest bin down."""
        return self.pos[::-1].cumsum(), self.neg[::-1].cumsum()

    def auroc(self) -> float:
        tp, fp = self._descending()
        if tp[-1] == 0 or fp[-1] == 0:
            return float("nan")
        return float(np.trapezoid(np.r_[0, tp / tp[-1]], np.r_[0, fp / fp[-1]]))

    def ap(self) -> float:
        tp, fp = self._descending()
        if tp[-1] == 0:
            return float("nan")
        flagged = np.maximum(tp + fp, 1)
        recall = np.r_[0, tp / tp[-1]]
        return float(np.sum(np.diff(recall) * tp / flagged))

    def aupro(self, fpr_max: float = 0.3) -> float:
        """Step-function integral of PRO against FPR up to fpr_max, divided by fpr_max (docs: decision A-06)."""
        if self.regions == 0 or self.neg.sum() == 0:
            return float("nan")
        _, fp = self._descending()
        fpr = np.r_[0, fp / fp[-1]]  # nothing flagged, then threshold bins from the top
        pro = np.r_[0, self.pro_sum[::-1] / self.regions]
        left = np.minimum(fpr, fpr_max)
        right = np.minimum(np.r_[fpr[1:], fpr_max], fpr_max)
        return float(np.sum(pro * (right - left)) / fpr_max)


def quantize(x: torch.Tensor, lo: float, hi: float, bins: int = 10_000) -> np.ndarray:
    return ((x.double() - lo) / (hi - lo) * bins).floor().clamp(0, bins - 1).long().numpy()


def _exact(scores: np.ndarray) -> tuple[np.ndarray, int]:
    u, inv = np.unique(scores, return_inverse=True)
    return inv.reshape(scores.shape), len(u)


def roc_auc(scores, labels) -> float:
    s = np.asarray(scores, float).ravel()
    idx, n = _exact(s)
    c = ScoreCounts(n)
    c.add(idx, np.asarray(labels, bool).ravel())
    return c.auroc()


def average_precision(scores, labels) -> float:
    s = np.asarray(scores, float).ravel()
    idx, n = _exact(s)
    c = ScoreCounts(n)
    c.add(idx, np.asarray(labels, bool).ravel())
    return c.ap()


def aupro(heatmaps, masks, fpr_max: float = 0.3) -> float:
    h, m = np.asarray(heatmaps, float), np.asarray(masks, bool)
    idx, n = _exact(h)
    c = ScoreCounts(n)
    for i in range(len(h)):
        c.add(idx[i], m[i], regions=True)
    return c.aupro(fpr_max)
