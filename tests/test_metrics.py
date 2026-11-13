import math
from collections import deque

import numpy as np
import pytest
import torch

from loomguard.eval.metrics import (ScoreCounts, aupro, average_precision, connected_components, quantize,
                                    roc_auc)


def flood_fill(mask):
    """Reference 8-connected labelling by BFS."""
    h, w = mask.shape
    lab = np.zeros((h, w), int)
    n = 0
    for y in range(h):
        for x in range(w):
            if mask[y, x] and not lab[y, x]:
                n += 1
                lab[y, x] = n
                q = deque([(y, x)])
                while q:
                    cy, cx = q.popleft()
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not lab[ny, nx]:
                                lab[ny, nx] = n
                                q.append((ny, nx))
    return lab, n


def brute_force_aupro(heatmaps, masks, fpr_max=0.3):
    """Loop over every distinct threshold; regions from the BFS reference; step integral written out directly."""
    regions = []
    for h, m in zip(heatmaps, masks):
        lab, n = flood_fill(m)
        regions += [(h[lab == k], (lab == k).sum()) for k in range(1, n + 1)]
    neg = np.concatenate([h[~m] for h, m in zip(heatmaps, masks)])
    pts = [(0.0, 0.0)]
    for t in sorted(set(np.concatenate([h.ravel() for h in heatmaps])), reverse=True):
        fpr = (neg >= t).sum() / len(neg)
        pro = np.mean([(s >= t).sum() / n for s, n in regions])
        pts.append((fpr, pro))
    area = 0.0
    for k, (f, p) in enumerate(pts):
        nxt = pts[k + 1][0] if k + 1 < len(pts) else fpr_max
        area += p * (min(nxt, fpr_max) - min(f, fpr_max))
    return area / fpr_max


def toy(rng, n=3, size=14, levels=None):
    masks = np.zeros((n, size, size), bool)
    for m in masks:
        for _ in range(rng.integers(1, 4)):
            y, x = rng.integers(0, size - 3, 2)
            m[y:y + rng.integers(1, 4), x:x + rng.integers(1, 4)] = True
    heat = rng.random((n, size, size)) + 0.6 * masks
    if levels:
        heat = np.round(heat * levels) / levels
    return heat, masks


def test_roc_auc_and_ap_hand_values():
    s, y = [0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]
    assert roc_auc(s, y) == pytest.approx(0.75)
    assert average_precision(s, y) == pytest.approx(5 / 6)
    # tie: positive and negative both score 0.5 -> AUROC counts one half; AP: one threshold, precision 1/2
    assert roc_auc([0.5, 0.5], [0, 1]) == pytest.approx(0.5)
    assert average_precision([0.5, 0.5], [0, 1]) == pytest.approx(0.5)
    assert roc_auc([0.1, 0.2, 0.9, 0.8], [0, 0, 1, 1]) == 1.0
    assert roc_auc([0.9, 0.8, 0.1, 0.2], [0, 0, 1, 1]) == 0.0
    assert math.isnan(roc_auc([0.1, 0.2], [0, 0])) and math.isnan(average_precision([0.1, 0.2], [0, 0]))


def test_aupro_hand_value():
    # one region (2 pixels: scores 0.9, 0.5) and 8 normal pixels. Thresholds descending:
    # 0.9: fpr 0 pro .5 | 0.8: 1/8 .5 | 0.6: 2/8 .5 | 0.5: 2/8 1.0 | 0.4: 3/8 (beyond 0.3)
    # step area to 0.3: .5*(1/8) + .5*(1/8) + 1.0*(0.3 - 0.25) = 0.175
    heat = np.array([[0.9, 0.5, 0.8, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05]])
    mask = np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0]], bool)
    assert aupro(heat[None], mask[None]) == pytest.approx(0.175 / 0.3, abs=1e-12)
    assert brute_force_aupro(heat[None], mask[None]) == pytest.approx(0.175 / 0.3, abs=1e-12)


@pytest.mark.parametrize("levels", [None, 5])
def test_aupro_matches_brute_force(levels):
    rng = np.random.default_rng(3)
    for _ in range(6):
        heat, masks = toy(rng, levels=levels)
        assert aupro(heat, masks) == pytest.approx(brute_force_aupro(heat, masks), abs=1e-6)
        assert aupro(heat, masks, 0.1) == pytest.approx(brute_force_aupro(heat, masks, 0.1), abs=1e-6)


def test_aupro_perfect_and_constant():
    rng = np.random.default_rng(0)
    _, masks = toy(rng)
    assert aupro(masks.astype(float), masks) == pytest.approx(1.0, abs=1e-12)
    assert aupro(np.full(masks.shape, 0.7), masks) == 0.0
    assert aupro(1.0 - masks, masks) < 0.01  # inverted: nothing in the defects is flagged before the normals
    _, many = toy(rng, n=60, size=30)
    assert 0.08 < aupro(rng.random(many.shape), many) < 0.25  # random scores: PRO ~ FPR, so about 0.15


def test_histogram_matches_exact():
    rng = np.random.default_rng(5)
    heat, masks = toy(rng, n=4, size=20)
    # one bin per distinct value: identical to the exact functions
    vals = np.unique(heat)
    idx = np.searchsorted(vals, heat)
    c = ScoreCounts(len(vals))
    for i in range(len(heat)):
        c.add(idx[i], masks[i], regions=True)
    assert c.auroc() == pytest.approx(roc_auc(heat, masks), abs=1e-12)
    assert c.ap() == pytest.approx(average_precision(heat, masks), abs=1e-12)
    assert c.aupro() == pytest.approx(aupro(heat, masks), abs=1e-12)
    # 10,000 bins over the range: within 1e-3
    c = ScoreCounts(10_000)
    for h, m in zip(heat, masks):
        c.add(quantize(torch.from_numpy(h), heat.min(), heat.max()), m, regions=True)
    assert c.auroc() == pytest.approx(roc_auc(heat, masks), abs=1e-3)
    assert c.ap() == pytest.approx(average_precision(heat, masks), abs=1e-3)
    assert c.aupro() == pytest.approx(aupro(heat, masks), abs=1e-3)


def test_quantize_range_and_clamp():
    q = quantize(torch.tensor([-1.0, 0.0, 0.5, 1.0, 2.0]), 0.0, 1.0, bins=10)
    assert q.tolist() == [0, 0, 5, 9, 9]


def test_connected_components():
    m = np.zeros((6, 8), bool)
    m[0, 0] = m[1, 1] = True      # diagonal: one component
    m[3, 0] = m[3, 2] = True      # one-pixel gap: two components
    m[5, 7] = True
    lab, n = connected_components(m)
    assert n == 4 and lab[0, 0] == lab[1, 1] != 0 and lab[3, 0] != lab[3, 2] and lab[2, 2] == 0
    rng = np.random.default_rng(1)
    for _ in range(20):
        mask = rng.random((17, 23)) < rng.uniform(0.1, 0.6)
        lab, n = connected_components(mask)
        ref, rn = flood_fill(mask)
        assert n == rn and (lab > 0).tolist() == (ref > 0).tolist()
        # same partition up to renaming
        pairs = {(a, b) for a, b in zip(lab.ravel(), ref.ravel())}
        assert len(pairs) == n + 1 or n == 0
    assert connected_components(np.zeros((4, 4), bool))[1] == 0
