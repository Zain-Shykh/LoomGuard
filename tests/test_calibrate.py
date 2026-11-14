import numpy as np
import pytest

from loomguard.eval.calibrate import calibrate


def test_order_statistic():
    # n = 9, far 0.2: k = ceil(10 * 0.8) = 8, so tau is the 8th smallest
    c = calibrate([5, 1, 9, 3, 7, 2, 8, 4, 6], 0.2)
    assert c["tau"] == 8 and c["n_val_tiles"] == 9 and c["score"] == "heatmap_max"
    assert (c["score_min"], c["score_max"], c["score_median"]) == (1, 9, 5)


def test_target_rate():
    peaks = np.random.default_rng(0).permutation(np.arange(1, 1001))
    c = calibrate(peaks)  # k = ceil(1001 * 0.99) = 991
    assert c["tau"] == 991 and (peaks > c["tau"]).sum() == 9 and not c["low_confidence"]


def test_low_confidence():
    assert calibrate(range(499))["low_confidence"] and not calibrate(range(500))["low_confidence"]


def test_float_edge():
    # (99 + 1) * 0.99 is 99.00000000000001 in floating point: k must be 99 (the maximum), not 100
    assert calibrate(range(1, 100))["tau"] == 99
    # the quantile level exceeds 1 for tiny n: tau is the maximum, nothing is flagged
    assert calibrate([3.0, 1.0])["tau"] == 3.0


def test_empty_raises():
    with pytest.raises(ValueError):
        calibrate([])
