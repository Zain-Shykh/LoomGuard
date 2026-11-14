import numpy as np
import pytest

from loomguard.inspection.postprocess import clean_mask, measure, min_rect_length


def blank(h=40, w=60):
    return np.zeros((h, w), bool)


def test_close_merges_3px_gap():
    m = blank()
    m[10:20, 10:20] = True
    m[10:20, 23:33] = True  # columns 20, 21, 22 are background: a 3 px gap
    out = clean_mask(m)
    assert len(measure(out, 1.0)) == 1 and out[15, 21]  # the gap is filled


def test_gap_5px_not_merged():
    m = blank()
    m[10:20, 10:20] = True
    m[10:20, 25:35] = True
    assert len(measure(clean_mask(m), 1.0)) == 2


def test_small_blob_removed():
    m = blank()
    m[5, 5:15] = True  # 10 px line: closing does not grow it, so it stays under 16 px
    assert not clean_mask(m).any()
    m[20:30, 20:30] = True  # a 100 px blob next to it survives
    out = clean_mask(m)
    assert out[25, 25] and not out[5].any() and len(measure(out, 1.0)) == 1


def test_border_not_eroded():
    m = blank()
    m[:, :20] = True  # touches the image border
    assert clean_mask(m).sum() == m.sum()


def test_min_rect_length_axis_line_and_rectangle():
    ys, xs = np.nonzero(np.pad(np.ones((1, 200), bool), 3))
    assert min_rect_length(ys, xs) == 200
    ys, xs = np.nonzero(np.pad(np.ones((30, 90), bool), 3))
    assert min_rect_length(ys, xs) == 90


def test_min_rect_length_diagonal():
    i = np.arange(100)
    assert min_rect_length(i, i) == pytest.approx(100 * 2 ** 0.5, abs=2)  # 45 degree line: length of the diagonal


def test_measure_fields():
    m = blank()
    m[10:14, 20:30] = True  # 4 x 10 rectangle
    (d,) = measure(m, 2.0)
    assert d.bbox == (20, 10, 30, 14) and d.area_px == 40 and d.length_px == 10
    assert d.centre == (24.5, 11.5) and d.length_mm == 5 and d.area_mm2 == 10
