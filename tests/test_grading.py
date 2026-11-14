import numpy as np
import pytest

from loomguard.inspection.grading import SQ_YD_MM2, grade_roll, points
from loomguard.inspection.postprocess import clean_mask, measure

PX = 4.0  # px per mm


def line_points(length_mm):
    m = np.zeros((40, 1400), bool)
    m[20:23, 10:10 + int(length_mm * PX)] = True  # 3 px thick horizontal line
    (d,) = measure(clean_mask(m), PX)
    return d.length_mm, points(d.length_mm)


@pytest.mark.parametrize("mm, pts", [(50, 1), (100, 2), (200, 3), (300, 4)])
def test_line_points(mm, pts):
    length, got = line_points(mm)
    assert length == mm and got == pts


def test_band_limits():
    assert [points(x) for x in (0.1, 75, 75.01, 150, 150.01, 230, 230.01, 5000)] == [1, 1, 2, 2, 3, 3, 4, 4]


def test_empty_mask_grade_a():
    assert measure(clean_mask(np.zeros((50, 50), bool)), PX) == []
    g = grade_roll(0, 114688.0)
    assert (g.points, g.points_per_100, g.grade) == (0, 0, "A")


def test_points_per_100():
    # one RAW-FABRID frame, 1792 x 1024 px at 4 px/mm = 448 x 256 mm = 114688 mm2 = 0.13717 sq yd
    area = 448 * 256
    g = grade_roll(1, area)
    assert g.area_sq_yd == pytest.approx(area / SQ_YD_MM2) and g.area_sq_yd == pytest.approx(0.13717, abs=1e-5)
    assert g.points_per_100 == pytest.approx(729.0, abs=0.1) and g.grade == "Reject"  # ARCHITECTURE 9: "730"
    assert grade_roll(2, 836127.36).points_per_100 == pytest.approx(200 / 1)  # exactly 1 sq yd


def test_cutoffs():
    one_sq_yd = SQ_YD_MM2
    # points per 100 sq yd for 100 sq yd of fabric = points
    big = 100 * one_sq_yd
    assert [grade_roll(p, big).grade for p in (20, 21, 40, 41, 60, 61)] == ["A", "B", "B", "C", "C", "Reject"]
    assert [grade_roll(p, big, (5, 10, 15)).grade for p in (5, 6, 15, 16)] == ["A", "B", "C", "Reject"]


def test_zero_area_raises():
    with pytest.raises(ValueError):
        grade_roll(1, 0)
