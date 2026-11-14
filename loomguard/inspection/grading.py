"""Four-point penalty and roll grade (docs/ARCHITECTURE.md section 9, decisions D-40 to D-43)."""
from dataclasses import dataclass
from typing import Sequence

POINT_BANDS = ((75.0, 1), (150.0, 2), (230.0, 3))  # upper limit in mm (inclusive) -> points; longer is 4
SQ_YD_MM2 = 914.4 ** 2


@dataclass(frozen=True)
class RollGrade:
    points: int
    area_sq_yd: float
    points_per_100: float
    grade: str  # "A" | "B" | "C" | "Reject"


def points(length_mm: float) -> int:
    return next((p for limit, p in POINT_BANDS if length_mm <= limit), 4)


def grade_roll(total_points: int, area_mm2: float, cutoffs: Sequence[float] = (20, 40, 60)) -> RollGrade:
    if area_mm2 <= 0:
        raise ValueError("inspected area must be positive")
    area = area_mm2 / SQ_YD_MM2
    per_100 = total_points * 100 / area
    grade = next((g for g, c in zip("ABC", cutoffs) if per_100 <= c), "Reject")
    return RollGrade(total_points, area, per_100, grade)
