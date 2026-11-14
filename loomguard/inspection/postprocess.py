"""Mask cleanup and defect measurement (docs/ARCHITECTURE.md section 9). NumPy and torch only."""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from loomguard.eval.metrics import connected_components


@dataclass(frozen=True)
class Defect:
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 (exclusive), pixels
    centre: tuple[float, float]      # x, y of the pixel centroid
    area_px: int
    length_px: float                 # long side of the minimum-area rectangle
    length_mm: float
    area_mm2: float
    points: int = 0                  # four-point penalty, filled in by Inspector


def clean_mask(mask: np.ndarray, closing: int = 5, min_area: int = 16) -> np.ndarray:
    """Closing with a closing x closing square (the image border is not eroded), then drop components under min_area px."""
    m = torch.from_numpy(np.asarray(mask, bool))[None, None].float()
    p = closing // 2
    m = F.max_pool2d(m, closing, 1, p)          # dilation; max_pool pads with -inf, so nothing grows from outside
    m = 1 - F.max_pool2d(1 - m, closing, 1, p)  # erosion = dilation of the background; outside counts as foreground
    lab, n = connected_components(m[0, 0].numpy() > 0.5)
    keep = np.bincount(lab.ravel(), minlength=n + 1) >= min_area
    keep[0] = False
    return keep[lab]


def min_rect_length(ys: np.ndarray, xs: np.ndarray) -> float:
    """Long side (pixels) of the minimum-area rectangle around the pixels.
    ponytail: 1 degree rotation search over row-end pixels, about 1 px error on long thin defects; rotating calipers if that matters."""
    order = np.lexsort((xs, ys))
    y, x = np.asarray(ys)[order], np.asarray(xs)[order]
    first = np.r_[True, y[1:] != y[:-1]]
    ends = np.stack([y, x], 1)[first | np.r_[first[1:], True]]  # leftmost and rightmost pixel of every row
    t = np.deg2rad(np.arange(90))
    u = ends[:, 1:2] * np.cos(t) + ends[:, 0:1] * np.sin(t)    # points x angles
    v = -ends[:, 1:2] * np.sin(t) + ends[:, 0:1] * np.cos(t)
    w, h = np.ptp(u, 0) + 1, np.ptp(v, 0) + 1
    i = int(np.argmin(w * h))
    return float(max(w[i], h[i]))


def measure(mask: np.ndarray, px_per_mm: float) -> list[Defect]:
    """One Defect per 8-connected component of an already cleaned mask."""
    lab, n = connected_components(mask)
    out = []
    if n == 0:
        return out
    flat = lab.ravel()
    order = np.argsort(flat, kind="stable")[len(flat) - np.count_nonzero(flat):]  # foreground pixels grouped by label
    for idx in np.split(order, np.cumsum(np.bincount(flat, minlength=n + 1)[1:-1])):
        ys, xs = np.divmod(idx, lab.shape[1])
        length = min_rect_length(ys, xs)
        out.append(Defect((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1), (float(xs.mean()), float(ys.mean())),
                          len(ys), length, length / px_per_mm, len(ys) / px_per_mm ** 2))
    return out
