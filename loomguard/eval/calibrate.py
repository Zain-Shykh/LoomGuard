"""Threshold calibration (docs/ARCHITECTURE.md section 8): tau from the heatmap maxima of normal validation tiles."""
import math
from typing import Sequence

import numpy as np
from torch.utils.data import DataLoader

from loomguard.data.dataset import TileDataset
from loomguard.data.manifest import TileRow

LOW_CONFIDENCE_TILES = 500  # EVALUATION 4.1 rule 5


def calibrate(peaks: Sequence[float], target_far: float = 0.01) -> dict:
    """tau = the ceil((n+1)(1-far))-th smallest peak, so a tile flagged by `peak > tau` is a false alarm for about far of normal tiles."""
    p = np.sort(np.asarray(peaks, np.float64))
    n = len(p)
    if n == 0:
        raise ValueError("calibration needs at least one validation tile")
    k = math.ceil(round((n + 1) * (1 - target_far), 9))  # rounding: 100 * 0.99 must be 99, not 99.00000000000001
    return {"tau": float(p[min(k, n) - 1]), "target_far": target_far, "n_val_tiles": n, "low_confidence": n < LOW_CONFIDENCE_TILES,
            "score": "heatmap_max", "score_median": float(np.median(p)), "score_p99": float(np.quantile(p, 0.99)),
            "score_min": float(p[0]), "score_max": float(p[-1])}


def val_peaks(model, rows: Sequence[TileRow], root, batch_size: int = 16, num_workers: int = 0) -> list[float]:
    """Heatmap maximum of each tile in rows (give val rows only), in row order."""
    out: list[float] = []
    for x, _, _ in DataLoader(TileDataset(rows, root), batch_size=batch_size, shuffle=False, num_workers=num_workers):
        out += model.score(x)[1].amax((1, 2, 3)).cpu().tolist()
    return out
