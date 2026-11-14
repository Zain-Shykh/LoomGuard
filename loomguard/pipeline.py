"""Inspector: the single path from image + bundle to heatmap, mask, defects, points and grade (CLAUDE.md rule 5).
Used by scripts/evaluate.py and by the app. docs/ARCHITECTURE.md sections 3, 8, 9, 10."""
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import yaml
from PIL import Image
from torch import Tensor

from loomguard.data.dataset import preprocess
from loomguard.data.tiling import stitch, tile
from loomguard.inspection.grading import RollGrade, grade_roll, points
from loomguard.inspection.postprocess import Defect, clean_mask, measure

GRADING = {"closing": 5, "min_area_px": 16, "cutoffs": [20, 40, 60]}


@dataclass
class Scored:
    """Model output for one image, before any threshold."""
    heat: Tensor                     # H x W float32, tiles stitched (overlaps averaged)
    scores: Tensor                   # n_tiles, model.score()[0]: the published tile score (threshold-free metrics)
    peaks: Tensor                    # n_tiles, max of each tile's heatmap: the operating tile score (A-15)
    origins: list[tuple[int, int]]   # (y0, x0) per tile, row-major
    size: tuple[int, int]            # H, W of the (possibly resized) image


@dataclass
class Inspection:
    heat: Tensor
    mask: np.ndarray                 # H x W bool, cleaned
    defects: list[Defect]
    peaks: Tensor
    flagged_tiles: np.ndarray        # n_tiles bool
    flagged: bool                    # any tile peak above tau
    score: float                     # max tile peak
    tau: float
    px_per_mm: float
    scale_assumed: bool
    points: int
    grade: RollGrade                 # this one frame as if it were a roll: an extrapolation (D-42)


class Inspector:
    def __init__(self, model, tau: float, px_per_mm: float = 4.0, scale_assumed: bool = True, resize: int | None = None,
                 grading: dict | None = None, batch_size: int = 16):
        self.model, self.tau, self.px_per_mm, self.scale_assumed = model, tau, px_per_mm, scale_assumed
        self.resize, self.grading, self.batch_size = resize, {**GRADING, **(grading or {})}, batch_size

    @classmethod
    def load(cls, bundle: Path | str, device: str | torch.device = "cpu", backbone=None) -> "Inspector":
        from loomguard.models.backbone import Backbone
        from loomguard.runs import load_model
        b = Path(bundle)
        cfg = yaml.safe_load((b / "config.yaml").read_text())
        tau = json.loads((b / "calibration.json").read_text())["tau"]
        model = load_model(cfg, b / "model.pt", backbone or Backbone(cfg["backbone_weights"]), torch.device(device))  # weights_only=True inside
        return cls(model, tau, cfg["px_per_mm"], cfg["scale_assumed"], cfg.get("resize"), cfg.get("grading"),
                   cfg.get(cfg["method"], {}).get("batch_size", 16))

    def score_image(self, img: Image.Image) -> Scored:
        if self.resize:  # MVTec: the whole image is one tile, resized exactly as TileDataset does
            img = img.resize((self.resize, self.resize), Image.BILINEAR)
        x = preprocess(img)
        tiles, origins = tile(x)
        scores, heats = [], []
        for b in tiles.split(self.batch_size):
            s, h = self.model.score(b)
            scores.append(s.cpu()), heats.append(h.cpu())
        heats = torch.cat(heats)
        return Scored(stitch(heats, origins, tuple(x.shape[1:]))[0], torch.cat(scores), heats.amax((1, 2, 3)), origins, tuple(x.shape[1:]))

    def measure(self, mask: np.ndarray) -> tuple[np.ndarray, list[Defect]]:
        """Clean any bool mask and measure its defects, with points. Used for predicted and for ground-truth masks."""
        g = self.grading
        clean = clean_mask(mask, g["closing"], g["min_area_px"])
        return clean, [replace(d, points=points(d.length_mm)) for d in measure(clean, self.px_per_mm)]

    def decide(self, s: Scored, tau: float | None = None) -> Inspection:
        tau = self.tau if tau is None else tau
        mask, defects = self.measure((s.heat > tau).numpy())
        total = sum(d.points for d in defects)
        flagged_tiles = (s.peaks > tau).numpy()
        h, w = s.size
        return Inspection(s.heat, mask, defects, s.peaks, flagged_tiles, bool(flagged_tiles.any()), float(s.peaks.max()), tau,
                          self.px_per_mm, self.scale_assumed, total,
                          grade_roll(total, h * w / self.px_per_mm ** 2, self.grading["cutoffs"]))

    def inspect(self, img: Image.Image, tau: float | None = None) -> Inspection:
        return self.decide(self.score_image(img), tau)

    def inspect_roll(self, frames: Iterable[Image.Image]) -> tuple[list[Inspection], RollGrade]:
        """Points and area summed over the frames. ponytail: keeps every frame's heatmap; stream with inspect() for long rolls."""
        results = [self.inspect(f) for f in frames]
        area = sum(r.heat.numel() for r in results) / self.px_per_mm ** 2
        return results, grade_roll(sum(r.points for r in results), area, self.grading["cutoffs"])
