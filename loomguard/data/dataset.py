"""Dataset that crops tiles out of parent images at load time (decision D-15). No augmentation (D-22)."""
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from loomguard.data.manifest import TileRow

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def preprocess(img: Image.Image) -> Tensor:
    """PIL image -> 3 x H x W float32: grayscale repeated to 3 channels, /255, ImageNet normalisation."""
    x = torch.from_numpy(np.asarray(img.convert("RGB")).copy()).permute(2, 0, 1).float() / 255
    mean, std = torch.tensor(IMAGENET_MEAN)[:, None, None], torch.tensor(IMAGENET_STD)[:, None, None]
    return (x - mean) / std


class TileDataset(Dataset):
    def __init__(self, rows: Sequence[TileRow], root: Path | str):
        self.rows, self.root = list(rows), Path(root)
        self._last: tuple[str, Image.Image] | None = None  # consecutive tiles share a parent: decode it once

    def __len__(self) -> int:
        return len(self.rows)

    def _open(self, rel: str) -> Image.Image:
        if self._last is None or self._last[0] != rel:
            with Image.open(self.root / rel) as im:
                self._last = (rel, im.copy())
        return self._last[1]

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor, int]:
        r = self.rows[i]
        box = (r.x0, r.y0, r.x0 + r.w, r.y0 + r.h)
        img = self._open(r.image).crop(box)
        if r.mask:
            with Image.open(self.root / r.mask) as mi:
                mask = mi.convert("L").crop(box)
        else:
            mask = Image.new("L", (r.w, r.h))
        if r.resize:
            img = img.resize((r.resize, r.resize), Image.BILINEAR)
            mask = mask.resize((r.resize, r.resize), Image.NEAREST)
        return preprocess(img), torch.from_numpy((np.asarray(mask) > 0).astype(np.float32))[None], r.label
