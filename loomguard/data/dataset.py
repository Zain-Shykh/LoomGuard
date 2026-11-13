"""Dataset that crops tiles out of parent images at load time (decision D-15). No augmentation (D-22)."""
import random
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from loomguard.data.manifest import TileRow
from loomguard.data.tiling import TILE

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
        self._last: dict[str, tuple[str, Image.Image]] = {}  # consecutive tiles share a parent: decode image and mask once

    def __len__(self) -> int:
        return len(self.rows)

    def _open(self, rel: str, mode: str) -> Image.Image:
        if self._last.get(mode, ("",))[0] != rel:
            with Image.open(self.root / rel) as im:
                self._last[mode] = (rel, im.convert(mode))
        return self._last[mode][1]

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor, int]:
        return self.get(self.rows[i])

    def get(self, r: TileRow) -> tuple[Tensor, Tensor, int]:
        box = (r.x0, r.y0, r.x0 + r.w, r.y0 + r.h)
        img = self._open(r.image, "RGB").crop(box)
        if r.mask:
            mask = self._open(r.mask, "L").crop(box)
        else:
            mask = Image.new("L", (r.w, r.h))
        if r.resize:
            img = img.resize((r.resize, r.resize), Image.BILINEAR)
            mask = mask.resize((r.resize, r.resize), Image.NEAREST)
        return preprocess(img), torch.from_numpy((np.asarray(mask) > 0).astype(np.float32))[None], r.label


class RandomCropDataset(Dataset):
    """RD4AD training stream (D-22): item i is a TILE x TILE crop at a random position of a random parent image.
    No flips, no rotations. Deterministic per (seed, i), so it is safe with DataLoader workers. Give it train rows only.
    A row with `resize` (MVTec) is a whole resized image and is returned as it is."""

    def __init__(self, rows: Sequence[TileRow], root: Path | str, length: int, seed: int):
        self.base, self.length, self.seed = TileDataset(rows, root), length, seed
        parents: dict[str, list[TileRow]] = {}
        for r in rows:
            parents.setdefault(r.parent, []).append(r)
        self.parents = [(g[0], max(r.x0 + r.w for r in g), max(r.y0 + r.h for r in g)) for g in parents.values()]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor, int]:
        rng = random.Random(f"{self.seed}/{i}")
        r, width, height = rng.choice(self.parents)
        if not r.resize:
            r = replace(r, x0=rng.randint(0, max(0, width - TILE)), y0=rng.randint(0, max(0, height - TILE)), w=TILE, h=TILE)
        return self.base.get(r)
