"""Tiler and stitcher (docs/ARCHITECTURE.md section 3). Tensors are C x H x W, origins are (y, x)."""
import math

import numpy as np
import torch
from torch import Tensor

TILE = 256


def tile_origins(length: int, tile_size: int = TILE, min_overlap: int = 0) -> list[int]:
    assert 0 <= min_overlap < tile_size
    if length <= tile_size:
        return [0]
    n = math.ceil((length - tile_size) / (tile_size - min_overlap)) + 1
    span = length - tile_size
    # round half up in integer arithmetic: floor(i * span / (n - 1) + 1/2)
    return [(2 * i * span + (n - 1)) // (2 * (n - 1)) for i in range(n)]


def grid(h: int, w: int, tile_size: int = TILE, min_overlap: int = 0) -> list[tuple[int, int]]:
    """(y0, x0) of every tile, row-major. The one place the grid is defined; manifests use it too."""
    return [(y, x) for y in tile_origins(h, tile_size, min_overlap) for x in tile_origins(w, tile_size, min_overlap)]


def tile(x: Tensor, tile_size: int = TILE, min_overlap: int = 0) -> tuple[Tensor, list[tuple[int, int]]]:
    _, h, w = x.shape
    ph, pw = max(tile_size - h, 0), max(tile_size - w, 0)
    if ph or pw:
        x = torch.from_numpy(np.pad(x.numpy(), ((0, 0), (0, ph), (0, pw)), mode="reflect"))  # repeats if pad > axis
    origins = grid(h, w, tile_size, min_overlap)
    return torch.stack([x[:, y:y + tile_size, xx:xx + tile_size] for y, xx in origins]), origins


def stitch(tiles: Tensor, origins: list[tuple[int, int]], size: tuple[int, int]) -> Tensor:
    n, c, t, _ = tiles.shape
    h, w = size
    H, W = max(h, t), max(w, t)
    # float64 accumulation: k copies of v sum and divide back to v exactly, so the round trip is bit-exact
    total = torch.zeros(c, H, W, dtype=torch.float64)
    count = torch.zeros(1, H, W, dtype=torch.float64)
    for tl, (y, x) in zip(tiles, origins):
        total[:, y:y + t, x:x + t] += tl.double()
        count[:, y:y + t, x:x + t] += 1
    return (total / count)[:, :h, :w].to(tiles.dtype)
