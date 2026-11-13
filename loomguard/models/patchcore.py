"""PatchCore (docs/ARCHITECTURE.md section 5), from the paper: Roth et al., arXiv 2106.08265. No gradient training."""
import time
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor

from loomguard.models.backbone import Backbone, gaussian_blur


@torch.no_grad()
def greedy_coreset(feats: Tensor, k: int, proj_dim: int | None = 128, seed: int = 0,
                   device: str | torch.device = "cpu") -> Tensor:
    """Greedy k-center on a seeded random projection of feats (N x D). Returns min(k, N) distinct row indices."""
    n, d = feats.shape
    k = min(k, n)
    g = torch.Generator().manual_seed(seed)
    proj = torch.randn(d, proj_dim, generator=g) / proj_dim ** 0.5 if proj_dim and proj_dim < d else None
    z = torch.cat([(c.to(device).float() @ proj.to(device)) if proj is not None else c.to(device).float()
                   for c in feats.split(65_536)])
    zz = (z * z).sum(1)
    sel = torch.empty(k, dtype=torch.long, device=device)
    i = torch.randint(n, (1,), generator=g).to(device)[0]
    mind = torch.full((n,), float("inf"), device=device)
    for j in range(k):
        sel[j] = i
        mind = torch.minimum(mind, zz - 2 * (z @ z[i]) + zz[i])  # squared distance to the newest centre
        mind[i] = -1  # never pick a centre twice, even with duplicate points
        i = mind.argmax()
    return sel.cpu()


class PatchCore:
    def __init__(self, backbone: Backbone, coreset_ratio: float = 0.10, proj_dim: int = 128, seed: int = 0,
                 device: str | torch.device = "cpu"):
        self.device = torch.device(device)
        self.backbone = backbone.to(self.device)
        self.coreset_ratio, self.proj_dim, self.seed = coreset_ratio, proj_dim, seed
        self.bank: Tensor | None = None  # M x 1536 float16
        self._bank32: Tensor | None = None

    @torch.no_grad()
    def embed(self, x: Tensor) -> Tensor:
        _, f2, f3 = self.backbone(x)
        f2, f3 = F.avg_pool2d(f2, 3, 1, 1), F.avg_pool2d(f3, 3, 1, 1)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([f2, f3], 1)

    @torch.no_grad()
    def fit(self, batches: Iterable[Tensor]) -> dict:
        chunks, n_tiles = [], 0
        for x in batches:
            z = self.embed(x.to(self.device))
            chunks.append(z.permute(0, 2, 3, 1).reshape(-1, z.shape[1]).half().cpu())
            n_tiles += len(x)
        feats = torch.cat(chunks)
        del chunks
        t0 = time.perf_counter()
        keep = greedy_coreset(feats, max(1, round(self.coreset_ratio * len(feats))), self.proj_dim, self.seed, self.device)
        seconds = time.perf_counter() - t0
        self.bank, self._bank32 = feats[keep].clone(), None
        return {"n_tiles": n_tiles, "n_patches": len(feats), "bank_size": len(self.bank), "coreset_seconds": seconds}

    @torch.no_grad()
    def score(self, x: Tensor, chunk: int = 2048) -> tuple[Tensor, Tensor]:
        if self._bank32 is None:
            self._bank32 = self.bank.to(self.device).float()
        z = self.embed(x.to(self.device))
        b, c, h, w = z.shape
        q = z.permute(0, 2, 3, 1).reshape(-1, c)
        d = torch.cat([torch.cdist(p, self._bank32).min(1).values for p in q.split(chunk)])
        grid = d.reshape(b, 1, h, w)
        heat = gaussian_blur(F.interpolate(grid, size=x.shape[-2:], mode="bilinear", align_corners=False), 4.0)
        return grid.amax((1, 2, 3)), heat

    def save(self, path: Path) -> None:
        torch.save(self.bank, path)

    @classmethod
    def load(cls, path: Path, backbone: Backbone, device: str | torch.device = "cpu", **kw) -> "PatchCore":
        pc = cls(backbone, device=device, **kw)
        pc.bank = torch.load(path, weights_only=True)
        return pc
