import pytest
import torch
import torch.nn.functional as F
from torch import nn

from loomguard.models.backbone import Backbone, gaussian_blur
from loomguard.models.patchcore import PatchCore, greedy_coreset


class FakeBackbone(nn.Module):
    """Right shapes, random linear patch features: fast stand-in for the WideResNet."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.c = nn.ModuleList([nn.Conv2d(3, ch, s, s, bias=False) for ch, s in ((256, 4), (512, 8), (1024, 16))])
        for c in self.c:
            c.weight.data = torch.randn(c.weight.shape, generator=g) * 0.1

    @torch.no_grad()
    def forward(self, x):
        return tuple(c(x) for c in self.c)


def normal_tiles(n, seed=0):
    return torch.randn(n, 3, 256, 256, generator=torch.Generator().manual_seed(seed)) * 0.3


def test_embed_shape():
    pc = PatchCore(Backbone(weights=None))
    assert pc.embed(torch.randn(1, 3, 256, 256)).shape == (1, 1536, 32, 32)


def brute_greedy(z, k, first):
    d = torch.cdist(z, z)
    sel = [first]
    mind = d[first].clone()
    for _ in range(k - 1):
        i = int(mind.argmax())
        sel.append(i)
        mind = torch.minimum(mind, d[i])
    return sel


def test_coreset_equals_bruteforce_farthest_point():
    z = torch.randn(300, 16, generator=torch.Generator().manual_seed(1))
    first = int(torch.randint(300, (1,), generator=torch.Generator().manual_seed(7))[0])
    got = greedy_coreset(z, 25, proj_dim=None, seed=7)
    assert got.tolist() == brute_greedy(z, 25, first)
    assert len(set(got.tolist())) == 25


def test_coreset_covers_clusters_and_edge_cases():
    centres = torch.tensor([[0.0, 0], [100, 0], [0, 100], [100, 100]])
    z = torch.cat([c + torch.randn(50, 2) * 0.1 for c in centres])
    for proj in (None, 2):
        idx = greedy_coreset(z, 4, proj_dim=proj, seed=0)
        assert sorted((idx // 50).tolist()) == [0, 1, 2, 3]  # one point per far-apart cluster
    assert sorted(greedy_coreset(z, 10_000, None).tolist()) == list(range(200))  # k >= N returns everything
    assert torch.equal(greedy_coreset(z.half(), 20, 2, seed=3), greedy_coreset(z.half(), 20, 2, seed=3))
    assert len(set(greedy_coreset(torch.zeros(10, 4), 5, None).tolist())) == 5  # duplicates stay distinct


def test_coreset_projection_keeps_coverage():
    z = torch.randn(2000, 12, generator=torch.Generator().manual_seed(2))
    sel = greedy_coreset(z, 200, proj_dim=8)
    cover = torch.cdist(z, z[sel]).min(1).values.max()
    rand = torch.cdist(z, z[torch.randperm(2000, generator=torch.Generator().manual_seed(0))[:200]]).min(1).values.max()
    assert cover < rand  # farthest-point sampling covers the cloud better than a random subset


def test_bank_size_and_dtype():
    pc = PatchCore(FakeBackbone(), coreset_ratio=0.01)
    stats = pc.fit(normal_tiles(2, s) for s in range(2))  # two batches of two tiles... (generator expression of batches)
    assert stats["n_tiles"] == 4 and stats["n_patches"] == 4096
    assert pc.bank.dtype == torch.float16 and pc.bank.shape == (round(0.01 * 4096), 1536) == (41, 1536)
    tiny = PatchCore(FakeBackbone(), coreset_ratio=1e-9)
    tiny.fit([normal_tiles(1)])
    assert tiny.bank.shape == (1, 1536)  # never empty


def test_score_matches_bruteforce_and_tile_score_is_grid_max():
    pc = PatchCore(FakeBackbone(), coreset_ratio=0.01)
    pc.fit([normal_tiles(4)])
    x = normal_tiles(2, seed=9)
    score, heat = pc.score(x)
    q = pc.embed(x).permute(0, 2, 3, 1).reshape(2, 1024, 1536)
    bank = pc.bank.float()
    grid = torch.stack([((q[b][:, None] - bank[None]) ** 2).sum(-1).sqrt().min(1).values for b in range(2)]).reshape(2, 1, 32, 32)
    assert score.shape == (2,) and score.dtype == torch.float32 and heat.shape == (2, 1, 256, 256)
    assert torch.allclose(score, grid.amax((1, 2, 3)), rtol=1e-3)
    expected = gaussian_blur(F.interpolate(grid, size=(256, 256), mode="bilinear", align_corners=False), 4.0)
    assert torch.allclose(heat, expected, rtol=1e-3, atol=1e-3)


def test_heatmap_localises_anomaly_and_save_load(tmp_path):
    pc = PatchCore(FakeBackbone(), coreset_ratio=0.1)
    pc.fit(normal_tiles(4, s) for s in range(3))
    x = normal_tiles(2, seed=11)
    x[1, :, 64:96, 160:192] += 4.0  # anomaly in tile 1 at y 64..96, x 160..192
    score, heat = pc.score(x)
    assert score[1] > 2 * score[0]
    y, xx = divmod(int(heat[1, 0].argmax()), 256)
    assert 64 <= y < 96 and 160 <= xx < 192
    pc.save(tmp_path / "model.pt")
    again = PatchCore.load(tmp_path / "model.pt", FakeBackbone())  # torch.load(weights_only=True) inside
    assert again.bank.dtype == torch.float16 and torch.equal(again.bank, pc.bank)
    s2, h2 = again.score(x)
    assert torch.equal(s2, score) and torch.equal(h2, heat)


def test_fit_is_deterministic():
    a, b = (PatchCore(FakeBackbone(), coreset_ratio=0.01, seed=4) for _ in range(2))
    a.fit([normal_tiles(3)])
    b.fit([normal_tiles(3)])
    assert torch.equal(a.bank, b.bank)
