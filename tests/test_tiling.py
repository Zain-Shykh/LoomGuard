import pytest
import torch

from loomguard.data.tiling import stitch, tile, tile_origins

# (H, W) of the four datasets: RAW-FABRID, ZJU-Leaper, AITEX, TSFabrics
SIZES = {(1024, 1792): 28, (512, 512): 4, (256, 4096): 16, (640, 800): 12}


def test_origins():
    assert tile_origins(1792) == [0, 256, 512, 768, 1024, 1280, 1536]
    assert tile_origins(1024) == [0, 256, 512, 768]
    assert tile_origins(800) == [0, 181, 363, 544]
    assert tile_origins(640) == [0, 192, 384]
    assert tile_origins(256) == [0] and tile_origins(100) == [0]
    for length, ov in [(1792, 32), (800, 64), (4096, 0), (777, 10)]:
        o = tile_origins(length, 256, ov)
        assert o[0] == 0 and o[-1] == length - 256
        assert all(b - a <= 256 - ov for a, b in zip(o, o[1:]))  # overlap at least min_overlap
        assert all(b > a for a, b in zip(o, o[1:]))
    assert len(tile_origins(1792, 256, 32)) == 8


def test_tile_counts_and_order():
    for (h, w), n in SIZES.items():
        tiles, origins = tile(torch.rand(3, h, w))
        assert tiles.shape == (n, 3, 256, 256) and len(origins) == n
        assert origins == sorted(origins)  # row-major: y outer, x inner
    _, origins = tile(torch.rand(1, 1024, 1792))
    assert origins[:2] == [(0, 0), (0, 256)] and origins[7] == (256, 0)


@pytest.mark.parametrize("overlap", [0, 32])
@pytest.mark.parametrize("h,w", list(SIZES))
def test_roundtrip_exact(h, w, overlap):
    x = torch.rand(3, h, w)
    tiles, origins = tile(x, min_overlap=overlap)
    assert torch.equal(stitch(tiles, origins, (h, w)), x)


def test_roundtrip_integer_dtype():
    x = torch.randint(0, 256, (1, 640, 800), dtype=torch.uint8)
    tiles, origins = tile(x)
    out = stitch(tiles, origins, (640, 800))
    assert out.dtype == torch.uint8 and torch.equal(out, x)


def test_roundtrip_short_axis():
    x = torch.rand(1, 100, 300)
    tiles, origins = tile(x)
    assert tiles.shape == (2, 1, 256, 256)
    assert torch.equal(tiles[0, :, :100], x[:, :, :256])
    assert torch.equal(tiles[0, :, 100], x[:, 98, :256]) and torch.equal(tiles[0, :, 101], x[:, 97, :256])  # reflect pad
    assert torch.equal(stitch(tiles, origins, (100, 300)), x)


def test_stitch_averages_overlap():
    # 256x300: two tiles at x = 0 and 44, overlap x in [44, 256)
    tiles = torch.stack([torch.full((1, 256, 256), 2.0), torch.full((1, 256, 256), 4.0)])
    out = stitch(tiles, [(0, 0), (0, 44)], (256, 300))
    assert out.shape == (1, 256, 300)
    assert (out[:, :, :44] == 2).all() and (out[:, :, 44:256] == 3).all() and (out[:, :, 256:] == 4).all()
