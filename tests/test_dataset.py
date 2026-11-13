import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from loomguard.data.dataset import IMAGENET_MEAN, IMAGENET_STD, TileDataset, preprocess
from loomguard.data.manifest import build_mvtec, build_raw_fabrid
from loomguard.data.tiling import tile


def test_preprocess_gray_to_three_channels_and_normalise():
    gray = Image.fromarray(np.full((4, 5), 128, np.uint8))
    x = preprocess(gray)
    assert x.shape == (3, 4, 5) and x.dtype == torch.float32
    for c in range(3):
        assert torch.allclose(x[c], torch.tensor((128 / 255 - IMAGENET_MEAN[c]) / IMAGENET_STD[c]))
    rgb = Image.fromarray(np.stack([np.full((2, 2), v, np.uint8) for v in (0, 255, 51)], -1))
    assert torch.allclose(preprocess(rgb)[:, 0, 0], torch.tensor([(0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0],
                                                                  (1 - IMAGENET_MEAN[1]) / IMAGENET_STD[1],
                                                                  (0.2 - IMAGENET_MEAN[2]) / IMAGENET_STD[2]]))


def test_tile_equals_tiler_output(raw_root):
    rows = build_raw_fabrid(raw_root)
    ds = TileDataset(rows, raw_root)
    assert len(ds) == len(rows)
    for parent in ("images/RolloA_d000.png", "images/RolloB_n002.png"):
        idx = [i for i, r in enumerate(rows) if r.parent == parent]
        tiles, origins = tile(preprocess(Image.open(raw_root / parent)))
        assert len(idx) == 2
        for i in idx:  # the dataset's crop is the tiler's tile at the same origin
            r = rows[i]
            x, mask, label = ds[i]
            assert x.shape == (3, 256, 256) and mask.shape == (1, 256, 256) and mask.dtype == torch.float32
            assert torch.equal(x, tiles[origins.index((r.y0, r.x0))])
            assert label == r.label and int(mask.sum()) == r.defect_px
            assert set(mask.unique().tolist()) <= {0.0, 1.0}


def test_mask_position(raw_root):
    rows = build_raw_fabrid(raw_root)
    i = next(i for i, r in enumerate(rows) if r.parent.endswith("RolloB_d000.png") and r.x0 == 256)
    _, mask, label = TileDataset(rows, raw_root)[i]
    # box (y 20..30, x 270..280) is at x 14..24 inside the second tile
    assert label == 1 and mask[0, 20:30, 14:24].all() and mask.sum() == 100


def test_normal_tile_has_empty_mask(raw_root):
    rows = build_raw_fabrid(raw_root)
    _, mask, label = TileDataset([r for r in rows if r.split == "train"], raw_root)[0]
    assert label == 0 and mask.sum() == 0


def test_mvtec_rows_are_resized_to_256(mvtec_root):
    rows = build_mvtec(mvtec_root, ("tile",))  # 48x48 images
    ds = TileDataset(rows, mvtec_root)
    defect = next(i for i, r in enumerate(rows) if r.label)
    x, mask, label = ds[defect]
    assert x.shape == (3, 256, 256) and mask.shape == (1, 256, 256) and label == 1
    assert set(mask.unique().tolist()) == {0.0, 1.0}  # nearest resize keeps the mask binary
    assert 300 < mask.sum() < 600  # a 4x4 box scaled by 256/48 is about 21x21


def test_dataloader_batches(raw_root):
    rows = build_raw_fabrid(raw_root)
    x, mask, label = next(iter(DataLoader(TileDataset(rows, raw_root), batch_size=4)))
    assert x.shape == (4, 3, 256, 256) and mask.shape == (4, 1, 256, 256) and label.shape == (4,)
