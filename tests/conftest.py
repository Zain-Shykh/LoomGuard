"""Synthetic dataset trees written to tmp_path. No real dataset is ever needed."""
import csv

import numpy as np
import pytest
from PIL import Image

H, W = 256, 512  # two tiles per RAW-FABRID-like image


def _png(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def _mask(box):
    m = np.zeros((H, W), np.uint8)
    if box:
        y0, y1, x0, x1 = box
        m[y0:y1, x0:x1] = 255
    return m


# roll -> (normal images, [defect box per defective image]); boxes are (y0, y1, x0, x1)
RAW_LAYOUT = {
    "RolloA": (10, [(10, 15, 250, 260), (0, 4, 300, 303), (100, 101, 5, 6)]),  # straddles the tile border; tile 1 only; tile 0 only
    "RolloB": (6, [(20, 30, 270, 280), (50, 52, 10, 12)]),
}


@pytest.fixture
def raw_root(tmp_path):
    """A mini RAW-FABRID: images/, masks/, high-res metadata CSV, patch metadata CSV (computed independently)."""
    root = tmp_path / "RAW_FABRID"
    rng = np.random.default_rng(0)
    meta, patches = [], []
    for roll, (n_normal, boxes) in RAW_LAYOUT.items():
        items = [(f"{roll}_n{i:03d}.png", None) for i in range(n_normal)] + [
            (f"{roll}_d{i:03d}.png", b) for i, b in enumerate(boxes)]
        for name, box in items:
            _png(root / "images" / name, rng.integers(0, 256, (H, W), dtype=np.uint8))
            meta.append({"filename": name, "roll_id": roll, "has_defect": "True" if box else "False"})
            m = _mask(box)
            if box:
                _png(root / "masks" / name, m)
            for k, x0 in enumerate((0, 256)):
                area = int((m[:, x0:x0 + 256] > 0).sum())
                kind = "train_good" if not box else "test_defect" if area else "test_good"
                patches.append({"patch_name": f"{name[:-4]}_{k}.png", "parent_raw": name, "type": kind,
                                "defect_area": area})
    # a mask for an image the CSV calls defect-free must be ignored
    _png(root / "masks" / "RolloA_n000.png", _mask((0, 5, 0, 5)))
    for fname, rows in (("RAW_FABRID_HighRes_Metadata.csv", meta), ("RAW_FABRID_Patches_Metadata.csv", patches)):
        with open(root / fname, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
    return root


@pytest.fixture
def mvtec_root(tmp_path):
    """Mini MVTec: the five textures with small, differently sized images."""
    root = tmp_path / "mvtec"
    rng = np.random.default_rng(1)
    for cat, size in (("carpet", 64), ("grid", 32), ("leather", 32), ("tile", 48), ("wood", 32)):
        for i in range(20):
            _png(root / cat / "train" / "good" / f"{i:03d}.png", rng.integers(0, 256, (size, size, 3), dtype=np.uint8))
        for i in range(3):
            _png(root / cat / "test" / "good" / f"{i:03d}.png", rng.integers(0, 256, (size, size, 3), dtype=np.uint8))
        for kind in ("crack", "hole"):
            for i in range(2):
                _png(root / cat / "test" / kind / f"{i:03d}.png", rng.integers(0, 256, (size, size, 3), dtype=np.uint8))
                m = np.zeros((size, size), np.uint8)
                m[4:8, 4:8] = 255
                _png(root / cat / "ground_truth" / kind / f"{i:03d}_mask.png", m)
    return root
