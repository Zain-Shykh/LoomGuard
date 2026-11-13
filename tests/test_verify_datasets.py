import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

spec = importlib.util.spec_from_file_location("verify_datasets", Path(__file__).parents[1] / "scripts" / "verify_datasets.py")
vd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vd)


def _png(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


@pytest.fixture
def zju_root(tmp_path):
    root = tmp_path / "zju"
    for n in range(1, 20):
        ids = {"normal": {"train": [f"n{n}a", f"n{n}b"], "test": [f"n{n}c"]},
               "defect": {"train": [f"d{n}a"], "test": [f"d{n}b"]}}
        (root / "ImageSets" / "Patterns").mkdir(parents=True, exist_ok=True)
        (root / "ImageSets" / "Patterns" / f"pattern{n}.json").write_text(json.dumps(ids))
        for k, d in ids.items():
            for s, lst in d.items():
                for i in lst:
                    _png(root / "Images" / f"{i}.jpg", np.zeros((64, 64), np.uint8))
        if n != 3:  # pattern 3 has no mask for its defective test image
            _png(root / "Annotations" / "masks" / f"d{n}b.png", np.zeros((64, 64), np.uint8))
    return root


@pytest.fixture
def aitex_root(tmp_path):
    root = tmp_path / "aitex"
    a = np.full((256, 512), 200, np.uint8)
    border = a.copy()
    border[:, 256:] = 0  # second tile is all black
    _png(root / "Defect_images" / "0001_002_00.png", a)
    _png(root / "Defect_images" / "0002_003_00.png", border)
    _png(root / "NODefect_images" / "0001_000_00.png", a)
    _png(root / "Mask_images" / "0001_002_00_mask.png", a)
    _png(root / "Mask_images" / "0002_003_00_mask.png", a)
    _png(root / "Mask_images" / "0002_003_00_mask2.png", a)
    _png(root / "Mask_images" / "9999_999_00_mask.png", a)
    return root


def test_raw_fabrid_report(raw_root, tmp_path):
    text = "\n".join(vd.check_raw_fabrid(raw_root, tmp_path))
    assert "images on disk 21, in CSV 21" in text and "masks on disk 6" in text and "CSV defective 5" in text
    assert "masks of images the CSV calls defect-free: ['RolloA_n000.png']" in text
    assert "defective images without mask: []" in text
    assert "512x256: 21" in text
    assert "test good" in text and "MISMATCH" in text  # the mini dataset is not the official one
    assert "patch metadata comparison: 0 problems" in text
    assert (tmp_path / "raw_fabrid.csv").is_file()


def test_mvtec_report(mvtec_root, tmp_path):
    text = "\n".join(vd.check_mvtec(mvtec_root, tmp_path))
    assert "carpet: train/good 20, test/good 3, defective 4 in ['crack', 'hole'], unpaired masks [], sizes 64x64: 20" in text
    assert "wood: train/good 20" in text and "MVTec manifest: {'test': 35, 'train': 90, 'val': 10}" in text
    assert (tmp_path / "mvtec.csv").is_file()
    (mvtec_root / "tile" / "ground_truth" / "hole" / "001_mask.png").unlink()
    text = "\n".join(vd.check_mvtec(mvtec_root))
    assert "unpaired masks ['001.png']" in text and "MVTec manifest not built" in text


def test_zju_report(zju_root):
    text = "\n".join(vd.check_zju(zju_root))
    assert "pattern1: {'normal/train': 2, 'normal/test': 1, 'defect/train': 1, 'defect/test': 1}" in text
    assert "defective test ids without a mask: 1" in text
    assert "listed ids without an image file: 0" in text and "64x64" in text and ".jpg" in text


def test_aitex_report(aitex_root):
    text = "\n".join(vd.check_aitex(aitex_root))
    assert "defect images 2, defect-free 1, mask files 4" in text
    assert "defect images with 0 masks: []" in text
    assert "'0002_003_00': ['0002_003_00_mask.png', '0002_003_00_mask2.png']" in text
    assert "['9999_999_00_mask.png']" in text
    assert "more than half near-black: 1 of 6" in text


def test_main_isolates_failures(raw_root, mvtec_root, capsys, tmp_path):
    code = vd.main(["--raw-fabrid", str(raw_root), "--zju", str(tmp_path / "nowhere"), "--mvtec", str(mvtec_root),
                    "--manifest-dir", str(tmp_path / "m")])
    out = capsys.readouterr().out
    assert code == 1
    assert "== ZJU-Leaper" in out and "ERROR" in out
    assert "patch metadata comparison" in out and "MVTec manifest:" in out  # the other datasets still ran
    assert (tmp_path / "m" / "raw_fabrid.csv").is_file() and (tmp_path / "m" / "mvtec.csv").is_file()
    assert vd.main([]) == 0
