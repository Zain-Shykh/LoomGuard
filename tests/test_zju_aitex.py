import json
from collections import Counter

import numpy as np
import pytest
import torch
from PIL import Image

from loomguard.data.dataset import TileDataset, load_mask
from loomguard.data.manifest import build_aitex, build_raw_fabrid, build_zju, select, zju_group
from loomguard.pipeline import Inspector
from loomguard.runs import profiles


def test_zju_splits(zju_root):
    rows = build_zju(zju_root)
    assert {r.fabric for r in rows} == {"pattern1", "pattern2", "pattern4", "pattern5"}
    for fab in {r.fabric for r in rows}:
        parents = {s: {r.parent for r in select(rows, split=s, fabric=fab)} for s in ("train", "val", "test")}
        assert (len(parents["train"]), len(parents["val"]), len(parents["test"])) == (9, 1, 4)  # 10% of 10; 2 + 2 test
        assert not (parents["train"] & parents["val"] or parents["train"] & parents["test"] or parents["val"] & parents["test"])
        assert not any("_dtr" in p for s in parents.values() for p in s)  # defect/train is not used (Setting 1)
    assert len(select(rows, fabric="pattern1", split="test")) == 8  # 4 images x 2 tiles
    # p1_dte0 has its box at x 300..310 (tile 1 only); p1_dte1 straddles x = 256 (both tiles)
    lab = {(r.parent.split("/")[-1], r.x0): (r.label, r.defect_px) for r in rows if "p1_dte" in r.parent}
    assert lab[("p1_dte0.png", 0)] == (0, 0) and lab[("p1_dte0.png", 256)] == (1, 100)
    assert lab[("p1_dte1.png", 0)][0] == 1 and lab[("p1_dte1.png", 256)][0] == 1
    assert all(r.label == 0 for r in rows if "_ntr" in r.parent or "_nte" in r.parent)
    assert [zju_group(f) for f in ("pattern1", "pattern3", "pattern4", "pattern7", "pattern8", "pattern12", "pattern19")] == \
        ["group1", "group1", "group2", "group2", "group3", "group3", "group5"]
    assert build_zju(zju_root) == rows and build_zju(zju_root, ["pattern2"]) == [r for r in rows if r.fabric == "pattern2"]


def test_zju_problems(zju_root):
    p = zju_root / "ImageSets" / "Patterns" / "pattern1.json"
    good = p.read_text()
    sets = json.loads(good)
    mask = zju_root / "Annotations" / "masks" / "p1_dte0.png"
    saved = mask.read_bytes()
    mask.unlink()
    with pytest.raises(FileNotFoundError, match="p1_dte0 has no mask"):
        build_zju(zju_root, ["pattern1"])
    mask.write_bytes(saved)
    sets["normal"]["test"].append(sets["normal"]["train"][0])  # one image in two sets
    sets["normal"]["train"].append("ghost")
    p.write_text(json.dumps(sets))
    with pytest.raises(FileNotFoundError) as e:
        build_zju(zju_root, ["pattern1"])
    assert "listed twice" in str(e.value) and "id ghost has no image" in str(e.value)
    p.write_text(json.dumps([1, 2]))
    with pytest.raises(ValueError, match="expected"):
        build_zju(zju_root, ["pattern1"])
    with pytest.raises(FileNotFoundError, match="no pattern"):
        build_zju(zju_root / "nowhere")


def test_zju_resize_and_mismatched_mask(zju_root):
    rows = build_zju(zju_root, ["pattern1"], resize=256)
    assert all(r.resize == 256 and (r.x0, r.y0, r.w, r.h) == (0, 0, 512, 256) for r in rows)
    assert len(rows) == 14 and sum(r.label for r in rows) == 2
    Image.fromarray(np.zeros((8, 8), np.uint8)).save(zju_root / "Annotations" / "masks" / "p1_dte0.png")
    with pytest.raises(ValueError, match="does not match"):
        build_zju(zju_root, ["pattern1"])


def test_aitex_adapter(aitex_root):
    rows = build_aitex(aitex_root)
    assert {r.fabric for r in rows} == {"fabric00", "fabric01"}
    for f in ("fabric00", "fabric01"):
        par = {s: {r.parent for r in select(rows, split=s, fabric=f)} for s in ("train", "val", "test")}
        assert (len(par["train"]), len(par["val"]), len(par["test"])) == (12, 8, 3)  # 12 / 8 of 20; three defective images
    # the black last tile of normal image 0 is dropped (whatever split it went to); every other normal image keeps four tiles
    counts = Counter(r.parent for r in rows if "NODefect" in r.parent)
    assert [v for p, v in counts.items() if "0000_019_" in p] == [3, 3]          # image 0 of each fabric
    assert Counter(v for p, v in counts.items() if "0000_019_" not in p) == {4: 38}
    # two mask files of one image are united: defects at x 300 (tile 1) and x 900 (tile 3)
    d0 = {r.x0: r.label for r in rows if r.parent.endswith("0000_002_00.png")}
    assert d0 == {0: 0, 256: 1, 512: 0, 768: 1}
    r = next(r for r in rows if r.parent.endswith("0000_002_00.png") and r.x0 == 768)
    assert "|" in r.mask
    m = np.asarray(load_mask(aitex_root, r.mask)) > 0
    assert m[10:20, 300:310].all() and m[100:110, 900:910].all() and m.sum() == 200
    _, mk, lab = TileDataset([r], aitex_root)[0]
    assert lab == 1 and mk.sum() == 100
    (aitex_root / "Mask_images" / "0001_002_01_mask.png").unlink()
    with pytest.raises(FileNotFoundError, match="0001_002_01.png: no mask"):
        build_aitex(aitex_root)


def test_aitex_layout_errors(tmp_path, aitex_root):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="Defect_images"):
        build_aitex(tmp_path / "empty")
    Image.new("L", (1024, 256)).save(aitex_root / "NODefect_images" / "oops.png")
    with pytest.raises(ValueError, match="oops.png"):
        build_aitex(aitex_root)


def cfg_of(regime, **kw):
    return {"regime": regime, "dataset": "zju_leaper", "held_out_scope": "group", "fabrics": None, **kw}


def test_profiles_regimes(zju_root):
    rows = build_zju(zju_root)
    g = profiles(rows, cfg_of("per_group"))
    assert set(g) == {"group1", "group2"} and {r.fabric for r in g["group1"]} == {"pattern1", "pattern2"}
    got = profiles(rows, cfg_of("held_out", fabrics=["pattern2"]))
    assert list(got) == ["pattern2_held_out"]
    got = got["pattern2_held_out"]
    assert {r.fabric for r in select(got, split="test")} == {"pattern2"}              # tested on the held-out fabric
    assert {r.fabric for r in got if r.split != "test"} == {"pattern1"}               # trained on the rest of its group
    assert not [r for r in got if r.fabric == "pattern2" and r.split != "test"]      # its own train/val rows never appear
    wide = profiles(rows, cfg_of("held_out", held_out_scope="all", fabrics=["pattern2"]))["pattern2_held_out"]
    assert {r.fabric for r in wide if r.split != "test"} == {"pattern1", "pattern4", "pattern5"}
    assert set(profiles(rows, cfg_of("held_out"))) == {f"pattern{n}_held_out" for n in (1, 2, 4, 5)}
    with pytest.raises(ValueError, match="zju_leaper only"):
        profiles(rows, cfg_of("per_group", dataset="raw_fabrid"))


def test_min_overlap(raw_root):
    base, over = build_raw_fabrid(raw_root), build_raw_fabrid(raw_root, min_overlap=32)
    assert set(Counter(r.parent for r in base).values()) == {2} and set(Counter(r.parent for r in over).values()) == {3}
    assert {(r.parent, r.split) for r in base} == {(r.parent, r.split) for r in over}  # the split does not depend on the overlap
    assert sorted({r.x0 for r in over}) == [0, 128, 256]
    # RolloA's first defect box is at x 250..260: with overlap it lies in the tiles at 0 and 128 and 256
    assert sum(r.label for r in over if r.parent.endswith("RolloA_d000.png")) == 3

    class Fake:
        def score(self, b):
            return torch.zeros(len(b)), torch.zeros(len(b), 1, 256, 256)

    img = Image.new("L", (512, 256))
    assert len(Inspector(Fake(), 1.0).score_image(img).origins) == 2
    s = Inspector(Fake(), 1.0, min_overlap=32).score_image(img)
    assert [o[1] for o in s.origins] == [0, 128, 256] and s.heat.shape == (256, 512)
