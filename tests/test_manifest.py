import csv
from collections import defaultdict
from functools import partial

import pytest

from loomguard.data.manifest import (assign_val, build_mvtec, build_raw_fabrid, compare_with_patch_metadata,
                                     read_manifest, select, write_manifest)

MINI = ("carpet", "tile")
BUILDERS = {"raw": ("raw_root", build_raw_fabrid), "mvtec": ("mvtec_root", partial(build_mvtec, categories=MINI))}


@pytest.fixture(params=BUILDERS)
def built(request):
    fixture, builder = BUILDERS[request.param]
    root = request.getfixturevalue(fixture)
    return lambda seed=0: builder(root, seed=seed)


def test_assign_val_rule():
    kw = dict(val_frac=0.15, min_val_tiles=200, max_frac=0.40, seed=0, stratum="r")
    n = lambda count, tpp: len(assign_val([f"p{i}" for i in range(count)], tpp, **kw))
    assert n(13, 28) == 5      # round(1.95) = 2, raised to 8, capped at floor(5.2) = 5
    assert n(20, 16) == 8      # AITEX: 3, raised to 13, capped at 8
    assert n(167, 28) == 25    # 15% already above the minimum
    assert n(66, 28) == 10 and n(55, 28) == 8 and n(18, 28) == 7 and n(96, 28) == 14 and n(92, 28) == 14
    assert n(1, 28) == 0 and n(2, 28) == 0 and n(3, 28) == 1   # train is never empty (cap 40% of 2 is 0)
    assert len(assign_val([f"p{i}" for i in range(280)], 1, val_frac=0.10, min_val_tiles=0,
                                            max_frac=0.40, seed=0, stratum="c")) == 28


def test_raw_fabrid_totals_from_the_documented_roll_table():
    rolls = {"1": 66, "2": 55, "3": 13, "4": 92, "5": 18, "6": 96, "7": 167}
    val = sum(len(assign_val([f"{r}/{i}" for i in range(n)], 28, val_frac=0.15, min_val_tiles=200, max_frac=0.4,
                             seed=0, stratum=r)) for r, n in rolls.items())
    assert (val, val * 28, (507 - val) * 28) == (83, 2324, 11872)


def test_no_parent_in_two_splits(built):
    for seed in range(5):
        rows = built(seed)
        splits = defaultdict(set)
        for r in rows:
            splits[r.parent].add(r.split)
        assert all(len(s) == 1 for s in splits.values())
        pools = {s: {r.parent for r in rows if r.split == s} for s in ("train", "val", "test")}
        assert not (pools["train"] & pools["val"]) and not (pools["train"] & pools["test"]) \
            and not (pools["val"] & pools["test"])
        assert all(pools.values())


def test_train_val_are_normal_only(built):
    rows = built()
    assert all(r.label == 0 and r.mask == "" for r in rows if r.split != "test")
    assert any(r.label == 1 for r in rows if r.split == "test")


def test_deterministic_and_seed_dependent(built):
    a, b = built(0), built(0)
    assert a == b
    assert any(built(s) != a for s in range(1, 6))  # some other seed picks other validation parents
    assert [(r.fabric, r.split, r.parent, r.y0, r.x0) for r in a] == sorted(
        (r.fabric, r.split, r.parent, r.y0, r.x0) for r in a)


def test_csv_roundtrip(built, tmp_path):
    rows = built()
    write_manifest(rows, tmp_path / "m.csv")
    assert read_manifest(tmp_path / "m.csv") == rows
    assert select(rows, split="val") == [r for r in rows if r.split == "val"]
    assert {r.fabric for r in select(rows, fabric=rows[0].fabric)} == {rows[0].fabric}


def test_raw_fabrid_labels_from_mask(raw_root):
    rows = build_raw_fabrid(raw_root)
    assert {r.fabric for r in rows} == {"RolloA", "RolloB"}
    assert len(rows) == (10 + 3 + 6 + 2) * 2
    by = {(r.parent, r.x0): r for r in rows}
    # box (10,15,250,260): 5 rows x 6 cols in tile 0, 5 rows x 4 cols in tile 1
    assert (by["images/RolloA_d000.png", 0].defect_px, by["images/RolloA_d000.png", 256].defect_px) == (30, 20)
    assert by["images/RolloA_d000.png", 0].label == by["images/RolloA_d000.png", 256].label == 1
    # box in tile 1 only
    assert (by["images/RolloA_d001.png", 0].label, by["images/RolloA_d001.png", 256].label) == (0, 1)
    assert by["images/RolloA_d001.png", 256].defect_px == 12
    assert by["images/RolloA_d002.png", 0].defect_px == 1
    assert all(r.w == r.h == 256 and r.resize == 0 and r.y0 == 0 for r in rows)
    assert {r.split for r in rows if "_d" in r.parent} == {"test"}
    assert {r.split for r in rows if "_n" in r.parent} <= {"train", "val"}
    # D-30 per roll: 2 tiles per parent, so the cap (40%) decides: 4 of 10 and 2 of 6 parents
    for roll, n_val in (("RolloA", 4), ("RolloB", 2)):
        assert len({r.parent for r in rows if r.fabric == roll and r.split == "val"}) == n_val


def test_raw_fabrid_csv_is_truth(raw_root):
    rows = build_raw_fabrid(raw_root)  # RolloA_n000 has a stray mask but the CSV says defect-free
    assert all(r.mask == "" and r.label == 0 for r in rows if r.parent.endswith("RolloA_n000.png"))
    (raw_root / "masks" / "RolloB_d001.png").unlink()
    with pytest.raises(FileNotFoundError):
        build_raw_fabrid(raw_root)


def test_raw_fabrid_bad_header(raw_root):
    (raw_root / "RAW_FABRID_HighRes_Metadata.csv").write_text("name,roll\nx,y\n")
    with pytest.raises(KeyError):
        build_raw_fabrid(raw_root)


def test_mvtec_builder(mvtec_root):
    rows = build_mvtec(mvtec_root, MINI)
    assert {r.fabric for r in rows} == {"carpet", "tile"}
    test = [r for r in rows if r.split == "test"]
    assert len(test) == 2 * (3 + 4)
    assert all(r.parent.split("/")[1] == "test" for r in test) and not any(
        r.split == "test" for r in rows if "/train/" in r.parent)
    assert sum(r.label for r in test) == 8 and all(r.label == (r.parent.split("/")[2] != "good") for r in test)
    assert all(r.mask.endswith("_mask.png") for r in test if r.label) and all(r.mask == "" for r in rows if not r.label)
    assert all((r.x0, r.y0, r.resize) == (0, 0, 256) for r in rows)
    assert {(r.fabric, r.w) for r in rows} == {("carpet", 64), ("tile", 48)}  # native size read from each file
    assert sum(r.split == "val" for r in rows if r.fabric == "carpet") == 2  # 10% of 20
    (mvtec_root / "tile" / "ground_truth" / "hole" / "000_mask.png").unlink()
    with pytest.raises(FileNotFoundError):
        build_mvtec(mvtec_root, MINI)


def test_compare_with_patch_metadata(raw_root, tmp_path):
    rows = build_raw_fabrid(raw_root)
    good = raw_root / "RAW_FABRID_Patches_Metadata.csv"
    assert compare_with_patch_metadata(rows, good) == []

    def altered(change):
        with open(good, newline="") as f:
            official = list(csv.DictReader(f))
        change(official)
        out = tmp_path / "p.csv"
        with open(out, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(official[0]))
            wr.writeheader()
            wr.writerows(official)
        return compare_with_patch_metadata(rows, out)

    def area(o):
        next(p for p in o if p["defect_area"] == "30")["defect_area"] = "31"
    assert any("defect areas differ" in p for p in altered(area))
    assert any("tiles" in p for p in altered(lambda o: o.pop(0)))
    assert any("unexpected patch types" in p for p in altered(lambda o: o[0].update(type="val_good")))
    assert any("test_defect" in p for p in altered(lambda o: next(
        p for p in o if p["type"] == "test_defect").update(type="test_good")))
