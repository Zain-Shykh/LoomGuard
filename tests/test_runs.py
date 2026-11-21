from collections import Counter
from pathlib import Path

import pytest
import yaml

from loomguard.data.manifest import TileRow, build_raw_fabrid
from loomguard.runs import RD4AD_DEFAULTS, bundle_dir, load_config, profiles, result_path, subsample

ROOT = Path(__file__).parents[1]


def rows_of(parents, per=8):
    return [TileRow("raw_fabrid", "R", "train", f"p{p}", f"p{p}", "", 256 * t, 0, 256, 256, 0, 0, 0)
            for p in range(parents) for t in range(per)]


def test_subsample():
    rows = rows_of(10)  # 80 tiles
    assert subsample(rows, 100, 0) is rows
    got = subsample(rows, 25, 0)
    assert len(got) == 25 and len(set(got)) == 25
    counts = Counter(r.parent for r in got)
    assert set(counts.values()) <= {2, 3} and len(counts) == 10  # stratified: every parent contributes 2 or 3
    assert got == subsample(rows, 25, 0) and got != subsample(rows, 25, 1)  # deterministic, seed dependent
    assert [rows.index(r) for r in got] == sorted(rows.index(r) for r in got)  # manifest order kept
    assert len({r.parent for r in subsample(rows_of(30, per=2), 30, 0)}) == 30  # fewer tiles than parents: one each


def test_profiles_and_paths(raw_root):
    rows = build_raw_fabrid(raw_root)
    cfg = {"regime": "per_fabric", "dataset": "raw_fabrid", "method": "patchcore", "artifacts": "A", "results": "R", "tag": ""}
    per = profiles(rows, cfg)
    assert set(per) == {"RolloA", "RolloB"} and all(r.fabric == k for k, v in per.items() for r in v)
    uni = profiles(rows, {**cfg, "regime": "unified"})
    assert list(uni) == ["unified"] and len(uni["unified"]) == len(rows)
    assert bundle_dir(cfg, "RolloA").as_posix() == "A/raw_fabrid/RolloA/patchcore"
    assert result_path(cfg, "RolloA").as_posix() == "R/raw_fabrid/per_fabric/patchcore/RolloA.json"
    tagged = {**cfg, "method": "rd4ad", "tag": "s1"}  # runs of one method (seeds) get separate directories
    assert bundle_dir(tagged, "RolloA").as_posix() == "A/raw_fabrid/RolloA/rd4ad_s1"
    assert result_path(tagged, "RolloA").as_posix() == "R/raw_fabrid/per_fabric/rd4ad_s1/RolloA.json"


def test_load_config(tmp_path):
    p = tmp_path / "c.yaml"
    base = {"dataset": "mvtec", "root": "x", "regime": "per_fabric", "method": "patchcore"}
    p.write_text(yaml.safe_dump({**base, "patchcore": {"coreset_ratio": 0.01}}))
    cfg = load_config(p)
    assert cfg["patchcore"] == {"max_tiles": 600, "coreset_ratio": 0.01, "proj_dim": 128, "batch_size": 16}
    assert cfg["seed"] == 0 and cfg["hist_bins"] == 10_000 and cfg["config"] == str(p) and cfg["tag"] == ""
    assert cfg["rd4ad"]["iterations"] == 5000 and cfg["rd4ad"]["seed"] == 0  # rd4ad seed defaults to the config seed
    p.write_text(yaml.safe_dump({**base, "method": "rd4ad", "seed": 3, "rd4ad": {"iterations": 7}}))
    assert load_config(p)["rd4ad"] == {**RD4AD_DEFAULTS, "iterations": 7, "seed": 3}
    for bad, exc in (({"dataset": "mvtec"}, KeyError), ({**base, "method": "padim"}, ValueError),
                     ({**base, "regime": "weird"}, ValueError)):
        p.write_text(yaml.safe_dump(bad))
        with pytest.raises(exc):
            load_config(p)


def test_shipped_configs_load():
    for name in ("mvtec_patchcore", "raw_fabrid_patchcore"):
        cfg = load_config(ROOT / "configs" / f"{name}.yaml")
        assert cfg["method"] == "patchcore" and cfg["patchcore"]["max_tiles"] == 600
    mv = load_config(ROOT / "configs" / "mvtec_patchcore.yaml")
    assert mv["targets"] == {"tile_auroc": 97.9, "pixel_auroc": 96.6, "aupro_30": 90.7}
    assert mv["fabrics"] == ["carpet", "grid", "leather", "tile", "wood"]
    rd = load_config(ROOT / "configs" / "mvtec_rd4ad.yaml")  # D-24 and D-33
    assert rd["method"] == "rd4ad" and rd["regime"] == "per_fabric" and rd["fabrics"] == mv["fabrics"]
    assert rd["rd4ad"]["iterations"] == 5000 and rd["targets"] == {"tile_auroc": 98.5, "pixel_auroc": 96.7, "aupro_30": 92.0}
    for k in range(3):  # three seeds of the unified RAW-FABRID run: same split, different training seed and output tag
        c = load_config(ROOT / "configs" / f"raw_fabrid_rd4ad_s{k}.yaml")
        assert c["regime"] == "unified" and c["rd4ad"]["iterations"] == 20000 and c["seed"] == 0
        assert (c["rd4ad"]["seed"], c["tag"]) == (k, f"s{k}")
