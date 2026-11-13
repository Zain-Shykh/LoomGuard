from collections import Counter
from pathlib import Path

import pytest
import yaml

from loomguard.data.manifest import TileRow, build_raw_fabrid
from loomguard.runs import bundle_dir, load_config, profiles, result_path, subsample

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
    cfg = {"regime": "per_fabric", "dataset": "raw_fabrid", "method": "patchcore", "artifacts": "A", "results": "R"}
    per = profiles(rows, cfg)
    assert set(per) == {"RolloA", "RolloB"} and all(r.fabric == k for k, v in per.items() for r in v)
    uni = profiles(rows, {**cfg, "regime": "unified"})
    assert list(uni) == ["unified"] and len(uni["unified"]) == len(rows)
    assert str(bundle_dir(cfg, "RolloA")) == "A/raw_fabrid/RolloA/patchcore"
    assert str(result_path(cfg, "RolloA")) == "R/raw_fabrid/per_fabric/patchcore/RolloA.json"


def test_load_config(tmp_path):
    p = tmp_path / "c.yaml"
    base = {"dataset": "mvtec", "root": "x", "regime": "per_fabric", "method": "patchcore"}
    p.write_text(yaml.safe_dump({**base, "patchcore": {"coreset_ratio": 0.01}}))
    cfg = load_config(p)
    assert cfg["patchcore"] == {"max_tiles": 600, "coreset_ratio": 0.01, "proj_dim": 128, "batch_size": 16}
    assert cfg["seed"] == 0 and cfg["hist_bins"] == 10_000 and cfg["config"] == str(p)
    for bad, exc in (({"dataset": "mvtec"}, KeyError), ({**base, "method": "rd4ad"}, ValueError),
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
