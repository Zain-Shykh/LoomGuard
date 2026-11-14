"""Phase 4 config keys: defaults, MVTec resize policy, RAW-FABRID scale."""
from pathlib import Path

import yaml

from loomguard.runs import load_config

CONFIGS = Path(__file__).parents[1] / "configs"


def test_defaults(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"dataset": "mvtec", "root": "x", "regime": "per_fabric", "method": "patchcore",
                                 "grading": {"cutoffs": [1, 2, 3]}}))
    c = load_config(p)
    assert c["target_far"] == 0.01 and c["resize"] == 256 and c["scale_assumed"] and not c["grade_rolls"]
    assert c["grading"] == {"closing": 5, "min_area_px": 16, "cutoffs": [1, 2, 3]}  # partial block keeps the other defaults


def test_raw_fabrid_configs_state_the_scale():
    for p in sorted(CONFIGS.glob("raw_fabrid_*.yaml")):
        c = load_config(p)
        assert c["px_per_mm"] == 4.0 and c["scale_assumed"] is False and c["grade_rolls"] is True and c["resize"] is None, p
    assert load_config(CONFIGS / "mvtec_patchcore.yaml")["grade_rolls"] is False
