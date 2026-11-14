"""Study driver, new evaluate metrics, study configs and failure-case panels, on synthetic data (CPU, random backbone)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import failure_cases  # noqa: E402
import study  # noqa: E402

from loomguard.data.dataset import preprocess  # noqa: E402
from loomguard.data.manifest import select  # noqa: E402
from loomguard.data.tiling import tile  # noqa: E402
from loomguard.eval.metrics import ScoreCounts, roc_auc  # noqa: E402
from loomguard.models.backbone import Backbone  # noqa: E402
from loomguard.models.patchcore import PatchCore  # noqa: E402
from loomguard.runs import build_rows, load_config  # noqa: E402

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import summarize  # noqa: E402

ROOT = Path(__file__).parents[1]
TINY = {"method": "patchcore", "seed": 0, "backbone_weights": None, "device": "cpu", "hist_bins": 2000,
        "patchcore": {"max_tiles": 8, "coreset_ratio": 0.005, "proj_dim": 32, "batch_size": 4}}


def seeded_backbone(weights=None):
    torch.manual_seed(0)
    return Backbone(weights)


@pytest.fixture(autouse=True)
def same_random_backbone(monkeypatch):
    for mod in (study.fit, study.evaluate, failure_cases):
        monkeypatch.setattr(mod, "Backbone", seeded_backbone)


def write(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(obj))
    return str(path)


def zju_configs(tmp_path, zju_root):
    """Four tiny ZJU configs (one per regime); the common keys come from a base file through `extends`."""
    base = write(tmp_path / "cfgs" / "base.yaml", {**TINY, "dataset": "zju_leaper", "root": str(zju_root),
                                                    "artifacts": str(tmp_path / "artifacts"), "results": str(tmp_path / "results")})
    out = [write(tmp_path / "cfgs" / "pf.yaml", {"extends": "base.yaml", "regime": "per_fabric", "fabrics": ["pattern1", "pattern4"]}),
           write(tmp_path / "cfgs" / "pg.yaml", {"extends": [Path(base).name], "regime": "per_group"}),
           write(tmp_path / "cfgs" / "un.yaml", {"extends": "base.yaml", "regime": "unified"}),
           write(tmp_path / "cfgs" / "ho.yaml", {"extends": "base.yaml", "regime": "held_out", "held_out_scope": "group",
                                                  "fabrics": ["pattern2", "pattern5"]})]
    lst = tmp_path / "study.txt"
    lst.write_text("# a study\n" + "\n".join(out[:2]) + "   # trailing comment\n\n" + "\n".join(out[2:]) + "\n")
    return lst, out


def test_driver_end_to_end(tmp_path, zju_root, capsys, monkeypatch):
    lst, cfgs = zju_configs(tmp_path, zju_root)
    assert study.read_list(lst) == cfgs
    assert study.main([str(lst), "--dry-run"]) == 0
    dry = capsys.readouterr().out
    assert not (tmp_path / "results").exists() and dry.count(" results, e.g. ") == 4 and "1 results" in dry
    assert study.main([str(lst)]) == 0
    res = tmp_path / "results" / "zju_leaper"
    want = {"per_fabric": ["pattern1", "pattern4"], "per_group": ["group1", "group2"], "unified": ["unified"],
            "held_out": ["pattern2_held_out", "pattern5_held_out"]}
    for regime, names in want.items():
        assert sorted(p.stem for p in (res / regime / "patchcore").glob("*.json")) == sorted(names)
    r = json.loads((res / "per_fabric" / "patchcore" / "pattern1.json").read_text())
    assert r["config"] == cfgs[0] and r["git_commit"] and r["tag"] == "" and "by_fabric" not in r  # one fabric: no breakdown
    g = json.loads((res / "per_group" / "patchcore" / "group1.json").read_text())
    assert set(g["by_fabric"]) == {"pattern1", "pattern2"} and g["n_test_good"] + g["n_test_defect"] == 16
    h = json.loads((res / "held_out" / "patchcore" / "pattern2_held_out.json").read_text())
    assert h["fabric"] == "pattern2_held_out" and h["n_test_good"] + h["n_test_defect"] == 8 and "by_fabric" not in h
    # a second run skips everything and does not call fit
    calls = []
    monkeypatch.setattr(study.fit, "main", lambda argv: calls.append(argv))
    capsys.readouterr()
    assert study.main([str(lst)]) == 0 and calls == []
    assert capsys.readouterr().out.count("skipped") == 4
    assert study.main([str(lst), "--force"]) == 0 and len(calls) == 4


def test_driver_continues_after_a_failure(tmp_path, zju_root, capsys):
    lst, cfgs = zju_configs(tmp_path, zju_root)
    broken = write(tmp_path / "cfgs" / "broken.yaml", {**TINY, "dataset": "zju_leaper", "root": str(tmp_path / "nowhere"),
                                                        "regime": "unified", "artifacts": str(tmp_path / "a"), "results": str(tmp_path / "r")})
    lst.write_text(broken + "\n" + cfgs[2] + "\n")
    assert study.main([str(lst)]) == 1
    out = capsys.readouterr().out
    assert "FAILED" in out and broken in out and "ok       " in out
    assert (tmp_path / "results" / "zju_leaper" / "unified" / "patchcore" / "unified.json").is_file()


def test_held_out_fit_reads_no_held_out_rows(tmp_path, zju_root, monkeypatch):
    """CLAUDE.md rule 2 for the held-out regime: the held-out pattern's train and val rows never reach fit or calibration."""
    seen = []

    class Spy(study.fit.TileDataset):
        def __init__(self, rows, root):
            seen.extend(rows)
            super().__init__(rows, root)

    monkeypatch.setattr(study.fit, "TileDataset", Spy)
    _, cfgs = zju_configs(tmp_path, zju_root)
    study.fit.main(["--config", cfgs[3]])
    assert seen and {r.split for r in seen} <= {"train", "val"}
    for r in seen:  # held-out fabrics are pattern2 and pattern5, each fitted on the other pattern of its group
        assert r.fabric in {"pattern1", "pattern4"}
    assert {r.fabric for r in seen} == {"pattern1", "pattern4"}


def test_evaluate_image_and_by_fabric(tmp_path, zju_root):
    _, cfgs = zju_configs(tmp_path, zju_root)
    study.fit.main(["--config", cfgs[2]])
    res = study.evaluate.main(["--config", cfgs[2]])["unified"]
    cfg = load_config(cfgs[2])
    pc = PatchCore.load(Path(cfg["artifacts"]) / "zju_leaper" / "unified" / "patchcore" / "model.pt", seeded_backbone(), "cpu")
    per = {}  # fabric -> (image scores, image labels), recomputed with plain code
    for r in sorted({(r.fabric, r.image, r.mask) for r in select(build_rows(cfg), split="test")}):
        tiles, origins = tile(preprocess(Image.open(zju_root / r[1])))
        s = pc.score(tiles)[0].tolist()
        gt = np.asarray(Image.open(zju_root / r[2])) > 0 if r[2] else np.zeros((256, 512), bool)
        per.setdefault(r[0], ([], []))
        per[r[0]][0].append(max(s))
        per[r[0]][1].append(int(gt.any()))
    for f, (sc, lab) in per.items():
        assert res["by_fabric"][f]["image_auroc"] == pytest.approx(roc_auc(sc, lab), abs=1e-6)
    allsc, alllab = [x for v in per.values() for x in v[0]], [x for v in per.values() for x in v[1]]
    assert res["image_auroc"] == pytest.approx(roc_auc(allsc, alllab), abs=1e-6)
    assert res["n_test_good"] + res["n_test_defect"] == 4 * 8
    assert set(res["by_fabric"]) == {"pattern1", "pattern2", "pattern4", "pattern5"}


def test_dump_predictions(tmp_path, zju_root):
    base = write(tmp_path / "c.yaml", {**TINY, "dataset": "zju_leaper", "root": str(zju_root), "regime": "unified",
                                       "fabrics": ["pattern1"], "dump_predictions": True,
                                       "artifacts": str(tmp_path / "artifacts"), "results": str(tmp_path / "results")})
    study.fit.main(["--config", base])
    study.evaluate.main(["--config", base])
    d = tmp_path / "artifacts" / "predictions" / "zju_leaper" / "unified" / "patchcore" / "unified"
    rows = (d / "scores.csv").read_text().strip().split("\n")
    assert rows[0] == "parent,fabric,label,score,flagged" and len(rows) == 5 and len(list(d.glob("*.png"))) == 4
    m = np.asarray(Image.open(d / "p1_dte0.png"))
    assert m.shape == (256, 512) and set(np.unique(m)) <= {0, 255}


def test_scorecounts_merge():
    rng = np.random.default_rng(0)
    parts = [(rng.integers(0, 50, (20, 20)), rng.random((20, 20)) > 0.9) for _ in range(3)]
    merged, one = ScoreCounts(50), ScoreCounts(50)
    for idx, m in parts:
        a = ScoreCounts(50)
        a.add(idx, m, regions=True)
        merged.merge(a)
        one.add(idx, m, regions=True)
    assert merged.auroc() == one.auroc() and merged.ap() == one.ap() and merged.aupro(0.3) == one.aupro(0.3)


def study_configs():
    out = []
    for lst in sorted((ROOT / "configs" / "studies").glob("*.txt")):
        for c in study.read_list(lst):
            assert (ROOT / c).is_file(), f"{lst.name}: {c} does not exist"
            out.append(load_config(ROOT / c))
    return out


def method_dir(c):
    return c["method"] + (f"_{c['tag']}" if c["tag"] else "")


def test_every_cell_has_a_config():
    cfgs = study_configs()
    for ds, regime, names in summarize.CELLS:
        for m in summarize.METHODS:
            d = summarize.headline(ds, regime, m)
            hit = [c for c in cfgs if (c["dataset"], c["regime"], method_dir(c)) == (ds, regime, d)]
            assert hit, f"no config for {ds}/{regime}/{d}"
            if regime in ("per_fabric", "held_out") and isinstance(names, list):
                have = {f for c in hit for f in (c["fabrics"] or [n.removesuffix("_held_out") for n in names])}
                assert {n.removesuffix("_held_out") for n in names} <= have, f"{ds}/{regime}/{d}: fabrics {sorted(have)}"
            if regime == "held_out" and ds == "zju_leaper":
                assert all(c["held_out_scope"] == "group" for c in hit)
    for name, (ds, regime, spec) in summarize.ABLATIONS.items():  # every ablation variant has a config (or is the headline)
        for variants in spec.values():
            for _, d in variants:
                assert any((c["dataset"], c["regime"], method_dir(c)) == (ds, regime, d) for c in cfgs), f"{name}: {ds}/{regime}/{d}"
    mini = [c for c in cfgs if c["dataset"] == "zju_leaper" and c["regime"] == "per_fabric" and not c["tag"]]
    pats = [f for c in mini if c["method"] == "patchcore" for f in c["fabrics"]]
    assert sorted(pats) == sorted(set(pats)) == sorted(summarize.PATTERNS)  # mini + rest: 19 patterns, each fitted once


def test_headline_settings_fixed():
    """D-35: the headline settings are fixed in advance. Tagged configs change exactly one named setting."""
    budget = {"per_fabric": 5000, "per_group": 10000, "unified": 20000}
    for c in study_configs():
        assert c["seed"] == 0 and c["backbone_weights"] == "IMAGENET1K_V1" and c["target_far"] == 0.01
        tag = c["tag"]
        pc, rd = c["patchcore"], c["rd4ad"]
        if c["method"] == "patchcore":
            want = {"coreset_ratio": {"cr1": 0.01, "cr25": 0.25}.get(tag, 0.10), "max_tiles": {"n50": 50, "n200": 200}.get(tag, 600)}
            assert {k: pc[k] for k in want} == want and (pc["proj_dim"], pc["batch_size"]) == (128, 16)
        else:
            held = 10000 if c["dataset"] == "zju_leaper" else 20000
            want = 1000 if tag == "it1k" else held if c["regime"] == "held_out" else budget[c["regime"]]
            assert rd["iterations"] == want and (rd["lr"], rd["betas"], rd["batch_size"]) == (0.005, [0.5, 0.999], 16)
        assert c["min_overlap"] == (32 if tag == "ov32" else 0)
        assert (c["resize"] == 256) == (tag == "res256" or c["dataset"] == "mvtec")
        assert tag in ("", "res256", "cr1", "cr25", "n50", "n200", "it1k", "ov32", "s0", "s1", "s2")


def test_failure_cases(tmp_path, raw_root):
    cfg = write(tmp_path / "c.yaml", {**TINY, "dataset": "raw_fabrid", "root": str(raw_root), "regime": "per_fabric",
                                       "artifacts": str(tmp_path / "artifacts"), "results": str(tmp_path / "results")})
    study.fit.main(["--config", cfg])
    out = tmp_path / "fc"
    files = failure_cases.main(["--config", cfg, "--profile", "RolloA", "--n", "2", "--out", str(out)])
    names = sorted(p.name for p in files)
    assert sum("_miss" in n for n in names) == 2 and all(n.startswith("raw_fabrid_per_fabric_patchcore_RolloA_") for n in names)
    for p in files:
        assert Image.open(p).size == (4 * 448 + 12, 224)  # 512 x 256 images, panels scaled to 448 wide
    # render keeps the four panels in order: image | ground truth | heatmap | mask
    img = Image.new("L", (64, 32), 100)
    gt = np.zeros((32, 64), bool)
    gt[:, :8] = True
    panel = np.asarray(failure_cases.render(img, gt, np.zeros((32, 64), np.float32), gt, 1.0))
    assert panel.shape == (32, 4 * 64 + 12, 3) and panel[0, 4, 1] == 100 and panel[0, 64 + 4 + 2, 0] == 255
    assert panel[0, 2 * 68 + 2].tolist() == [0, 0, 255]  # zero heat is blue
