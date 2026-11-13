"""fit.py and evaluate.py end to end on tiny synthetic datasets, CPU, random backbone weights, tiny coreset."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from loomguard.data.dataset import preprocess
from loomguard.data.tiling import stitch, tile
from loomguard.eval.metrics import aupro, average_precision, roc_auc
from loomguard.models.backbone import Backbone
from loomguard.data.manifest import select
from loomguard.models.patchcore import PatchCore
from loomguard.models.rd4ad import RD4AD
from loomguard.runs import build_rows, load_config


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fit, evaluate = load_script("fit"), load_script("evaluate")


def seeded_backbone(weights=None):
    """Random-weight backbone, identical on every call: fit, evaluate and the test must see the same network."""
    torch.manual_seed(0)
    return Backbone(weights)


@pytest.fixture(autouse=True)
def same_random_backbone(monkeypatch):
    monkeypatch.setattr(fit, "Backbone", seeded_backbone)
    monkeypatch.setattr(evaluate, "Backbone", seeded_backbone)
COMMON = {"dataset", "regime", "fabric", "method", "git_commit", "seed", "config", "n_train_tiles", "n_val_tiles",
          "n_test_good", "n_test_defect", "tile_auroc", "tile_ap", "aupro_30", "pixel_auroc", "pixel_ap",
          "ms_per_tile_cpu", "peak_vram_gb", "peak_ram_gb", "fit_minutes", "bundle_mb", "hist_bins", "device"}
KEYS = {"patchcore": COMMON | {"coreset_seconds", "coreset_ratio", "bank_size"},
        "rd4ad": COMMON | {"iterations", "iters_per_second", "final_train_loss", "final_val_loss"}}
MODEL = {"patchcore": PatchCore, "rd4ad": RD4AD}


def write_cfg(tmp_path, **kw):
    cfg = {"method": "patchcore", "seed": 0, "backbone_weights": None, "device": "cpu", "hist_bins": 10000,
           "patchcore": {"max_tiles": 8, "coreset_ratio": 0.005, "proj_dim": 32, "batch_size": 4},
           "rd4ad": {"iterations": 2, "batch_size": 2, "val_tiles": 2, "val_every": 1, "log_every": 1},
           "artifacts": str(tmp_path / "artifacts"), "results": str(tmp_path / "results"), **kw}
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def check_bundle_and_result(tmp_path, cfg, profile, res):
    method = cfg["method"]
    out = Path(cfg["artifacts"]) / cfg["dataset"] / profile / method
    assert {p.name for p in out.iterdir()} == {"model.pt", "config.yaml", "metrics.json"}
    saved = torch.load(out / "model.pt", weights_only=True)
    if method == "patchcore":
        assert saved.dtype == torch.float16 and saved.shape[1] == 1536 and len(saved) == res["bank_size"]
    else:  # student state dict, floating tensors in float16
        assert all(v.dtype == torch.float16 for v in saved.values() if v.is_floating_point())
        assert res["iterations"] == 2 and res["iters_per_second"] > 0 and res["final_val_loss"] is not None
        assert yaml.safe_load((out / "config.yaml").read_text())["layers"] == ["layer1", "layer2", "layer3"]
    on_disk = json.loads((Path(cfg["results"]) / cfg["dataset"] / cfg["regime"] / method / f"{profile}.json").read_text())
    assert set(on_disk) == KEYS[method] and on_disk == json.loads(json.dumps(res))
    assert (Path(cfg["artifacts"]) / "manifests" / f"{cfg['dataset']}.csv").is_file()
    assert all(0 <= res[k] <= 1 for k in ("tile_auroc", "tile_ap", "aupro_30", "pixel_auroc", "pixel_ap"))
    return out


@pytest.mark.parametrize("method", ["patchcore", "rd4ad"])
def test_mvtec_end_to_end(mvtec_root, tmp_path, method):
    p = write_cfg(tmp_path, dataset="mvtec", root=str(mvtec_root), regime="per_fabric", fabrics=["tile"],
                  targets={"tile_auroc": 0.0}, method=method)
    fit.main(["--config", str(p)])
    res = evaluate.main(["--config", str(p)])["tile"]
    cfg = yaml.safe_load(p.read_text()) | {"dataset": "mvtec", "regime": "per_fabric"}
    out = check_bundle_and_result(tmp_path, cfg, "tile", res)
    # 18 train tiles (PatchCore caps them to 8, RD4AD samples crops from all 18), 2 val images, 3 good + 4 defective test images
    assert res["n_train_tiles"] == (8 if method == "patchcore" else 18) and res["n_val_tiles"] == 2
    assert (res["n_test_good"], res["n_test_defect"]) == (3, 4)
    # independent recomputation: resize, score, exact metrics
    pc = MODEL[method].load(out / "model.pt", seeded_backbone(), "cpu")
    scores, heats, masks, labels = [], [], [], []
    for kind, label in (("good", 0), ("crack", 1), ("hole", 1)):
        for f in sorted((mvtec_root / "tile" / "test" / kind).glob("*.png")):
            x = preprocess(Image.open(f).convert("RGB").resize((256, 256), Image.BILINEAR))
            s, h = pc.score(x[None])
            scores.append(float(s[0])); heats.append(h[0, 0].numpy()); labels.append(label)
            gt = mvtec_root / "tile" / "ground_truth" / kind / f"{f.stem}_mask.png"
            m = np.asarray(Image.open(gt).resize((256, 256), Image.NEAREST)) > 0 if label else np.zeros((256, 256), bool)
            masks.append(m)
    assert res["tile_auroc"] == pytest.approx(roc_auc(scores, labels), abs=1e-6)
    assert res["tile_ap"] == pytest.approx(average_precision(scores, labels), abs=1e-6)
    heats, masks = np.stack(heats), np.stack(masks)
    assert res["pixel_auroc"] == pytest.approx(roc_auc(heats, masks), abs=5e-3)
    assert res["aupro_30"] == pytest.approx(aupro(heats, masks), abs=5e-3)


@pytest.mark.parametrize("method", ["patchcore", "rd4ad"])
def test_raw_fabrid_unified_end_to_end(raw_root, tmp_path, method):
    p = write_cfg(tmp_path, dataset="raw_fabrid", root=str(raw_root), regime="unified", method=method)
    fit.main(["--config", str(p)])
    res = evaluate.main(["--config", str(p)])["unified"]
    cfg = yaml.safe_load(p.read_text()) | {"dataset": "raw_fabrid", "regime": "unified"}
    out = check_bundle_and_result(tmp_path, cfg, "unified", res)
    n_train = len(select(build_rows(load_config(p)), split="train"))
    assert res["n_train_tiles"] == (8 if method == "patchcore" else n_train) and n_train > 8
    assert res["n_val_tiles"] == 12 and (res["n_test_good"], res["n_test_defect"]) == (4, 6)
    # independent recomputation: tile whole parent images, score, stitch, exact metrics on full images
    pc = MODEL[method].load(out / "model.pt", seeded_backbone(), "cpu")
    scores, labels, heats, masks = [], [], [], []
    for f in sorted((raw_root / "masks").glob("*_d*.png")):
        gt = np.asarray(Image.open(f)) > 0
        tiles, origins = tile(preprocess(Image.open(raw_root / "images" / f.name)))
        s, h = pc.score(tiles)
        scores += s.tolist()
        labels += [int(gt[y:y + 256, x:x + 256].any()) for y, x in origins]
        heats.append(stitch(h, origins, gt.shape)[0].numpy())
        masks.append(gt)
    assert sorted(labels) == [0] * 4 + [1] * 6
    assert res["tile_auroc"] == pytest.approx(roc_auc(scores, labels), abs=1e-6)
    assert res["tile_ap"] == pytest.approx(average_precision(scores, labels), abs=1e-6)
    heats, masks = np.stack(heats), np.stack(masks)
    assert res["pixel_auroc"] == pytest.approx(roc_auc(heats, masks), abs=5e-3)
    assert res["pixel_ap"] == pytest.approx(average_precision(heats, masks), abs=5e-3)
    assert res["aupro_30"] == pytest.approx(aupro(heats, masks), abs=5e-3)


def test_fit_reads_only_train_rows(raw_root, tmp_path, monkeypatch):
    seen = []

    class Spy(fit.TileDataset):
        def __init__(self, rows, root):
            seen.extend(rows)
            super().__init__(rows, root)

    monkeypatch.setattr(fit, "TileDataset", Spy)
    p = write_cfg(tmp_path, dataset="raw_fabrid", root=str(raw_root), regime="per_fabric")
    fit.main(["--config", str(p)])
    assert seen and {r.split for r in seen} == {"train"}
    assert len({r.fabric for r in seen}) == 2  # per_fabric: both rolls fitted


def test_rd4ad_fit_reads_no_test_rows(raw_root, tmp_path, monkeypatch):
    """CLAUDE.md rule 2: only train rows reach the crop stream, only val rows the validation batches, no test row is read."""
    seen = {"crops": [], "val": []}

    class SpyCrops(fit.RandomCropDataset):
        def __init__(self, rows, root, length, seed):
            seen["crops"].extend(rows)
            super().__init__(rows, root, length, seed)

    class SpyTiles(fit.TileDataset):
        def __init__(self, rows, root):
            seen["val"].extend(rows)
            super().__init__(rows, root)

    monkeypatch.setattr(fit, "RandomCropDataset", SpyCrops)
    monkeypatch.setattr(fit, "TileDataset", SpyTiles)
    p = write_cfg(tmp_path, dataset="raw_fabrid", root=str(raw_root), regime="per_fabric", method="rd4ad")
    fit.main(["--config", str(p)])
    assert seen["crops"] and {r.split for r in seen["crops"]} == {"train"}
    assert seen["val"] and {r.split for r in seen["val"]} == {"val"} and len(seen["val"]) <= 2 * 2  # val_tiles 2 per profile
    assert len({r.fabric for r in seen["crops"]}) == 2  # per_fabric: both rolls fitted
