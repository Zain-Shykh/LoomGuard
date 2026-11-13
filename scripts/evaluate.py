"""Evaluate fitted PatchCore bundles on the test split and write results/<dataset>/<regime>/<method>/<profile>.json.

    python scripts/evaluate.py --config configs/mvtec_patchcore.yaml

Tiles are scored, grouped by parent image and stitched back to the full image, so pixel metrics (and AUPRO regions)
are computed on whole images. The test tiles are scored twice: pass 1 finds the heatmap range, tile scores and timing;
pass 2 fills the score histograms. Metrics are threshold-free; nothing here picks a threshold or a setting.
"""
import argparse
import json
import math
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from loomguard.data.dataset import TileDataset
from loomguard.data.manifest import select
from loomguard.data.tiling import stitch
from loomguard.eval.metrics import ScoreCounts, average_precision, quantize, roc_auc
from loomguard.models.backbone import Backbone
from loomguard.models.patchcore import PatchCore
from loomguard.runs import build_rows, bundle_dir, git_commit, load_config, pick_device, profiles, result_path

KEYS = ("tile_auroc", "tile_ap", "aupro_30", "pixel_auroc", "pixel_ap")


def _tiles(pc, rows, cfg, timing=None):
    """Yield (score, heatmap 1x256x256, mask 1x256x256) per row, in row order."""
    loader = DataLoader(TileDataset(rows, cfg["root"]), batch_size=cfg["patchcore"]["batch_size"], shuffle=False,
                        num_workers=cfg["num_workers"])
    for x, m, _ in loader:
        t0 = time.perf_counter()
        score, heat = pc.score(x)
        if pc.device.type == "cuda":
            torch.cuda.synchronize()
        if timing is not None:
            timing[0] += time.perf_counter() - t0
        yield from zip(score.cpu(), heat.cpu(), m)


def _parents(pc, rows, cfg, timing=None):
    """Yield (tile scores, stitched heatmap H x W, stitched mask H x W bool) per parent image."""
    stream, i = _tiles(pc, rows, cfg, timing), 0
    while i < len(rows):
        j = i
        while j < len(rows) and rows[j].parent == rows[i].parent:
            j += 1
        group = rows[i:j]
        got = [next(stream) for _ in group]
        if group[0].resize:  # MVTec: the whole image is one resized tile
            origins, size = [(0, 0)], (group[0].resize, group[0].resize)
        else:
            origins = [(r.y0, r.x0) for r in group]
            size = (max(r.y0 + r.h for r in group), max(r.x0 + r.w for r in group))
        heat = stitch(torch.stack([g[1] for g in got]), origins, size)[0]
        mask = stitch(torch.stack([g[2] for g in got]), origins, size)[0] > 0.5
        yield [float(g[0]) for g in got], heat, mask
        i = j


def evaluate_profile(cfg, name, prows, device) -> dict:
    out = bundle_dir(cfg, name)
    pc = PatchCore.load(out / "model.pt", Backbone(cfg["backbone_weights"]), device)
    test = select(prows, split="test")
    timing, scores, labels, lo, hi = [0.0], [], [], np.inf, -np.inf
    for s, heat, _ in _parents(pc, test, cfg, timing):  # pass 1
        scores += s
        lo, hi = min(lo, float(heat.min())), max(hi, float(heat.max()))
    labels = [r.label for r in test]
    if hi <= lo:
        hi = lo + 1.0
    counts = ScoreCounts(cfg["hist_bins"])
    for _, heat, mask in _parents(pc, test, cfg):  # pass 2
        counts.add(quantize(heat, lo, hi, cfg["hist_bins"]), mask.numpy(), regions=True)
    fit = json.loads((out / "metrics.json").read_text())
    res = {"dataset": cfg["dataset"], "regime": cfg["regime"], "fabric": name, "method": cfg["method"],
           "git_commit": git_commit(), "seed": cfg["seed"], "config": cfg["config"],
           "n_train_tiles": fit["n_train_tiles"], "n_val_tiles": len(select(prows, split="val")),
           "n_test_good": labels.count(0), "n_test_defect": labels.count(1),
           "tile_auroc": roc_auc(scores, labels), "tile_ap": average_precision(scores, labels),
           "aupro_30": counts.aupro(0.3), "pixel_auroc": counts.auroc(), "pixel_ap": counts.ap(),
           f"ms_per_tile_{device.type}": 1000 * timing[0] / len(test), "peak_vram_gb": fit["peak_vram_gb"],
           "peak_ram_gb": fit["peak_ram_gb"], "fit_minutes": fit["fit_minutes"], "coreset_seconds": fit["coreset_seconds"],
           "bundle_mb": fit["bundle_mb"], "coreset_ratio": cfg["patchcore"]["coreset_ratio"],
           "bank_size": fit["bank_size"], "hist_bins": cfg["hist_bins"], "device": device.type}
    res = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in res.items()}
    p = result_path(cfg, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2))
    (out / "metrics.json").write_text(json.dumps({**fit, **{k: v for k, v in res.items() if k in KEYS}}, indent=2))
    return res


def main(argv: list[str] | None = None) -> dict[str, dict]:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args(argv).config)
    device, results = pick_device(cfg), {}
    for name, prows in profiles(build_rows(cfg), cfg).items():
        results[name] = evaluate_profile(cfg, name, prows, device)
        print(name, " ".join(f"{k}={results[name][k]:.4f}" for k in KEYS if results[name][k] is not None))
    if len(results) > 1:
        mean = {k: float(np.mean([r[k] for r in results.values()])) for k in KEYS}
        print("mean over", len(results), "profiles:", " ".join(f"{k}={v:.4f}" for k, v in mean.items()))
        for k, target in (cfg["targets"] or {}).items():  # reported only, never used for a decision
            print(f"  {k}: {100 * mean[k]:.2f} vs target {target}: {'PASS' if 100 * mean[k] >= target else 'FAIL'}")
    return results


if __name__ == "__main__":
    main()
