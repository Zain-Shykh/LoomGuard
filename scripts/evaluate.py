"""Evaluate fitted PatchCore or RD4AD bundles on the test split and write results/<dataset>/<regime>/<method>/<profile>.json.

    python scripts/evaluate.py --config configs/mvtec_patchcore.yaml

Every test parent image goes through the Inspector (tile, score, stitch, threshold, clean, measure), so pixel metrics
(and AUPRO regions) are computed on whole images and the operating point is the one the app uses. The images are scored
twice: pass 1 finds the heatmap range and tile scores and computes the operating-point and grading metrics at the
calibrated tau; pass 2 fills the score histograms. Nothing here picks a threshold or a setting.
"""
import argparse
import itertools
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from loomguard.data.manifest import select
from loomguard.eval.metrics import ScoreCounts, average_precision, quantize, roc_auc
from loomguard.inspection.grading import grade_roll
from loomguard.models.backbone import Backbone
from loomguard.pipeline import Inspector
from loomguard.runs import build_rows, bundle_dir, git_commit, load_config, pick_device, profiles, result_path

KEYS = ("tile_auroc", "tile_ap", "aupro_30", "pixel_auroc", "pixel_ap")
METHOD_KEYS = {"patchcore": ("coreset_ratio", "bank_size", "coreset_seconds"),
               "rd4ad": ("iterations", "iters_per_second", "final_train_loss", "final_val_loss")}  # from fit's metrics.json


def _gt_mask(r, size, root) -> np.ndarray:
    if not r.mask:
        return np.zeros(size, bool)
    with Image.open(root / r.mask) as im:
        im = im.convert("L")
        if r.resize:
            im = im.resize((r.resize, r.resize), Image.NEAREST)
        gt = np.asarray(im) > 0
    assert gt.shape == size, f"{r.mask}: mask {gt.shape} does not match image {size}"
    return gt


def _images(inspector, rows, cfg, timing=None):
    """Yield (rows of one parent, Scored, ground-truth mask H x W bool) per parent image, in row order."""
    root = Path(cfg["root"])
    for _, g in itertools.groupby(rows, key=lambda r: r.parent):
        g = list(g)
        t0 = time.perf_counter()
        with Image.open(root / g[0].image) as im:
            scored = inspector.score_image(im)
        if timing is not None:
            timing[0] += time.perf_counter() - t0
            timing[1] += len(scored.origins)
        yield g, scored, _gt_mask(g[0], scored.size, root)


def _operating(inspector, g, scored, gt, acc, grade_rolls) -> None:
    """Operating-point counts of one parent at the calibrated tau: tile flags, Dice and IoU pixels, roll points."""
    dec = inspector.decide(scored)
    where = {o: i for i, o in enumerate(scored.origins)}
    region = np.zeros(gt.shape, bool)  # pixels of the defective tiles
    for r in g:
        k = "rec" if r.label else "far"
        acc[k][0] += bool(dec.flagged_tiles[where[(r.y0, r.x0)]])
        acc[k][1] += 1
        if r.label:
            if r.resize:  # MVTec: the one tile is the whole resized image
                region[:] = True
            else:
                region[r.y0:r.y0 + r.h, r.x0:r.x0 + r.w] = True
    p, t = dec.mask & region, gt & region
    acc["inter"] += int((p & t).sum())
    acc["pred"] += int(p.sum())
    acc["gt"] += int(t.sum())
    if grade_rolls:
        _, gt_defects = inspector.measure(gt)
        roll = acc["rolls"][g[0].fabric]
        for k, v in (("frames", 1), ("area_px", gt.size), ("defects_pred", len(dec.defects)), ("defects_gt", len(gt_defects)),
                     ("points_pred", dec.points), ("points_gt", sum(d.points for d in gt_defects))):
            roll[k] += v


def _rolls(inspector, rolls) -> dict:
    out = {}
    for fabric, v in sorted(rolls.items()):
        area = v["area_px"] / inspector.px_per_mm ** 2
        pred, gt = (grade_roll(v[f"points_{k}"], area, inspector.grading["cutoffs"]) for k in ("pred", "gt"))
        out[fabric] = {"frames": v["frames"], "defects_pred": v["defects_pred"], "defects_gt": v["defects_gt"],
                       "points_pred": v["points_pred"], "points_gt": v["points_gt"], "points_per_100_pred": pred.points_per_100,
                       "points_per_100_gt": gt.points_per_100, "grade_pred": pred.grade, "grade_gt": gt.grade}
    return out


def _ratio(a, b):
    return a / b if b else None


def evaluate_profile(cfg, name, prows, device) -> dict:
    out = bundle_dir(cfg, name)
    if not (out / "calibration.json").is_file():
        raise SystemExit(f"{out}/calibration.json is missing: this bundle predates Phase 4, refit it with scripts/fit.py")
    inspector = Inspector.load(out, device, Backbone(cfg["backbone_weights"]))
    test = select(prows, split="test")
    timing, scores, labels, lo, hi = [0.0, 0], [], [], np.inf, -np.inf
    acc = {"far": [0, 0], "rec": [0, 0], "inter": 0, "pred": 0, "gt": 0, "rolls": defaultdict(lambda: defaultdict(int))}
    for g, s, gt in _images(inspector, test, cfg, timing):  # pass 1
        where = {o: i for i, o in enumerate(s.origins)}
        scores += [float(s.scores[where[(r.y0, r.x0)]]) for r in g]
        labels += [r.label for r in g]
        lo, hi = min(lo, float(s.heat.min())), max(hi, float(s.heat.max()))
        _operating(inspector, g, s, gt, acc, cfg["grade_rolls"])
    if hi <= lo:
        hi = lo + 1.0
    counts = ScoreCounts(cfg["hist_bins"])
    for _, s, gt in _images(inspector, test, cfg):  # pass 2
        counts.add(quantize(s.heat, lo, hi, cfg["hist_bins"]), gt, regions=True)
    fit = json.loads((out / "metrics.json").read_text())
    res = {"dataset": cfg["dataset"], "regime": cfg["regime"], "fabric": name, "method": cfg["method"],
           "git_commit": git_commit(), "seed": cfg["seed"], "config": cfg["config"],
           "n_train_tiles": fit["n_train_tiles"], "n_val_tiles": len(select(prows, split="val")),
           "n_test_good": labels.count(0), "n_test_defect": labels.count(1),
           "tile_auroc": roc_auc(scores, labels), "tile_ap": average_precision(scores, labels),
           "aupro_30": counts.aupro(0.3), "pixel_auroc": counts.auroc(), "pixel_ap": counts.ap(),
           "tau": inspector.tau, "false_alarms_per_1000": _ratio(1000 * acc["far"][0], acc["far"][1]),
           "tile_recall": _ratio(acc["rec"][0], acc["rec"][1]), "dice": _ratio(2 * acc["inter"], acc["pred"] + acc["gt"]),
           "iou": _ratio(acc["inter"], acc["pred"] + acc["gt"] - acc["inter"]),
           f"ms_per_tile_{device.type}": 1000 * timing[0] / timing[1], "peak_vram_gb": fit["peak_vram_gb"],
           "peak_ram_gb": fit["peak_ram_gb"], "fit_minutes": fit["fit_minutes"], "bundle_mb": fit["bundle_mb"],
           **{k: fit.get(k) for k in METHOD_KEYS[cfg["method"]]}, "hist_bins": cfg["hist_bins"], "device": device.type}
    if cfg["grade_rolls"]:
        rolls = _rolls(inspector, acc["rolls"])
        res |= {"rolls": rolls, "grade_agreement": float(np.mean([v["grade_pred"] == v["grade_gt"] for v in rolls.values()])),
                "points_error_mean": float(np.mean([abs(v["points_per_100_pred"] - v["points_per_100_gt"]) for v in rolls.values()])),
                "defect_count_error_mean": float(np.mean([abs(v["defects_pred"] - v["defects_gt"]) for v in rolls.values()]))}
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
        mean = {k: float(np.mean([r[k] for r in results.values() if r[k] is not None] or [np.nan])) for k in KEYS}
        print("mean over", len(results), "profiles:", " ".join(f"{k}={v:.4f}" for k, v in mean.items()))
        for k, target in (cfg["targets"] or {}).items():  # reported only, never used for a decision
            print(f"  {k}: {100 * mean[k]:.2f} vs target {target}: {'PASS' if 100 * mean[k] >= target else 'FAIL'}")
    return results


if __name__ == "__main__":
    main()
