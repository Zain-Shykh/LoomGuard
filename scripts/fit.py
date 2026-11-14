"""Fit PatchCore or RD4AD (config key `method`): one bundle per fabric profile, from the train split only
(RD4AD also logs the loss on normal validation tiles, never on test data).

    python scripts/fit.py --config configs/mvtec_patchcore.yaml
"""
import argparse
import json
import resource
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from loomguard.data.dataset import RandomCropDataset, TileDataset
from loomguard.data.manifest import select, write_manifest
from loomguard.eval.calibrate import calibrate, val_peaks
from loomguard.models.backbone import Backbone
from loomguard.models.patchcore import PatchCore
from loomguard.models.rd4ad import RD4AD
from loomguard.runs import build_rows, bundle_dir, git_commit, load_config, pick_device, profiles, subsample


def fit_patchcore(cfg, prows, device):
    pc_cfg = cfg["patchcore"]
    train = subsample(select(prows, split="train"), pc_cfg["max_tiles"], cfg["seed"])  # train rows only (rule 2)
    loader = DataLoader(TileDataset(train, cfg["root"]), batch_size=pc_cfg["batch_size"], shuffle=False,
                        num_workers=cfg["num_workers"])
    pc = PatchCore(Backbone(cfg["backbone_weights"]), pc_cfg["coreset_ratio"], pc_cfg["proj_dim"], cfg["seed"], device)
    stats = pc.fit(x for x, _, _ in loader)
    return pc, {"n_train_tiles": stats["n_tiles"], "bank_size": stats["bank_size"], "coreset_seconds": stats["coreset_seconds"],
                "coreset_ratio": pc_cfg["coreset_ratio"]}, f"{stats['n_patches']} patches -> bank {stats['bank_size']}, " \
                                                           f"coreset {stats['coreset_seconds']:.1f}s"


def fit_rd4ad(cfg, prows, device):
    rd_cfg = cfg["rd4ad"]
    train = select(prows, split="train")  # train rows only (rule 2); no test row is read, no checkpoint is chosen
    val = subsample(select(prows, split="val"), rd_cfg["val_tiles"], cfg["seed"])  # normal tiles, divergence check only
    val_batches = [x for x, _, _ in DataLoader(TileDataset(val, cfg["root"]), batch_size=rd_cfg["batch_size"], shuffle=False)]
    n = rd_cfg["iterations"] * rd_cfg["batch_size"]
    loader = DataLoader(RandomCropDataset(train, cfg["root"], n, rd_cfg["seed"]), batch_size=rd_cfg["batch_size"],
                        shuffle=False, num_workers=cfg["num_workers"])
    rd = RD4AD(Backbone(cfg["backbone_weights"]), device, rd_cfg["seed"])
    stats = rd.fit((x for x, _, _ in loader), rd_cfg["iterations"], rd_cfg["lr"], tuple(rd_cfg["betas"]), val_batches,
                   rd_cfg["val_every"], rd_cfg["log_every"])
    for it, tr, va in stats["loss_curve"]:
        print(f"  iter {it}: train loss {tr:.4f}" + (f", val loss {va:.4f}" if va is not None else ""))
    keep = ("iterations", "iters_per_second", "final_train_loss", "final_val_loss", "loss_curve")
    return rd, {"n_train_tiles": len(train), **{k: stats[k] for k in keep}}, \
        f"{stats['iterations']} iterations at {stats['iters_per_second']:.2f}/s, final train loss {stats['final_train_loss']:.4f}"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args(argv).config)
    device = pick_device(cfg)
    rows = build_rows(cfg)
    manifests = Path(cfg["artifacts"]) / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    tag = f"_{cfg['tag']}" if cfg["tag"] else ""  # ablations with overlap or resize change the manifest
    write_manifest(rows, manifests / f"{cfg['dataset']}{tag}.csv")
    for name, prows in profiles(rows, cfg).items():
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        model, stats, line = (fit_patchcore if cfg["method"] == "patchcore" else fit_rd4ad)(cfg, prows, device)
        out = bundle_dir(cfg, name)
        out.mkdir(parents=True, exist_ok=True)
        model.save(out / "model.pt")
        # tau from normal validation tiles only (rule 2): heatmap maxima, the same quantity the Inspector thresholds
        cal = calibrate(val_peaks(model, select(prows, split="val"), cfg["root"], cfg[cfg["method"]]["batch_size"], cfg["num_workers"]),
                        cfg["target_far"])
        (out / "calibration.json").write_text(json.dumps(cal, indent=2))
        with open(out / "config.yaml", "w") as f:
            yaml.safe_dump({**cfg, "profile": name, "git_commit": git_commit(), "tile_size": 256,
                            "layers": ["layer2", "layer3"] if cfg["method"] == "patchcore" else ["layer1", "layer2", "layer3"]}, f)
        metrics = {**stats, "fit_minutes": (time.perf_counter() - t0) / 60,
                   "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0,
                   "peak_ram_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,  # KiB on Linux
                   "bundle_mb": (out / "model.pt").stat().st_size / 2**20}
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(f"{name}: {stats['n_train_tiles']} train tiles, {line}, peak RAM {metrics['peak_ram_gb']:.1f} GB, "
              f"peak VRAM {metrics['peak_vram_gb']:.1f} GB, tau {cal['tau']:.4f}"
              f"{' (LOW-CONFIDENCE calibration)' if cal['low_confidence'] else ''} -> {out}")


if __name__ == "__main__":
    main()
