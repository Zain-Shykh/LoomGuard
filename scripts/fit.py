"""Fit PatchCore from a config: one bundle per fabric profile, from the train split only.

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

from loomguard.data.dataset import TileDataset
from loomguard.data.manifest import select, write_manifest
from loomguard.models.backbone import Backbone
from loomguard.models.patchcore import PatchCore
from loomguard.runs import build_rows, bundle_dir, git_commit, load_config, pick_device, profiles, subsample


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args(argv).config)
    pc_cfg, device = cfg["patchcore"], pick_device(cfg)
    rows = build_rows(cfg)
    manifests = Path(cfg["artifacts"]) / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    write_manifest(rows, manifests / f"{cfg['dataset']}.csv")
    for name, prows in profiles(rows, cfg).items():
        train = subsample(select(prows, split="train"), pc_cfg["max_tiles"], cfg["seed"])  # train rows only (rule 2)
        loader = DataLoader(TileDataset(train, cfg["root"]), batch_size=pc_cfg["batch_size"], shuffle=False,
                            num_workers=cfg["num_workers"])
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        pc = PatchCore(Backbone(cfg["backbone_weights"]), pc_cfg["coreset_ratio"], pc_cfg["proj_dim"], cfg["seed"], device)
        stats = pc.fit(x for x, _, _ in loader)
        out = bundle_dir(cfg, name)
        out.mkdir(parents=True, exist_ok=True)
        pc.save(out / "model.pt")
        with open(out / "config.yaml", "w") as f:
            yaml.safe_dump({**cfg, "profile": name, "git_commit": git_commit(), "tile_size": 256,
                            "layers": ["layer2", "layer3"]}, f)
        metrics = {"n_train_tiles": stats["n_tiles"], "bank_size": stats["bank_size"],
                   "fit_minutes": (time.perf_counter() - t0) / 60, "coreset_seconds": stats["coreset_seconds"],
                   "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0,
                   "peak_ram_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,  # KiB on Linux
                   "bundle_mb": (out / "model.pt").stat().st_size / 2**20}
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(f"{name}: {stats['n_tiles']} tiles, {stats['n_patches']} patches -> bank {stats['bank_size']}, "
              f"coreset {stats['coreset_seconds']:.1f}s, peak RAM {metrics['peak_ram_gb']:.1f} GB, "
              f"peak VRAM {metrics['peak_vram_gb']:.1f} GB -> {out}")


if __name__ == "__main__":
    main()
