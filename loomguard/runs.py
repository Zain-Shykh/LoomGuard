"""Plumbing shared by scripts/fit.py and scripts/evaluate.py: config, manifests, fabric profiles, tile cap, paths."""
import random
import subprocess
from collections import defaultdict
from pathlib import Path

import torch
import yaml

from loomguard.data.manifest import MVTEC_TEXTURES, TileRow, build_mvtec, build_raw_fabrid

DEFAULTS = {"seed": 0, "backbone_weights": "IMAGENET1K_V1", "device": "auto", "num_workers": 0, "hist_bins": 10_000,
            "artifacts": "artifacts", "results": "results", "fabrics": None, "targets": None}
PATCHCORE_DEFAULTS = {"max_tiles": 600, "coreset_ratio": 0.10, "proj_dim": 128, "batch_size": 16}


def load_config(path: Path | str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    missing = {"dataset", "root", "regime", "method"} - set(cfg)
    if missing:
        raise KeyError(f"{path} lacks required keys {sorted(missing)}")
    if cfg["method"] != "patchcore":
        raise ValueError(f"method {cfg['method']!r} is not implemented yet (only 'patchcore')")
    if cfg["regime"] not in ("per_fabric", "unified"):
        raise ValueError(f"regime must be per_fabric or unified, got {cfg['regime']!r}")
    cfg = {**DEFAULTS, **cfg, "patchcore": {**PATCHCORE_DEFAULTS, **cfg.get("patchcore", {})}}
    cfg["config"] = str(path)
    return cfg


def build_rows(cfg: dict) -> list[TileRow]:
    if cfg["dataset"] == "mvtec":
        return build_mvtec(Path(cfg["root"]), cfg["fabrics"] or MVTEC_TEXTURES, seed=cfg["seed"])
    if cfg["dataset"] == "raw_fabrid":
        rows = build_raw_fabrid(Path(cfg["root"]), seed=cfg["seed"])
        return [r for r in rows if not cfg["fabrics"] or r.fabric in cfg["fabrics"]]
    raise ValueError(f"unknown dataset {cfg['dataset']!r}")


def profiles(rows: list[TileRow], cfg: dict) -> dict[str, list[TileRow]]:
    if cfg["regime"] == "unified":
        return {"unified": list(rows)}
    out: dict[str, list[TileRow]] = defaultdict(list)
    for r in rows:
        out[r.fabric].append(r)
    return dict(out)


def subsample(rows: list[TileRow], max_tiles: int, seed: int) -> list[TileRow]:
    """Tile cap stratified by parent image: take one random tile per parent in turn until max_tiles."""
    if len(rows) <= max_tiles:
        return rows
    rng = random.Random(seed)
    by_parent: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_parent[r.parent].append(i)
    order = sorted(by_parent)
    rng.shuffle(order)
    for p in order:
        rng.shuffle(by_parent[p])
    keep: list[int] = []
    while len(keep) < max_tiles:
        for p in order:
            if by_parent[p] and len(keep) < max_tiles:
                keep.append(by_parent[p].pop())
    return [rows[i] for i in sorted(keep)]


def bundle_dir(cfg: dict, profile: str) -> Path:
    return Path(cfg["artifacts"]) / cfg["dataset"] / profile / cfg["method"]


def result_path(cfg: dict, profile: str) -> Path:
    return Path(cfg["results"]) / cfg["dataset"] / cfg["regime"] / cfg["method"] / f"{profile}.json"


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def pick_device(cfg: dict) -> torch.device:
    d = cfg["device"]
    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(d)
