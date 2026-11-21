"""Plumbing shared by scripts/fit.py and scripts/evaluate.py: config, manifests, fabric profiles, tile cap, paths."""
import argparse
import hashlib
import json
import random
import subprocess
from collections import defaultdict
from pathlib import Path

import torch
import yaml

from loomguard.data.manifest import (FIELDS, MVTEC_TEXTURES, TileRow, build_aitex, build_mvtec, build_raw_fabrid, build_zju,
                                     zju_group)

DEFAULTS = {"seed": 0, "backbone_weights": "IMAGENET1K_V1", "device": "auto", "num_workers": 0, "hist_bins": 10_000,
            "artifacts": "artifacts", "results": "results", "fabrics": None, "targets": None, "tag": "",
            "target_far": 0.01, "px_per_mm": 4.0, "scale_assumed": True, "grade_rolls": False,
            "min_overlap": 0, "held_out_scope": "all", "dump_predictions": False}
GRADING_DEFAULTS = {"closing": 5, "min_area_px": 16, "cutoffs": [20, 40, 60]}
PATCHCORE_DEFAULTS = {"max_tiles": 600, "coreset_ratio": 0.10, "proj_dim": 128, "batch_size": 16}
RD4AD_DEFAULTS = {"iterations": 5000, "batch_size": 16, "lr": 0.005, "betas": [0.5, 0.999], "seed": None,
                  "val_tiles": 64, "val_every": 500, "log_every": 100}
METHODS = ("patchcore", "rd4ad")
REGIMES = ("per_fabric", "per_group", "unified", "held_out")


def _merge(a: dict, b: dict) -> dict:
    return {**a, **{k: _merge(a[k], v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v for k, v in b.items()}}


def _read(path: Path | str) -> dict:
    """YAML with `extends: base.yaml | [a.yaml, b.yaml]` (paths relative to the file): bases first, the file wins."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    bases = cfg.pop("extends", [])
    out: dict = {}
    for b in [bases] if isinstance(bases, str) else bases:
        out = _merge(out, _read(Path(path).parent / b))
    return _merge(out, cfg)


def load_config(path: Path | str) -> dict:
    cfg = _read(path)
    missing = {"dataset", "root", "regime", "method"} - set(cfg)
    if missing:
        raise KeyError(f"{path} lacks required keys {sorted(missing)}")
    if cfg["method"] not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {cfg['method']!r}")
    if cfg["regime"] not in REGIMES:
        raise ValueError(f"regime must be one of {REGIMES}, got {cfg['regime']!r}")
    cfg = {**DEFAULTS, **cfg, "patchcore": {**PATCHCORE_DEFAULTS, **cfg.get("patchcore", {})},
           "rd4ad": {**RD4AD_DEFAULTS, **cfg.get("rd4ad", {})}, "grading": {**GRADING_DEFAULTS, **cfg.get("grading", {})}}
    cfg.setdefault("resize", 256 if cfg["dataset"] == "mvtec" else None)  # MVTec protocol: whole image resized (D-21)
    if cfg["rd4ad"]["seed"] is None:
        cfg["rd4ad"]["seed"] = cfg["seed"]
    cfg["config"] = str(path)
    return cfg


def build_rows(cfg: dict) -> list[TileRow]:
    if cfg["dataset"] == "mvtec":
        return build_mvtec(Path(cfg["root"]), cfg["fabrics"] or MVTEC_TEXTURES, seed=cfg["seed"])
    if cfg["dataset"] == "raw_fabrid":
        rows = build_raw_fabrid(Path(cfg["root"]), seed=cfg["seed"], min_overlap=cfg["min_overlap"])
    elif cfg["dataset"] == "zju_leaper":
        # held_out needs the other fabrics of the scope too; profiles() picks the held-out ones
        keep = None if cfg["regime"] == "held_out" else cfg["fabrics"]
        rows = build_zju(Path(cfg["root"]), keep, seed=cfg["seed"], resize=cfg["resize"] or 0, min_overlap=cfg["min_overlap"])
    elif cfg["dataset"] == "aitex":
        rows = build_aitex(Path(cfg["root"]), seed=cfg["seed"], min_overlap=cfg["min_overlap"])
    else:
        raise ValueError(f"unknown dataset {cfg['dataset']!r}")
    return [r for r in rows if not cfg["fabrics"] or cfg["regime"] == "held_out" or r.fabric in cfg["fabrics"]]


def _scope(fabric: str, cfg: dict) -> str:
    return zju_group(fabric) if cfg["held_out_scope"] == "group" else "all"


def profiles(rows: list[TileRow], cfg: dict) -> dict[str, list[TileRow]]:
    """Fabric profiles to fit. held_out: `<f>_held_out` = train and val rows of the other fabrics in the scope plus the
    test rows of f only, so f's own train and val rows never reach fit and tau (D-32) comes from the other fabrics."""
    regime = cfg["regime"]
    if regime == "unified":
        return {"unified": list(rows)}
    if regime == "held_out":
        fabrics = cfg["fabrics"] or sorted({r.fabric for r in rows})
        return {f"{f}_held_out": [r for r in rows if (r.fabric == f and r.split == "test") or
                                  (r.fabric != f and r.split != "test" and _scope(r.fabric, cfg) == _scope(f, cfg))]
                for f in fabrics}
    if regime == "per_group" and cfg["dataset"] != "zju_leaper":
        raise ValueError("per_group is defined for zju_leaper only")
    out: dict[str, list[TileRow]] = defaultdict(list)
    for r in rows:
        out[zju_group(r.fabric) if regime == "per_group" else r.fabric].append(r)
    return dict(out)


def selected_profiles(rows: list[TileRow], cfg: dict, selection: str | None) -> dict[str, list[TileRow]]:
    available = profiles(rows, cfg)
    if selection is None:
        return available
    names = [s.strip() for s in selection.split(",")]
    if not all(names) or len(set(names)) != len(names):
        raise ValueError("--profiles requires distinct, nonempty profile names")
    unknown = set(names) - available.keys()
    if unknown:
        raise ValueError(f"unknown profiles {sorted(unknown)}; available: {sorted(available)}")
    return {name: available[name] for name in names}


def manifest_digest(rows: list[TileRow]) -> str:
    """Fingerprint ordered tile rows, including splits, independent of dataset location and OS."""
    digest = hashlib.sha256()
    for row in rows:
        digest.update((json.dumps([getattr(row, k) for k in FIELDS], separators=(",", ":")) + "\n").encode())
    return digest.hexdigest()


def run_inputs(ap, argv):
    """Load unchanged scientific inputs and optionally route writes to a runner's staging directory."""
    ap.add_argument("--config", required=True)
    ap.add_argument("--profiles", help="comma-separated exact names of complete model profiles")
    ap.add_argument("--output-root", type=Path, help=argparse.SUPPRESS)
    ap.add_argument("--manifest-sha256", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    canonical = load_config(args.config)
    rows = build_rows(canonical)
    if args.manifest_sha256 and manifest_digest(rows) != args.manifest_sha256:
        raise ValueError(f"{args.config}: manifest differs from checkpoint 0")
    selected = selected_profiles(rows, canonical, args.profiles)
    cfg = canonical.copy()
    if args.output_root is not None:
        for key in ("artifacts", "results"):
            path = Path(cfg[key])
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"cannot stage non-relative {key}: {path}")
            cfg[key] = str(args.output_root / path)
    return canonical, cfg, rows, selected, args.profiles


def predictions_dir(cfg: dict, profile: str) -> Path:
    return Path(cfg["artifacts"]) / "predictions" / cfg["dataset"] / cfg["regime"] / _method_dir(cfg) / profile


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


def _method_dir(cfg: dict) -> str:
    return f"{cfg['method']}_{cfg['tag']}" if cfg["tag"] else cfg["method"]  # tag: separate runs of one method (seeds)


def bundle_dir(cfg: dict, profile: str) -> Path:
    return Path(cfg["artifacts"]) / cfg["dataset"] / profile / _method_dir(cfg)


def result_path(cfg: dict, profile: str) -> Path:
    return Path(cfg["results"]) / cfg["dataset"] / cfg["regime"] / _method_dir(cfg) / f"{profile}.json"


def load_model(cfg: dict, path: Path, backbone, device: torch.device):
    """The fitted PatchCore or RD4AD of cfg["method"], loaded from a bundle's model.pt."""
    from loomguard.models.patchcore import PatchCore
    from loomguard.models.rd4ad import RD4AD
    return {"patchcore": PatchCore, "rd4ad": RD4AD}[cfg["method"]].load(path, backbone, device)


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
