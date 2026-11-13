"""Manifests: one CSV row per tile, crop boxes only (decision D-15). Split by parent image first, then tile.

Paths are relative to the dataset root with '/' separators, so a manifest works on any machine.
"""
import csv
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

from loomguard.data.tiling import TILE, grid

MVTEC_TEXTURES = ("carpet", "grid", "leather", "tile", "wood")
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
RAW_META = "RAW_FABRID_HighRes_Metadata.csv"


@dataclass(frozen=True)
class TileRow:
    dataset: str    # "raw_fabrid" | "mvtec"
    fabric: str     # fabric profile: RAW-FABRID roll_id, MVTec category
    split: str      # "train" | "val" | "test"
    parent: str     # parent image id, unique within the dataset (= image path)
    image: str      # path relative to the dataset root
    mask: str       # relative path of the parent's ground-truth mask, "" if none
    x0: int         # crop box in the parent, native pixels
    y0: int
    w: int
    h: int
    resize: int     # 0 = none; N = resize crop (and mask) to N x N (MVTec: 256)
    label: int      # 1 if the crop contains a defect
    defect_px: int  # defect pixels in the crop, native resolution; 0 where not computed (MVTec)


FIELDS = tuple(f.name for f in fields(TileRow))
_INT_FIELDS = {f.name for f in fields(TileRow) if f.type is int}


def write_manifest(rows: Iterable[TileRow], path: Path) -> None:
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(asdict(r) for r in rows)


def read_manifest(path: Path) -> list[TileRow]:
    with open(path, newline="") as f:
        return [TileRow(**{k: int(v) if k in _INT_FIELDS else v for k, v in r.items()}) for r in csv.DictReader(f)]


def select(rows: Iterable[TileRow], split: str | None = None, fabric: str | None = None) -> list[TileRow]:
    return [r for r in rows if (split is None or r.split == split) and (fabric is None or r.fabric == fabric)]


def _round_half_up(x: float) -> int:
    return math.floor(x + 0.5)


def assign_val(parents: Sequence[str], tiles_per_parent: int, *, val_frac: float, min_val_tiles: int,
               max_frac: float, seed: int, stratum: str) -> set[str]:
    n = len(parents)
    n_val = max(_round_half_up(val_frac * n), math.ceil(min_val_tiles / tiles_per_parent))
    n_val = min(n_val, math.floor(max_frac * n), n - 1)  # cap wins over the minimum; train never empty
    order = sorted(parents)
    random.Random(f"{seed}/{stratum}").shuffle(order)
    return set(order[:max(n_val, 0)])


def _size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size  # (W, H), header only


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _split_normals(items: dict[str, list[str]], tiles_of: dict[str, int], **rule) -> dict[str, str]:
    """items: fabric -> normal parents. Returns parent -> 'train' | 'val', decided per fabric."""
    out = {}
    for fabric, parents in items.items():
        val = assign_val(parents, tiles_of[parents[0]], stratum=fabric, **rule)
        out.update({p: "val" if p in val else "train" for p in parents})
    return out


def _sorted(rows: list[TileRow]) -> list[TileRow]:
    return sorted(rows, key=lambda r: (r.fabric, r.split, r.parent, r.y0, r.x0))


def build_mvtec(root: Path, categories: Sequence[str] = MVTEC_TEXTURES, *,
                val_frac: float = 0.10, seed: int = 0) -> list[TileRow]:
    root = Path(root)
    rows, normals, tiles_of = [], defaultdict(list), {}
    info = {}  # parent -> (category, split or None for the normal pool, mask, label, path)
    for cat in categories:
        train = sorted(p for p in (root / cat / "train" / "good").glob("*") if p.suffix.lower() in IMG_EXT)
        if not train:
            raise FileNotFoundError(f"no images in {root / cat / 'train' / 'good'}")
        for p in train:
            normals[cat].append(_rel(p, root))
            tiles_of[_rel(p, root)] = 1
            info[_rel(p, root)] = (cat, None, "", 0, p)
        for d in sorted(x for x in (root / cat / "test").iterdir() if x.is_dir()):
            for p in sorted(x for x in d.glob("*") if x.suffix.lower() in IMG_EXT):
                mask = ""
                if d.name != "good":
                    m = root / cat / "ground_truth" / d.name / f"{p.stem}_mask.png"
                    if not m.is_file():
                        raise FileNotFoundError(f"missing mask {m}")
                    mask = _rel(m, root)
                info[_rel(p, root)] = (cat, "test", mask, int(d.name != "good"), p)
    split_of = _split_normals(normals, tiles_of, val_frac=val_frac, min_val_tiles=0, max_frac=0.40, seed=seed)
    for parent, (cat, split, mask, label, p) in info.items():
        w, h = _size(p)
        rows.append(TileRow("mvtec", cat, split or split_of[parent], parent, parent, mask, 0, 0, w, h, TILE, label, 0))
    return _sorted(rows)


def _truthy(v: str) -> bool:
    return v.strip().lower() in {"1", "true", "yes"}


def build_raw_fabrid(root: Path, *, val_frac: float = 0.15, seed: int = 0) -> list[TileRow]:
    root = Path(root)
    with open(root / RAW_META, newline="") as f:
        reader = csv.DictReader(f)
        meta = list(reader)
    missing = {"filename", "roll_id", "has_defect"} - set(reader.fieldnames or [])
    if missing:
        raise KeyError(f"{RAW_META} lacks columns {sorted(missing)}; found {reader.fieldnames}")
    normals, tiles_of, parents = defaultdict(list), {}, []
    for m in meta:
        image = f"images/{m['filename']}"
        w, h = _size(root / image)
        if h < TILE or w < TILE:
            raise ValueError(f"{image} is {w}x{h}, smaller than a tile")
        origins = grid(h, w)
        tiles_of[image] = len(origins)
        defective = _truthy(m["has_defect"])
        parents.append((m["roll_id"], image, origins, defective))
        if not defective:
            normals[m["roll_id"]].append(image)
    split_of = _split_normals(normals, tiles_of, val_frac=val_frac, min_val_tiles=200, max_frac=0.40, seed=seed)
    rows = []
    for roll, image, origins, defective in parents:
        if not defective:
            rows += [TileRow("raw_fabrid", roll, split_of[image], image, image, "", x, y, TILE, TILE, 0, 0, 0)
                     for y, x in origins]
            continue
        mask_path = root / "masks" / Path(image).name
        if not mask_path.is_file():
            raise FileNotFoundError(f"defective image without mask: {mask_path}")
        with Image.open(mask_path) as mi:
            mask = np.asarray(mi.convert("L")) > 0
        if mask.shape != _size(root / image)[::-1]:
            raise ValueError(f"mask {mask_path} does not match its image size")
        for y, x in origins:
            px = int(mask[y:y + TILE, x:x + TILE].sum())
            rows.append(TileRow("raw_fabrid", roll, "test", image, image, _rel(mask_path, root), x, y, TILE, TILE, 0,
                                int(px > 0), px))
    return _sorted(rows)


def compare_with_patch_metadata(rows: Sequence[TileRow], patches_csv: Path) -> list[str]:
    with open(patches_csv, newline="") as f:
        reader = csv.DictReader(f)
        official = list(reader)
    need = {"patch_name", "parent_raw", "type", "defect_area"}
    if need - set(reader.fieldnames or []):
        return [f"patch CSV lacks columns {sorted(need - set(reader.fieldnames or []))}; found {reader.fieldnames}"]
    problems = []
    ours, theirs = defaultdict(list), defaultdict(list)
    for r in rows:
        ours[Path(r.parent).stem].append(r.defect_px)
    for o in official:
        theirs[Path(o["parent_raw"]).stem].append(int(float(o["defect_area"] or 0)))
    for p in sorted(ours.keys() | theirs.keys()):
        if p not in ours or p not in theirs:
            problems.append(f"{p}: only in {'manifest' if p in ours else 'patch CSV'}")
        elif len(ours[p]) != len(theirs[p]):
            problems.append(f"{p}: {len(ours[p])} tiles, patch CSV has {len(theirs[p])}")
        elif sorted(a for a in ours[p] if a) != sorted(a for a in theirs[p] if a):
            problems.append(f"{p}: defect areas differ: {sorted(a for a in ours[p] if a)} vs {sorted(a for a in theirs[p] if a)}")
    types = Counter(o["type"].strip().lower() for o in official)
    unknown = set(types) - {"train_good", "test_good", "test_defect"}
    if unknown:
        problems.append(f"unexpected patch types {sorted(unknown)}")
    mine = Counter({"train_good": sum(r.split != "test" for r in rows),
                    "test_good": sum(r.split == "test" and not r.label for r in rows),
                    "test_defect": sum(r.split == "test" and bool(r.label) for r in rows)})
    for k in mine:
        if mine[k] != types.get(k, 0):
            problems.append(f"{k}: manifest {mine[k]}, patch CSV {types.get(k, 0)}")
    return problems
