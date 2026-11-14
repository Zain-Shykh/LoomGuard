"""Manifests: one CSV row per tile, crop boxes only (decision D-15). Split by parent image first, then tile.

Paths are relative to the dataset root with '/' separators, so a manifest works on any machine.
"""
import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Callable, Iterable, Sequence

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
        per_parent = {tiles_of[p] for p in parents}
        if len(per_parent) != 1:
            raise ValueError(f"{fabric}: images give different tile counts {sorted(per_parent)}")
        val = assign_val(parents, per_parent.pop(), stratum=fabric, **rule)
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


def _image_rows(dataset: str, fabric: str, split: str, parent: str, mask_rel: str, mask: np.ndarray | None, w: int, h: int,
                min_overlap: int = 0, keep: Callable[[int, int], bool] | None = None) -> list[TileRow]:
    """Tile rows of one parent image on the grid; label = the tile holds a mask pixel. `keep(y, x)` can drop tiles."""
    rows = []
    for y, x in grid(h, w, TILE, min_overlap):
        if keep is None or keep(y, x):
            px = int(mask[y:y + TILE, x:x + TILE].sum()) if mask is not None else 0
            rows.append(TileRow(dataset, fabric, split, parent, parent, mask_rel, x, y, TILE, TILE, 0, int(px > 0), px))
    return rows


def _mask_array(paths: Sequence[Path], size: tuple[int, int]) -> np.ndarray:
    """Union of mask files as H x W bool; size is (W, H) of the image."""
    out = np.zeros(size[::-1], bool)
    for p in paths:
        with Image.open(p) as mi:
            m = np.asarray(mi.convert("L")) > 0
        if m.shape != out.shape:
            raise ValueError(f"mask {p} does not match its image size {size}")
        out |= m
    return out


def build_raw_fabrid(root: Path, *, val_frac: float = 0.15, seed: int = 0, min_overlap: int = 0) -> list[TileRow]:
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
        tiles_of[image] = len(grid(h, w))  # the split never depends on the overlap
        defective = _truthy(m["has_defect"])
        parents.append((m["roll_id"], image, w, h, defective))
        if not defective:
            normals[m["roll_id"]].append(image)
    split_of = _split_normals(normals, tiles_of, val_frac=val_frac, min_val_tiles=200, max_frac=0.40, seed=seed)
    rows = []
    for roll, image, w, h, defective in parents:
        if not defective:
            rows += _image_rows("raw_fabrid", roll, split_of[image], image, "", None, w, h, min_overlap)
            continue
        mask_path = root / "masks" / Path(image).name
        if not mask_path.is_file():
            raise FileNotFoundError(f"defective image without mask: {mask_path}")
        rows += _image_rows("raw_fabrid", roll, "test", image, _rel(mask_path, root), _mask_array([mask_path], (w, h)),
                            w, h, min_overlap)
    return _sorted(rows)


ZJU_GROUPS = {1: range(1, 4), 2: range(4, 8), 3: range(8, 13), 4: range(13, 16), 5: range(16, 20)}  # DATASETS 2


def zju_group(fabric: str) -> str:
    n = int(fabric.removeprefix("pattern"))
    return next(f"group{g}" for g, r in ZJU_GROUPS.items() if n in r)


def _index(d: Path) -> dict[str, Path]:
    """stem and file name -> path of every image under d (ids in the JSON files may or may not carry an extension)."""
    out: dict[str, Path] = {}
    for p in sorted(d.rglob("*")) if d.is_dir() else []:
        if p.suffix.lower() in IMG_EXT:
            out.setdefault(p.stem, p)
            out.setdefault(p.name, p)
    return out


def _find(idx: dict[str, Path], i: str) -> Path | None:
    return idx.get(i) or idx.get(Path(i).name)


def build_zju(root: Path, patterns: Sequence[str] | None = None, *, val_frac: float = 0.10, seed: int = 0, resize: int = 0,
              min_overlap: int = 0) -> list[TileRow]:
    """ZJU-Leaper, official Setting 1: train/val from `normal/train`, test = `normal/test` + `defect/test`. Layout unverified."""
    root = Path(root)
    pdir = root / "ImageSets" / "Patterns"
    files = {f"pattern{n}": pdir / f"pattern{n}.json" for n in range(1, 20) if (pdir / f"pattern{n}.json").is_file()}
    files = {k: v for k, v in files.items() if not patterns or k in patterns}
    if not files:
        raise FileNotFoundError(f"no pattern<n>.json under {pdir} (patterns wanted: {list(patterns) if patterns else 'all'})")
    images, masks = _index(root / "Images"), _index(root / "Annotations" / "masks")
    info: dict[str, tuple] = {}  # parent -> (fabric, kind, mask rel, image path)
    normals, problems = defaultdict(list), []
    for fabric, f in files.items():
        sets = json.loads(f.read_text())
        try:
            lists = {"train": sets["normal"]["train"], "ntest": sets["normal"]["test"], "dtest": sets["defect"]["test"]}
        except (KeyError, TypeError):
            raise ValueError(f"{f}: expected {{'normal': {{'train', 'test'}}, 'defect': {{'test'}}}}, found "
                             f"{list(sets) if isinstance(sets, dict) else type(sets).__name__}")
        for kind, ids in lists.items():
            for i in ids:
                img = _find(images, str(i))
                mask = _find(masks, str(i)) if kind == "dtest" else None
                if img is None or (kind == "dtest" and mask is None):
                    problems.append(f"{fabric}: id {i} has no {'image' if img is None else 'mask'} file")
                    continue
                parent = _rel(img, root)
                if parent in info:
                    problems.append(f"{fabric}: image {parent} is listed twice")
                    continue
                info[parent] = (fabric, kind, _rel(mask, root) if mask else "", img)
                if kind == "train":
                    normals[fabric].append(parent)
    if problems:
        raise FileNotFoundError(f"ZJU-Leaper layout problems ({len(problems)}): " + "; ".join(problems[:10]))
    sizes = {p: _size(v[3]) for p, v in info.items()}
    tiles_of = {p: 1 if resize else len(grid(*sizes[p][::-1])) for p in sizes}
    split_of = _split_normals(normals, tiles_of, val_frac=val_frac, min_val_tiles=0, max_frac=0.40, seed=seed)
    rows = []
    for parent, (fabric, kind, mask_rel, _) in info.items():
        (w, h), split = sizes[parent], split_of.get(parent, "test")
        if resize:  # whole image, resized like MVTec (resolution ablation)
            _mask_array([root / mask_rel], (w, h)) if mask_rel else None  # size check only
            rows.append(TileRow("zju_leaper", fabric, split, parent, parent, mask_rel, 0, 0, w, h, resize, int(kind == "dtest"), 0))
        else:
            mask = _mask_array([root / mask_rel], (w, h)) if mask_rel else None
            rows += _image_rows("zju_leaper", fabric, split, parent, mask_rel, mask, w, h, min_overlap)
    return _sorted(rows)


def build_aitex(root: Path, *, val_frac: float = 0.15, seed: int = 0, black_level: int = 10, max_black: float = 0.5,
                min_overlap: int = 0) -> list[TileRow]:
    """AITEX: per fabric 12 train / 8 val of the 20 defect-free images (A-01 rule), all defective images are test.
    Tiles with more than `max_black` of their pixels below `black_level` are dropped. Layout unverified."""
    root = Path(root)
    dirs = {}
    for name in ("Defect_images", "NODefect_images", "Mask_images"):
        dirs[name] = next((p for p in sorted(root.rglob("*")) if p.is_dir() and p.name.lower() == name.lower()), None)
        if dirs[name] is None:
            raise FileNotFoundError(f"AITEX: no folder {name} under {root}; found {sorted(p.name for p in root.iterdir())}")

    def files(d: Path) -> list[Path]:
        return sorted(p for p in d.rglob("*") if p.suffix.lower() in IMG_EXT)

    def fabric_of(p: Path) -> str:
        m = re.search(r"_(\d+)$", p.stem)
        if not m:
            raise ValueError(f"AITEX: cannot read the fabric number from the file name {p.name} (expected ..._<NN>)")
        return f"fabric{m.group(1)}"

    pairs = defaultdict(list)  # image stem -> mask files
    for m in files(dirs["Mask_images"]):
        pairs[re.sub(r"_mask.*$", "", m.stem)].append(m)
    normals, tiles_of, items, problems = defaultdict(list), {}, [], []
    for p in files(dirs["NODefect_images"]):
        normals[fabric_of(p)].append(_rel(p, root))
        tiles_of[_rel(p, root)] = len(grid(*_size(p)[::-1]))
        items.append((fabric_of(p), _rel(p, root), p, []))
    for p in files(dirs["Defect_images"]):
        if p.stem not in pairs:
            problems.append(f"{p.name}: no mask file")
        items.append((fabric_of(p), _rel(p, root), p, pairs.get(p.stem, [])))
    if problems:
        raise FileNotFoundError(f"AITEX layout problems ({len(problems)}): " + "; ".join(problems[:10]))
    split_of = _split_normals(normals, tiles_of, val_frac=val_frac, min_val_tiles=200, max_frac=0.40, seed=seed)
    rows = []
    for fabric, parent, p, mask_files in items:
        w, h = _size(p)
        with Image.open(p) as im:
            gray = np.asarray(im.convert("L"))
        keep = lambda y, x, g=gray: (g[y:y + TILE, x:x + TILE] < black_level).mean() <= max_black
        mask = _mask_array(mask_files, (w, h)) if mask_files else None
        rows += _image_rows("aitex", fabric, split_of.get(parent, "test"), parent, "|".join(_rel(m, root) for m in mask_files),
                            mask, w, h, min_overlap, keep)
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
