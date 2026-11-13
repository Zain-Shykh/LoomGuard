"""Compare what is on disk with docs/DATASETS.md: counts, image sizes, mask pairing. Read-only on the datasets.

    python scripts/verify_datasets.py --raw-fabrid DIR --zju DIR --mvtec DIR --aitex DIR [--manifest-dir DIR]

Every dataset is optional and one failing dataset does not stop the others. Layouts of ZJU-Leaper and AITEX are
taken from the official loader and from DATASETS.md and have not been seen on real files yet.
"""
import argparse
import csv
import json
import sys
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from loomguard.data.manifest import (IMG_EXT, MVTEC_TEXTURES, RAW_META, build_mvtec, build_raw_fabrid,
                                     compare_with_patch_metadata, write_manifest)
from loomguard.data.tiling import TILE, grid

OFFICIAL_RAW = {"train_pool": 14196, "test_good": 4969, "test_defect": 687}  # DATASETS.md section 1


def _images(d: Path) -> list[Path]:
    return sorted(p for p in d.glob("*") if p.suffix.lower() in IMG_EXT) if d.is_dir() else []


def _sizes(paths) -> str:
    c = Counter()
    for p in paths:
        with Image.open(p) as im:
            c[im.size] += 1
    return ", ".join(f"{w}x{h}: {n}" for (w, h), n in c.most_common()) or "none"


def check_raw_fabrid(root: Path, manifest_dir: Path | None = None) -> list[str]:
    images, masks = _images(root / "images"), _images(root / "masks")
    with open(root / RAW_META, newline="") as f:
        meta = list(csv.DictReader(f))
    defective = [m["filename"] for m in meta if m["has_defect"].strip().lower() in {"1", "true", "yes"}]
    names = {p.name for p in masks}
    out = [f"images on disk {len(images)}, in CSV {len(meta)}; masks on disk {len(masks)}; CSV defective {len(defective)}",
           f"CSV roll_id values: {dict(Counter(m['roll_id'] for m in meta))}",
           f"image sizes: {_sizes(images)}; mask sizes: {_sizes(masks)}",
           f"defective images without mask: {sorted(set(defective) - names)}",
           f"masks of images the CSV calls defect-free: {sorted(names - set(defective))}"]
    rows = build_raw_fabrid(root)
    pool = sum(r.split != "test" for r in rows)
    got = {"train_pool": pool, "test_good": sum(r.split == "test" and not r.label for r in rows),
           "test_defect": sum(r.split == "test" and bool(r.label) for r in rows)}
    out += ["RAW-FABRID manifest:",
            f"  tiles per parent image: {dict(Counter(Counter(r.parent for r in rows).values()))}",
            f"  train {sum(r.split == 'train' for r in rows)}, val {sum(r.split == 'val' for r in rows)} "
            f"(train pool {pool}), test good {got['test_good']}, test defect {got['test_defect']}",
            f"  official {OFFICIAL_RAW} -> {'MATCH' if got == OFFICIAL_RAW else 'MISMATCH, got ' + str(got)}"]
    problems = compare_with_patch_metadata(rows, root / "RAW_FABRID_Patches_Metadata.csv")
    out.append(f"  patch metadata comparison: {len(problems)} problems" + "".join(f"\n    {p}" for p in problems[:20]))
    if manifest_dir:
        write_manifest(rows, manifest_dir / "raw_fabrid.csv")
    return out


def check_mvtec(root: Path, manifest_dir: Path | None = None) -> list[str]:
    out = []
    for cat in MVTEC_TEXTURES:
        base = root / cat
        test_dirs = sorted(d for d in (base / "test").iterdir() if d.is_dir()) if (base / "test").is_dir() else []
        defects = [p for d in test_dirs if d.name != "good" for p in _images(d)]
        unpaired = [p.name for d in test_dirs if d.name != "good" for p in _images(d)
                    if not (base / "ground_truth" / d.name / f"{p.stem}_mask.png").is_file()]
        out.append(f"{cat}: train/good {len(_images(base / 'train' / 'good'))}, test/good "
                   f"{len(_images(base / 'test' / 'good'))}, defective {len(defects)} in {[d.name for d in test_dirs if d.name != 'good']}, "
                   f"unpaired masks {unpaired}, sizes {_sizes(_images(base / 'train' / 'good'))}")
    try:
        rows = build_mvtec(root)
    except FileNotFoundError as e:  # unpaired masks are already listed above
        return out + [f"MVTec manifest not built: {e}"]
    out.append(f"MVTec manifest: {dict(Counter(r.split for r in rows))}")
    if manifest_dir:
        write_manifest(rows, manifest_dir / "mvtec.csv")
    return out


def check_zju(root: Path) -> list[str]:
    def find(i: str) -> Path | None:
        return next((p for p in ((root / "Images" / f"{i}{e}") for e in (".jpg", ".png")) if p.is_file()), None)

    out, exts, seen, no_img, no_mask = [], Counter(), [], 0, 0
    for n in range(1, 20):
        sets = json.loads((root / "ImageSets" / "Patterns" / f"pattern{n}.json").read_text())
        counts = {f"{k}/{s}": len(v) for k, d in sets.items() for s, v in d.items()}
        out.append(f"pattern{n}: {counts}")
        for k, d in sets.items():
            for s, ids in d.items():
                for i in ids[:50] if not (k == "defect" and s == "test") else ids:
                    p = find(i)
                    if p is None:
                        no_img += 1
                        continue
                    exts[p.suffix] += 1
                    seen.append(p)
                    if k == "defect" and s == "test" and not (root / "Annotations" / "masks" / f"{i}.png").is_file():
                        no_mask += 1
    return out + [f"image extensions (sampled): {dict(exts)}; sizes (sampled): {_sizes(seen[:300])}",
                  f"listed ids without an image file: {no_img}; defective test ids without a mask: {no_mask}"]


def check_aitex(root: Path) -> list[str]:
    def folder(name: str) -> Path:
        return next(p for p in root.rglob("*") if p.is_dir() and p.name.lower() == name.lower())

    defect, normal, mask = (_images(folder(n)) for n in ("Defect_images", "NODefect_images", "Mask_images"))
    per_mask = {d.stem: [m.name for m in mask if m.stem.startswith(d.stem)] for d in defect}
    border = tot = 0
    for p in defect + normal:
        a = np.asarray(Image.open(p).convert("L"))
        for y, x in grid(*a.shape):
            tot += 1
            border += (a[y:y + TILE, x:x + TILE] < 10).mean() > 0.5
    return [f"defect images {len(defect)}, defect-free {len(normal)}, mask files {len(mask)}",
            f"sizes: {_sizes(defect + normal)}",
            f"defect images with 0 masks: {[k for k, v in per_mask.items() if not v]}",
            f"defect images with several masks: { {k: v for k, v in per_mask.items() if len(v) > 1} }",
            f"mask files not matching any defect image: "
            f"{[m.name for m in mask if not any(m.stem.startswith(k) for k in per_mask)]}",
            f"tiles that are more than half near-black: {border} of {tot}"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for name in ("raw-fabrid", "zju", "mvtec", "aitex"):
        ap.add_argument(f"--{name}", type=Path)
    ap.add_argument("--manifest-dir", type=Path, help="write the RAW-FABRID and MVTec manifests here")
    a = ap.parse_args(argv)
    if a.manifest_dir:
        a.manifest_dir.mkdir(parents=True, exist_ok=True)
    jobs = [("RAW-FABRID", a.raw_fabrid, lambda r: check_raw_fabrid(r, a.manifest_dir)),
            ("ZJU-Leaper", a.zju, check_zju),
            ("MVTec AD textures", a.mvtec, lambda r: check_mvtec(r, a.manifest_dir)),
            ("AITEX", a.aitex, check_aitex)]
    failed = 0
    for title, root, fn in jobs:
        if root is None:
            continue
        print(f"== {title}: {root}")
        try:
            print("\n".join(fn(root)))
        except Exception:
            failed += 1
            print("ERROR (the checks for the other datasets continue):\n" + traceback.format_exc())
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
