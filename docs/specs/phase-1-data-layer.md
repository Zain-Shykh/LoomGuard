# Phase 1: Repository skeleton and data layer

| | |
|---|---|
| Phase | 1 |
| Status | Written |
| Spec written | 2026-11-13 |
| Written by | developer session, phase 1 |
| Depends on | Phase 0 (Done) |
| Plan entry | `docs/IMPLEMENTATION_PLAN.md`, Phase 1 |

## 1. Objective

Any dataset can be turned into a manifest of tiles with leak-free splits, and any image can be tiled and stitched back exactly. At the end there are a tiler and stitcher, manifest builders for MVTec AD textures and RAW-FABRID, a dataset class that crops tiles at load time, and a dataset verification script with a runbook for the training PC.

## 2. Background

The package skeleton, `pyproject.toml`, `.gitignore` and the smoke test already exist (plan task 1, done 2026-11-12). What remains on the laptop is plan tasks 2 to 4. Plan tasks 5 to 7 need the training PC and are in the runbook (section 8).

Read:

- `docs/ARCHITECTURE.md` section 3 (tiler and stitcher, table of tiles per image, MVTec exception), section 12 (layout)
- `docs/EVALUATION.md` section 4 (split rules, per-dataset table)
- `docs/DATASETS.md` sections 1 to 4 and "Preprocessing rules common to all datasets"
- Decisions: D-15 (manifest of crop boxes, tiles never written to disk), D-21 (native 256 tiles, MVTec is the one resize), D-22 (no flips or rotations), D-30 (split by parent first; val 15%, at least 200 tiles, at most 40%), D-12, D-52

## 3. Scope

**In scope**

- `loomguard/data/tiling.py`: tile origins, tile, stitch with overlap averaging.
- `loomguard/data/manifest.py`: manifest row type, CSV read/write, split assignment by parent, adapters for MVTec AD textures and RAW-FABRID, and a comparison against the official RAW-FABRID patch metadata.
- `loomguard/data/dataset.py`: preprocessing and the PyTorch `Dataset` that crops at load time.
- `scripts/verify_datasets.py`: prints counts, image sizes and mask pairing for RAW-FABRID, ZJU-Leaper, MVTec AD and AITEX, and optionally writes manifests.
- CPU tests with synthetic data for all of it.
- The training-PC runbook.

**Out of scope**

- ZJU-Leaper and AITEX manifest adapters (Phase 5). The verification script only inspects those two datasets; it does not build manifests for them.
- Random-crop sampling for RD4AD training (Phase 3). Models, metrics, calibration.
- Any dataset download, any training, any GPU use.
- Images smaller than a tile as dataset rows. The tiler handles them (reflect pad, ARCHITECTURE section 3); the manifest builders reject them because no core dataset has any.

## 4. Design

### 4.1 Files

| File | New or changed | Purpose |
|---|---|---|
| `loomguard/data/tiling.py` | new | `tile_origins`, `tile`, `stitch` |
| `loomguard/data/manifest.py` | new | `TileRow`, CSV I/O, `assign_val`, `select`, `build_mvtec`, `build_raw_fabrid`, `compare_with_patch_metadata` |
| `loomguard/data/dataset.py` | new | `preprocess`, `TileDataset` |
| `scripts/verify_datasets.py` | new | dataset verification on the training PC |
| `tests/test_tiling.py` | new | tiler and stitcher tests |
| `tests/test_manifest.py` | new | split, leakage, adapter tests |
| `tests/test_dataset.py` | new | dataset class tests |
| `tests/test_verify_datasets.py` | new | the verification script on synthetic trees |
| `docs/specs/phase-1-data-layer.md` | new | this spec |
| `docs/IMPLEMENTATION_PLAN.md`, `docs/PROGRESS_LOG.md`, `docs/DECISIONS.md`, `docs/EVALUATION.md`, `README.md`, `CLAUDE.md` | changed | status, evidence, decisions, corrected numbers |

No change to `pyproject.toml`.

### 4.2 Interfaces

Coordinates are `(y, x)` for origins and `(H, W)` for sizes throughout; image tensors are `C x H x W`. Dataset sizes quoted as "1792x1024" in the docs are width x height, i.e. a `1 x 1024 x 1792` tensor.

```python
# loomguard/data/tiling.py
TILE = 256

def tile_origins(length: int, tile_size: int = TILE, min_overlap: int = 0) -> list[int]:
    """Start offsets along one axis. n = 1 if length <= tile_size else
    ceil((length - tile_size) / (tile_size - min_overlap)) + 1;
    origin_i = round_half_up(i * (length - tile_size) / (n - 1)); the last tile ends exactly at length."""

def tile(x: Tensor, tile_size: int = TILE, min_overlap: int = 0) -> tuple[Tensor, list[tuple[int, int]]]:
    """x: C x H x W, floating point. Axes shorter than tile_size are reflect-padded at the
    bottom/right to tile_size. Returns tiles N x C x T x T (same dtype as x), row-major
    (y outer, x inner), and the N (y0, x0) origins."""

def stitch(tiles: Tensor, origins: list[tuple[int, int]], size: tuple[int, int]) -> Tensor:
    """tiles: N x C x T x T; size: original (H, W). Sums tiles on a canvas in float64, divides by the
    per-pixel tile count, crops to (H, W), returns C x H x W in tiles.dtype.
    Works for any C, so it stitches images (C=3) and heatmaps (C=1) alike."""
```

```python
# loomguard/data/manifest.py
MVTEC_TEXTURES = ("carpet", "grid", "leather", "tile", "wood")
FIELDS = ("dataset", "fabric", "split", "parent", "image", "mask",
          "x0", "y0", "w", "h", "resize", "label", "defect_px")

@dataclass(frozen=True)
class TileRow:
    dataset: str      # "raw_fabrid" | "mvtec"
    fabric: str       # RAW-FABRID roll_id, MVTec category: the fabric profile
    split: str        # "train" | "val" | "test"
    parent: str       # parent image id, unique within the dataset: image path relative to the root
    image: str        # path relative to the dataset root, POSIX separators
    mask: str         # relative path of the ground-truth mask of the parent, "" if none
    x0: int; y0: int  # crop box top-left in the parent image, native pixels
    w: int; h: int    # crop box size (256 x 256 for a tile; whole image for MVTec)
    resize: int       # 0 = none; N = resize the crop (and mask) to N x N after cropping (MVTec: 256)
    label: int        # 1 if the crop contains a defect, else 0
    defect_px: int    # defect pixels inside the crop at native resolution; 0 where not computed (MVTec)

def write_manifest(rows: Iterable[TileRow], path: Path) -> None: ...
def read_manifest(path: Path) -> list[TileRow]: ...
def select(rows: Iterable[TileRow], split: str | None = None, fabric: str | None = None) -> list[TileRow]: ...

def assign_val(parents: Sequence[str], tiles_per_parent: int, *, val_frac: float, min_val_tiles: int,
               max_frac: float, seed: int, stratum: str) -> set[str]:
    """Which of the normal parents of one stratum (one fabric profile) go to val.
    n_val = round_half_up(val_frac * n); raised to ceil(min_val_tiles / tiles_per_parent);
    capped at floor(max_frac * n) and at n - 1 (train never empty).
    The choice is a shuffle of the sorted parents with random.Random(f"{seed}/{stratum}"):
    deterministic and independent of directory listing order."""

def build_mvtec(root: Path, categories: Sequence[str] = MVTEC_TEXTURES, *,
                val_frac: float = 0.10, seed: int = 0) -> list[TileRow]:
    """root/<category>/{train/good, test/<type>, ground_truth/<type>/<stem>_mask.png}.
    One row per image (whole image, resize=256). train/good -> train and val (10% of images,
    no tile minimum); every image of test/ -> test, label 0 for test/good, 1 otherwise."""

def build_raw_fabrid(root: Path, *, val_frac: float = 0.15, seed: int = 0) -> list[TileRow]:
    """root/{images, masks, RAW_FABRID_HighRes_Metadata.csv}. Images with has_defect false go to the
    normal pool, split into train/val per roll with assign_val(val_frac, min_val_tiles=200, max_frac=0.40);
    images with has_defect true go to test. Each parent is cut by the tiler into its grid (28 tiles for
    1792x1024). Tile label and defect_px come from the parent's mask (label 1 iff defect_px > 0);
    normal-pool tiles are label 0 and have no mask."""

def compare_with_patch_metadata(rows: Sequence[TileRow], patches_csv: Path) -> list[str]:
    """Compare a RAW-FABRID manifest with RAW_FABRID_Patches_Metadata.csv
    (patch_name, parent_raw, type, defect_area). Returns human-readable problems, empty if consistent:
    per parent (file stem), number of tiles and the sorted list of nonzero defect areas must equal the CSV's;
    pool counts (train+val, test good, test defect) must equal the CSV's counts per `type`."""
```

```python
# loomguard/data/dataset.py
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def preprocess(img: PIL.Image.Image) -> Tensor:
    """Any PIL mode that converts to RGB (grayscale is repeated to 3 channels).
    Returns 3 x H x W float32: /255, then ImageNet normalisation."""

class TileDataset(torch.utils.data.Dataset):
    def __init__(self, rows: Sequence[TileRow], root: Path | str): ...
    def __len__(self) -> int: ...
    def __getitem__(self, i: int) -> tuple[Tensor, Tensor, int]:
        """(image 3 x 256 x 256 float32 normalised, mask 1 x 256 x 256 float32 in {0, 1}, label int).
        Crops the box (x0, y0, w, h) out of the parent, resizes if row.resize (image bilinear,
        mask nearest). A row without a mask gives an all-zero mask."""
```

### 4.3 Behaviour and edge cases

- **Tiler.** For 1792x1024 the origins are 7 x 4 at multiples of 256 (28 tiles); 512x512 gives 2 x 2; 4096x256 gives 16 x 1; 800x640 gives `x = [0, 181, 363, 544]`, `y = [0, 192, 384]` (12 tiles, overlap 75 and 64 px, matching ARCHITECTURE section 3). Rounding is round-half-up in integer arithmetic, so it does not depend on Python's banker's rounding.
- **Stitch is exact.** Accumulation is in float64, so for float32 tiles cut from one image `stitch(tile(x)) == x` bit for bit (`torch.equal`), including overlaps, because each averaged value is `k*v/k` computed in float64 and rounded back to float32. Integer tensors also round-trip exactly (tile slicing only; reflect padding needs a floating dtype).
- **Padding.** Axis shorter than 256: reflect pad bottom/right, origin 0, stitch crops back to the original `size`. Reflect padding needs pad < axis length; otherwise `torch` raises, which is acceptable.
- **`min_overlap`** (default 0) is the calibration knob from ARCHITECTURE section 3. It must be below `tile_size`.
- **Split by parent first.** Both builders assign a split to each parent image, then emit its tiles with the parent's split. There is no code path that splits tiles. Splits are per fabric (stratum), so per-roll bundles and the unified bundle share one assignment.
- **Normal-only train and val.** Rows with split `train` or `val` always have label 0 and an empty mask. Test rows come only from the dataset's official test side: RAW-FABRID defective images (per `has_defect` in the high-res metadata CSV, which is the source of truth for labels, DATASETS "Changes from the starter file"), MVTec `test/`.
- **RAW-FABRID details.** Fabric = `roll_id` column as given. A defective parent without a mask file raises `FileNotFoundError`. A mask on disk for an image the CSV calls defect-free is ignored (204 masks versus 202 defective images in the CSV). Mask pixels count as defect if `> 0`. `has_defect` is parsed case-insensitively; `1`, `true`, `yes` are true. Image size is read from the file header, not assumed.
- **MVTec details.** Image size is read from the file; masks are looked up as `ground_truth/<type>/<stem>_mask.png` and a missing one raises `FileNotFoundError`. Parent id for `train/good` images is the relative path, so no image is in two splits by construction. `defect_px` is 0 and `label` comes from the folder (MVTec detection is per image).
- **Val-size rule (D-30), applied per fabric.** With 15% of n parents, at least 200 tiles and at most 40%. The cap wins over the minimum, which is how small rolls end up with a "low-confidence calibration" later (fewer than 500 val tiles, a Phase 4 flag). Worked examples with 28 tiles per parent: roll with 13 normal images gives `round(1.95) = 2`, raised to 8, capped at `floor(5.2) = 5`; AITEX fabric with 20 normal images of 16 tiles gives 3, raised to 13, capped at 8 (the 12/8 split in EVALUATION section 4.2). For RAW-FABRID this gives 83 val images (2,324 tiles) and 424 train images (11,872 tiles), not the "about 76 / 2,128 / 431 / 12,068" of EVALUATION, which came from pure 15% per roll without the 200-tile minimum. EVALUATION is corrected in the doc sync (decision A-01).
- **Determinism.** Same inputs and seed give the same manifest, byte for byte. Rows are ordered by (fabric, split, parent, y0, x0). Each parent's tiles are consecutive, which is what the dataset's one-image cache relies on.
- **Manifest paths** are relative to the dataset root and use `/`, so one manifest works on both machines. The root is given to `TileDataset`.
- **No flips, rotations or other augmentation** in the dataset (D-22).
- **Dataset cache.** `TileDataset` keeps the last decoded parent image, so consecutive tiles of one parent do not decode a 1792x1024 PNG again. Single-entry, per worker.
- **`verify_datasets.py`.** Read-only. Each section is a function returning printable lines, and one failing dataset does not stop the others. It never writes into the dataset directories. Only for the training PC, but its logic runs on synthetic trees in tests.

### 4.4 Configuration

No config files in this phase. Builder arguments are the defaults of section 4.2 (`val_frac` 0.15 for RAW-FABRID, 0.10 for MVTec, `seed` 0, `min_overlap` 0). `configs/` stays empty until Phase 2.

### 4.5 Dependencies

None added. Uses `torch`, `numpy`, `pillow` (already declared or installed). CSV and JSON use the standard library. No pandas, no scikit-learn.

## 5. Tasks

### Laptop (developer session)

1. Tick plan task 1 (skeleton) with its evidence. Already done, nothing to build.
2. `tiling.py` and `tests/test_tiling.py`.
3. `manifest.py` (row type, CSV I/O, `assign_val`, `select`, both builders, `compare_with_patch_metadata`) and `tests/test_manifest.py` on synthetic trees written under `tmp_path`.
4. `dataset.py` and `tests/test_dataset.py`.
5. `scripts/verify_datasets.py` and `tests/test_verify_datasets.py`.
6. Full `pytest` run; commit `phase 1: implementation`.
7. Review loop, doc sync, commit `phase 1: review fixes and doc sync`.

### Training PC (operator; not done by the developer session)

8. Environment: PyTorch 2.7 or newer with CUDA 12.8 wheels; check `sm_120` in `torch.cuda.get_arch_list()` (plan task 5).
9. Download RAW-FABRID, ZJU-Leaper, MVTec AD, AITEX (plan task 6).
10. Run `scripts/verify_datasets.py`, paste its output into `PROGRESS_LOG.md`, resolve or correct every "unverified" item in `DATASETS.md` (plan task 6).
11. Look at the seven RAW-FABRID rolls and record whether they are different fabrics (plan task 7).

## 6. Acceptance criteria

The first six rows are the plan's criteria; their meaning is unchanged, only made precise. A7 to A10 are added.

| # | Criterion | Where | Verified by | Met |
|---|---|---|---|---|
| A1 | `stitch(tile(x)) == x` exactly for sizes 1792x1024, 512x512, 4096x256 and 800x640 (unit test) | Laptop | `pytest tests/test_tiling.py::test_roundtrip_exact` (float32, `torch.equal`, all four sizes, also with `min_overlap=32`) | [ ] |
| A2 | Unit test proves no parent image appears in more than one split, for every manifest builder | Laptop | `pytest tests/test_manifest.py::test_no_parent_in_two_splits` (parametrised over `build_mvtec` and `build_raw_fabrid`, several seeds) | [ ] |
| A3 | RAW-FABRID manifest reproduces the official numbers: 14,196 train-pool tiles, 4,969 test-good, 687 test-defect, and every tile's label matches `RAW_FABRID_Patches_Metadata.csv` | Training PC (logic tested on the laptop) | Laptop: `pytest tests/test_manifest.py -k "raw_fabrid or compare"` on a synthetic dataset with a synthetic patch CSV, including a deliberately wrong CSV that must be reported. Training PC: runbook step 5, counts and an empty problem list recorded | [ ] |
| A4 | `torch.cuda.get_arch_list()` on the training PC contains `sm_120` | Training PC | runbook step 1 | [ ] |
| A5 | `verify_datasets.py` output is pasted into `PROGRESS_LOG.md`. Every "unverified" item in DATASETS.md is resolved or corrected | Training PC (script tested on the laptop) | Laptop: `pytest tests/test_verify_datasets.py`. Training PC: runbook steps 4 to 7 | [ ] |
| A6 | `pytest` passes on the laptop with no dataset present | Laptop | `pytest` in the project venv, full suite | [ ] |
| A7 | Tile geometry matches ARCHITECTURE section 3: 28, 4, 16 and 12 tiles for the four sizes, origins as in 4.3, last tile ends at the edge, axis shorter than 256 is reflect-padded and stitched back to its original size | Laptop | `pytest tests/test_tiling.py` | [ ] |
| A8 | Overlapping tiles are averaged: a canvas where two tiles disagree in their overlap stitches to their mean there | Laptop | `pytest tests/test_tiling.py::test_stitch_averages_overlap` | [ ] |
| A9 | Val-size rule of D-30 holds per fabric (15%, at least 200 tiles, at most 40%, train never empty), the split is deterministic for a seed, train and val rows are label 0 with no mask, test rows come only from the dataset's test side | Laptop | `pytest tests/test_manifest.py` (`test_assign_val_*`, `test_train_val_are_normal_only`, `test_deterministic`) | [ ] |
| A10 | `TileDataset` returns exactly the tile `tile()` would cut, gray to 3 channels, ImageNet-normalised, mask `1 x 256 x 256` in {0,1}, and resizes MVTec rows to 256 | Laptop | `pytest tests/test_dataset.py` | [ ] |

## 7. Test plan

All CPU, synthetic, no dataset. Image trees are written to `tmp_path` with Pillow (small images such as 512x256 for RAW-FABRID-like data).

| Test | Proves |
|---|---|
| `test_tiling.py::test_origins` | Origins for 1792, 1024, 512, 4096, 256, 800, 640 and `min_overlap=32`; last tile ends at the edge; short axis gives `[0]` |
| `test_tiling.py::test_tile_counts_and_order` | 28 / 4 / 16 / 12 tiles, row-major order, shape `N x C x 256 x 256` |
| `test_tiling.py::test_roundtrip_exact` | A1 |
| `test_tiling.py::test_roundtrip_short_axis` | Reflect pad then stitch returns the original size and values |
| `test_tiling.py::test_stitch_averages_overlap` | A8; also a 1-channel stack (heatmap) |
| `test_manifest.py::test_assign_val_rule` | D-30 arithmetic on the worked examples (13 parents of 28 tiles gives 5; 20 of 16 gives 8; 1 parent gives 0; 167 gives 25) |
| `test_manifest.py::test_no_parent_in_two_splits` | A2, checked independently of the builders' own logic by grouping rows by parent |
| `test_manifest.py::test_train_val_are_normal_only`, `test_deterministic`, `test_csv_roundtrip` | A9 and manifest I/O |
| `test_manifest.py::test_raw_fabrid_labels_from_mask` | Tile labels and `defect_px` equal a hand count on a mask with defects in known tiles; defect-free images yield only train/val; defective yield only test |
| `test_manifest.py::test_raw_fabrid_csv_is_truth` | A mask for a CSV-defect-free image is ignored; a defective image without mask raises |
| `test_manifest.py::test_mvtec_builder` | Official test side stays test, `test/good` label 0, others label 1, mask paired, whole-image row with `resize=256` |
| `test_manifest.py::test_compare_with_patch_metadata` | A3 logic: consistent CSV gives no problems; a changed area, a missing patch and a wrong type each give a problem |
| `test_dataset.py::*` | A10: crop equals the tiler's tile; gray to 3 channels; normalisation values; mask; resize; DataLoader collation |
| `test_verify_datasets.py::*` | Script sections run on tiny synthetic RAW-FABRID, MVTec, ZJU-Leaper and AITEX trees, report counts, sizes and unpaired masks, and a missing dataset is reported without a crash |

## 8. Training-PC runbook

```bash
# 0. Setup (CLAUDE.md "Setup"); pull main
git pull
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[dev]"
pytest                                          # must pass with no dataset present

# 1. A4
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_arch_list())"
#    arch list must contain sm_120

# 2. Downloads into data/ (git-ignored); see DATASETS.md "Download checklist"
#    RAW-FABRID: Mendeley DOI 10.17632/db6g85xsyg.1 ; sha256sum RAW_FABRID.zip must equal
#      f6b4f82cce2f0220bbab3968a74b5337ebf2bbfd5e33d0548d34e1ebf80f85b9
#    ZJU-Leaper: gdown 1zzoU3Aj21V62GWS8LBm217TsE5uFeMVB ; 7z x ZJU-Leaper.7z
#    MVTec AD: form on mvtec.com ; extract carpet grid leather tile wood
#    AITEX: three 7z files from https://www.aitex.es/afid/ ; extract each
#    Suggested roots: data/RAW_FABRID  data/ZJU-Leaper  data/mvtec_ad  data/aitex

# 3. Verify and write manifests (A3, A5). Takes a few minutes; read-only on the datasets.
python scripts/verify_datasets.py \
  --raw-fabrid data/RAW_FABRID --zju data/ZJU-Leaper --mvtec data/mvtec_ad --aitex data/aitex \
  --manifest-dir artifacts/manifests | tee artifacts/verify_datasets.txt

# 4. Paste artifacts/verify_datasets.txt into docs/PROGRESS_LOG.md (A5)

# 5. A3: the output has a "RAW-FABRID manifest" block. Record: train-pool tiles (14196 expected),
#    test-good (4969), test-defect (687), and the list of problems from
#    compare_with_patch_metadata (must be empty). If the patch CSV `type` values or filename columns
#    differ from the assumptions in section 4.3, the script says so; fix manifest.py and its tests.

# 6. Resolve the "unverified" items of DATASETS.md using the script output: ZJU-Leaper counts per
#    pattern and split, image format, mask for every defective test image; MVTec archive size;
#    AITEX mask-to-image pairing and border tiles; RAW-FABRID masks versus CSV defective count.
#    Edit DATASETS.md where reality differs.

# 7. Rolls: view one image per roll and note whether the seven rolls are different fabrics
python - <<'EOF'
import csv
seen = {}
for r in csv.DictReader(open("data/RAW_FABRID/RAW_FABRID_HighRes_Metadata.csv")):
    seen.setdefault(r["roll_id"], r["filename"])
for k, v in sorted(seen.items()): print(k, "data/RAW_FABRID/images/" + v)
EOF
#    Record the verdict in PROGRESS_LOG.md and DATASETS.md (decides whether leave-one-roll-out is an
#    unseen-fabric test).

# 8. Commit results (`phase 1: results`): PROGRESS_LOG.md, DATASETS.md, ticked criteria.
```

| Step | Expected | Record in |
|---|---|---|
| 1 | `sm_120` in the arch list | `PROGRESS_LOG.md` |
| 3 to 5 | RAW-FABRID: 14,196 train-pool tiles, 4,969 test-good, 687 test-defect, no problems from the patch-metadata comparison | `PROGRESS_LOG.md` |
| 6 | Each "unverified" item resolved or corrected | `DATASETS.md` |
| 7 | One sentence per roll, and a verdict | `PROGRESS_LOG.md`, `DATASETS.md` |

## 9. Documents to update

- `docs/EVALUATION.md` section 4.2: RAW-FABRID val and train image and tile counts (83 / 2,324 and 424 / 11,872 with the D-30 200-tile minimum applied per roll), and the note on the manifest columns where relevant.
- `docs/ARCHITECTURE.md`: manifest columns and the tiler API only if they differ from what sections 3 and 12 already say; add the `preprocess` home (`loomguard/data/dataset.py`).
- `docs/DECISIONS.md`: "Awaiting confirmation" rows (see section 10).
- `README.md` "Current state", `CLAUDE.md` only if layout or rules text became inaccurate.

## 10. Risks and open points

| Risk or open point | Handling |
|---|---|
| Names of the CSV columns and values of `has_defect`, `roll_id` and `type` are known from DATASETS.md but their exact spelling was not seen by us | The builder parses tolerantly and fails loudly with the column names it found; the runbook says to fix it on the PC. Tests use the documented column names |
| How official patch names encode grid position is not documented | `compare_with_patch_metadata` compares per parent (tile count and sorted defect-area list), which does not need the position. If the training PC finds the naming scheme, it may add an exact per-position check |
| A-01: D-30's 200-tile minimum is applied per fabric profile, which changes RAW-FABRID counts from the "about" numbers in EVALUATION | Recorded in DECISIONS "Awaiting confirmation"; EVALUATION corrected. Easy to reverse (`min_val_tiles=0`) |
| A-02: manifest schema (13 columns, relative POSIX paths, one row per tile, MVTec as a whole-image row with `resize=256`) | Documents did not fix the columns. Recorded in DECISIONS "Awaiting confirmation". Phases 2 and 3 read it through `read_manifest`, so a change stays in one file |
| Averaging overlaps at 3 or more copies is exact only because of float64 accumulation | Tested for 800x640 (2x2 overlap) and `min_overlap=32` |
| ZJU-Leaper and AITEX layouts in the verification script are from the official loader and from DATASETS.md, not from files we have seen | Script reports what it cannot find instead of crashing; the operator corrects it |

Nothing in this spec depends on an item under "Open" in `DECISIONS.md` (O-01, O-02).

## 11. Completion record

| | |
|---|---|
| Spec commit | |
| Implementation commit | |
| Review commit | |
| Review rounds and verdict | |
| Tests | |
| Results commit (training PC) | |
| Decisions made | |
| Deviations from this spec | |
| Closed by owner | |
