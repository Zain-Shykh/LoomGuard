# Phase 2: Metrics and PatchCore

| | |
|---|---|
| Phase | 2 |
| Status | Code complete (training-PC runbook outstanding) |
| Spec written | 2026-11-13 |
| Written by | developer session, phase 2 |
| Depends on | Phase 1 (Code complete; its training-PC runbook is queued and does not block this phase, decision D-58) |
| Plan entry | `docs/IMPLEMENTATION_PLAN.md`, Phase 2 |

## 1. Objective

A tested metrics module, a frozen WideResNet-50-2 backbone wrapper and a PatchCore implementation, plus `scripts/fit.py` and `scripts/evaluate.py` that run from one YAML config and work end to end on a tiny synthetic dataset on CPU. At the end the training-PC operator has configs and a runbook to reproduce the published MVTec AD texture numbers and to produce the first RAW-FABRID unified results.

## 2. Background

Metrics come first so that every later number can be trusted. PatchCore needs no gradient training: a pass over normal tiles fills a memory bank, a greedy k-center coreset thins it, and a test tile is scored by nearest-neighbour distance. Phase 1's data layer (tiler, manifests, `TileDataset`) supplies the tiles.

Read:

- `docs/ARCHITECTURE.md` section 4 (backbone), section 5 (PatchCore: 5.1 bank, 5.2 scoring, simplifications), section 10 (bundle), section 12 (layout)
- `docs/EVALUATION.md` section 2 (metrics), section 3 (reference numbers and pass criteria), section 4 (splits), section 5 (result files)
- Decisions: D-02 (own implementations), D-20 (backbone), D-21 (native 256 tiles, MVTec resized), D-23 (PatchCore settings), D-30, D-31, D-33, D-35 (no tuning on test), A-01 to A-05 (Phase 1 manifest and split choices, unchanged)

## 3. Scope

**In scope**

- `loomguard/eval/metrics.py`: AUROC, AP, pixel AUROC, pixel AP, AUPRO at FPR 0.3, histogram-based versions for large test sets, 8-connected component labelling (no scipy available).
- `loomguard/models/backbone.py`: frozen WideResNet-50-2 returning layer1, layer2, layer3; the shared Gaussian-blur helper for heatmaps.
- `loomguard/models/patchcore.py`: pooling, concatenation, random projection, greedy k-center coreset, nearest-neighbour scoring, heatmap, bank save and load.
- `loomguard/runs.py`: plumbing shared by the two scripts (config, manifest dispatch, fabric profiles, tile cap, paths).
- `scripts/fit.py`, `scripts/evaluate.py`, `configs/mvtec_patchcore.yaml`, `configs/raw_fabrid_patchcore.yaml`.
- CPU tests, including a synthetic end-to-end fit and evaluate run.
- The training-PC runbook (section 8).

**Out of scope**

- Calibration (`tau`), operating-point metrics (false alarms, Dice, IoU), grading, `Inspector`, bundle `calibration.json` (Phase 4). Until Phase 4, `scripts/evaluate.py` is the one place that scores test tiles; Phase 4 moves that flow behind `Inspector` (CLAUDE.md rule 5).
- Reverse Distillation (Phase 3), ZJU-Leaper and AITEX adapters (Phase 5), the per-roll breakdown of a unified run (Phase 5 summary), FAISS and the paper's adaptive 1,024-d pooling (only if the MVTec check fails).
- Any dataset download, training, GPU use or pretrained-weight download on the laptop.

## 4. Design

### 4.1 Files

| File | New or changed | Purpose |
|---|---|---|
| `loomguard/eval/metrics.py` | new | metrics |
| `loomguard/models/backbone.py` | new | `Backbone`, `gaussian_blur` |
| `loomguard/models/patchcore.py` | new | `PatchCore`, `greedy_coreset` |
| `loomguard/runs.py` | new | shared plumbing for the scripts (addition to the ARCHITECTURE layout, decision A-07) |
| `scripts/fit.py`, `scripts/evaluate.py` | new | driven by `--config` |
| `configs/mvtec_patchcore.yaml`, `configs/raw_fabrid_patchcore.yaml` | new | the two training-PC runs |
| `tests/test_metrics.py`, `tests/test_backbone.py`, `tests/test_patchcore.py`, `tests/test_runs.py`, `tests/test_fit_evaluate.py` | new | CPU tests |
| `docs/specs/phase-2-metrics-patchcore.md` | new | this spec |
| `docs/IMPLEMENTATION_PLAN.md`, `PROGRESS_LOG.md`, `DECISIONS.md`, `EVALUATION.md`, `ARCHITECTURE.md`, `README.md` | changed | status, evidence, decisions, corrected text |

No change to `pyproject.toml` (numpy, torch, torchvision, pillow, pyyaml already present; torchvision is installed with torch as the setup says).

### 4.2 Interfaces

Conventions as in Phase 1: image tensors `C x H x W`, batches `B x C x H x W`.

```python
# loomguard/eval/metrics.py
def roc_auc(scores: ArrayLike, labels: ArrayLike) -> float:
    """Exact. scores float, labels bool/0-1, same length. Ties count one half. nan if one class is empty."""
def average_precision(scores: ArrayLike, labels: ArrayLike) -> float:
    """Exact, step-wise definition (sklearn's): sum over distinct thresholds of (R_k - R_{k-1}) * P_k."""
def aupro(heatmaps: ArrayLike, masks: ArrayLike, fpr_max: float = 0.3) -> float:
    """Exact. heatmaps float N x H x W, masks bool N x H x W (ground truth). Area under PRO against pixel FPR
    up to fpr_max, divided by fpr_max."""
def connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """mask: bool H x W. 8-connectivity. Returns int32 labels H x W (0 = background, 1..n) and n."""

class ScoreCounts:
    """Streaming histogram counts for pixel metrics on large test sets."""
    def __init__(self, bins: int = 10_000): ...
    def add(self, idx: np.ndarray, mask: np.ndarray, regions: bool = False) -> None:
        """idx: int bin index per pixel, any shape; mask: bool same shape, True = defect pixel.
        regions=True also labels the connected components of mask (2-D only) for AUPRO."""
    def auroc(self) -> float: ...
    def ap(self) -> float: ...
    def aupro(self, fpr_max: float = 0.3) -> float: ...

def quantize(x: Tensor, lo: float, hi: float, bins: int = 10_000) -> np.ndarray:
    """x float tensor, any shape. Returns int64 numpy bin index in [0, bins): floor((x - lo) / (hi - lo) * bins), clamped."""
```

The exact functions are thin wrappers: map scores to ranks with `np.unique(return_inverse=True)` and push them through the same `ScoreCounts`, with one bin per distinct value, so exact and histogram versions share one implementation of the curves.

```python
# loomguard/models/backbone.py
class Backbone(nn.Module):
    def __init__(self, weights: str | None = "IMAGENET1K_V1"):
        """torchvision wide_resnet50_2 up to layer3. `weights` is a torchvision weights-enum member name
        (e.g. "IMAGENET1K_V1", loaded with the usual torchvision download) or None for random initialisation
        (tests). All parameters requires_grad=False; the module stays in eval() even after .train()."""
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """x: B x 3 x 256 x 256 float32, ImageNet-normalised. No gradients are tracked.
        Returns layer1 B x 256 x 64 x 64, layer2 B x 512 x 32 x 32, layer3 B x 1024 x 16 x 16."""

def gaussian_blur(x: Tensor, sigma: float = 4.0) -> Tensor:
    """x: B x 1 x H x W. Separable Gaussian, radius int(4 * sigma + 0.5), reflect padding. Same shape out."""
```

```python
# loomguard/models/patchcore.py
def greedy_coreset(feats: Tensor, k: int, proj_dim: int | None = 128, seed: int = 0,
                   device: str | torch.device = "cpu") -> Tensor:
    """feats: N x D (any float dtype, may be on CPU). Greedy k-center on a seeded Gaussian random projection
    to proj_dim (None or >= D: no projection). First centre chosen at random from the seed. Returns LongTensor
    of min(k, N) distinct row indices (selection order). Selection runs on `device`."""

class PatchCore:
    def __init__(self, backbone: Backbone, coreset_ratio: float = 0.10, proj_dim: int = 128,
                 seed: int = 0, device: str | torch.device = "cpu"): ...
    bank: Tensor | None          # M x 1536 float16, M = max(1, round(ratio * N))
    def embed(self, x: Tensor) -> Tensor:
        """x: B x 3 x 256 x 256. layer2 and layer3 each avg-pooled 3x3 stride 1 pad 1; layer3 bilinearly
        upsampled to 32 x 32; channel concat. Returns B x 1536 x 32 x 32 float32."""
    def fit(self, batches: Iterable[Tensor]) -> dict:
        """batches of B x 3 x 256 x 256 normal tiles. Collects all patch vectors (N = tiles * 1024, stored
        as float16 on the CPU), runs greedy_coreset, keeps the selected vectors as `bank`.
        Returns {"n_tiles", "n_patches", "bank_size", "coreset_seconds"}."""
    def score(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """x: B x 3 x 256 x 256. Returns (tile score B float32, heatmap B x 1 x 256 x 256 float32).
        Grid = Euclidean distance of each of the 1024 query vectors to its nearest bank vector
        (torch.cdist, queries processed in chunks), B x 1 x 32 x 32. Tile score = max over the grid.
        Heatmap = gaussian_blur(bilinear upsample of the grid to 256 x 256, sigma 4)."""
    def save(self, path: Path) -> None:      # torch.save(bank) : a plain float16 tensor
    @classmethod
    def load(cls, path: Path, backbone: Backbone, device="cpu", **kw) -> "PatchCore":   # torch.load(weights_only=True)
```

```python
# loomguard/runs.py   (plumbing for scripts/fit.py and scripts/evaluate.py)
def load_config(path: Path | str) -> dict:           # yaml.safe_load; fills the defaults of 4.4; checks required keys
def build_rows(cfg: dict) -> list[TileRow]:          # dataset "mvtec" -> build_mvtec(root, fabrics or textures, seed=seed);
                                                     # "raw_fabrid" -> build_raw_fabrid(root, seed=seed), then filtered to cfg fabrics if given
def profiles(rows: list[TileRow], cfg: dict) -> dict[str, list[TileRow]]:
    """regime "per_fabric": one profile per fabric, named by the fabric. regime "unified": {"unified": all rows}."""
def subsample(rows: list[TileRow], max_tiles: int, seed: int) -> list[TileRow]:
    """Tile cap, stratified by parent image (ARCHITECTURE 5.1): if len(rows) <= max_tiles return rows unchanged;
    else repeatedly take one random not-yet-taken tile from each parent (parents in random order) until
    max_tiles are chosen. Result keeps manifest order (consecutive tiles of a parent stay consecutive)."""
def bundle_dir(cfg: dict, profile: str) -> Path:     # <artifacts>/<dataset>/<profile>/<method>
def result_path(cfg: dict, profile: str) -> Path:    # <results>/<dataset>/<regime>/<method>/<profile>.json
def git_commit() -> str:                             # short hash, "unknown" if git is unavailable
def pick_device(cfg: dict) -> torch.device:          # "auto" -> cuda if available else cpu
```

```python
# scripts/fit.py      python scripts/fit.py --config configs/X.yaml
def main(argv: list[str] | None = None) -> None
# scripts/evaluate.py python scripts/evaluate.py --config configs/X.yaml
def main(argv: list[str] | None = None) -> dict[str, dict]     # profile -> result dict
```

### 4.3 Behaviour and edge cases

**Metrics**

- *AUROC and AP*: exact, from cumulative counts at distinct thresholds (flagged means score `>= t`). AUROC is the trapezoid area of the (FPR, TPR) curve from (0, 0), so ties count one half (equals Mann-Whitney). AP is the sklearn step-wise definition.
- *AUPRO* (EVALUATION 2.1): the ground-truth regions are the 8-connected components of each image's mask (the MVTec convention). For each threshold `t` (flagged means score `>= t`): `PRO(t)` is the mean over all regions of all images of the fraction of the region's pixels flagged; `FPR(t)` is the fraction of all non-defect pixels of all images flagged. The curve starts at (0, 0), the point where nothing is flagged. It is integrated as a **step function**: the value at false-positive rate `f` is the largest `PRO(t)` with `FPR(t) <= f`, so the area up to `fpr_max` is `sum_k PRO_k * (min(FPR_{k+1}, F) - min(FPR_k, F))`, divided by `F`. This makes a perfect heatmap score exactly 1.0 and a constant heatmap exactly 0 (a constant map can flag everything or nothing but never a fraction; linear interpolation between those two points would credit it 0.15, the value of a random map). It differs from linear interpolation only by the width of tied score steps, which vanishes for continuous scores. No GT region at all gives `nan`. Decision A-06.
- *Histogram versions*: `quantize` maps scores to bins over `[lo, hi]` and `ScoreCounts` accumulates per-bin counts of defect and normal pixels and, with `regions=True`, the running sum over regions of the fraction of region pixels at or above each bin, so no per-region state is kept. With 10,000 bins the pixel AUROC, AP and AUPRO differ from the exact values by far less than 0.001 on continuous scores; they equal them exactly when each distinct score has its own bin.
- *Connected components*: run-length based union-find on rows, vectorised run detection, no scipy. 8-connected: runs on adjacent rows are joined if their column extents touch or overlap including diagonally.
- Pixel and region metrics need the heatmap range. `evaluate.py` therefore scores the test tiles twice: pass 1 gets the global minimum and maximum of all test heatmaps (and the tile scores and timing); pass 2 fills the histograms. The extra GPU time is small and nothing is clipped. The range comes from the test heatmaps' own extent only to place bins; it does not influence any threshold, checkpoint or hyperparameter (the headline metrics are threshold-free).

**Backbone**

- `forward` runs under `torch.no_grad()`; `train(True)` is overridden to keep eval mode (CLAUDE.md rule 4); `layer4`, the pool and `fc` are dropped. The used part has 24.86 M parameters (ARCHITECTURE section 4).
- `weights` is looked up as `torchvision.models.Wide_ResNet50_2_Weights[name]`; an unknown name raises `KeyError`. The default is the real ImageNet weights, so forgetting the argument cannot silently produce random features; laptop tests always pass `None`.

**PatchCore**

- Local averaging, concatenation, bank size, float16 storage, projection and exact search as ARCHITECTURE 5.1 and 5.2. The tile score is the max of the *unblurred* grid, as ARCHITECTURE 5.2 states; the heatmap is the blurred upsampled grid.
- Coreset: `greedy_coreset` keeps a running minimum squared distance to the chosen set in the projected space (one matrix-vector product per iteration) and picks the farthest point next. The projection matrix is `N(0, 1/proj_dim)` from a `torch.Generator` seeded with `seed`. The stored bank is the original 1,536-d vectors, not the projections.
- Memory: all patches of up to 600 tiles are held as float16 on the CPU (614,400 x 1,536 x 2 B = 1.9 GB); only the 128-d projection (315 MB float32) and per-iteration temporaries go to the device. The cast of the bank to float32 for scoring is cached on first use.
- Scoring chunks queries (2,048 rows at a time) so the distance matrix against a 61,000-row bank stays near 0.5 GB.
- `fit` consumes the iterable once and never sees labels or masks.

**Scripts**

- `fit.py`: builds the manifest (deterministic from `root` and `seed`), writes it to `<artifacts>/manifests/<dataset>.csv`, and for every profile takes the `train` rows only, applies `subsample(..., max_tiles, seed)`, loads them with `TileDataset` (`shuffle=False`, so a parent's tiles stay consecutive), fits PatchCore and writes the bundle directory `bundle_dir(cfg, profile)`: `model.pt`, `config.yaml` (the resolved config plus `profile`, `git_commit`), `metrics.json` (fit-side system metrics: `n_train_tiles`, `bank_size`, `fit_minutes`, `coreset_seconds`, `peak_vram_gb` (0 on CPU), `peak_ram_gb` (process peak RSS), `bundle_mb`). `calibration.json` is Phase 4. It prints one line per profile.
- `evaluate.py`: rebuilds the manifest, and for every profile loads the bundle (`torch.load(weights_only=True)` via `PatchCore.load`) and scores only the profile's `test` rows. Tiles are grouped by parent (rows are consecutive per parent, Phase 1), the tile heatmaps and tile masks are stitched back to the full parent image with `tiling.stitch` (a mask stitched from identical copies is exact), and pixel metrics are computed on the full image, so defects that straddle tile borders are one region (MVTec: the single 256 x 256 resized image). Tile metrics use the per-tile scores and labels (EVALUATION 2.1: a tile is defective if its mask has a defect pixel; the manifest `label` says so). It writes `results/<dataset>/<regime>/<method>/<profile>.json` and merges the test metrics into the bundle's `metrics.json`. With more than one profile it prints the mean table (profiles whose metric is undefined are skipped), and if the config has `targets` it prints PASS or FAIL per target (D-33); targets are reported, never used for any decision.
- Result JSON keys (a subset of EVALUATION 5; the operating-point keys `tau`, `false_alarms_per_1000`, `tile_recall`, `dice`, `iou` arrive in Phase 4): `dataset, regime, fabric, method, git_commit, seed, config, n_train_tiles, n_val_tiles, n_test_good, n_test_defect, tile_auroc, tile_ap, aupro_30, pixel_auroc, pixel_ap, ms_per_tile_<cuda|cpu>, peak_vram_gb, peak_ram_gb, fit_minutes, coreset_seconds, bundle_mb, coreset_ratio, bank_size, hist_bins, device`. Metrics are fractions in [0, 1]; `targets` in the config are in percent.
- `ms_per_tile_*` is the wall time of `PatchCore.score` over pass 1 divided by the number of tiles, with `torch.cuda.synchronize()` around it on CUDA.
- Rule 2 (no test influence): `fit.py` reads no row whose split is not `train`; `evaluate.py` only reads `test` rows and chooses nothing from them.

### 4.4 Configuration

```yaml
dataset: mvtec                 # mvtec | raw_fabrid
root: data/mvtec_ad            # dataset root (git-ignored)
regime: per_fabric             # per_fabric | unified
fabrics: [carpet, grid, leather, tile, wood]   # optional; default: all fabrics of the dataset
method: patchcore
seed: 0
backbone_weights: IMAGENET1K_V1   # null = random weights (tests only)
patchcore:
  max_tiles: 600               # tile cap per bank (D-23)
  coreset_ratio: 0.10          # 0.10 for benchmarks, 0.01 for the CPU demo
  proj_dim: 128
  batch_size: 16
device: auto                   # auto | cpu | cuda
num_workers: 0
hist_bins: 10000
artifacts: artifacts
results: results
targets: {tile_auroc: 97.9, pixel_auroc: 96.6, aupro_30: 90.7}   # optional, percent, report only
```

Required keys: `dataset`, `root`, `regime`, `method` (only `patchcore` in this phase; anything else raises). Everything else has the default shown. `configs/raw_fabrid_patchcore.yaml` has `dataset: raw_fabrid`, `root: data/RAW_FABRID`, `regime: unified`, no `fabrics`, no `targets`.

### 4.5 Dependencies

None added. `torchvision` is installed together with `torch` (CLAUDE.md "Setup") and is already part of the environment; it is not declared in `pyproject.toml`, as for `torch` (D-51). No scikit-learn and no scipy (not installed, decision A-06): every metric is our own code, tested against hand-computed and brute-force values.

## 5. Tasks

### Laptop (developer session)

1. `metrics.py` and `tests/test_metrics.py` (plan task 1).
2. `backbone.py` and `tests/test_backbone.py` (plan task 2).
3. `patchcore.py` and `tests/test_patchcore.py` (plan task 3).
4. `runs.py`, `fit.py`, `evaluate.py`, the two configs, `tests/test_runs.py` and `tests/test_fit_evaluate.py` (plan task 4).
5. Full `pytest`; commit `phase 2: implementation`.
6. Review loop, doc sync, commit `phase 2: review fixes and doc sync`.

### Training PC (operator; not done by the developer session)

7. Environment as in the Phase 1 runbook; `pytest` passes.
8. MVTec AD textures: `fit.py` and `evaluate.py` with `configs/mvtec_patchcore.yaml`; compare with the targets (plan task 5, first half).
9. RAW-FABRID unified: `fit.py` and `evaluate.py` with `configs/raw_fabrid_patchcore.yaml`; record coreset time, peak RAM and VRAM (plan task 5, second half).
10. If the MVTec check misses its targets, follow the upgrade path of ARCHITECTURE 5.2 one item at a time (1,024-d pooling, re-weighted score, centre crop) and log each; that is a new decision, not part of this runbook.

## 6. Acceptance criteria

A1 to A5 are the plan's criteria with unchanged meaning. A6 onward are added.

| # | Criterion | Where | Verified by | Met |
|---|---|---|---|---|
| A1 | AUPRO matches a brute-force implementation to 1e-6 on toy masks. A perfect heatmap scores 1.0 and a constant heatmap scores about 0 (we make it exactly 0) | Laptop | `pytest tests/test_metrics.py::test_aupro_matches_brute_force tests/test_metrics.py::test_aupro_perfect_and_constant tests/test_metrics.py::test_aupro_hand_value` | [x] |
| A2 | Backbone unit test: for a 256x256 input the three outputs have shapes 256x64x64, 512x32x32, 1024x16x16, and no parameter requires gradients | Laptop | `pytest tests/test_backbone.py` | [x] |
| A3 | PatchCore on MVTec textures, mean over five categories: image AUROC 97.9 or more, pixel AUROC 96.6 or more, AUPRO 90.7 or more (EVALUATION 3) | Training PC (code, config and targets check tested on the laptop) | Laptop: `pytest tests/test_fit_evaluate.py`. Training PC: runbook step 3, the mean table printed by `evaluate.py` shows PASS for all three | [ ] |
| A4 | Coreset selection for 600 tiles finishes in under 10 minutes on the training PC. Peak RAM and VRAM recorded | Training PC (code tested on the laptop) | Laptop: `pytest tests/test_patchcore.py -k coreset`. Training PC: runbook step 5, `coreset_seconds`, `peak_ram_gb`, `peak_vram_gb` in the bundle `metrics.json` | [ ] |
| A5 | RAW-FABRID unified PatchCore results are in `results/` and summarised in `PROGRESS_LOG.md` | Training PC | runbook steps 4 to 6, `results/raw_fabrid/unified/patchcore/unified.json` committed | [ ] |
| A6 | AUROC and AP equal hand-computed values on toy scores, including ties, and a one-class input gives `nan` | Laptop | `pytest tests/test_metrics.py::test_roc_auc_and_ap_hand_values` | [x] |
| A7 | The histogram versions of pixel AUROC, AP and AUPRO equal the exact ones when each distinct score has its own bin and are within 1e-3 on 10,000 bins of random continuous scores | Laptop | `pytest tests/test_metrics.py::test_histogram_matches_exact` | [x] |
| A8 | Connected components are 8-connected and agree with a flood-fill reference on random masks | Laptop | `pytest tests/test_metrics.py::test_connected_components` | [x] |
| A9 | PatchCore parts are correct: `embed` is B x 1536 x 32 x 32; `greedy_coreset` equals a brute-force farthest-point reference and covers clusters; the bank is float16 with `max(1, round(ratio * N))` rows; scores equal a brute-force nearest-neighbour distance; tile score is the grid max; heatmap is B x 1 x 256 x 256 and peaks on an injected anomaly; a bank saved by `save` loads with `weights_only=True` and gives the same scores | Laptop | `pytest tests/test_patchcore.py` | [x] |
| A10 | The tile cap is stratified by parent image, deterministic for a seed, and keeps manifest order | Laptop | `pytest tests/test_runs.py::test_subsample` | [x] |
| A11 | `fit.py` then `evaluate.py` run end to end on tiny synthetic MVTec-like and RAW-FABRID-like datasets on CPU with a tiny coreset, produce the bundle files and a result JSON with all keys, and the reported tile and pixel metrics equal an independent recomputation (tile, score, stitch, exact metrics) | Laptop | `pytest tests/test_fit_evaluate.py` | [x] |
| A12 | `fit.py` reads only `train` rows and `evaluate.py` only `test` rows (CLAUDE.md rule 2) | Laptop | `pytest tests/test_fit_evaluate.py::test_fit_reads_only_train_rows` | [x] |
| A13 | The whole test suite passes on the laptop with no dataset present | Laptop | `pytest` in the project venv | [x] |

## 7. Test plan

All CPU, synthetic, no dataset, no downloaded weights (`Backbone(weights=None)` or a small fake backbone).

| Test | Proves |
|---|---|
| `test_metrics.py::test_roc_auc_and_ap_hand_values` | A6: `[0.1, 0.4, 0.35, 0.8]` vs `[0, 0, 1, 1]` gives AUROC 0.75 and AP 5/6; a tie case; one class gives nan |
| `test_metrics.py::test_aupro_hand_value` | A1: a hand-computed curve (derivation in the test) gives 0.175 / 0.3 |
| `test_metrics.py::test_aupro_matches_brute_force` | A1: random toy heatmaps and masks (several regions per image, several images, quantised and continuous scores): equals a loop-based brute force with its own flood fill, to 1e-6 |
| `test_metrics.py::test_aupro_perfect_and_constant` | A1: perfect map 1.0, constant map 0.0, inverted map near 0 |
| `test_metrics.py::test_histogram_matches_exact` | A7 |
| `test_metrics.py::test_connected_components` | A8: diagonal pixels join, a one-pixel gap does not, random masks agree with a flood fill |
| `test_backbone.py::test_shapes_and_frozen` | A2, plus 24.86 M parameters, eval mode survives `.train()`, no graph on the outputs |
| `test_backbone.py::test_unknown_weights_name` | A bad `weights` name fails loudly |
| `test_backbone.py::test_gaussian_blur` | Preserves a constant, sums to one, spreads an impulse symmetrically, same shape |
| `test_patchcore.py::test_embed_shape` | A9: with the real `Backbone(None)`, B x 1536 x 32 x 32 |
| `test_patchcore.py::test_coreset_*` | A4 and A9: equals brute-force farthest-point selection; picks one point per far-apart cluster; returns distinct indices; `k >= N` returns all; deterministic for a seed; projection variant keeps covering |
| `test_patchcore.py::test_bank_size_and_dtype`, `test_score_matches_bruteforce_and_tile_score_is_grid_max`, `test_heatmap_localises_anomaly_and_save_load` | A9, with a small fake backbone that gives the right shapes |
| `test_runs.py::test_subsample`, `test_profiles_and_paths`, `test_load_config` | A10; per-fabric and unified grouping; defaults and required keys |
| `test_fit_evaluate.py::test_mvtec_end_to_end`, `test_raw_fabrid_unified_end_to_end` | A11 (the RAW-FABRID tree has defects straddling a tile border, so stitched full-image regions are exercised) |
| `test_fit_evaluate.py::test_fit_reads_only_train_rows` | A12 |

## 8. Training-PC runbook

Prerequisite: the Phase 1 environment (PyTorch 2.7 or newer with CUDA 12.8 wheels, `sm_120`), the datasets downloaded as in the Phase 1 runbook, and the Phase 1 verification run done (so the manifest builders are confirmed against the real CSV columns). Real ImageNet weights download on first use (about 130 MB), which needs internet.

```bash
# 0. Update and test
git pull
source .venv/bin/activate
pytest                                           # must pass

# 1. Weights sanity: the real backbone loads and is frozen
python - <<'EOF'
from loomguard.models.backbone import Backbone
b = Backbone("IMAGENET1K_V1"); print(sum(p.numel() for p in b.parameters()) / 1e6, "M params,",
      any(p.requires_grad for p in b.parameters()), "requires grad")
EOF
# expected: 24.86 M params, False

# 2. MVTec AD textures, PatchCore 10 percent (roots as in the Phase 1 runbook: data/mvtec_ad)
python scripts/fit.py      --config configs/mvtec_patchcore.yaml | tee artifacts/phase2_mvtec_fit.txt
python scripts/evaluate.py --config configs/mvtec_patchcore.yaml | tee artifacts/phase2_mvtec_eval.txt

# 3. A3: the end of phase2_mvtec_eval.txt prints the mean over the five categories and PASS/FAIL against
#    97.9 / 96.6 / 90.7. Record the table in PROGRESS_LOG.md. Per-category files: results/mvtec/per_fabric/patchcore/*.json

# 4. RAW-FABRID unified, PatchCore 10 percent (data/RAW_FABRID)
python scripts/fit.py      --config configs/raw_fabrid_patchcore.yaml | tee artifacts/phase2_raw_fit.txt
python scripts/evaluate.py --config configs/raw_fabrid_patchcore.yaml | tee artifacts/phase2_raw_eval.txt

# 5. A4: the unified fit uses the 600-tile cap. Read coreset_seconds, peak_ram_gb, peak_vram_gb
cat artifacts/raw_fabrid/unified/patchcore/metrics.json

# 6. A5: results/raw_fabrid/unified/patchcore/unified.json ; summarise in PROGRESS_LOG.md
#    (tile AUROC, AP, AUPRO, pixel AUROC, pixel AP, latency, bank size)

# 7. Commit (phase 2: results): results/**.json, PROGRESS_LOG.md, ticked criteria in this spec and the plan.
#    If any target is missed: do not tune on these results. Log the miss, then see runbook note 10.
```

| Step | Expected | Record in |
|---|---|---|
| 1 | 24.86 M parameters, no `requires_grad` | `PROGRESS_LOG.md` |
| 2 to 3 | Mean over 5 textures: image AUROC at least 97.9, pixel AUROC at least 96.6, AUPRO at least 90.7 (percent) | `PROGRESS_LOG.md`, `results/mvtec/...` |
| 5 | `coreset_seconds` under 600 for 600 tiles; peak RAM and VRAM noted | `PROGRESS_LOG.md` |
| 6 | One result JSON for RAW-FABRID unified, summarised | `PROGRESS_LOG.md`, `results/raw_fabrid/...` |

## 9. Documents to update

- `docs/EVALUATION.md` section 2.1: metrics come from our own code, not `sklearn.metrics`; AUPRO step-integration convention and 8-connectivity; result JSON keys available in Phase 2.
- `docs/ARCHITECTURE.md`: layout gains `loomguard/runs.py`; the `Backbone` and `gaussian_blur` homes; PatchCore API names.
- `README.md` "Current state".
- `docs/DECISIONS.md`: "Awaiting confirmation" rows A-06 onward (see section 10).
- `CLAUDE.md` layout block (add `runs.py`).

## 10. Risks and open points

| Risk or open point | Handling |
|---|---|
| A-06: metrics are our own code (no scikit-learn, none installed); AUPRO integrates the PRO curve as a step function and uses 8-connectivity | ARCHITECTURE and EVALUATION said "from `sklearn.metrics`"; corrected. Tested against hand values and a brute force. Differences from linear interpolation vanish for continuous scores |
| A-07: new file `loomguard/runs.py`, config schema of 4.4, bundle and result paths, `metrics.json` holds fit-side system metrics until evaluation merges the test metrics | The documents fixed the bundle directory but not the plumbing. Recorded in DECISIONS |
| A-08: evaluation runs two scoring passes and builds pixel histograms over the stitched full parent images | Gives exact bin range and region-correct AUPRO at bounded memory |
| Tile score is the max of the unblurred grid (ARCHITECTURE 5.2) while the mask uses the blurred heatmap, so a tile can score above `tau` without any pixel above `tau` | Followed as written. Phase 4 must decide how the tile decision and the mask relate (D-32 says one `tau` for both); not decided here |
| The PatchCore MVTec targets may be missed because of the documented simplifications (256x256 instead of 224 crop, 1,536-d instead of 1,024-d, no density re-weighting) | The upgrade path is in ARCHITECTURE 5.2 and is a decision for after the run. No tuning on test results (D-35) |
| The real backbone and the real MVTec and RAW-FABRID files are not available on the laptop, so the numbers, timings and memory in A3 to A5 cannot be seen here | Runbook with exact commands; integration tests on synthetic trees with the real network architecture at random weights |
| Greedy k-center for 614,400 points and 61,440 centres is a Python loop of 61k GPU steps | Estimated 1 to 3 minutes; A4 measures it. If too slow the fallback is a larger `proj_dim` stride or fewer tiles, not in this phase |

Nothing in this spec depends on an item under "Open" in `DECISIONS.md` (O-01, O-02).

## 11. Completion record

| | |
|---|---|
| Spec commit | b39877e `phase 2: spec` |
| Implementation commit | 28ce03e `phase 2: implementation` (then a small `phase 2: docs in line before review` commit with the decision rows and doc corrections) |
| Review commit | `phase 2: review fixes and doc sync` (the last commit of the phase) |
| Review rounds and verdict | 2 rounds. Round 1: CHANGES REQUIRED (progress log entry and plan status not yet written, mean over undefined metrics; two minor). Round 2: PASS, two minor doc findings, fixed |
| Tests | `pytest -q`: 67 passed, 0 failed, no dataset present (metrics 8, backbone 3, patchcore 8, runs 4, fit/evaluate 3, Phase 1 and smoke 41) |
| Results commit (training PC) | pending: runbook steps 0 to 7, then `phase 2: results` |
| Decisions made | A-06 to A-09 (Awaiting confirmation) |
| Deviations from this spec | None in design. Test-only: with random backbone weights the tests seed the network identically in `fit`, `evaluate` and the independent recomputation (`seeded_backbone` in `tests/test_fit_evaluate.py`). `evaluate.py` skips undefined (nan) metrics in the printed mean. A3, A4 and A5 stay unticked until the training-PC run |
| Closed by owner | |
