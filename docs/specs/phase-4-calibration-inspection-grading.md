# Phase 4: Calibration, inspection and grading

| | |
|---|---|
| Phase | 4 |
| Status | Code complete (training-PC runbook outstanding) |
| Spec written | 2026-11-14 |
| Written by | developer session, phase 4 |
| Depends on | Phases 1 to 3 (Code complete; their training-PC runbooks are queued and do not block this phase, decision D-58) |
| Plan entry | `docs/IMPLEMENTATION_PLAN.md`, Phase 4 |

## 1. Objective

From an image and a bundle to a stitched heatmap, a cleaned mask, a defect table in millimetres, four-point penalties and a roll grade, through one `Inspector` class that `scripts/evaluate.py` already uses (and the app will). `scripts/fit.py` calibrates `tau` on normal validation tiles and writes `calibration.json`; `scripts/evaluate.py` adds the operating-point and grading metrics. All tested on CPU with synthetic data and fake models.

## 2. Background

Phases 2 and 3 produced `PatchCore` and `RD4AD` (both `fit/score/save/load`), the tiler, manifests, metrics and the two scripts. `evaluate.py` currently tiles, scores and stitches by itself; CLAUDE.md rule 5 says evaluation and the app both go through `Inspector`, so this phase moves it.

Read:

- `docs/ARCHITECTURE.md` sections 3 (flow, tiler), 5.2 and 6.3 (scoring), 8 (calibration), 9 (inspection and grading), 10 (bundle), 12 (layout)
- `docs/EVALUATION.md` sections 2.2, 2.3 (operating-point and grading metrics), 4.1 rule 5 (low-confidence flag), 5 (result files)
- Decisions: D-30, D-32 (one `tau`), D-40 to D-43 (grading), D-53 (Inspector), D-58, A-01 to A-14 (unchanged)

**Open question inherited from Phase 2 (A-09), settled here.** PatchCore's tile score is the max of the *unblurred* nearest-neighbour grid; the mask comes from the *blurred* heatmap. With one `tau` (D-32) a tile could be flagged while no pixel of the mask exceeds `tau`. Resolution: **the operating tile score is the maximum of the (blurred) tile heatmap, for both methods.** Calibration takes the maxima of exactly that quantity on validation tiles, the tile decision compares that quantity with `tau`, and the mask thresholds the same heatmap with the same `tau`; so a tile is flagged if and only if at least one of its pixels is above `tau` (before cleanup). This is what ARCHITECTURE 8 already says ("take each tile's heatmap maximum"). The first return value of `model.score` is not changed: it remains the published score (PatchCore: unblurred grid max; RD4AD: heatmap max) used for the threshold-free `tile_auroc` and `tile_ap` and so for the MVTec pass criteria. ARCHITECTURE 5.2 gets a sentence stating the two roles. No change to Phase 2 scoring or numbers; recorded as A-15.

## 3. Scope

**In scope**

- `loomguard/eval/calibrate.py`, `loomguard/inspection/postprocess.py`, `loomguard/inspection/grading.py`, `loomguard/pipeline.py` (`Inspector`).
- `scripts/fit.py` writes `calibration.json`; `scripts/evaluate.py` runs on `Inspector` and adds `tau`, `false_alarms_per_1000`, `tile_recall`, `dice`, `iou`, and for RAW-FABRID the per-roll grading comparison.
- Config keys `target_far`, `px_per_mm`, `scale_assumed`, `grading`, `grade_rolls`, `resize` (4.4).
- Training-PC runbook, including bundle upload to the Hugging Face Hub with its CLI.

**Out of scope**

- The Streamlit app, roll-stream UI, the sensitivity slider UI (Phase 6; `decide(..., tau=)` is the hook). A `huggingface_hub` dependency (the app's download code, Phase 6). Group and held-out regimes, ZJU-Leaper, AITEX (Phase 5). The ZLEval metrics (Phase 5). Tile overlap other than 0. Defect-type classification (hole rule, D-43).
- Any dataset download, training run, GPU use or pretrained-weight download on the laptop.

## 4. Design

### 4.1 Files

| File | New or changed | Purpose |
|---|---|---|
| `loomguard/eval/calibrate.py` | new | `calibrate`, `val_peaks` |
| `loomguard/inspection/postprocess.py` | new | `Defect`, `clean_mask`, `measure`, `min_rect_length` |
| `loomguard/inspection/grading.py` | new | `points`, `grade_roll`, `RollGrade` |
| `loomguard/pipeline.py` | new | `Inspector`, `Scored`, `Inspection` |
| `loomguard/runs.py` | changed | new config keys and defaults |
| `scripts/fit.py` | changed | writes `calibration.json` |
| `scripts/evaluate.py` | changed | scoring through `Inspector`; operating-point and grading metrics |
| `configs/raw_fabrid_*.yaml` | changed | `px_per_mm: 4.0`, `scale_assumed: false`, `grade_rolls: true` |
| `tests/test_calibrate.py`, `test_postprocess.py`, `test_grading.py`, `test_pipeline.py` | new | section 7 |
| `tests/test_fit_evaluate.py` | changed | new bundle file and result keys, independent recomputation |
| `tests/test_inspection_config.py` | new | config defaults and RAW-FABRID scale |
| docs | changed | section 9 |

### 4.2 Interfaces

```python
# loomguard/eval/calibrate.py
def calibrate(peaks: Sequence[float], target_far: float = 0.01) -> dict:
    """peaks: heatmap maximum of every normal validation tile (n >= 1). k = ceil((n+1)*(1-target_far)) (rounded to 9
    decimals first); tau = sorted(peaks)[min(k, n) - 1]. Tiles are flagged when peak > tau, so at most about target_far of
    normal tiles exceed it. Returns {"tau", "target_far", "n_val_tiles", "low_confidence" (n < 500), "score": "heatmap_max",
    "score_median", "score_p99", "score_min", "score_max"}."""

def val_peaks(model, rows: Sequence[TileRow], root, batch_size: int = 16, num_workers: int = 0) -> list[float]:
    """Heatmap maximum per tile of `rows` (val rows only), in row order. model.score(x)[1].amax((1, 2, 3))."""

# loomguard/inspection/postprocess.py
@dataclass(frozen=True)
class Defect:
    bbox: tuple[int, int, int, int]   # x0, y0, x1, y1, x1/y1 exclusive, pixels
    centre: tuple[float, float]       # x, y of the pixel centroid
    area_px: int
    length_px: float                  # long side of the minimum-area rectangle
    length_mm: float
    area_mm2: float
    points: int = 0                   # filled by Inspector

def clean_mask(mask: np.ndarray, closing: int = 5, min_area: int = 16) -> np.ndarray:
    """H x W bool -> H x W bool. Morphological closing with a closing x closing square (no border erosion), then remove
    8-connected components with fewer than min_area pixels."""

def min_rect_length(ys: np.ndarray, xs: np.ndarray) -> float:
    """Long side, in pixels, of the minimum-area rectangle around the pixels (ys, xs). Rotation search in 1 degree steps
    over the left and right end pixel of each row; a side is (extent of pixel centres + 1)."""

def measure(mask: np.ndarray, px_per_mm: float) -> list[Defect]:
    """8-connected components of an already cleaned mask, in label order."""

# loomguard/inspection/grading.py
POINT_BANDS = ((75.0, 1), (150.0, 2), (230.0, 3))   # upper limits in mm, inclusive; longer: 4
SQ_YD_MM2 = 914.4 ** 2
def points(length_mm: float) -> int: ...
@dataclass(frozen=True)
class RollGrade:
    points: int; area_sq_yd: float; points_per_100: float; grade: str    # "A" | "B" | "C" | "Reject"
def grade_roll(total_points: int, area_mm2: float, cutoffs: Sequence[float] = (20, 40, 60)) -> RollGrade:
    """points_per_100 = total_points * 100 / area_sq_yd. A if <= cutoffs[0], B if <= cutoffs[1], C if <= cutoffs[2],
    else Reject. area_mm2 <= 0 raises ValueError."""

# loomguard/pipeline.py
@dataclass
class Scored:        # model output for one image, before any threshold
    heat: Tensor     # H x W float32, stitched (overlaps averaged)
    scores: Tensor   # n_tiles float32, model.score(...)[0]: the published tile score
    peaks: Tensor    # n_tiles float32, max of each tile's heatmap: the operating tile score
    origins: list[tuple[int, int]]   # (y0, x0) per tile, row-major, tiling.grid order
    size: tuple[int, int]            # H, W of the (possibly resized) image

@dataclass
class Inspection:
    heat: Tensor; mask: np.ndarray   # mask: H x W bool, cleaned
    defects: list[Defect]; peaks: Tensor; flagged_tiles: np.ndarray   # n bool
    flagged: bool          # any tile peak > tau
    score: float           # max over tile peaks
    tau: float; px_per_mm: float; scale_assumed: bool
    points: int            # sum of defect points
    grade: RollGrade       # this one frame treated as a roll: an extrapolation (D-42), area = H*W / px_per_mm**2

class Inspector:
    def __init__(self, model, tau: float, px_per_mm: float = 4.0, scale_assumed: bool = True, resize: int | None = None,
                 grading: dict | None = None, batch_size: int = 16): ...
    @classmethod
    def load(cls, bundle: Path | str, device="cpu", backbone=None) -> "Inspector":
        """Reads config.yaml, calibration.json and model.pt (torch.load weights_only=True through the model class)."""
    def score_image(self, img: PIL.Image.Image) -> Scored     # resize (PIL bilinear) if set, preprocess, tile, score, stitch
    def decide(self, s: Scored, tau: float | None = None) -> Inspection   # threshold, clean, measure, points; no model call
    def inspect(self, img, tau=None) -> Inspection             # decide(score_image(img))
    def measure(self, mask: np.ndarray) -> tuple[np.ndarray, list[Defect]]   # clean + measure + points, for any bool mask
    def inspect_roll(self, frames: Iterable[Image]) -> tuple[list[Inspection], RollGrade]
```

### 4.3 Behaviour and edge cases

- `peaks` of a tile are the maxima of the per-tile heatmaps, not of the stitched one; with the default overlap 0 both are the same.
- `decide` thresholds the stitched heatmap with `heat > tau`, then `clean_mask` (closing 5, min area 16 by default), then `measure`. Raising `tau` never creates a defect (monotone).
- Millimetres: `length_mm = length_px / px_per_mm`, `area_mm2 = area_px / px_per_mm**2`. Points: length <= 75 mm 1, <= 150 mm 2, <= 230 mm 3, else 4. A line of `L` pixels has `length_px == L`.
- Roll area is the sum of `H * W / px_per_mm**2` over the frames; `inspect_roll` sums points over frames. An empty mask gives 0 points and grade A. A single `Inspection.grade` is only an extrapolation; the app labels it so (Phase 6).
- Closing joins blobs separated by up to 4 background pixels; a gap of 3 px is merged. Blobs smaller than 16 px (after closing) are dropped.
- `Inspector.load` needs ImageNet weights for the teacher unless `config.yaml` says `backbone_weights: null` (tests only). The bundle never contains backbone weights.
- `Inspector` resizes with `PIL.Image.BILINEAR` exactly as `TileDataset` does for MVTec, so Phase 2 and 3 numbers are unaffected by the move.
- `evaluate.py` per test parent: `score_image`, tile map by `(y0, x0)` to the manifest rows, tile metrics from `scores` (unchanged), pixel metrics from `heat`, operating metrics from `decide` at the calibrated `tau`. If `calibration.json` is missing it stops with a message to refit.
- Operating metrics: `false_alarms_per_1000` = 1000 * (normal test tiles with peak > tau) / (normal test tiles); `tile_recall` = flagged defective tiles / defective tiles; `dice`, `iou` pooled over the pixels of all defective test tiles: `2I/(P+G)` and `I/(P+G-I)` of the cleaned predicted mask against the ground-truth mask. Undefined values are `null`.
- Grading metrics (only if `grade_rolls`): for each fabric present in the test pool, over the test parent images of that fabric, `grade_roll` is applied to the predicted cleaned masks and to the cleaned ground-truth masks (same `Inspector.measure`). Result key `rolls`: per fabric `frames, defects_pred, defects_gt, points_pred, points_gt, points_per_100_pred, points_per_100_gt, grade_pred, grade_gt`; plus `grade_agreement` (fraction of rolls with equal grade), `points_error_mean` (mean absolute difference of points per 100 sq yd) and `defect_count_error_mean`. RAW-FABRID's test pool holds only defective images (EVALUATION 4.2), so these rolls are test-pool rolls, not full rolls (A-17).
- `ms_per_tile_<device>` is now the time of `score_image` per scored tile (includes tiling and stitching).

### 4.4 Configuration

New keys with defaults (in `runs.DEFAULTS`):

| Key | Default | Meaning |
|---|---|---|
| `target_far` | 0.01 | calibration false-alarm rate (D-32) |
| `px_per_mm` | 4.0 | pixels per millimetre of the fabric profile |
| `scale_assumed` | true | true unless the dataset states the scale (RAW-FABRID: false) |
| `grading` | `{closing: 5, min_area_px: 16, cutoffs: [20, 40, 60]}` | cleanup and grade cut-offs (D-41) |
| `grade_rolls` | false | compute per-roll grading metrics in `evaluate.py` (true in the RAW-FABRID configs) |
| `resize` | 256 for `mvtec`, else null | whole-image resize policy; written into the bundle's `config.yaml` |

`calibration.json`: the dict returned by `calibrate`. `config.yaml` already holds the whole config, so the Inspector reads `method`, `backbone_weights`, `px_per_mm`, `scale_assumed`, `resize`, `grading` from it.

### 4.5 Dependencies

None. Connected components come from `loomguard/eval/metrics.py`; closing uses `torch.nn.functional.max_pool2d`; the minimum-area rectangle is a small NumPy rotation search. The Hub upload uses the `huggingface-cli` that the training-PC operator installs for the upload; it is not a project dependency (the app's download code adds `huggingface_hub` in Phase 6).

## 5. Tasks

### Laptop (developer session)

1. `calibrate.py` and its test.
2. `postprocess.py` and `grading.py` with hand-computed tests.
3. `pipeline.py` (`Inspector`) with the fake-model equality test.
4. Config keys in `runs.py`; `fit.py` writes `calibration.json`; configs updated.
5. `evaluate.py` on `Inspector`, operating-point and grading metrics; extend the end-to-end tests (including independent recomputation).
6. Bundle round trip test (fit, `Inspector.load`, inspect) with `weights_only=True`.
7. Run the full suite, review loop, documents.

### Training PC (operator)

1. `git pull`, run the suite. Refit is required for bundles made before this phase (no `calibration.json`).
2. Rerun fit and evaluate for MVTec and RAW-FABRID, both methods (section 8). Check the false-alarm band and the per-roll grading table.
3. Upload bundles to the Hugging Face Hub; download one on the laptop and run it on CPU.

## 6. Acceptance criteria

| # | Criterion | Where | Verified by | Met |
|---|---|---|---|---|
| A1 | Grading unit tests with synthetic masks: a 50 mm line scores 1 point, 100 mm scores 2, 200 mm scores 3, 300 mm scores 4, at 4 px/mm. An empty mask gives 0 points and grade A. | Laptop | `tests/test_grading.py::test_line_points`, `::test_empty_mask_grade_a` | [x] |
| A2 | Two blobs 3 px apart are merged into one defect; a 10 px blob is removed. | Laptop | `tests/test_postprocess.py::test_close_merges_3px_gap`, `::test_small_blob_removed` | [x] |
| A3 | `Inspector` output on a full 1792x1024 image equals tiling, scoring and stitching done by hand (deterministic fake model). | Laptop | `tests/test_pipeline.py::test_inspector_matches_manual_pipeline` | [x] |
| A4 | On RAW-FABRID, false alarms on test-normal tiles are between 5 and 20 per 1,000 for the 10 per 1,000 target, for both methods. If not, the cause is analysed and written up. | Training PC | runbook step 3; `false_alarms_per_1000` in `results/raw_fabrid/unified/*/unified.json` | [ ] |
| A5 | Per-roll grade from predicted masks and from ground-truth masks reported for all 7 rolls (points error and grade agreement). | Training PC | runbook step 4; `rolls`, `grade_agreement`, `points_error_mean` in the same files | [ ] |
| A6 | A bundle saved on the training PC loads and runs on the laptop CPU with `weights_only=True`. | Laptop (test) and Training PC (real bundle) | `tests/test_pipeline.py::test_bundle_round_trip` (synthetic); runbook step 5 (real bundle) | [x] synthetic; real bundle pending |
| A7 | Added: `calibrate` returns the `ceil((n+1)*(1-far))`-th order statistic, flags n < 500 as low confidence, and about `target_far` of the calibration tiles exceed it. | Laptop | `tests/test_calibrate.py` | [x] |
| A8 | Added: tile decision and mask agree: a tile has `peak > tau` iff its heatmap has a pixel above `tau`; `score()` of both models is unchanged. | Laptop | `tests/test_pipeline.py::test_tile_flag_matches_mask`; `tests/test_patchcore.py`, `tests/test_rd4ad.py` unchanged and passing | [x] |
| A9 | Added: `fit.py` writes `calibration.json` from val tiles only; `evaluate.py` goes through `Inspector` (no own tiling or stitching), and its operating-point and grading numbers equal an independent recomputation. | Laptop | `tests/test_fit_evaluate.py` | [x] |
| A10 | Added: monotone in `tau`; grade cut-offs read from config; points bands at the exact limits (75, 150, 230 mm). | Laptop | `tests/test_pipeline.py::test_tau_monotone`, `tests/test_grading.py` | [x] |
| A11 | Added: all Phase 1 to 3 tests still pass. | Laptop | `pytest` | [x] |

## 7. Test plan

| Test | Proves |
|---|---|
| `test_calibrate.py::test_order_statistic`, `::test_target_rate`, `::test_low_confidence`, `::test_float_edge` | tau rule, rate on 1..1000, flag, `n = 99` does not overshoot to k = 100 |
| `test_postprocess.py::test_close_merges_3px_gap`, `::test_gap_5px_not_merged`, `::test_small_blob_removed`, `::test_border_not_eroded`, `::test_min_rect_length_*` (axis line, 45 degree line, rectangle), `::test_measure_fields` | cleanup and measurement against hand-computed values |
| `test_grading.py::test_line_points`, `::test_band_limits`, `::test_points_per_100`, `::test_cutoffs`, `::test_empty_mask_grade_a`, `::test_zero_area_raises` | hand-computed points, 1 frame = 0.1372 sq yd, cut-offs including custom ones |
| `test_pipeline.py::test_inspector_matches_manual_pipeline` | full image through Inspector equals hand tiling, scoring, stitching, threshold |
| `::test_tile_flag_matches_mask`, `::test_tau_monotone`, `::test_decide_without_model` | A8, A10, slider hook |
| `::test_resize_policy` | MVTec-style resize gives a 256 x 256 heatmap |
| `::test_bundle_round_trip` | fit then `Inspector.load` then inspect equals the fitted in-memory model, `weights_only=True` |
| `::test_roll_grade` | two frames with known defects sum to hand-computed points and area |
| `test_fit_evaluate.py` (extended) | `calibration.json` present with val-only tiles, new result keys, operating and grading numbers recomputed independently |

## 8. Training-PC runbook

Prerequisite: the Phase 1 to 3 environment and datasets. Bundles fitted before this phase lack `calibration.json`: refit them (run in phase order; this runbook's fit commands replace the earlier ones).

```bash
# 0. Update and test
git pull
source .venv/bin/activate
pytest

# 1. MVTec AD, both methods (Phase 2 and 3 pass criteria still apply; tile_auroc etc. are unchanged by this phase)
for m in patchcore rd4ad; do
  python scripts/fit.py      --config configs/mvtec_$m.yaml | tee artifacts/phase4_mvtec_${m}_fit.txt
  python scripts/evaluate.py --config configs/mvtec_$m.yaml | tee artifacts/phase4_mvtec_${m}_eval.txt
done

# 2. RAW-FABRID unified: PatchCore, and RD4AD seed 0 (the other seeds as in the Phase 3 runbook)
for c in raw_fabrid_patchcore raw_fabrid_rd4ad_s0; do
  python scripts/fit.py      --config configs/$c.yaml | tee artifacts/phase4_${c}_fit.txt
  python scripts/evaluate.py --config configs/$c.yaml | tee artifacts/phase4_${c}_eval.txt
done

# 3. A4: false alarms per 1,000 normal test tiles (target 10, accepted band 5 to 20)
python - <<'EOF'
import json
for f in ("patchcore", "rd4ad_s0"):
    r = json.load(open(f"results/raw_fabrid/unified/{f}/unified.json"))
    print(f, "tau", round(r["tau"], 4), "FA/1000", round(r["false_alarms_per_1000"], 1), "recall", round(r["tile_recall"], 3),
          "dice", round(r["dice"], 3), "iou", round(r["iou"], 3))
EOF
# Outside 5 to 20: analyse and write up (val and test tiles come from different images; roll 3 has only 140 val tiles).

# 4. A5: per-roll grading, predicted against ground-truth masks
python - <<'EOF'
import json
for f in ("patchcore", "rd4ad_s0"):
    r = json.load(open(f"results/raw_fabrid/unified/{f}/unified.json"))
    print(f, "grade_agreement", r["grade_agreement"], "points_error_mean", round(r["points_error_mean"], 1))
    for roll, v in r["rolls"].items():
        print(" ", roll, v["frames"], "frames", "pred", v["points_per_100_pred"], v["grade_pred"], "gt", v["points_per_100_gt"], v["grade_gt"])
EOF

# 5. A6 and bundle upload (huggingface_hub is installed on the PC only for this: pip install huggingface_hub)
huggingface-cli login
huggingface-cli upload <user>/loomguard-bundles artifacts/raw_fabrid/unified/patchcore raw_fabrid/unified/patchcore
huggingface-cli upload <user>/loomguard-bundles artifacts/raw_fabrid/unified/rd4ad_s0 raw_fabrid/unified/rd4ad_s0
# On the laptop (CPU; the ImageNet weights download on first use):
#   huggingface-cli download <user>/loomguard-bundles --include "raw_fabrid/unified/patchcore/*" --local-dir artifacts_dl
#   python -c "from loomguard.pipeline import Inspector; from PIL import Image; import glob; \
#     i = Inspector.load('artifacts_dl/raw_fabrid/unified/patchcore'); \
#     r = i.inspect(Image.open(sorted(glob.glob('data/RAW_FABRID/images/*.png'))[0])); print(r.flagged, r.score, r.tau, len(r.defects), r.points)"
```

| Step | Expected | Record in |
|---|---|---|
| 1 | tile and pixel metrics as in Phases 2 and 3 (same `score()`) | `PROGRESS_LOG.md` |
| 3 | `false_alarms_per_1000` 5 to 20 for both methods | `PROGRESS_LOG.md`, tick A4 |
| 4 | 7 rolls, grade agreement and points error | `PROGRESS_LOG.md`, tick A5 |
| 5 | the laptop prints a verdict and a defect count | `PROGRESS_LOG.md`, tick A6 |

Commit `phase 4: results` with `results/**.json`, the log entry, ticks in this spec and in the plan. Do not tune on these results (D-35).

## 9. Documents to update

- `ARCHITECTURE.md`: 5.2 (operating tile score, A-15), 8 (calibration.json fields, low confidence), 10 (bundle files), 12 (Inspector line).
- `EVALUATION.md`: 2.2, 5 (result keys; A-17 note on test-pool rolls).
- `DECISIONS.md`: A-15 onward. `CLAUDE.md` layout note if needed. `README.md` status line.

## 10. Risks and open points

| Risk or open point | Handling |
|---|---|
| Tile decision defined on the blurred heatmap max; ARCHITECTURE 5.2 says unblurred grid for PatchCore | Settled above (A-15); `score()` untouched. If the owner prefers otherwise, only `Scored.peaks` and `val_peaks` change |
| RAW-FABRID test pool has no normal images, so test-pool "rolls" are all defective | Grade agreement is still a valid predicted-against-truth comparison on the same frames; stated in the result docs (A-17) |
| Minimum-area rectangle by 1 degree search is approximate (about 1 pixel on long thin defects) | Acceptable against the 75 mm bands (300 px); `ponytail` note in code |
| Val and test maxima differ in distribution (val is whole normal images, test-good tiles sit beside defects) | A4 measures it; a miss is analysed, not tuned (D-35) |
| `evaluate.py` previously batched across parents; now batches within an image | Metrics unchanged to numeric precision; tests recompute independently |

## 11. Completion record

| | |
|---|---|
| Spec commit | 082a2c9 |
| Implementation commit | 4cead6a |
| Review commit | 704c2c4 (`phase 4: review fixes and doc sync`) plus the round 2 record fix |
| Review rounds and verdict | round 1 CHANGES REQUIRED (documents only), round 2 PASS |
| Tests | `pytest -q`: 110 passed, 0 failed, 1 slow deselected |
| Results commit (training PC) | pending |
| Decisions made | A-15 to A-20 |
| Deviations from this spec | `Inspector.load` has an extra `backbone=None` argument (lets `evaluate.py` and tests inject the backbone; A-18). `tests/test_inspection_config.py` is new, `tests/test_runs.py` unchanged. A4, A5 and the real-bundle half of A6 are training-PC criteria and stay unticked. |
| Closed by owner | |
