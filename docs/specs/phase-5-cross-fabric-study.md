# Phase 5: Cross-fabric study

| | |
|---|---|
| Phase | 5 |
| Status | Code complete |
| Spec written | 2026-11-14 |
| Written by | developer session, phase 5 |
| Depends on | Phases 1 to 4 (Code complete, runbooks queued, D-58) |
| Plan entry | `docs/IMPLEMENTATION_PLAN.md`, Phase 5 |

## 1. Objective

The laptop side of the cross-fabric study: ZJU-Leaper and AITEX manifest adapters, the new regimes (per group, held-out fabric) and ablation switches in the existing `fit.py` / `evaluate.py` path, a config for every cell of EVALUATION 4.3 and every ablation of 4.4, a resumable study driver, a results-to-tables-and-plots script, a failure-case image script and the full training-PC runbook. The runs themselves (Q2, Q3, Q4, Q7 answers) are training-PC work.

## 2. Background

- `docs/EVALUATION.md` sections 2.2, 4 (splits, regimes 4.3, ablations 4.4), 5 (result files), 6 (compute).
- `docs/DATASETS.md` sections 2 (ZJU-Leaper) and 4 (AITEX): layouts are **unverified** on real files.
- `docs/ARCHITECTURE.md` section 3 (tiler, `min_overlap`), 10 (bundle).
- Decisions: D-11, D-13, D-15, D-21 to D-25, D-30 to D-35, D-58, A-01 to A-20.
- Existing code reused, not duplicated: `loomguard/data/*`, `runs.py`, `pipeline.py` (`Inspector`), `scripts/fit.py`, `scripts/evaluate.py`, `eval/metrics.py`.

## 3. Scope

**In scope**

- `build_zju` and `build_aitex` adapters (tolerant, strict about reporting problems).
- Regimes `per_group` and `held_out`; config keys `extends`, `min_overlap`, `held_out_scope`, `dump_predictions`; `resize` for ZJU-Leaper.
- `evaluate.py`: image-level metrics (max over a parent's tiles, needed for ZJU-Leaper), a per-fabric breakdown for multi-fabric bundles (needed to compare per-fabric, per-group, unified and held-out models on the same fabric), optional dump of predicted masks (input for the official ZJU `ZLEval`).
- Configs, study lists, `scripts/study.py`, `scripts/summarize.py`, `scripts/failure_cases.py`, runbook.

**Out of scope**

- Running anything on real data, installing packages, ZLEval glue code (format unverified), Phase 6 and 7 work, the written findings (training PC).

## 4. Design

### 4.1 Files

| File | New or changed | Purpose |
|---|---|---|
| `loomguard/data/manifest.py` | changed | `ZJU_GROUPS`, `zju_group`, `build_zju`, `build_aitex`, shared `_image_rows`; `build_raw_fabrid(min_overlap=)` |
| `loomguard/data/dataset.py` | changed | `load_mask(root, rel)`: `rel` may hold several paths joined by `\|` (union), used by `TileDataset` and `evaluate.py` |
| `loomguard/eval/metrics.py` | changed | `ScoreCounts.merge` |
| `loomguard/pipeline.py` | changed | `Inspector(min_overlap=)`, read from the bundle config |
| `loomguard/runs.py` | changed | `extends`, regimes, `build_rows` for the new datasets, `profiles` for `per_group` and `held_out`, `predictions_dir` |
| `scripts/fit.py` | changed | manifest file name carries the tag |
| `scripts/evaluate.py` | changed | image metrics, `by_fabric`, `tag`, `dump_predictions` |
| `scripts/study.py` | new | driver |
| `scripts/summarize.py` | new | tables, SVG plots, coverage check |
| `scripts/failure_cases.py` | new | failure-case panels |
| `configs/base/*.yaml`, `configs/*.yaml`, `configs/studies/*.txt` | new | one config per run, study lists |
| `tests/test_zju_aitex.py`, `tests/test_study.py`, `tests/test_summarize.py`, `tests/conftest.py` and existing tests | new / changed | see section 7 |

### 4.2 Interfaces

```python
ZJU_GROUPS: dict[int, range]                      # group -> pattern numbers (DATASETS 2): 1:1-3, 2:4-7, 3:8-12, 4:13-15, 5:16-19
def zju_group(fabric: str) -> str                 # "pattern12" -> "group3"
def build_zju(root, patterns: Sequence[str] | None = None, *, val_frac=0.10, seed=0, resize=0, min_overlap=0) -> list[TileRow]
def build_aitex(root, *, val_frac=0.15, seed=0, black_level=10, max_black=0.5, min_overlap=0) -> list[TileRow]
def load_mask(root: Path, rel: str) -> PIL.Image   # mode "L"
ScoreCounts.merge(other: ScoreCounts) -> None
profiles(rows, cfg) -> dict[str, list[TileRow]]   # regimes per_fabric | per_group | unified | held_out
predictions_dir(cfg, profile) -> Path             # artifacts/predictions/<dataset>/<regime>/<method>/<profile>
```

Tensor shapes and dtypes are unchanged (tiles `3 x 256 x 256` float32, heatmaps `H x W` float32).

### 4.3 Behaviour and edge cases

**ZJU-Leaper adapter.** Fabric `pattern<n>`; reads `ImageSets/Patterns/pattern<n>.json` (`{"normal": {"train","test"}, "defect": {"train","test"}}`); every present pattern file is used (restricted by `patterns`). Ids resolve through an index of `Images/` and `Annotations/masks/` (by stem or file name, any image extension, subfolders allowed). Normal test ids and defect test ids form the test pool; `defect/train` is not used (Setting 1). Val is 10 percent of each pattern's `normal/train` ids (no tile minimum), shuffled with the existing `assign_val`, so one split serves every regime. Group, unified and held-out training sets are unions of the per-pattern train and val rows. Strict reporting: an id without an image, a defect test id without a mask, a mask whose size differs from its image, an id listed in two sets, or a pattern file of the wrong structure raises one error with counts and the first examples. `resize=256` gives one whole-image row per image with `resize=256` (resolution ablation, as MVTec). Tile grid from `tiling.grid` with `min_overlap`; the split uses the no-overlap tile count so overlap never changes the split.

**AITEX adapter.** Finds `Defect_images`, `NODefect_images`, `Mask_images` anywhere under the root (case-insensitive). Fabric `fabric<NN>` from the trailing `_<digits>` of the file stem (error naming the file if absent). Masks are paired by stem (`<stem>_mask*`); several masks of one image are joined with `|` and united by `load_mask`. A defective image without a mask or with a size mismatch raises. Normal images: per fabric, 12 train and 8 val (the A-01 rule: 15 percent, raised to 200 tiles, capped at 40 percent). All defective images are test. Tiles with more than `max_black` of their pixels below `black_level` (gray) are dropped from every split. Known limit: pixel metrics are computed on whole stitched images, so black border tiles still enter the pixel pool.

**Regimes** (D-34): `per_fabric` (one profile per fabric), `per_group` (ZJU only, profile `group<g>`), `unified` (profile `unified`), `held_out`: for each fabric `f` in `fabrics` (default all), the profile `<f>_held_out` holds the train and val rows of every other fabric in the scope (`held_out_scope: group` for ZJU, `all` for RAW-FABRID) and only the test rows of `f`. The held-out fabric's train and val rows never reach `fit`; `tau` comes from validation tiles of the training fabrics. `build_rows` does not filter by `fabrics` for `held_out`.

**evaluate.py.** `image_auroc` and `image_ap`: parent image score is the maximum of its tiles' model scores, label 1 if any tile is defective (null when one class only, e.g. RAW-FABRID, AITEX). If the test pool holds more than one fabric the result adds `by_fabric: {fabric: {tile_auroc, tile_ap, image_auroc, image_ap, aupro_30, pixel_auroc, pixel_ap, false_alarms_per_1000, tile_recall, dice, iou}}`; the top-level metrics equal the pooled values (`ScoreCounts.merge`). The result adds `tag`. With `dump_predictions: true` each test parent's cleaned predicted mask (`<stem>.png`, 0/255) and `scores.csv` (`parent,fabric,label,score,flagged`) are written under `predictions_dir`.

**Driver.** `python scripts/study.py LIST [--force] [--dry-run]`: LIST has one config path per line (`#` comments). For each config: skip if every expected result file exists (unless `--force`), else `fit.main` then `evaluate.main`; an exception or `SystemExit` marks the config failed and the study continues. Prints a table (ok / skipped / failed with reason); exit code 1 if any failed. `--dry-run` prints configs and expected result paths.

**Summary script.** `python scripts/summarize.py [--results results] [--out results/report] [--check]`. Reads every result JSON (method directory name taken from the path). Writes `regimes.md` (per regime tables; and the cross-regime table per fabric: per-fabric, per-group, unified, held-out values from the same fabric, `by_fabric` for the shared bundles), `ablations.md`, `all_results.csv` and SVG bar charts under `figures/` (stdlib only, A-21). Detection metric: `image_auroc` for ZJU-Leaper, else `tile_auroc`. Ablation tables compare variants on the fabrics present in every variant. `--check` lists every missing cell of EVALUATION 4.3 for both methods and exits 1 if any.

**Failure cases.** `python scripts/failure_cases.py --config C --profile P [--n 3] [--out DIR]` picks, among the test images of the profile, the `n` defective images with the lowest Dice (misses) and the `n` images with most predicted pixels outside the ground truth (false alarms) and writes one PNG panel each (image, ground truth, heatmap, predicted mask). Selection is for display only and feeds nothing.

**Headline settings (D-35), fixed here before any run:** PatchCore `coreset_ratio` 0.10, `max_tiles` 600, seed 0, native 256 tiles, `min_overlap` 0; RD4AD iterations 5,000 per fabric, 10,000 per group, 20,000 unified, last checkpoint (D-24, D-25); held-out RD4AD uses the budget of the bundle it is compared with (ZJU group 10,000; RAW-FABRID, 6 rolls, 20,000; A-22). Ablation tags: `res256`, `cr1`, `cr25`, `n50`, `n200`, `it1k`, `ov32`, `s1`, `s2`.

### 4.4 Configuration

New keys: `extends` (path or list, relative to the config file, deep-merged, later wins), `min_overlap` (0), `held_out_scope` ("all"), `dump_predictions` (false); `regime` also `per_group`, `held_out`; `dataset` also `zju_leaper`, `aitex`. `configs/base/` holds shared fragments.

### 4.5 Dependencies

None. `pyproject.toml` unchanged; plots are hand-written SVG.

## 5. Tasks

### Laptop (developer session)

1. Manifest helpers, `build_zju`, `build_aitex`, `load_mask`, `ScoreCounts.merge`, `min_overlap` in `Inspector`.
2. `runs.py` (`extends`, regimes, datasets); `evaluate.py` and `fit.py` changes.
3. Configs and study lists for every regime and ablation.
4. `study.py`, `summarize.py`, `failure_cases.py`.
5. Tests, synthetic ZJU and AITEX trees, end-to-end driver test.
6. Runbook (section 8), documents.

### Training PC (operator)

1. Verify ZJU-Leaper and AITEX layouts with `verify_datasets.py`; fix adapters if the real layout differs (log it).
2. ZJU-mini, RAW-FABRID, AITEX, full ZJU-Leaper, ablations (plan tasks 2 to 5).
3. ZJU `ZLEval` on the dumped predictions (plan task 7).
4. `summarize.py --check`, tables, plots; failure-case images; written findings (Q2, Q3, Q4, Q7).

## 6. Acceptance criteria

| # | Criterion | Where | Verified by | Met |
|---|---|---|---|---|
| A1 | Every cell of the regime table in EVALUATION 4.3 has a result file for both methods | Training PC | `python scripts/summarize.py --check` exits 0 | [ ] |
| A1L | Laptop: every cell has a config in a study list; the check reports missing cells | Laptop | `tests/test_study.py::test_every_cell_has_a_config`, `tests/test_summarize.py::test_check_reports_missing` | [x] |
| A2 | Every run reproducible with one command and one config; config path and git commit in each result JSON | Both | `tests/test_study.py::test_driver_end_to_end` checks `config`, `git_commit`; command `python scripts/study.py LIST` or `fit.py` + `evaluate.py --config C` | [x] |
| A3 | All ablations of 4.4 (the four of the plan: resolution, coreset ratio, training-set size, overlap; plus the seeds row, five in the table) run and plotted | Training PC | `results/report/ablations.md` and `figures/` contain every ablation of `summarize.ABLATIONS` | [ ] |
| A3L | Laptop: a config for every ablation variant; tables and SVGs are produced from results | Laptop | `test_every_cell_has_a_config` (ablations part), `tests/test_summarize.py::test_ablation_tables_and_plots` | [x] |
| A4 | Written findings answer Q2, Q3, Q4, Q7 with numbers and at least three failure-case images per method | Training PC | `docs/FINDINGS.md`, images from `failure_cases.py` | [ ] |
| A4L | Laptop: failure-case script writes the panels | Laptop | `tests/test_study.py::test_failure_cases` | [x] |
| A5 | Headline settings not changed after looking at test results; deviations logged | Both | `tests/test_study.py::test_headline_settings_fixed` (budgets, ratios); progress log | [x] |
| A6 | ZJU adapter: official-JSON splits, 10 percent val by parent, no parent in two splits, strict problem reports, resize rows | Laptop | `tests/test_zju_aitex.py::test_zju_*` | [x] |
| A7 | AITEX adapter: fabric from name, 12/8 split, border filter, several masks | Laptop | `tests/test_zju_aitex.py::test_aitex_*` | [x] |
| A8 | Regimes: per-group and held-out profiles; held-out fabric never in train or val rows | Laptop | `tests/test_zju_aitex.py::test_profiles_regimes`, `test_held_out_fit_reads_no_held_out_rows` | [x] |
| A9 | Image metrics and `by_fabric` equal an independent recomputation; pooled equals merged | Laptop | `tests/test_study.py::test_evaluate_image_and_by_fabric` | [x] |
| A10 | Overlap: manifest and `Inspector` use the overlapped grid, split unchanged | Laptop | `tests/test_zju_aitex.py::test_min_overlap` | [x] |
| A11 | No new dependency | Laptop | `git diff pyproject.toml` empty | [x] |
| A12 | Full suite passes | Laptop | `pytest -q` | [x] |
| A13 | Adapters check out on real data | Training PC | runbook step 0 | [ ] |
| A14 | ZLEval figures recorded | Training PC | runbook step 6 | [ ] |

## 7. Test plan

All CPU, synthetic.

| Test | Proves |
|---|---|
| `test_zju_*` | splits follow the JSON, 10 percent val, group and pattern membership, errors on missing mask or duplicated id, `resize` rows |
| `test_aitex_*` | fabric from name, split counts, black tiles dropped, two masks united |
| `test_profiles_regimes` | per_group, held_out composition |
| `test_held_out_fit_reads_no_held_out_rows` | rule 2 for the held-out regime |
| `test_min_overlap` | grid and Inspector with overlap, split unchanged |
| `test_evaluate_image_and_by_fabric` | new metrics |
| `test_driver_end_to_end` | tiny multi-fabric ZJU tree: four regimes via study lists, result paths, skip on rerun, failure continues |
| `test_every_cell_has_a_config`, `test_headline_settings_fixed` | configs cover the study and keep the fixed settings |
| `test_check_reports_missing`, `test_ablation_tables_and_plots` | summary script |
| `test_failure_cases` | panels written |

## 8. Training-PC runbook

For the proposed three-person schedule, see the [distributed training runbook](../DISTRIBUTED_TRAINING.md). It selects complete profiles from these configurations, gates larger ZJU work on both ZJU-mini methods and retains the full-study coverage check.

```bash
git pull && pip install -e ".[dev]" && pytest -q
# 0. layouts: read the output, compare with DATASETS.md, record mismatches in PROGRESS_LOG, adapt the adapters if needed
python scripts/verify_datasets.py --raw-fabrid data/RAW_FABRID --zju data/ZJU-Leaper --aitex data/AITEX --mvtec data/mvtec_ad --manifest-dir artifacts/manifests
python scripts/study.py configs/studies/01_zju_mini.txt --dry-run   # then without --dry-run
python scripts/study.py configs/studies/01_zju_mini.txt             # 1. debug on ZJU-mini first
python scripts/study.py configs/studies/02_raw_fabrid.txt           # 2. per roll, unified, leave-one-roll-out
python scripts/study.py configs/studies/03_aitex.txt                # 3.
python scripts/study.py configs/studies/04_zju_full.txt             # 4. rest of 19 patterns, groups, unified, held-out
python scripts/study.py configs/studies/05_ablations.txt            # 5. resolution, coreset, size, overlap, seeds
# 6. ZLEval: set dump_predictions: true in the ZJU configs (a copy, tag it) or run evaluate on a copy; feed
#    artifacts/predictions/zju_leaper/... to the dataset's ZLEval (format unverified); record F1 pixel/region/sample
python scripts/summarize.py --check
python scripts/summarize.py
python scripts/failure_cases.py --config configs/raw_fabrid_per_fabric_patchcore.yaml --profile Rollo4A   # also rd4ad; commit RAW-FABRID panels only (CC BY)
# write docs/FINDINGS.md: Q2, Q3, Q4, Q7 with numbers from results/report; commit results/ and report: "phase 5: results"
```

Each study list runs PatchCore before RD4AD so a path bug shows up in minutes. Studies are resumable: rerun the same command. Headline configs must not change after the first result; any change is logged with its reason.

Expected GPU time (EVALUATION 6 speeds, 4 it/s, batch 16): PatchCore about 100 bundles, 6 to 8 h with evaluation. RD4AD: ZJU per pattern 19 x 21 min = 6.6 h; per group 5 x 42 = 3.5 h; unified 1.4 h; ZJU held-out 3.5 h; RAW per roll 2.4 h; RAW unified 3 seeds 4.2 h; RAW held-out 7 x 83 min = 9.7 h; AITEX 2.4 h; ablations (res256 1.7, it1k 0.5, ov32 2.4 h) 4.6 h. RD4AD total about 38 h, above the 27 h of the compute table, mostly the RAW-FABRID held-out budget (A-22).

| Step | Expected | Record in |
|---|---|---|
| 0 | counts as DATASETS.md or mismatches noted | `PROGRESS_LOG.md` |
| 1 to 5 | study table all ok | `PROGRESS_LOG.md`, `results/` |
| 6 | F1 pixel / region / sample per pattern | `PROGRESS_LOG.md` |
| check | exit 0 | `PROGRESS_LOG.md` |

## 9. Documents to update

`EVALUATION.md` (regimes, image metrics, result keys, held-out profile names), `DATASETS.md` (adapter assumptions), `ARCHITECTURE.md` (layout, config keys), `README.md`, `CLAUDE.md` (layout), plan, progress log, decisions.

## 10. Risks and open points

| Risk or open point | Handling |
|---|---|
| ZJU and AITEX layouts unverified | Tolerant lookup, strict errors with counts; runbook step 0 |
| AITEX fabric from the file name is an assumption | Error names the file; `verify_datasets` prints fabric counts |
| ZLEval input format unknown | Predictions dumped; glue on the PC |
| Held-out RD4AD budget | A-22, reversible |
| Nothing depends on an item under "Open" in DECISIONS | n/a |

## 11. Completion record

| | |
|---|---|
| Spec commit | 186bca3 |
| Implementation commit | 1ad94e9 |
| Review commit | `phase 5: review fixes and doc sync` |
| Review rounds and verdict | 1 round, PASS (three minor findings fixed) |
| Tests | `pytest -q`: 128 passed, 1 slow test deselected (new: zju/aitex 7, study 9, summarize 2) |
| Results commit (training PC) | |
| Decisions made | A-21 to A-25 |
| Deviations from this spec | None. A2 and A5 are ticked for their laptop part; their training-PC part is in the runbook. |
| Closed by owner | |
