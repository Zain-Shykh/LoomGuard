# LoomGuard: Progress Log

Newest entry first. Add an entry whenever a task finishes, a number is measured, a decision is made or a plan changes.

Entry template:

```
## YYYY-MM-DD: short title
- Phase:
- Who / where: (name, laptop or training PC)
- Done:
- Evidence: (test output, result file, commit)
- Decisions:
- Problems / next:
```

---

## 2026-11-14: Phase 4 laptop work, calibration, inspection and grading

- **Phase:** 4 (Code complete; training-PC tasks outstanding)
- **Who / where:** developer session, laptop
- **Done:**
  - `loomguard/eval/calibrate.py` (`calibrate`, `val_peaks`), `loomguard/inspection/postprocess.py` (closing, area filter, components, minimum-area rectangle length), `loomguard/inspection/grading.py` (four-point bands, points per 100 sq yd, grade), `loomguard/pipeline.py` (`Inspector`, `Scored`, `Inspection`, bundle load, roll grading).
  - `scripts/fit.py` writes `calibration.json`; `scripts/evaluate.py` now runs every test image through `Inspector` and adds `tau`, `false_alarms_per_1000`, `tile_recall`, `dice`, `iou` and per-roll grading (`rolls`, `grade_agreement`, `points_error_mean`, `defect_count_error_mean`). New config keys and `px_per_mm: 4.0` in the RAW-FABRID configs.
  - Spec `docs/specs/phase-4-calibration-inspection-grading.md` with the training-PC runbook (false-alarm band, per-roll grading, Hub upload, laptop load).
  - The Phase 2 open question is settled (A-15): the operating tile score is the heatmap maximum for both methods; `score()` and the MVTec criteria are untouched.
- **Evidence:** `pytest -q`: 110 passed, 0 failed, 1 slow test deselected (new: calibrate 5, postprocess 7, grading 9 cases, pipeline 8, inspection config 2; fit/evaluate end-to-end tests extended). Hand-computed grading: 50, 100, 200, 300 mm lines at 4 px/mm score 1, 2, 3, 4 points; empty mask 0 points, grade A; one RAW-FABRID frame is 0.13717 sq yd. A 3 px gap is closed, a 5 px gap is not, a 10 px blob is removed. `Inspector` on a 1792x1024 image equals hand tiling, scoring and stitching bit for bit. End-to-end `fit.py` then `evaluate.py` (both methods, MVTec-like and RAW-FABRID-like synthetic trees): tau, false-alarm, recall, Dice, IoU and the per-roll grades equal an independent recomputation.
- **Decisions:** A-15 to A-20 in `DECISIONS.md`. ARCHITECTURE 5.2, 8, 9, 10 and EVALUATION 2.2, 5 updated.
- **Review:** round 1 CHANGES REQUIRED (documents only: progress log, README, spec record, a broken table in ARCHITECTURE 5.2, spec file table); fixed. Round 2 PASS, no blocking findings (two minor record items, fixed). Final suite: 110 passed.
- **Problems / next:** not seen on real data: the false-alarm rate against the 5 to 20 per 1,000 band, per-roll grade agreement on the 7 rolls, a real bundle on the laptop CPU. The `ms_per_tile` figure now includes tiling and stitching (A-18). Bundles fitted before this phase must be refitted. Next: the operator runs the Phase 4 runbook and commits `phase 4: results`.

## 2026-11-13: Phase 3 laptop work, Reverse Distillation

- **Phase:** 3 (Code complete; training-PC tasks outstanding)
- **Who / where:** developer session, laptop
- **Done:**
  - `loomguard/models/rd4ad.py`: bottleneck (OCBE), mirrored decoder, `Student`, `cosine_loss`, `anomaly_map`, `RD4AD` (fixed-budget Adam loop with bfloat16 autocast on CUDA only, validation loss logging, float16 checkpoint, `score`, `save`, `load`).
  - `RandomCropDataset` (D-22), `method: rd4ad` in `runs.py`, `scripts/fit.py` and `scripts/evaluate.py` (no parallel scripts), `tag` for separate seed runs, configs `mvtec_rd4ad.yaml` and `raw_fabrid_rd4ad_s{0,1,2}.yaml`, `slow` pytest marker.
  - Spec `docs/specs/phase-3-reverse-distillation.md` with the training-PC runbook.
- **Evidence:** `pytest -q`: 79 passed, 0 failed, 1 deselected (new: rd4ad 8, dataset 1, runs and fit/evaluate extended with rd4ad cases). Student has 83.80 M trainable parameters (fusion 10.62 + embedding 48.26 + decoder 24.92), the teacher none and bit-identical after training steps; embedding 2048x8x8, decoder outputs 256x64x64, 512x32x32, 1024x16x16. `fit.py` then `evaluate.py` run end to end for RD4AD on synthetic MVTec-like and RAW-FABRID-like trees; metrics equal an independent recomputation. Slow overfit test logic (run as a script with the identical data and settings, random-weight teacher, CPU): 8 weave tiles of 256x256, 300 iterations, mean loss per 25 iterations 0.599, 0.266, 0.221, 0.19, 0.154, 0.13, 0.093, 0.069, 0.057, 0.048, 0.043, 0.037 (2,498 s). White-noise tiles at 32x32 only reached 0.106 to 0.133 in 300 iterations.
- **Decisions:** A-10 to A-14 in `DECISIONS.md` (bfloat16 autocast; random crops from train parents; `rd4ad.seed` and `tag`; validation loss and the slow test; RD4AD tile score and result keys).
- **Review:** round 1 CHANGES REQUIRED (progress log entry, README, spec and plan ticks missing; minor: CLAUDE.md, a warning, the bundle section), all fixed. The developer session ended before its final report; the orchestrator ran round 2: PASS, no blocking findings, two minor (a missing `coreset_ratio` key in old PatchCore bundles now reads as null; this review line). Final suite: 79 passed, 1 slow test deselected.
- **Problems / next:**
  - Not seen on real data: the MVTec numbers (targets 98.5 / 96.7 / 92.0), VRAM at batch 16 (limit 11 GB), iterations per second, whether random-crop decoding bounds the speed, the RAW-FABRID results over 3 seeds. The EVALUATION compute table is still an estimate.
  - Next: the operator runs the Phase 3 runbook and commits `phase 3: results`.

## 2026-11-13: Phase 2 laptop work, metrics and PatchCore

- **Phase:** 2 (Code complete; training-PC tasks outstanding)
- **Who / where:** developer session, laptop
- **Done:**
  - `loomguard/eval/metrics.py`: exact AUROC and AP, `ScoreCounts` histograms for pixel AUROC, AP and AUPRO, 8-connected component labelling without scipy.
  - `loomguard/models/backbone.py` (frozen WideResNet-50-2 to layer3, `gaussian_blur`) and `loomguard/models/patchcore.py` (pooling, concatenation, seeded random projection, greedy k-center coreset, nearest-neighbour scoring, heatmap, float16 bank save and load).
  - `loomguard/runs.py`, `scripts/fit.py`, `scripts/evaluate.py`, `configs/mvtec_patchcore.yaml`, `configs/raw_fabrid_patchcore.yaml`.
  - Spec `docs/specs/phase-2-metrics-patchcore.md` with the training-PC runbook.
- **Evidence:** `pytest -q`: 67 passed, 0 failed, no dataset present (new: metrics 8, backbone 3, patchcore 8, runs 4, fit/evaluate 3). AUPRO equals a loop-based brute force with its own flood fill to 1e-6, a hand-derived value (0.175 / 0.3), 1.0 for a perfect map and 0.0 for a constant map. `fit.py` then `evaluate.py` run end to end on synthetic MVTec-like and RAW-FABRID-like trees (random backbone weights, tiny coreset); the reported tile metrics equal an independent tile, score, stitch, exact-metric recomputation and the pixel metrics agree within 5e-3. The backbone has 24.86 M parameters, none requiring gradients.
- **Decisions:** A-06 to A-09 in `DECISIONS.md` (own metrics code and the AUPRO step integration; `runs.py` and the config schema; two-pass evaluation on stitched images; backbone defaults and the tile-score / mask relation). EVALUATION 2.1 and 5, ARCHITECTURE 5 and 12, CLAUDE.md layout corrected.
- **Review:** 2 rounds. Round 1 CHANGES REQUIRED (progress log entry and plan status missing, spec not yet ticked, mean over undefined metrics), all fixed. Round 2 PASS with two minor doc findings (test names in the spec), fixed. Final suite: 67 passed.
- **Problems / next:**
  - Not seen on real data: the MVTec and RAW-FABRID numbers (targets 97.9 / 96.6 / 90.7), real ImageNet weights, coreset time and memory for 600 tiles, GPU latency. The laptop cannot download weights.
  - Open for Phase 4: PatchCore's tile score is the max of the unblurred grid while the mask uses the blurred heatmap (A-09), so one `tau` for both needs a rule (settled in Phase 4, A-15).
  - Next: the operator runs the Phase 2 runbook after the Phase 1 one and commits `phase 2: results`.

## 2026-11-13: Phase 1 laptop work, data layer

- **Phase:** 1 (Code complete; training-PC tasks outstanding)
- **Who / where:** developer session, laptop
- **Done:**
  - `loomguard/data/tiling.py`: `tile_origins`, `grid`, `tile`, `stitch` (float64 overlap averaging, reflect pad for short axes).
  - `loomguard/data/manifest.py`: `TileRow`, CSV I/O, `assign_val`, `select`, `build_mvtec`, `build_raw_fabrid`, `compare_with_patch_metadata`.
  - `loomguard/data/dataset.py`: `preprocess`, `TileDataset`.
  - `scripts/verify_datasets.py` for RAW-FABRID, ZJU-Leaper, MVTec AD and AITEX (untested on real data).
  - Spec `docs/specs/phase-1-data-layer.md` with the training-PC runbook.
- **Evidence:** `pytest -q`: 41 passed, 0 failed, no dataset present (tiling 13, manifest 16, dataset 6, verify script 5, smoke 1). `stitch(tile(x)) == x` bit-exact for 1792x1024, 512x512, 4096x256, 800x640, with and without `min_overlap=32`. Applying D-30 to the documented per-roll counts gives 83 val images (2,324 tiles) and 424 train images (11,872 tiles) for RAW-FABRID.
- **Decisions:** A-01 to A-05 in `DECISIONS.md` (Awaiting confirmation). EVALUATION 4.2 corrected to the A-01 numbers.
- **Review:** 2 rounds. Round 1 CHANGES REQUIRED (docs not yet synced, decision rows missing, three minor code points); round 2 PASS with five minor findings, all fixed. Final suite: 41 passed.
- **Problems / next:**
  - Not verified on real data: column names and value formats of the two RAW-FABRID metadata CSVs, the layouts assumed for ZJU-Leaper and AITEX, the patch-name format.
  - Next: the operator runs the runbook (environment with `sm_120`, downloads, `verify_datasets.py`, roll inspection) and commits `phase 1: results`. Phase 2 can start on the laptop in parallel.

## 2026-11-13: Laptop environment installed

- **Phase:** between 0 and 1 (preflight)
- **Who / where:** Zain, laptop
- **Done:** project virtual environment `.venv/` (git-ignored, 1.1 GB) with CPU PyTorch 2.15.0, torchvision 0.29.2, numpy 2.5.4, pillow 13.0.0, pyyaml 6.0.3, pytest 9.2.0, and `loomguard` installed in editable mode.
- **Evidence:** `pytest -q` gives 1 passed. An untrained WideResNet-50-2 gives layer1 256x64x64, layer2 512x32x32, layer3 1024x16x16 for a 256x256 input, as in `ARCHITECTURE.md`. `git status` is clean.
- **Decisions:** none. `WORKFLOW.md` preflight now says to activate `.venv` before launching a session.
- **Problems / next:**
  - Free disk is 2.5 GB. ImageNet weights for the backbone (about 130 MB) are not downloaded; laptop tests must build the model with `weights=None`.
  - Next: launch the Phase 1 developer session.

## 2026-11-13: Workflow revised, the developer session writes the spec

- **Phase:** between 0 and 1
- **Who / where:** Zain, laptop
- **Done:** `WORKFLOW.md`, the developer prompt, the reviewer agent and the spec template changed so that the whole phase runs in the headless session: spec, plan, implementation, review. The orchestrator only prepares, launches, verifies and pushes.
- **Evidence:** the `phase-reviewer` agent is now offered as an agent type in a Claude session opened in this repository, so its definition file loads. The workflow diagram renders.
- **Decisions:** D-55 and D-57 revised. No owner approval of the spec before coding; the safeguards are the plan the spec must cover, the reviewer's spec check, and the owner reading the spec with the final report.
- **Problems / next:**
  - Still unverified: whether the `effort` key in the reviewer's front matter takes effect.
  - Preflight for Phase 1: install `pytest` and CPU PyTorch on the laptop.
  - Next: launch the Phase 1 developer session.

## 2026-11-12: Workflow, decisions record and spec template

- **Phase:** 0 (closed)
- **Who / where:** Zain, laptop
- **Done:**
  - `docs/DECISIONS.md`: every decision agreed so far, the open ones, and what was dropped.
  - `docs/WORKFLOW.md`: spec-first process, one fresh headless developer session per phase (Sonnet 5.5, medium effort), review loop, orchestrator verification, commit points.
  - `docs/specs/TEMPLATE.md`: the template every phase spec follows.
  - `.claude/agents/phase-reviewer.md`: the reviewer agent used at the end of every phase.
- **Evidence:** all Mermaid diagrams render in headless Chrome. CLI flags used in the launch command (`-p`, `--model`, `--effort`, `--permission-mode`, `--allowedTools`, `--name`) checked against `claude --help`, version 2.1.322.
- **Decisions:** Lusitano dropped (D-14). Third method stays open (O-01). Workflow decisions D-55 to D-57.
- **Problems / next:**
  - The workflow has not been run yet. Phase 1 is its first use; adjust `WORKFLOW.md` after it if something does not work in practice.
  - Not verified: that the `phase-reviewer` agent file loads as written (in particular the `effort` key in its front matter). The first developer session will show it.
  - `pytest` and CPU PyTorch are not installed on the laptop. Needed in the Phase 1 preflight.
  - Next: write the Phase 1 spec.

## 2026-11-12: Repository skeleton and team decisions

- **Phase:** 0 (review) and 1 (task 1)
- **Who / where:** Zain, laptop
- **Done:**
  - Package skeleton: `loomguard/{data,models,eval,inspection}`, `scripts/`, `app/`, `configs/`, `tests/`, `results/`, `assets/samples/`.
  - `pyproject.toml`, `.gitignore`, `CLAUDE.md`, one import smoke test.
- **Evidence:** `tests/test_smoke.py` run directly with Python: all five packages import. `pytest` is not installed on the laptop yet.
- **Decisions:**
  - Deployment: Hugging Face Spaces, free CPU tier. Front end stays Streamlit (course requirement).
  - Model bundles: Hugging Face Hub model repository.
  - Grade cut-offs accepted: A up to 20, B up to 40, C up to 60 points per 100 square yards, Reject above.
  - Package renamed from `inspect` to `inspection`. A folder named `inspect` shadows the standard library when a script runs from inside `loomguard/`.
  - PyTorch is not a declared dependency; it is installed first from the CPU or CUDA 12.8 index.
- **Problems / next:**
  - Still open: third method (Dinomaly or none), Lusitano (keep as optional or drop).
  - Next: Phase 1 tasks 2 to 4 on the laptop (tiler, manifest builders, dataset class); tasks 5 to 7 on the training PC.

## 2026-11-12: Phase 0, design and dataset verification

- **Phase:** 0
- **Who / where:** Zain, laptop (no GPU, no dataset downloads)
- **Done:**
  - Verified all six candidate datasets without downloading them. Results in `DATASETS.md`.
  - Fixed the architecture of both methods with tensor shapes and parameter counts. `ARCHITECTURE.md`.
  - Fixed metrics, splits, regimes, ablations and pass criteria. `EVALUATION.md`.
  - Wrote the phased plan with acceptance criteria. `IMPLEMENTATION_PLAN.md`.
- **Evidence:**
  - RAW-FABRID: zip listing read remotely (21,457 files), both metadata CSVs analysed. Counts 14,196 / 4,969 / 687 confirmed. Official split is by parent image. Scale is 4 px/mm.
  - ZJU-Leaper: Google Drive file alive, 5.8 GB. Layout taken from the official loader `zl_fabric.py`.
  - AITEX: three archives alive, 77.7 MB + 98.3 MB + 88 KB.
  - TSFabrics: Figshare record 29573381, 9.19 GB, CC BY 4.0.
  - Lusitano: 5.96 GB + 13.3 GB. Host TLS certificate expired. Paper reports image-level metrics only.
  - Reference numbers copied from the PatchCore and RD4AD papers' tables.
  - Parameter counts computed from the layer definitions: teacher 24.86 M (frozen), student 83.80 M (trainable).
- **Decisions:**
  - RAW-FABRID promoted to primary dataset. Lusitano and TSFabrics moved to optional.
  - Tiles are always 256x256 at native resolution. MVTec AD is the only resized dataset.
  - PatchCore: at most 600 tiles per bank, 10 percent coreset for benchmarks, 1 percent for the CPU demo.
  - RD4AD: fixed iteration budget, last checkpoint, no test-set model selection.
  - One threshold per bundle from normal validation tiles at a 1 percent false-alarm target.
  - Grading in real millimetres where the dataset gives a scale; grade is a roll-level quantity.
  - Recommended third method is Dinomaly (ViT-Small), decided at the Phase 7 gate.
- **Problems / next:**
  - Not verified: ZJU-Leaper per-split counts, MVTec AD archive size, AITEX mask pairing, whether RAW-FABRID rolls are different fabrics, whether Lusitano ships masks. All are Phase 1 tasks on the training PC.
  - Open questions for the team are listed in `README.md`.
  - Next: team review of the documents, then Phase 1.
