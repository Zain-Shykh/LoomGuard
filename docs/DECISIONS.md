# LoomGuard: Decisions

The single record of what the team has agreed, what is still open, and what was dropped. If a decision is not here, it has not been made.

How to use this file:

- **Accepted** decisions are binding. To change one, add a new row that supersedes it and say why. Do not edit history silently.
- **Awaiting confirmation** holds decisions a developer session had to make on its own during a phase. The project owner confirms or reverses them when the phase is reviewed.
- **Open** decisions have not been made. Nobody implements anything that depends on an open decision.
- Detailed reasoning lives in the linked document. This file states the decision and the short reason.

## Accepted

### Scope and tools

| ID | Date | Decision | Reason | Details |
|---|---|---|---|---|
| D-01 | 2026-11-12 | Two core methods: PatchCore and Reverse Distillation (RD4AD), compared on the same data | One stores normal patches, one learns them; the comparison is the research result | ARCHITECTURE 1 |
| D-02 | 2026-11-12 | Implement both methods ourselves from the papers. No anomaly-detection library. | Learning is the project goal | ARCHITECTURE 14 |
| D-03 | 2026-11-12 | PyTorch. Training on one RTX 5070 (12 GB VRAM) with PyTorch 2.7 or newer (hardware superseded by D-59: RTX 5060 Ti, 16 GB) and CUDA 12.8 wheels | The only GPU available; older builds do not support the RTX 50 series | ARCHITECTURE 13 |
| D-04 | 2026-11-12 | The laptop does code, docs and CPU tests only. No dataset downloads, no training. | No GPU and about 3.6 GB of free disk | ARCHITECTURE 13 |
| D-05 | 2026-11-12 | Front end is Streamlit | Course requirement | ARCHITECTURE 11 |
| D-06 | 2026-11-12 | Deploy the app on Hugging Face Spaces, free CPU tier. Any free host is acceptable if that fails. | Free, and enough memory for PyTorch on CPU | ARCHITECTURE 11 |
| D-07 | 2026-11-12 | Model bundles are stored in a Hugging Face Hub model repository | GitHub rejects files over 100 MB; an RD4AD student is 168 MB | ARCHITECTURE 10 |
| D-08 | 2026-11-12 | No experiment tracker. Results are small JSON files in `results/`, committed to git. | Enough for two methods; results visible without a GPU | EVALUATION 5 |

### Data

| ID | Date | Decision | Reason | Details |
|---|---|---|---|---|
| D-10 | 2026-11-12 | RAW-FABRID is the primary dataset (localization, grading, demo) | Fully verified, CC BY 4.0, masks, 7 rolls, 4 px/mm | DATASETS 1 |
| D-11 | 2026-11-12 | ZJU-Leaper is the cross-fabric benchmark, official Setting 1 | 19 patterns in 5 groups with masks and official splits | DATASETS 2 |
| D-12 | 2026-11-12 | MVTec AD textures (carpet, grid, leather, tile, wood) are used only to check our implementations against published numbers. Never shipped in the demo. | Non-commercial licence | DATASETS 3 |
| D-13 | 2026-11-12 | AITEX is a small-data stress test, not a head-to-head comparison with papers | No official split exists | DATASETS 4 |
| D-14 | 2026-11-12 | **Lusitano is dropped** | No documented masks, 19 GB, no licence, expired TLS certificate on the host; no core question depends on it | DATASETS, Dropped |
| D-15 | 2026-11-12 | Datasets are never written out as tiles. A manifest CSV lists crop boxes and tiles are cropped at load time. | Saves disk; one loading path for every dataset | ARCHITECTURE 3 |

### Models

| ID | Date | Decision | Reason | Details |
|---|---|---|---|---|
| D-20 | 2026-11-12 | Shared backbone: WideResNet-50-2, ImageNet weights, frozen, always in `eval()` mode. PatchCore uses layer2 and layer3. RD4AD uses layer1, layer2, layer3. | Same features for both methods makes the comparison fair | ARCHITECTURE 4 |
| D-21 | 2026-11-12 | Models see 256x256 tiles at native resolution. Images are tiled, never shrunk. MVTec AD is the one exception (whole image resized to 256). | Thin defects vanish when downscaled; MVTec must follow the published protocol | ARCHITECTURE 3 |
| D-22 | 2026-11-12 | No flips or rotations as augmentation. RD4AD trains on random crops only. | Weave direction is part of what is normal | ARCHITECTURE 3 |
| D-23 | 2026-11-12 | PatchCore: 1,536-d patch vectors, at most 600 tiles per bank, greedy k-center coreset on a 128-d projection, keep 10% for benchmarks and 1% for the CPU demo, bank stored in float16 | Bounded memory and search time | ARCHITECTURE 5 |
| D-24 | 2026-11-12 | RD4AD: Adam, lr 0.005, betas (0.5, 0.999), batch 16, mixed precision, 5,000 iterations per fabric (10,000 per group, 20,000 unified) | Paper settings, with an iteration budget instead of epochs | ARCHITECTURE 6 |
| D-25 | 2026-11-12 | RD4AD keeps the last checkpoint. The test split is never used to pick a checkpoint. | The official code selects the best epoch on the test set, which leaks | ARCHITECTURE 6.2 |

### Evaluation

| ID | Date | Decision | Reason | Details |
|---|---|---|---|---|
| D-30 | 2026-11-12 | Split by parent image first, then tile. Three pools: train (normal), val (normal), test. Val is 15% of normal parents, at least 200 tiles, at most 40%. | No tile leakage; threshold set without defect data | EVALUATION 4 |
| D-31 | 2026-11-12 | Primary metrics: AUROC for detection, AUPRO up to FPR 0.3 for localization. Secondary: AP, pixel AUROC, pixel AP. | AUPRO weighs every defect region equally | EVALUATION 2 |
| D-32 | 2026-11-12 | One threshold per bundle: the 99% quantile of heatmap maxima on normal validation tiles. Used for both the tile decision and the pixel mask. | One source of truth; targets 1% false alarms | ARCHITECTURE 8 |
| D-33 | 2026-11-12 | Implementation pass criteria on MVTec textures: PatchCore 97.9 / 96.6 / 90.7 and RD4AD 98.5 / 96.7 / 92.0 (image AUROC / pixel AUROC / AUPRO, mean of five categories) | Published means minus 1.0 (AUROC) and 3.0 (AUPRO) | EVALUATION 3 |
| D-34 | 2026-11-12 | Training regimes: per fabric, per group, unified, and held-out fabric | These answer the cross-fabric questions | EVALUATION 4.3 |
| D-35 | 2026-11-12 | Headline settings are fixed in advance. Ablations are reported as ablations and never used to tune the headline configuration on test data. | Keeps test results honest | EVALUATION 4.1 |

### Grading

| ID | Date | Decision | Reason | Details |
|---|---|---|---|---|
| D-40 | 2026-11-12 | Four-point system on connected defects, measured in millimetres. Real scale for RAW-FABRID (4 px/mm) and TSFabrics (5 px/mm); elsewhere an assumed scale, labelled as such. | Two datasets state a physical scale | ARCHITECTURE 9 |
| D-41 | 2026-11-12 | Grade cut-offs in points per 100 square yards: A up to 20, B up to 40, C up to 60, Reject above. Kept in the config. | 40 is the usual acceptance limit; the rest are project assumptions the team accepted | ARCHITECTURE 9 |
| D-42 | 2026-11-12 | A grade is a roll-level result. A single image shows defects and points, with the grade marked as an extrapolation. | One frame is 0.137 square yards | ARCHITECTURE 9 |
| D-43 | 2026-11-12 | Holes are scored by length like any other defect | No defect-type labels | ARCHITECTURE 9 |

### Repository and process

| ID | Date | Decision | Reason | Details |
|---|---|---|---|---|
| D-50 | 2026-11-12 | The package is `loomguard/inspection`, not `inspect` | `inspect` shadows the standard library when a script runs from inside `loomguard/` | CLAUDE.md |
| D-51 | 2026-11-12 | PyTorch is not a declared dependency. It is installed first from the CPU or the CUDA 12.8 index. | A plain install pulls the multi-gigabyte CUDA build onto the laptop | README |
| D-52 | 2026-11-12 | Datasets, model bundles and MVTec images are never committed | Size and licence | ARCHITECTURE 10 |
| D-53 | 2026-11-12 | Evaluation and the app share one code path, the `Inspector` | Report numbers come from the code the demo runs | ARCHITECTURE 12 |
| D-54 | 2026-11-12 | Progress is judged by acceptance criteria with evidence, not by dates | | IMPLEMENTATION_PLAN |
| D-55 | 2026-11-12, revised 2026-11-13 | Spec-first workflow: every phase gets a spec in `docs/specs/` from the template before any code. **The developer session of that phase writes the spec itself**, from the plan and the design documents; it must keep every acceptance criterion of the plan. (First version: the orchestrator wrote the spec and the owner approved it before coding.) | Context lives in documents, not in chat; the whole phase runs in one fresh context | WORKFLOW |
| D-56 | 2026-11-12 | Each phase is done by a fresh headless Claude session (Sonnet 5.5, medium effort): spec, plan, implementation, then a review loop against a reviewer agent that also checks the spec against the plan | A clean context per phase; independent check against drift and stale docs | WORKFLOW |
| D-57 | 2026-11-12, revised 2026-11-13 | The main session (Fable 5.2) is the orchestrator: launches sessions with the prompt, verifies their reports, pushes. It does not write specs or phase code. Three commits per phase (spec, implementation, review), made locally by the developer session and pushed by the orchestrator after verification. | High-level control stays in one place | WORKFLOW |
| D-58 | 2026-11-13 | All laptop work of every phase is done first, phase after phase, with the training-PC runbooks queued. The training-PC operator then runs all runbooks in phase order. A phase is `Code complete` until its runbook has been run. | The owner wants the implementation finished on the laptop before the teammate's GPU time is needed. The cost is that integration bugs with real data appear late, so every phase must carry synthetic end-to-end tests and a runbook the operator can follow without the developer. | WORKFLOW |
| D-59 | 2026-11-17 | The training PC is an RTX 5060 Ti with 16 GB VRAM, not an RTX 5070 with 12 GB. Supersedes the hardware in D-03; the rest of D-03 holds (PyTorch 2.7 or newer, CUDA 12.8 wheels, `sm_120`, since it is the same Blackwell generation). All fixed settings stay unchanged, including batch 16 and the 11 GB peak-VRAM bar, which was set for a 12 GB card and is kept so results stay portable. Run times in the docs were estimated for a faster card: expect the card to be roughly a third slower (fewer cores, lower memory bandwidth; not measured), so replace the estimates with measured speeds. | The owner's teammate has this card | README, CLAUDE.md |
| D-60 | 2026-11-21 | RD4AD mixed precision is `torch.autocast` with float16 plus `torch.amp.GradScaler` on CUDA, for every RD4AD run on every machine (the 5060 Ti included, so all cells share one protocol). Supersedes A-10 (bfloat16). Reason: Kaggle's T4 and P100 lack native bfloat16; on a T4 emulated bf16 ran at 0.78 it/s against 3.47 it/s for fp16 (fp32 1.30). The MVTec RD4AD gate checks accuracy; if it misses its target, record it and revisit (options: an Ampere or newer GPU for bf16, or all RD4AD on the 5060 Ti), without tuning on test results. | Lets the Kaggle accounts run RD4AD; nothing was trained yet, so no reruns | PROGRESS_LOG 2026-11-21 |

### Made by developer sessions, confirmed by the owner on 2026-11-14

| ID | Date | Phase | Decision | Reason |
|---|---|---|---|---|
| A-01 | 2026-11-13 | 1 | D-30's validation rule (15% of normal parents, at least 200 tiles, at most 40%) is applied per fabric profile (per roll, per AITEX fabric), the cap winning over the minimum. For RAW-FABRID this gives 83 val images (2,324 tiles) and 424 train images (11,872 tiles), not the "about 76 / 2,128 / 431 / 12,068" first written in EVALUATION 4.2, which was a plain 15% per roll. | Bundles and thresholds are per fabric, and the rule reproduces the AITEX 12/8 split in EVALUATION. EVALUATION corrected. Reverse by passing `min_val_tiles=0` to `build_raw_fabrid`'s call of `assign_val`. |
| A-02 | 2026-11-13 | 1 | Manifest schema: one CSV row per tile with columns dataset, fabric, split, parent, image, mask, x0, y0, w, h, resize, label, defect_px; paths relative to the dataset root with `/`; MVTec is one whole-image row with `resize=256`; split assigned per parent, per fabric, shared by the per-fabric and unified regimes. | The documents fixed D-15 (crop boxes) but not the columns. All later phases read manifests through `read_manifest`, so a change stays in `loomguard/data/manifest.py`. |
| A-03 | 2026-11-13 | 1 | Split seed 0 by default; the val parents are a shuffle of the sorted parent ids with `random.Random(f"{seed}/{fabric}")`. MVTec val is 10% of `train/good` with no tile minimum (EVALUATION 4.2). | Deterministic and independent of directory order. |
| A-04 | 2026-11-13 | 1 | `tiling.grid()` is a public helper (the one definition of the tile grid, used by the tiler, the manifest builder and the verification script). Tiles are in row-major order. Origins use round-half-up in integer arithmetic. Stitching accumulates in float64. | One definition of the grid; exact round trip. |
| A-05 | 2026-11-13 | 1 | `compare_with_patch_metadata` checks the RAW-FABRID manifest against the official patch CSV per parent image (tile count and sorted nonzero defect areas) and per `type` pool counts, because how patch names encode the grid position is not documented. | Avoids guessing the naming scheme; a position-exact check can be added on the training PC. |
| A-06 | 2026-11-13 | 2 | All metrics are our own code (no scikit-learn, none installed): exact AUROC and AP, plus histogram counts (`ScoreCounts`) for pixel metrics. AUPRO uses the 8-connected components of each mask as regions and integrates the PRO curve as a step function (value at FPR f = best PRO with FPR <= f), so a constant heatmap scores exactly 0 and a perfect one exactly 1. | EVALUATION 2.1 said "from `sklearn.metrics`". Linear interpolation would give a constant map 0.15, but the plan says it must score about 0; the step form differs from interpolation only by the width of tied score steps. Reverse by changing `ScoreCounts.aupro`. EVALUATION corrected. |
| A-07 | 2026-11-13 | 2 | New file `loomguard/runs.py` (config loading, manifest dispatch, fabric profiles, stratified tile cap, bundle and result paths) shared by `scripts/fit.py` and `scripts/evaluate.py`. Config keys as in the Phase 2 spec 4.4. `fit.py` writes `model.pt`, `config.yaml` and a `metrics.json` with fit-side system metrics (tiles, bank size, time, coreset seconds, peak RAM and VRAM, size); `evaluate.py` merges the test metrics into it and writes the result JSON. | The documents fixed the bundle layout but not the plumbing. Keeps the scripts thin and testable. Phase 4 moves scoring behind `Inspector`. |
| A-08 | 2026-11-13 | 2 | `evaluate.py` scores the test tiles twice (pass 1: heatmap range, tile scores, timing; pass 2: histograms) and computes pixel metrics on whole parent images stitched from their tiles, 10,000 bins. | Exact histogram range with nothing clipped, region-correct AUPRO for defects that straddle tile borders, bounded memory. The range only places bins; metrics are threshold-free. |
| A-09 | 2026-11-13 | 2 | `Backbone(weights="IMAGENET1K_V1")` defaults to the real weights; `None` (random) is for tests only. `gaussian_blur` (sigma 4, shared by PatchCore and RD4AD) lives in `backbone.py`. PatchCore's tile score is the max of the unblurred distance grid (ARCHITECTURE 5.2), so a tile can exceed `tau` while no pixel of the blurred mask does; how Phase 4 reconciles the tile decision with the mask under D-32 is left to Phase 4 (settled in A-15). | A forgotten argument must not silently give random features. |
| A-10 | 2026-11-13 | 3 | **Superseded by D-60 (2026-11-21): fp16 + GradScaler.** RD4AD mixed precision is `torch.autocast` with bfloat16 on CUDA only, no GradScaler; the cosine loss is computed in float32; autocast is off on CPU so the tests run the same code. | D-24 says "mixed precision" without a dtype. bfloat16 has float32's range, needs no loss scaler, and the RTX 5070 supports it. Switching to float16 plus a scaler is a few lines. |
| A-11 | 2026-11-13 | 3 | RD4AD trains on `RandomCropDataset`: item i is a 256 x 256 window at a uniformly random origin of a uniformly random train parent image (extent = union of its manifest crop boxes), deterministic in (seed, i); an MVTec row is the whole resized image. `TileDataset` gains `get(row)`. | D-22 says random crops; the manifest only holds grid tiles. Crops come from train parents only, so rules 1 and 2 are untouched. |
| A-12 | 2026-11-13 | 3 | Config keys `rd4ad.seed` (default: the config seed) and `tag` (appended to the method directory: `rd4ad_s1`). The three RAW-FABRID seeds share the split (config seed 0) and differ in initialisation and crops. | EVALUATION says "3 seeds for RD4AD" without saying what varies. A fixed split keeps seeds comparable with PatchCore and measures training variance. |
| A-13 | 2026-11-13 | 3 | RD4AD validation loss is taken on up to `rd4ad.val_tiles` (64) normal `val` tiles with the student in `eval()`, every `val_every` steps and at the end, logged only. The 300-iteration overfit test is marked `slow` (deselected by default, `pytest -m slow`) and runs at 256 x 256 on 8 fabric-like weave tiles (white-noise tiles at 32 x 32 only reached 0.106 in 300 iterations). | D-25 only says the validation loss is logged to spot divergence. An 84 M-parameter Adam step costs about a second on CPU, too slow for the default suite. |
| A-14 | 2026-11-13 | 3 | RD4AD's tile score is the max of the blurred heatmap (ARCHITECTURE 6.3), so for RD4AD the tile decision and the mask share one map; PatchCore keeps the unblurred-grid max (A-09). Result JSON for RD4AD replaces `coreset_*`/`bank_size` by `iterations`, `iters_per_second`, `final_train_loss`, `final_val_loss`. | Follows ARCHITECTURE as written; keeps Phase 2 result files unchanged. |
| A-15 | 2026-11-14 | 4 | Settles the open point of A-09. The operating tile score is the maximum of the blurred tile heatmap for both methods: calibration takes its maxima on val tiles, a tile is flagged iff `peak > tau`, the mask is `heatmap > tau`, so decision and mask cannot disagree under D-32. `model.score()[0]` is unchanged (PatchCore: unblurred grid max; RD4AD: heatmap max) and still feeds `tile_auroc` and `tile_ap`, so the Phase 2 MVTec pass criteria are untouched. ARCHITECTURE 5.2 and 8 say so. | ARCHITECTURE 8 already said "take each tile's heatmap maximum"; only 5.2 said otherwise. If the owner prefers the unblurred grid for the decision, only `Scored.peaks` and `val_peaks` change, but a flagged tile could then have an empty mask. |
| A-16 | 2026-11-14 | 4 | Config keys `target_far` (0.01), `px_per_mm` (4.0), `scale_assumed` (true; RAW-FABRID false), `grading: {closing 5, min_area_px 16, cutoffs [20, 40, 60]}`, `grade_rolls` (false; RAW-FABRID true), `resize` (256 for MVTec). `calibration.json` holds tau, target_far, n_val_tiles, low_confidence (n < 500, EVALUATION 4.1) and score statistics. `fit.py` writes it; `evaluate.py` refuses a bundle without it. | The documents named the knobs but not the keys or the file schema. |
| A-17 | 2026-11-14 | 4 | Per-roll grading metrics (EVALUATION 2.3) are computed over the test-pool frames of each fabric: predicted and ground-truth masks go through the same cleanup and `grade_roll`. RAW-FABRID's test pool has defective images only, so these are partial rolls, mostly Reject; the comparison of predicted against ground truth on the same frames stays valid. Train and val frames are not scored for grading (the model was fitted on train; val sets tau). | The documents do not say which frames form a roll. Scoring frames the model was fitted on would flatter it. |
| A-18 | 2026-11-14 | 4 | `Inspector` takes PIL images (resizing MVTec with PIL bilinear exactly as `TileDataset` does) and `evaluate.py` scores whole parent images through it, mapping tiles to manifest rows by (y0, x0). Consequences: batches are per image, `ms_per_tile_<device>` now includes tiling and stitching, all grid tiles of a parent are scored. `Inspector.load(bundle, device, backbone=None)` reads the bundle's own `config.yaml`. | CLAUDE.md rule 5. Metrics are unchanged to numeric precision (tests recompute independently). |
| A-19 | 2026-11-14 | 4 | Mask cleanup without opencv or scipy: closing by max-pool (5 x 5, border not eroded), area filter on `connected_components` from `metrics.py`, defect length from a 1 degree rotation search for the minimum-area rectangle over each row's end pixels (a side is extent + 1 pixel). A point band is an inclusive upper limit (75 mm scores 1). A single frame's grade is carried in `Inspection.grade` and is an extrapolation (D-42). | Nothing else is installed; the error is about 1 px on a 300 px defect. Upgrade path: rotating calipers. |
| A-20 | 2026-11-14 | 4 | Bundle upload to the Hugging Face Hub is a runbook step with `huggingface-cli`, not code and not a dependency. The app's download code adds `huggingface_hub` to `pyproject.toml` in Phase 6 (superseded by A-26: the app downloads with `urllib`, no new dependency). | Rule 7: a dependency belongs to the phase that needs it. |
| A-21 | 2026-11-14 | 5 | Plots are hand-written SVG from the standard library (`scripts/summarize.py`), tables are markdown and CSV. No matplotlib. | Rule 7 and the laptop cannot install packages; two-method bar charts need nothing more. If richer plots are wanted, add matplotlib to `pyproject.toml` and replace `bar_svg`. |
| A-22 | 2026-11-14 | 5 | Held-out regime: ZJU-Leaper holds out the last pattern of each group (3, 7, 12, 15, 19) and trains on the rest of the group (`held_out_scope: group`); RAW-FABRID holds out each roll and trains on the other six. RD4AD iterations match the bundle compared with: 10,000 (ZJU group) and 20,000 (RAW-FABRID unified). Tested on the held-out fabric's official test pool only; `tau` from the validation tiles of the training fabrics. Profile and result name `<fabric>_held_out`. | EVALUATION 4.3 leaves the patterns and the budget open. Matching the budget isolates seen against unseen fabric, but RAW-FABRID held-out RD4AD alone costs about 9.7 GPU hours (the compute table's 8 h covered all held-out and ablation runs). Reverse by editing `iterations` in `configs/raw_fabrid_held_out_rd4ad.yaml` (10,000 halves it) before any run. |
| A-23 | 2026-11-14 | 5 | Ablation datasets and regimes where EVALUATION 4.4 names only a dataset: coreset ratio on RAW-FABRID per roll and ZJU-mini per pattern; training-set size and overlap on RAW-FABRID per roll; resolution on ZJU-mini per pattern (both methods); seeds on RAW-FABRID unified, seed 0 being the headline. Tags `res256`, `cr1`, `cr25`, `n50`, `n200`, `it1k`, `ov32`, `s1`, `s2`. Overlap never changes the split (the split uses the no-overlap tile count). | Per-roll runs are cheap for PatchCore and give 7 data points per ablation. |
| A-24 | 2026-11-14 | 5 | ZJU-Leaper adapter: fabric `pattern<n>`; split per pattern with 10 percent val and no tile minimum (EVALUATION 4.2); per-group, unified and held-out sets are unions of the per-pattern files (the official group and total JSON files are not read); group membership is hard-coded from DATASETS 2; ids resolve through a file index; any inconsistency stops the run with counts. ZJU-mini runs are the headline results of those five patterns (configs `zju_per_fabric_mini` and `zju_per_fabric_rest`). AITEX adapter: fabric from the trailing `_<NN>` of the file name; masks paired by `<stem>_mask*` and united (` | `-joined paths in the manifest, `load_mask`); 12/8 split by the A-01 rule; black-tile filter (below gray 10 on more than half the pixels). |
| A-25 | 2026-11-14 | 5 | `evaluate.py` adds `image_auroc` and `image_ap` (max over the parent's tiles) and `by_fabric` for bundles covering several fabrics; `dump_predictions: true` writes predicted masks and `scores.csv` under `artifacts/predictions/` as input for ZLEval. Configs may `extends` base fragments. The driver `scripts/study.py` skips configs whose result files exist (`--force` reruns). | ZJU-Leaper detects per image (EVALUATION 4.2); Q3 and Q4 need the same fabric under every regime. The ZLEval input format is unverified, so the glue code is training-PC work. |
| A-26 | 2026-11-14 | 6 | The app downloads bundles with `urllib` from `<hub>/<repo>/resolve/main/<dataset>/<fabric>/<method>/{model.pt,calibration.json,config.yaml}` (public repository, 600 MB cap per file, atomic rename into the bundles folder, which is the cache). `huggingface_hub` is not added. Supersedes the dependency part of A-20; the upload stays a runbook step with `huggingface-cli`. `assets/profiles.json` lists what the Space offers and is used only when a Hub repository is set; bundles on disk are always listed. | Rule 7 and "nothing else can be installed": three public files need no client library. Reverse by swapping `ensure_bundle`'s body for `huggingface_hub.snapshot_download` and adding the package to the `app` extra. Real Hub URL behaviour is checked in runbook step 2b. |
| A-27 | 2026-11-14 | 6 | "Unknown fabric" rule: warn when the cleaned mask covers at least 30 percent of the pixels, or when at least 50 percent of the tiles of an image with 4 or more tiles are flagged. The warning replaces the verdict, defect table, points and grade. In a roll, such frames are still counted (as in `evaluate.py`) and the number is reported. | ARCHITECTURE 11 says "nearly the whole image is flagged" without a number. The three constants are named in `app/streamlit_app.py` and are guesses until real wrong-fabric images are seen (runbook step 5). |
| A-28 | 2026-11-14 | 6 | Streamlit is the optional extra `app = ["streamlit>=1.45"]`, not a core dependency; `pip install -e ".[dev,app]"` runs all tests. Upload limits: 25 MB, 16 megapixels, both sides at least 256 px (one tile), format from the content (PNG, JPEG, TIFF, BMP). `.streamlit/config.toml` sets `maxUploadSize` to 32 MB so the app's own message is shown. Space files live in `deploy/` (copied into the Space repository), not at the repository root. | The documents name the formats but give no numbers; 16 MP covers the 5 MP ZJU-Leaper images with room. A root `requirements.txt` with the CPU index would break `pip install -r` on the training PC. |
| A-29 | 2026-11-14 | 6 | UI conventions: the sensitivity slider is a factor `s` from 0.5 to 2.0 with `tau_used = tau / s` (higher flags more); the scale is "stated by the dataset" only for a sample whose `samples.json` entry names a profile, every upload is "assumed scale"; a demo roll is `assets/rolls/<name>/roll.json` (profile, ordered frames, credit) and its final grade is comparable with `evaluate.py`'s per-roll number only when it holds the test-pool frames of that fabric (A-17). Scale is not editable in the UI. | The documents fix the labels but not the control ranges or the file formats. |

## Awaiting confirmation

Decisions made by a developer session during a phase. A-01 to A-29 were confirmed by the owner on 2026-11-14 and moved to "Accepted". New rows start at A-30.

| ID | Date | Phase | Decision | Reason | Status |
|---|---|---|---|---|---|
| | | | | | |

The three-person training workflow and its initial ownership ledger are [proposed operational changes](DISTRIBUTED_TRAINING.md). The existing scientific decisions and full-study coverage govern the runs. Record team approval of the workflow, any reassignment, and the AITEX inclusion decision before using them. Any reduction to the accepted full-study scope requires a separate team decision.

## Open

| ID | Question | Options | Decide by | Notes |
|---|---|---|---|---|
| O-01 | Add a third method? (owner, 2026-11-14: deferred until the core study is finished; no third method for now) | (a) none, stop at two; (b) Dinomaly with ViT-Small; (c) EfficientAD | Phase 7 gate, after Phase 5 results | See below |
| O-02 | Which other stretch goals, if any? | TSFabrics stream with the cutline test; defect-type classification from a few labels; automatic fabric-profile selection in the app | Phase 7 gate | At most one or two |

### O-01 background: Dinomaly with ViT-Small

- **What it produces:** the same outputs as the other methods, a score per tile and a 256x256 heatmap. It plugs into the existing threshold, grading and app as a third model choice.
- **Value:** it is designed for one model covering many classes, which is the unified cross-fabric regime where RD4AD is expected to degrade. It also uses a self-supervised transformer, so the study can separate the effect of the method from the effect of the backbone. The paper reports 99.3 image AUROC on multi-class MVTec AD with ViT-Small.
- **Workload (estimates, not measured):** 300 to 400 lines of new code (DINOv2 encoder wrapper, MLP bottleneck, 8-layer transformer decoder with linear attention, loss, optimiser with warm-up). 1 to 1.5 weeks of developer time. About 25 GPU hours for the full study, or a few hours if run only in the unified and per-group regimes.
- **Fit on 12 GB and on the 16 GB card:** very likely. 37 M parameters in total against 84 M trainable for RD4AD. The paper gives no memory figure, so the first run must measure it.
- **Risks:** a 20x20 feature grid per tile may localize thin defects worse than RD4AD's 64x64 finest scale. It is the largest single scope addition.
- **Current recommendation:** decide after the two-method study. If RD4AD clearly drops in the unified regime, add Dinomaly for the unified and per-group runs only.
- EfficientAD remains an option but is not recommended: it needs an ImageNet image stream during training, has no official code, and its strength (latency) is not our bottleneck.

## Dropped or rejected

| Item | Date | Reason |
|---|---|---|
| Lusitano dataset | 2026-11-12 | See D-14. May be revisited at the Phase 7 gate if the host is fixed. |
| Tianchi, Roboflow fabric sets, TILDA | starter | Restricted access; boxes only and augmented; old and low resolution |
| EfficientAD as the default third method | 2026-11-12 | Replaced by open decision O-01 |
| A grade per single image | 2026-11-12 | See D-42 |
| Using an anomaly-detection library such as anomalib | 2026-11-12 | See D-02 |
| GitHub Releases for model bundles | 2026-11-12 | See D-07 |
| Streamlit Community Cloud | 2026-11-12 | Memory limit is tight for PyTorch; see D-06 |
| Week-by-week schedule | 2026-11-12 | See D-54 |
