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
| D-03 | 2026-11-12 | PyTorch. Training on one RTX 5070 (12 GB VRAM) with PyTorch 2.7 or newer and CUDA 12.8 wheels | The only GPU available; older builds do not support the RTX 50 series | ARCHITECTURE 13 |
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

## Awaiting confirmation

Decisions made by a developer session during a phase.

| ID | Date | Phase | Decision | Reason | Status |
|---|---|---|---|---|---|
| A-01 | 2026-11-13 | 1 | D-30's validation rule (15% of normal parents, at least 200 tiles, at most 40%) is applied per fabric profile (per roll, per AITEX fabric), the cap winning over the minimum. For RAW-FABRID this gives 83 val images (2,324 tiles) and 424 train images (11,872 tiles), not the "about 76 / 2,128 / 431 / 12,068" first written in EVALUATION 4.2, which was a plain 15% per roll. | Bundles and thresholds are per fabric, and the rule reproduces the AITEX 12/8 split in EVALUATION. EVALUATION corrected. Reverse by passing `min_val_tiles=0` to `build_raw_fabrid`'s call of `assign_val`. | Awaiting |
| A-02 | 2026-11-13 | 1 | Manifest schema: one CSV row per tile with columns dataset, fabric, split, parent, image, mask, x0, y0, w, h, resize, label, defect_px; paths relative to the dataset root with `/`; MVTec is one whole-image row with `resize=256`; split assigned per parent, per fabric, shared by the per-fabric and unified regimes. | The documents fixed D-15 (crop boxes) but not the columns. All later phases read manifests through `read_manifest`, so a change stays in `loomguard/data/manifest.py`. | Awaiting |
| A-03 | 2026-11-13 | 1 | Split seed 0 by default; the val parents are a shuffle of the sorted parent ids with `random.Random(f"{seed}/{fabric}")`. MVTec val is 10% of `train/good` with no tile minimum (EVALUATION 4.2). | Deterministic and independent of directory order. | Awaiting |
| A-04 | 2026-11-13 | 1 | `tiling.grid()` is a public helper (the one definition of the tile grid, used by the tiler, the manifest builder and the verification script). Tiles are in row-major order. Origins use round-half-up in integer arithmetic. Stitching accumulates in float64. | One definition of the grid; exact round trip. | Awaiting |
| A-05 | 2026-11-13 | 1 | `compare_with_patch_metadata` checks the RAW-FABRID manifest against the official patch CSV per parent image (tile count and sorted nonzero defect areas) and per `type` pool counts, because how patch names encode the grid position is not documented. | Avoids guessing the naming scheme; a position-exact check can be added on the training PC. | Awaiting |
| A-06 | 2026-11-13 | 2 | All metrics are our own code (no scikit-learn, none installed): exact AUROC and AP, plus histogram counts (`ScoreCounts`) for pixel metrics. AUPRO uses the 8-connected components of each mask as regions and integrates the PRO curve as a step function (value at FPR f = best PRO with FPR <= f), so a constant heatmap scores exactly 0 and a perfect one exactly 1. | EVALUATION 2.1 said "from `sklearn.metrics`". Linear interpolation would give a constant map 0.15, but the plan says it must score about 0; the step form differs from interpolation only by the width of tied score steps. Reverse by changing `ScoreCounts.aupro`. EVALUATION corrected. | Awaiting |
| A-07 | 2026-11-13 | 2 | New file `loomguard/runs.py` (config loading, manifest dispatch, fabric profiles, stratified tile cap, bundle and result paths) shared by `scripts/fit.py` and `scripts/evaluate.py`. Config keys as in the Phase 2 spec 4.4. `fit.py` writes `model.pt`, `config.yaml` and a `metrics.json` with fit-side system metrics (tiles, bank size, time, coreset seconds, peak RAM and VRAM, size); `evaluate.py` merges the test metrics into it and writes the result JSON. | The documents fixed the bundle layout but not the plumbing. Keeps the scripts thin and testable. Phase 4 moves scoring behind `Inspector`. | Awaiting |
| A-08 | 2026-11-13 | 2 | `evaluate.py` scores the test tiles twice (pass 1: heatmap range, tile scores, timing; pass 2: histograms) and computes pixel metrics on whole parent images stitched from their tiles, 10,000 bins. | Exact histogram range with nothing clipped, region-correct AUPRO for defects that straddle tile borders, bounded memory. The range only places bins; metrics are threshold-free. | Awaiting |
| A-09 | 2026-11-13 | 2 | `Backbone(weights="IMAGENET1K_V1")` defaults to the real weights; `None` (random) is for tests only. `gaussian_blur` (sigma 4, shared by PatchCore and RD4AD) lives in `backbone.py`. PatchCore's tile score is the max of the unblurred distance grid (ARCHITECTURE 5.2), so a tile can exceed `tau` while no pixel of the blurred mask does; how Phase 4 reconciles the tile decision with the mask under D-32 is left to Phase 4. | A forgotten argument must not silently give random features. | Awaiting |
| A-10 | 2026-11-13 | 3 | RD4AD mixed precision is `torch.autocast` with bfloat16 on CUDA only, no GradScaler; the cosine loss is computed in float32; autocast is off on CPU so the tests run the same code. | D-24 says "mixed precision" without a dtype. bfloat16 has float32's range, needs no loss scaler, and the RTX 5070 supports it. Switching to float16 plus a scaler is a few lines. | Awaiting |
| A-11 | 2026-11-13 | 3 | RD4AD trains on `RandomCropDataset`: item i is a 256 x 256 window at a uniformly random origin of a uniformly random train parent image (extent = union of its manifest crop boxes), deterministic in (seed, i); an MVTec row is the whole resized image. `TileDataset` gains `get(row)`. | D-22 says random crops; the manifest only holds grid tiles. Crops come from train parents only, so rules 1 and 2 are untouched. | Awaiting |
| A-12 | 2026-11-13 | 3 | Config keys `rd4ad.seed` (default: the config seed) and `tag` (appended to the method directory: `rd4ad_s1`). The three RAW-FABRID seeds share the split (config seed 0) and differ in initialisation and crops. | EVALUATION says "3 seeds for RD4AD" without saying what varies. A fixed split keeps seeds comparable with PatchCore and measures training variance. | Awaiting |
| A-13 | 2026-11-13 | 3 | RD4AD validation loss is taken on up to `rd4ad.val_tiles` (64) normal `val` tiles with the student in `eval()`, every `val_every` steps and at the end, logged only. The 300-iteration overfit test is marked `slow` (deselected by default, `pytest -m slow`) and runs at 256 x 256 on 8 fabric-like weave tiles (white-noise tiles at 32 x 32 only reached 0.106 in 300 iterations). | D-25 only says the validation loss is logged to spot divergence. An 84 M-parameter Adam step costs about a second on CPU, too slow for the default suite. | Awaiting |
| A-14 | 2026-11-13 | 3 | RD4AD's tile score is the max of the blurred heatmap (ARCHITECTURE 6.3), so for RD4AD the tile decision and the mask share one map; PatchCore keeps the unblurred-grid max (A-09). Result JSON for RD4AD replaces `coreset_*`/`bank_size` by `iterations`, `iters_per_second`, `final_train_loss`, `final_val_loss`. | Follows ARCHITECTURE as written; keeps Phase 2 result files unchanged. | Awaiting |

## Open

| ID | Question | Options | Decide by | Notes |
|---|---|---|---|---|
| O-01 | Add a third method? | (a) none, stop at two; (b) Dinomaly with ViT-Small; (c) EfficientAD | Phase 7 gate, after Phase 5 results | See below |
| O-02 | Which other stretch goals, if any? | TSFabrics stream with the cutline test; defect-type classification from a few labels; automatic fabric-profile selection in the app | Phase 7 gate | At most one or two |

### O-01 background: Dinomaly with ViT-Small

- **What it produces:** the same outputs as the other methods, a score per tile and a 256x256 heatmap. It plugs into the existing threshold, grading and app as a third model choice.
- **Value:** it is designed for one model covering many classes, which is the unified cross-fabric regime where RD4AD is expected to degrade. It also uses a self-supervised transformer, so the study can separate the effect of the method from the effect of the backbone. The paper reports 99.3 image AUROC on multi-class MVTec AD with ViT-Small.
- **Workload (estimates, not measured):** 300 to 400 lines of new code (DINOv2 encoder wrapper, MLP bottleneck, 8-layer transformer decoder with linear attention, loss, optimiser with warm-up). 1 to 1.5 weeks of developer time. About 25 GPU hours for the full study, or a few hours if run only in the unified and per-group regimes.
- **Fit on 12 GB:** very likely. 37 M parameters in total against 84 M trainable for RD4AD. The paper gives no memory figure, so the first run must measure it.
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
