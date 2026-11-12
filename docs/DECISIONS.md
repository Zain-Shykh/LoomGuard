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
| D-55 | 2026-11-12 | Spec-first workflow: every phase gets a spec in `docs/specs/` from the template, approved before any code | Context lives in documents, not in chat | WORKFLOW |
| D-56 | 2026-11-12 | Each phase is implemented by a fresh headless Claude session (Sonnet 5.5, medium effort) that ends with a review loop against a reviewer agent | A clean context per phase; independent check against drift and stale docs | WORKFLOW |
| D-57 | 2026-11-12 | The main session (Fable 5.2) is the orchestrator: writes specs, launches sessions, verifies, pushes. Three commits per phase: spec, implementation, review. | High-level control stays in one place | WORKFLOW |

## Awaiting confirmation

Decisions made by a developer session during a phase. Empty until Phase 1 runs.

| ID | Date | Phase | Decision | Reason | Status |
|---|---|---|---|---|---|
| | | | | | |

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
