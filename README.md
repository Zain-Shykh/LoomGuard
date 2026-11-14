# LoomGuard

Fabric inspection with deep learning. LoomGuard learns what normal fabric looks like from defect-free images only, flags defects it has never seen, shows where they are, and turns that into a quality grade.

Two methods are implemented from their papers in PyTorch and compared on the same data: **PatchCore** (stores normal patches) and **Reverse Distillation** (trains a decoder to rebuild normal features). A Streamlit app presents the result as a product: upload an image, get the heatmap, defect list and grade.

**Current state:** design phase complete; Phase 1 code complete (tiler, manifests, dataset class, dataset verification script); Phase 2 code complete (metrics, frozen backbone, PatchCore, `scripts/fit.py` and `scripts/evaluate.py`); Phase 3 code complete (RD4AD student, training loop, random-crop dataset, method dispatch in the scripts); Phase 4 code complete (calibration, mask cleanup, four-point grading, `Inspector`, operating-point and grading metrics); Phase 5 code complete (ZJU-Leaper and AITEX adapters, per-group and held-out regimes, ablation configs, `scripts/study.py`, `scripts/summarize.py`, `scripts/failure_cases.py`). Waiting for the training-PC runbooks of Phases 1 to 5 (`docs/specs/`). See the [progress log](docs/PROGRESS_LOG.md).

## Documents

| Document | Contents |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design, both networks with tensor shapes and diagrams, calibration, grading, app, repository layout |
| [docs/DATASETS.md](docs/DATASETS.md) | Every dataset: what was verified, size, licence, layout, role |
| [docs/EVALUATION.md](docs/EVALUATION.md) | Metrics, splits, training regimes, ablations, published reference numbers, pass criteria |
| [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) | Phases with objective, tasks and acceptance criteria |
| [docs/PROGRESS_LOG.md](docs/PROGRESS_LOG.md) | Dated log of work, measurements and decisions |
| [docs/DECISIONS.md](docs/DECISIONS.md) | Every agreed decision, the open ones, and what was dropped |
| [docs/WORKFLOW.md](docs/WORKFLOW.md) | How each phase is built: spec, headless developer session, review loop, commits |
| [docs/specs/](docs/specs/) | One spec per phase, written from `TEMPLATE.md` by that phase's developer session before it codes |
| [LoomGuard Project Starter.md](LoomGuard%20Project%20Starter.md) | The original idea. Where it disagrees with `docs/`, `docs/` wins. |

Diagrams are Mermaid. In VS Code press `Ctrl+Shift+V` (needs the `bierner.markdown-mermaid` extension). GitHub renders them directly.

## The design in one screen

| Item | Decision |
|---|---|
| Backbone | WideResNet-50-2, ImageNet weights, frozen. 24.9 M parameters used (stem to layer3). |
| Model input | 256x256 tiles at the dataset's native resolution. Large images are tiled, never shrunk. |
| PatchCore | layer2 + layer3 features, 1,536-d patch vectors, 1,024 per tile, greedy coreset to a 10% memory bank |
| Reverse Distillation | Bottleneck 58.9 M + decoder 24.9 M trainable parameters. Cosine loss on three scales. Adam, 5,000 iterations, batch 16. |
| Threshold | One per fabric, set on held-out normal tiles for 1% false alarms. No defect data used. |
| Grading | Four-point system on connected defects, in millimetres, summed over a roll |
| Primary dataset | RAW-FABRID (4.1 GB, CC BY 4.0, masks, 7 rolls, 4 px/mm) |
| Cross-fabric benchmark | ZJU-Leaper (5.8 GB, 19 patterns in 5 groups, masks) |
| Sanity check | MVTec AD textures against published numbers |
| Main metrics | AUROC for detection, AUPRO for localization, false alarms per 1,000 tiles at the calibrated threshold |
| Hardware | Training on one RTX 5070 (12 GB). Demo runs on CPU. |

## What changed from the starter file

| Starter | Now | Why |
|---|---|---|
| RAW-FABRID only to get the pipeline running | Primary dataset | Only core set with a permissive licence, verified masks, roll streams and a real physical scale |
| "No physical scale in public datasets" | Real millimetres on RAW-FABRID and TSFabrics | RAW-FABRID states 4 px/mm; TSFabrics states 127 PPI |
| Lusitano for realism and the demo | Dropped | No masks documented, 19 GB, expired certificate on the host |
| Tile size left open | 256 native tiles everywhere, tiler with even spacing | Thin defects vanish when images are shrunk |
| PatchCore memory grows with data | Capped at 600 tiles before coreset, float16 bank | Bounded memory and search time |
| RD4AD trained "by the paper" | Fixed iteration budget, last checkpoint | The official code picks the best epoch on the test set |
| Threshold at about 99% of normal tiles | Same idea, exact rule, one threshold for tile and pixel decisions | One source of truth; calibrated false-alarm rate is then tested |
| Grade per image | Grade per roll; single images show points only | One frame is 0.14 square yards, so a per-image grade is meaningless |
| Third method defaults to EfficientAD | Recommend Dinomaly (ViT-Small), decided after the main study | It targets the unified multi-fabric setting; EfficientAD needs ImageNet data in training and has no official code |
| Week-by-week plan | Phases with acceptance criteria | Progress is judged by evidence, not by dates |

## Team decisions (2026-11-12)

The full list is in [docs/DECISIONS.md](docs/DECISIONS.md). The ones that answer the questions raised by the design phase:

| Topic | Decision |
|---|---|
| Front end | Streamlit (course requirement) |
| Deployment | Hugging Face Spaces, free CPU tier |
| Model bundles | Hugging Face Hub model repository. GitHub rejects files over 100 MB and a Reverse Distillation student is 168 MB. |
| Grade cut-offs | A up to 20, B up to 40, C up to 60 points per 100 square yards, Reject above |
| Lusitano dataset | Dropped |

## Open decisions

- **Third method (O-01).** None, Dinomaly with ViT-Small, or EfficientAD. Decided at the Phase 7 gate, after the two-method study.
- **Other stretch goals (O-02).** Decided at the same gate.

## Setup

Install PyTorch first, from the index that fits the machine. `pip install -e .` does not pull it in.

```bash
# laptop (CPU only)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# training PC (RTX 5070): needs PyTorch 2.7 or newer with CUDA 12.8 wheels
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

pip install -e ".[dev]"
pytest
```

## Running the cross-fabric study (training PC)

Each run is one config file; a study is a list of configs and is resumable:

```bash
python scripts/study.py configs/studies/01_zju_mini.txt      # then 02 to 05, see docs/specs/phase-5-cross-fabric-study.md
python scripts/summarize.py --check                          # which cells of EVALUATION 4.3 are still missing
python scripts/summarize.py                                  # tables, CSV and SVG plots in results/report/
```

## Team workflow

- Laptop: code, documentation, CPU unit tests on synthetic data. No datasets, no training.
- Training PC (RTX 5070, 12 GB): datasets, fitting, evaluation.
- Results (`results/*.json`) are committed. Datasets and model bundles are never committed.
