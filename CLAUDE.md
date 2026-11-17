# LoomGuard

Fabric inspection with deep learning in PyTorch. Learns normal fabric from defect-free images only, then detects and localizes unseen defects and grades the fabric. Two methods implemented from their papers and compared: PatchCore and Reverse Distillation (RD4AD). Streamlit front end (a course requirement).

## Read first

The design is fixed in `docs/`. Read the relevant document before writing code, and do not re-decide what is written there without asking.

| File | Use it for |
|---|---|
| `docs/ARCHITECTURE.md` | Tensor shapes, layers, tiler, calibration, grading, bundle format, repo layout |
| `docs/EVALUATION.md` | Metrics, splits, regimes, pass criteria |
| `docs/DATASETS.md` | Dataset layouts, sizes, licences, what is still unverified |
| `docs/IMPLEMENTATION_PLAN.md` | Current phase, tasks, acceptance criteria |
| `docs/PROGRESS_LOG.md` | What has been done and measured |
| `docs/DECISIONS.md` | What is agreed, what is open, what was dropped |
| `docs/WORKFLOW.md` | The process: spec first, one fresh session per phase, review loop, commits |
| `docs/specs/` | The spec of each phase, written by that phase's session before it codes |

`LoomGuard Project Starter.md` is the original idea. Where it disagrees with `docs/`, `docs/` wins.

## Two machines, different jobs

- **Laptop (no GPU, almost no free disk):** code, docs, CPU unit tests on synthetic data. **Never download a dataset or train a model here.** Verify remote data with HEAD/range requests instead.
- **Training PC (RTX 5060 Ti, 16 GB VRAM):** datasets, fitting, evaluation. Needs PyTorch 2.7 or newer with CUDA 12.8 wheels (`torch.cuda.get_arch_list()` must contain `sm_120`).

If a task needs a GPU or a dataset and you are on the laptop, write the code and the test, then record the run as a training-PC task in `docs/PROGRESS_LOG.md`.

## Setup

Install PyTorch first, from the index that fits the machine. `pip install -e .` does not pull it in.

```bash
# laptop (CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# training PC (RTX 5060 Ti)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

pip install -e ".[dev,app]"   # app = Streamlit (Phase 6, needed by tests/test_app.py); the training PC can use ".[dev]"
pytest
```

## Layout

```
loomguard/
  data/        tiling, manifest builders (one adapter per dataset), dataset class
  models/      backbone, patchcore, rd4ad
  eval/        metrics, calibrate
  inspection/  postprocess, grading
  pipeline.py  Inspector: the single path used by evaluation and by the app
  runs.py      config, manifest dispatch, paths (incl. run tag) and model loading shared by scripts/fit.py and scripts/evaluate.py
scripts/       verify_datasets.py, fit.py, evaluate.py, study.py, summarize.py, failure_cases.py
app/           streamlit_app.py (thin UI over Inspector: upload validation, bundle download, three pages)
assets/        samples/ (images + samples.json attribution), rolls/<name>/roll.json demo rolls, profiles.json bundle registry
deploy/        requirements.txt and README_space.md for the Hugging Face Space
configs/       one YAML per run (base/ fragments via `extends`, studies/ run lists)
tests/         CPU-only, no dataset required
results/       small metric JSON files, committed
data/, artifacts/   git-ignored, never committed
```

The package is `inspection`, not `inspect`: a folder named `inspect` shadows the standard library module when a script is run from inside `loomguard/`.

## Rules that must not be broken

1. Split by parent image first, then tile. No parent image in two splits. Keep the unit test that proves it.
2. Training and validation use normal images only. The test split never influences a threshold, a checkpoint or a hyperparameter. RD4AD trains for a fixed iteration budget and keeps the last checkpoint.
3. Models see 256x256 tiles at native resolution. Do not shrink images (MVTec AD is the one documented exception).
4. The backbone is frozen and stays in `eval()` mode, including during RD4AD training.
5. Evaluation and the app both go through `Inspector`. Do not duplicate tiling, scoring or grading in the app.
6. Never commit datasets, model bundles or MVTec images. Bundles go to the Hugging Face Hub and are loaded with `torch.load(..., weights_only=True)`.
7. Add a dependency only in the phase that needs it, and add it to `pyproject.toml`.

## Working style

- Work happens phase by phase under `docs/WORKFLOW.md`. The developer session of a phase writes its spec in `docs/specs/` first; no code for a phase before its spec is written and committed. If you were launched as a developer session or as the reviewer, that file and your prompt define your job.
- Do not make or implement anything listed under "Open" in `docs/DECISIONS.md`. Record decisions you do make there.
- Implement the methods ourselves from the papers. No anomaly-detection library; learning is the goal.
- Smallest code that meets the acceptance criteria of the current phase. No scaffolding for later phases.
- Every piece of non-trivial logic gets one runnable CPU test.
- A phase is done only when its acceptance criteria are ticked in `docs/IMPLEMENTATION_PLAN.md` with evidence in `docs/PROGRESS_LOG.md`. Add a log entry whenever something is finished, measured or decided.
- If a measurement contradicts a number in `docs/`, fix the document in the same commit.
