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
