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
