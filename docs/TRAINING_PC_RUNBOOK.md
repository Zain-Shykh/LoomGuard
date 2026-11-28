# LoomGuard: Training-PC Runbook

One ordered checklist for Huzaima (Member A) on the RTX 5060 Ti (16 GB VRAM) training PC. It collects the training-PC steps of the six phase specs (`docs/specs/phase-N-*.md`, section 8) in the order they should be run, with stop points. If this file and a spec disagree, the spec's section 8 is the detail and this file is the order. Report disagreements to the project owner.

Zain (Member B) led the initial implementation and completed the five real MVTec PatchCore profiles on Kaggle. Hamza (Member C) prepared and verified the datasets. Huzaima independently verified and staged the handoff on the training PC.

The [approved distributed training runbook](DISTRIBUTED_TRAINING.md) governs assignments and checkpoints. This document supplies the single-machine procedure and its dataset, benchmark, grading and CPU app acceptance criteria. Use [Huzaima's local RD4AD handoff](TEAM_TASKS_MEMBER_A.md) for the pinned checkout, original pins and MVTec launch.

## Rules for the whole runbook

1. Do not tune anything on test results. Headline settings (iteration budgets, coreset ratios, thresholds) are fixed in advance (decision D-35). If a pass criterion is missed, log the miss, do not change a setting to fix it, and tell the project owner. A fix of a bug is fine; a change of a hyperparameter is a decision.
2. Never commit datasets, model bundles, `artifacts/` or `data/`. Results JSON files in `results/`, logs, and documents are committed.
3. Record every number a step asks for in `docs/PROGRESS_LOG.md` (template at the top of that file), newest entry first. If a measurement contradicts a number in `docs/`, correct the document in the same commit.
4. Commit after each part with the message given, and push. Pull before starting each part.
5. Keep the console output of every fit and evaluate run (`tee artifacts/<name>.txt`). If something fails, the output is the evidence.

## Part 0: Environment

```bash
git clone https://github.com/Zain-Shykh/LoomGuard.git && cd LoomGuard
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[dev,app]"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_arch_list())"
pytest -q          # about 7 minutes; must pass with no dataset
```

Expected: `torch.cuda.is_available()` is `True` and the architecture list contains `sm_120`. If not, the PyTorch build is too old for the RTX 50 series (the RTX 5060 Ti is Blackwell like the RTX 5070 and uses the same `sm_120` wheels; need 2.7 or newer with CUDA 12.8 wheels). Stop here until it is fixed.

Also run the slow overfit test once on the GPU (8 tiles, loss below 0.05 in 300 iterations): `pytest -m slow tests/test_rd4ad.py`.

Record in the log: PyTorch version, GPU name, `sm_120` present, test counts.

## Part 1: Datasets and verification (Phase 1)

Download into `data/` (git-ignored), or mount the verified handoff at the same paths. Sizes and licenses are in [Datasets](DATASETS.md). D-61 excludes AITEX. The distributed MVTec gate needs only the five MVTec textures and the gate pin. See [the distributed runbook](DISTRIBUTED_TRAINING.md#independent-mvtec-gate).

| Dataset | Source | Folder |
|---|---|---|
| RAW-FABRID (4.1 GB) | Mendeley Data, DOI 10.17632/db6g85xsyg.1. `sha256sum RAW_FABRID.zip` must equal `f6b4f82cce2f0220bbab3968a74b5337ebf2bbfd5e33d0548d34e1ebf80f85b9` | `data/RAW_FABRID` |
| ZJU-Leaper (5.8 GB) | Google Drive id `1zzoU3Aj21V62GWS8LBm217TsE5uFeMVB` (`gdown`), then `7z x ZJU-Leaper.7z`. Retry off-peak if the quota blocks it. | `data/ZJU-Leaper` |
| MVTec AD textures | form on mvtec.com; extract only `carpet`, `grid`, `leather`, `tile`, `wood` | `data/mvtec_ad` |
| AITEX | three 7z files from `https://www.aitex.es/afid/`; extract each | `data/AITEX` |

Then:

```bash
mkdir -p artifacts
set -o pipefail
python scripts/verify_datasets.py \
  --raw-fabrid data/RAW_FABRID --zju data/ZJU-Leaper --mvtec data/mvtec_ad \
  --manifest-dir artifacts/manifests | tee artifacts/verify_datasets.txt
```

Do all of these (full detail in section 8 of `docs/specs/phase-1-data-layer.md`):

1. Paste `artifacts/verify_datasets.txt` into the progress log.
2. Verify the two RAW-FABRID representations independently. The shipped patch benchmark has 14,196 training, 4,969 normal test and 687 defective test tiles. The high-resolution mask-crop manifest has 11,872 training, 2,324 validation, 4,944 normal test and 712 defective test tiles. Use [Datasets](DATASETS.md#1-raw-fabrid-primary) for the recorded cross-representation differences.
3. For ZJU-Leaper, confirm the layout (`Images/`, `Annotations/masks`, `ImageSets/...`), rebuild the full Setting 1 manifest and record pattern/split counts. Confirm terms for the proposed execution environment before mounting it in Kaggle.
4. Correct every "unverified" statement in `docs/DATASETS.md` that the script resolves.
5. Look at one image from each of the seven RAW-FABRID rolls. Are they visually different fabrics? Write one sentence per roll and a verdict in the log and in `DATASETS.md`. If they are the same fabric, leave-one-roll-out is not an unseen-fabric test, and the report uses ZJU-Leaper held-out patterns for that question (`docs/IMPLEMENTATION_PLAN.md`, risks).

Record the verifier output and inspection findings for each applicable queue. The independent MVTec gate can proceed after MVTec data, source and readiness checks. Use the full pin and the relevant checkpoints before later queues.

## Part 2: Implementation check on MVTec AD (Phases 2, 3, 4)

The MVTec gate evaluates both methods on all five textures and writes the calibration files. For the team study, use the [distributed gate commands](DISTRIBUTED_TRAINING.md#checkpoint-1-run-both-mvtec-benchmarks) so the results carry checkpoint provenance. The standalone commands below describe single-machine fitting and evaluation.

```bash
# real ImageNet weights (about 130 MB) download on first use
python - <<'EOF'
from loomguard.models.backbone import Backbone
b = Backbone("IMAGENET1K_V1"); print(sum(p.numel() for p in b.parameters()) / 1e6, "M params,",
      any(p.requires_grad for p in b.parameters()), "requires grad")
EOF
# expected: 24.86 M params, False

for m in patchcore rd4ad; do
  python scripts/fit.py      --config configs/mvtec_$m.yaml | tee artifacts/mvtec_${m}_fit.txt
  python scripts/evaluate.py --config configs/mvtec_$m.yaml | tee artifacts/mvtec_${m}_eval.txt
done
```

PatchCore takes minutes per category. RD4AD takes about 20 to 25 minutes per category (5,000 iterations at about 4 it/s; measure the real speed).

`evaluate.py` prints the mean over the five categories and PASS or FAIL against the targets (decision D-33):

| Method | Image AUROC | Pixel AUROC | AUPRO |
|---|---|---|---|
| PatchCore | at least 97.9 | at least 96.6 | at least 90.7 |
| RD4AD | at least 98.5 | at least 96.7 | at least 92.0 |

Also record:

- RD4AD speed and memory from `artifacts/mvtec/*/rd4ad/metrics.json`: `peak_vram_gb` must be under 11 and `iters_per_second` is the real speed. If the speed is far below what the GPU can do, the crop loader is the bottleneck: raise `num_workers` in the config and note it. Update the compute table in `docs/EVALUATION.md` section 6 with the measured numbers (they also change the time estimates below).
- The train and validation loss lines of the RD4AD fits, to check for divergence.

**If a target is missed:** do not tune. Write down which category fails and the numbers, then stop and ask the project owner. The documented upgrade path for PatchCore is in `docs/ARCHITECTURE.md` section 5.2, and it needs a new decision.

**Commit:** `phase 2: results`, `phase 3: results` (the MVTec rows), with `results/mvtec/**`.

## Part 3: RAW-FABRID headline runs (Phases 2, 3, 4)

```bash
# PatchCore unified, 10 percent coreset, 600-tile cap
python scripts/fit.py      --config configs/raw_fabrid_patchcore.yaml | tee artifacts/raw_patchcore_fit.txt
python scripts/evaluate.py --config configs/raw_fabrid_patchcore.yaml | tee artifacts/raw_patchcore_eval.txt
cat artifacts/raw_fabrid/unified/patchcore/metrics.json     # coreset_seconds (must be under 600), peak_ram_gb, peak_vram_gb

# RD4AD unified, 3 seeds, 20,000 iterations each (about 80 minutes each at 4 it/s)
for s in 0 1 2; do
  python scripts/fit.py      --config configs/raw_fabrid_rd4ad_s$s.yaml | tee artifacts/raw_rd4ad_s${s}_fit.txt
  python scripts/evaluate.py --config configs/raw_fabrid_rd4ad_s$s.yaml | tee artifacts/raw_rd4ad_s${s}_eval.txt
done
```

Then the checks of the specs:

1. RD4AD mean and spread over the three seeds (script in `docs/specs/phase-3-reverse-distillation.md`, section 8, step 4). Put the table in the log.
2. False alarms on normal test tiles (script in `docs/specs/phase-4-calibration-inspection-grading.md`, section 8, step 3): `false_alarms_per_1000` should be between 5 and 20 for both methods (target 10). Outside that band, analyse the cause and write it up; roll 3 has few validation tiles.
3. Per-roll grading, predicted against ground-truth masks, all seven rolls (same spec, step 4): record `grade_agreement` and `points_error_mean`. Roll frames in the official test pool are all defective, so most rolls grade Reject; this is expected and is a finding about the data, not a bug.

**Commit:** `phase 4: results` with `results/raw_fabrid/**` and the log entries.

## Part 4: The cross-fabric study (Phase 5)

Details: `docs/specs/phase-5-cross-fabric-study.md`, section 8.

```bash
python scripts/study.py configs/studies/01_zju_mini.txt --dry-run     # check the list, then run without --dry-run
python scripts/study.py configs/studies/01_zju_mini.txt               # 1. debug on ZJU-mini first
python scripts/study.py configs/studies/02_raw_fabrid.txt             # 2. per roll, unified, leave-one-roll-out
python scripts/study.py configs/studies/03_aitex.txt                  # 3. AITEX
python scripts/study.py configs/studies/04_zju_full.txt               # 4. rest of ZJU-Leaper, groups, unified, held-out
python scripts/study.py configs/studies/05_ablations.txt              # 5. ablations and seeds
python scripts/summarize.py --check                                   # must exit 0: every cell has a result
python scripts/summarize.py                                           # tables and SVG plots under results/report/
python scripts/failure_cases.py --config configs/raw_fabrid_per_fabric_patchcore.yaml --profile Rollo4A
python scripts/failure_cases.py --config configs/raw_fabrid_per_fabric_rd4ad.yaml --profile Rollo4A
```

Notes:

- Each list runs PatchCore before RD4AD so a path bug shows up in minutes. Lists are resumable: if a run dies, run the same command again; finished runs are skipped.
- Step 1 on ZJU-mini first: it is the cheapest place to find adapter and config bugs.
- Studies 2 and 3 reuse the Part 3 runs only if the result files exist; otherwise they run them.
- ZLEval (the dataset's official evaluation tool): set `dump_predictions: true` in a tagged copy of a ZJU config, run `evaluate.py`, and feed `artifacts/predictions/zju_leaper/...` to the dataset's ZLEval. The input format was not verified, so the glue code is not written; record pixel, region and sample F1 per pattern. Skip this if it takes more than a few hours; note that you skipped it.
- Commit only RAW-FABRID failure-case panels (CC BY). Do not commit MVTec images (non-commercial licence).
- Write `docs/FINDINGS.md`: answer Q2, Q3, Q4 and Q7 with numbers from `results/report/`, say where each method fails, and include at least three failure-case images per method.
- GPU time estimate with the Phase 5 spec numbers: PatchCore about 100 bundles, 6 to 8 hours; RD4AD about 38 hours in total (ZJU per pattern 6.6 h, per group 3.5 h, unified 1.4 h, ZJU held-out 3.5 h, RAW per roll 2.4 h, RAW unified 3 seeds 4.2 h, RAW held-out 9.7 h, AITEX 2.4 h, ablations 4.6 h). Replace these with your measured speed from Part 2. They were estimated for a faster card; the RTX 5060 Ti is probably about a third slower, so plan for roughly 50 hours of RD4AD and 8 to 10 hours of PatchCore until you have the measured speed. The largest single item is the RAW-FABRID held-out RD4AD at 20,000 iterations (decision A-22); the owner accepted it. If time is short, ask before reducing it.

**Commit:** `phase 5: results` with `results/**`, `results/report/**`, `docs/FINDINGS.md`.

## Part 5: The app (Phase 6)

Details: `docs/specs/phase-6-streamlit-app.md`, section 8, steps 1 to 6. Summary:

1. Upload the demo bundles (`huggingface-cli upload ...`; install `huggingface_hub` on this machine only for this). The repository is `Zain-1389/loomguard-bundles` (public, owned by the project owner). Set `LOOMGUARD_HUB_REPO=Zain-1389/loomguard-bundles` for the checks. Only at the very end, put the name in `assets/profiles.json` (`hub_repo`) and update `tests/test_app.py::test_shipped_assets_start_the_app`, which assumes it is empty.
2. On a laptop or any CPU-only machine: fresh clone, `pip install -e ".[app]"` with CPU PyTorch, `streamlit run app/streamlit_app.py`. Time a 1792x1024 image with the PatchCore 1 percent bundle: the median must be under 10 seconds.
3. Check that the app's score, flag and defect list equal `evaluate.py`'s dumped predictions for one real image (script in the spec).
4. Build a demo roll from one RAW-FABRID roll, play it in the Roll stream page, and compare the final points and grade with the result JSON.
5. Wrong-fabric test: the unknown-fabric warning must appear on images from another roll and must not appear on normal images of the right roll. If it misfires, adjust the three `UNKNOWN_*` constants in `app/streamlit_app.py` and record a decision.
6. Add three to five RAW-FABRID images (CC BY 4.0, with credit) to `assets/samples/` with entries in `samples.json`, then deploy to the existing Streamlit Space `https://huggingface.co/spaces/Zain-1389/loomguard` (free CPU tier, the owner creates the access for you) using `deploy/requirements.txt` and `deploy/README_space.md`. Record the Space URL.

**Commit:** `phase 6: results`.

## Part 6: Hand back

When all parts are done:

1. Push everything and tell the project owner. The next steps are on the laptop: tick the plan criteria with the evidence, the Phase 7 decision about extras (a third method was deferred by the owner on 2026-11-14 until the core study is finished), and Phase 8, the report and README.
2. Send the owner a short list: which criteria passed, which did not, every code fix you made, and any number that contradicted a document.

## Checklist of pass criteria

| Where | Criterion | Part |
|---|---|---|
| Phase 1 | `sm_120` in the arch list | 0 |
| Phase 1 | RAW-FABRID 14,196 / 4,969 / 687 tiles, labels match the metadata CSV | 1 |
| Phase 1 | verification output in the log; "unverified" items of `DATASETS.md` resolved | 1 |
| Phase 2 | PatchCore MVTec textures 97.9 / 96.6 / 90.7 | 2 |
| Phase 2 | coreset for 600 tiles under 10 minutes; RAM and VRAM recorded | 3 |
| Phase 2 | RAW-FABRID unified PatchCore result committed and summarised | 3 |
| Phase 3 | RD4AD MVTec textures 98.5 / 96.7 / 92.0 | 2 |
| Phase 3 | peak VRAM under 11 GB at batch 16 (bar kept from the 12 GB design; the card has 16 GB); speed recorded; compute table updated | 2 |
| Phase 3 | RAW-FABRID unified RD4AD, 3 seeds, mean and spread | 3 |
| Phase 4 | false alarms 5 to 20 per 1,000 for both methods | 3 |
| Phase 4 | per-roll grading predicted against ground truth for 7 rolls | 3 |
| Phase 4 | a trained bundle loads and runs on the laptop CPU | 5 |
| Phase 5 | every regime cell has a result; `summarize.py --check` exits 0 | 4 |
| Phase 5 | all ablations run and plotted; `docs/FINDINGS.md` with failure cases | 4 |
| Phase 6 | install and start on a clean CPU machine; under 10 s per image; equals `evaluate.py`; roll grade matches; Space deployed | 5 |
