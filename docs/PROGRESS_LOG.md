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

## 2026-11-28: ZJU-Leaper Kaggle upload verified; local dataset copies removed
- **Phase:** data handoff, Member B
- **Who / where:** Zain (Member B), private Kaggle notebook `notebook4db0e623ba`, CPU (no accelerator)
- **Done:** Hashed the private dataset `loomguard-zju-leaper` with `digest_tree` at commit `67fb7f6`. Result: 213,400 files, digest `c2d7b294f2d4c32ccdc0c4d59b5377ac23c3bc69357e5bdf3ea7de73e72995d0`, equal to the full pin. Deleted the local MVTec, RAW-FABRID and ZJU-Leaper copies and the upload staging files (MVTec and RAW-FABRID digests had passed Kaggle `--prepare` in B2 and B4). `data/AITEX` is kept.
- **Evidence:** log line `RESULT MATCH`; version ran successfully in 17,790.8 s. Free disk afterwards: 24 GB.
- **Problems / next:** Hashing ZJU-Leaper took about 4.9 h on a CPU session. `--prepare` for a queue that includes ZJU hashes the same tree, so its cost on a GPU session is a schedule risk (not measured on GPU). `gate.json` is still missing.

## 2026-11-26 final integration and local RD4AD preparation

- Fetched and synchronized development main through `d8940dd` before integration. The fetched refs supplied no newer teammate commit or verifier and AITEX patch. The original audit worktree was preserved. The final slider test uses full-precision calibration. The development test run returned 188 passed and 1 slow deselection. The plan still has 33 jobs and 185 canonical paths with gate dry runs of A 5, B 5 and C 0.
- A approved `67fb7f6ca2dec0365b63a7454ffb57b3138b6875` for the common experimental checkout subject to verification of B's original shared pins. B leads dataset preparation and verification and runs PatchCore. C handles the rest of its assigned RD4AD work. Completed result bytes, configs, the ledger and training code are preserved.
- Prepared the detached local worktree and the owner-only archive under ignored artifacts. Confirmed Python imports from the frozen worktree, CUDA 12.8 and `sm_120`, and cached the approved ImageNet weights with verified checksum. Verified the local transfer procedure on a small test file. The prior 237,391-file handoff verification and the four preserved manifests remain in place.
- B's original pins, five model bundle exports and original transfer checksums are pending. Fresh gate-scoped readiness awaits the original gate pin. No real training or combined checkpoint approval was run. AITEX compatibility, separate RAW-FABRID published-patch support and later workload review stay outside the immediate local gate preparation.

## 2026-11-26 audit of the teammate push

- Fetched `main` at `d8940dd9e0799000c2c98e2a915727c18b8595be`. The training code, configs and assignment ledger match `67fb7f6ca2dec0365b63a7454ffb57b3138b6875`. GitHub lists PR #1 as merged. The three feature/fix branch tips are ancestors of main. No tracked C verifier or AITEX diff was available.
- The exact candidate gave 187 non-slow passes, 1 known failure and 1 slow deselection. The failure reproduced 78.507 displayed versus 78.506 expected from a rounded input. Fetched main gave 188 passes and 1 deselection. The local test improvement reads the fixture's raw calibration threshold and compares the exact three-decimal display. It gave 188 passes and 1 deselection. The targeted suites gave 111 passes and 1 deselection.
- Plan validation returned 33 jobs and 185 unique paths. Gate dry runs returned A 5, B 5 and C 0. Synthetic regressions covered source, tree and manifest mismatch, stale and incomplete readiness, gate and mini dependencies, legacy full-pin receipts, duplicate outputs and interrupted publication. The real CLI also rejected a `d8940dd` checkout against a `67fb7f6` pin.
- Mounted MVTec contains 2,173 files. The 1,781 manifest rows match the handoff CSV, with 1,139 train, 127 validation and 515 test. All 382 defective images have matching, nonempty masks of the expected dimensions. The computed tree SHA-256 is `f1842104ce5f822aa5b1532e6006b21a387a7668687357cab6c110817f597920`. Both gate manifest digests are `5f15c6ef1001bc3e8231540f757d0e77e88a82757156b5950371b0759b1e2827`. Both gate-scoped and legacy full-pin A readiness passed on the local RTX 5060 Ti. A fresh gate-scoped readiness is required with the original shared pin.
- Comparison pins reproduced all three active tree hashes and 31 canonical full-study manifests on the training revision. The RAW-FABRID high-resolution and ZJU Setting 1 manifests match C's CSVs. All 687 published defective RAW-FABRID patches have the shipped mask. The local comparison pin files carry this audit's review reference. B's original gate and full pin bytes are pending transfer.
- All five B result JSONs passed the runner's metadata and provenance checks against the locally reproduced gate digests. Recomputed means were 99.1794 / 97.2354 / 90.8848 against 97.9 / 96.6 / 90.7. B reports downloading and checking the saved-version outputs. A still needs the privately transferred bundles, manifests, logs and version references. The runbook export procedure passed a synthetic copy and checksum check. No real fitting, freezing or checkpoint approval occurred during the audit.
- A confirms the common source and archive destination. B transfers the original pins and five exports. C confirms the first pin against the mounted copy. A launches the five RD4AD profiles after readiness and reviews all ten pairs. The full-pin approval and later queue authorizations follow the [handoff](TEAM_TASKS_MEMBER_A.md).

## 2026-11-26: B4 attempt 1: standard-queue prepare passes on Kaggle, run blocked by missing gate.json
- **Phase:** 2 to 4 (RAW-FABRID PatchCore), Member B step B4
- **Who / where:** Zain (Member B), Kaggle T4 (private notebook `loomguard-standard-patchcore`, version 1), driven from the laptop browser
- **Done:** Created the private notebook (local copy `artifacts/kaggle_upload/standard_raw_fabrid_patchcore.ipynb`, inputs `loomguard-raw-fabrid` and `loomguard-pin-full`, GPU T4 x2, internet on). Save & Run All with `RUN_REAL=True`, profile `raw_fabrid_patchcore/unified`. The notebook checked the pin SHA-256, the RAW_FABRID file count (21,457), the commit (`67fb7f6`), `--validate` (33 jobs, 185 paths) and the dry run (15 profiles). `--prepare --queue standard` passed ("Checkpoint 0 verified", 8 split manifests match). The profile run then stopped before fitting.
- **Evidence:** log line `raw_fabrid_patchcore/unified: waiting for shared checkpoint approval artifacts/distributed/full-shared/gate.json`. Version failed after 1,198 s (prepare alone 1,127.7 s). Environment as in B2 (torch 2.11.0+cu130, sm_120 in arch list). Weekly quota shown before the run: 29 h; about 20 min used, so about 28.6 h left (computed).
- **Decisions:** none. Nothing was changed; `--approve-gate` was not run (A's step).
- **Problems / next:** No `gate.json` exists for the full pin: A must run the second `--approve-gate` against `datasets.json` `144a80d3...`. Then `gate.json` goes into the Kaggle dataset `loomguard-pin-full` as a new version (beside `datasets.json`, which stays untouched); the notebook already copies it if present. `--prepare` costs about 19 min of GPU per session (readiness is stale each session), so group profiles per session. No results, no bundles, nothing to download yet. B4 acceptance boxes stay open.

## 2026-11-26: B3 full dataset pin created; Kaggle metadata set
- **Phase:** 5 (study preparation), Member B task B3
- **Who / where:** Zain (Member B), laptop, frozen worktree at `67fb7f6`
- **Done:** Created the full-study dataset pin (MVTec, RAW-FABRID, ZJU-Leaper; AITEX excluded by D-61, `aitex_approval` null) into the new shared directory `artifacts/distributed/full-shared/`. The gate pin was not touched. Uploaded the file as the private Kaggle dataset `loomguard-pin-full` (1 file). Set descriptions and licences on the private datasets `loomguard-mvtec-gate` (CC BY-NC-SA 4.0) and `loomguard-raw-fabrid` (CC BY 4.0, DOI 10.17632/db6g85xsyg.1); no new versions, still private.
- **Evidence:** `datasets.json` SHA-256 `144a80d3f518a46b0183df1c3b5707627e58e128c89413a07907ca247374d40b`. `--validate`: 33 jobs, 185 paths. Dry runs: standard 81, mini 30, zju_full 50 profiles, 0 conditional. `load_pin` accepts the pin (31 configs and manifests). Tree digests: mvtec `f1842104...7920` (equals the gate pin), raw_fabrid `ce7cc3bf...ce9e`, zju_leaper `c2d7b294...95d0`. Source commit `67fb7f6ca2dec0365b63a7454ffb57b3138b6875`. Kaggle: https://www.kaggle.com/datasets/muhammadzain1x0/loomguard-pin-full
- **Decisions:** none. The `--review` text records the gate approval as reported by the owner. No approval file was written or run by B.
- **Problems / next:** `datasets.json` is not committed. Send it and its hash to A and C. C cross-checks the RAW-FABRID and ZJU tree digests. A runs the second `--approve-gate` against the full pin. B then runs `--prepare --queue standard` (needs the RAW-FABRID dataset and this pin on Kaggle).

## 2026-11-26: B2 MVTec PatchCore gate, five textures run on Kaggle T4
- **Phase:** 2 (MVTec PatchCore gate), Member B step B2
- **Who / where:** Zain (Member B), Kaggle T4 (private notebook `loomguard-gate-patchcore`, versions 1 to 3), driven from the laptop browser
- **Done:** Ran PatchCore on carpet, grid, leather, tile, wood at SHA `67fb7f6ca2dec0365b63a7454ffb57b3138b6875`, pin `702a0913...981a0`. `--prepare` passed in the first session (it checks the MVTec tree hash against the pin; 2,173 files seen), then `--validate` (33 jobs, 185 paths) and dry run (5 profiles) passed. All five `status.json` say `passed`, every bundle has five non-empty files. Exports downloaded from the saved versions' Output tab and every file matched `MANIFEST.sha256` (14, 26 and 26 files checked, 0 mismatches). Result JSONs are on the laptop under `results/mvtec/per_fabric/patchcore/` (uncommitted); bundles stay in `~/Downloads/loomguard_exports/`.
- **Evidence:** image AUROC / pixel AUROC / AUPRO@0.3 per texture: carpet 98.80 / 99.02 / 94.65, grid 98.33 / 97.90 / 89.50, leather 100.00 / 99.27 / 96.93, tile 100.00 / 96.10 / 86.15, wood 98.77 / 93.89 / 87.21. **Mean 99.18 / 97.24 / 90.88** against the bars 97.9 / 96.6 / 90.7: all three met (AUPRO margin only 0.18). Tile and wood are below the bars alone. Fit 0.51 to 0.70 min, evaluate and export about 0.5 min, coreset 14 to 20 s (207 to 252 training tiles per texture, not 600), peak VRAM 0.8 GB, peak RAM 2.4 to 2.6 GB, bundle 62 to 76 MB. Session wall time 886 s (with pytest 425 s and prepare 315 s), 412 s (grid, leather), 301 s (tile, wood); about 27 GPU minutes of the 30 h weekly quota. Pytest on Kaggle: 187 passed, 1 failed (`tests/test_app.py::test_sensitivity_slider_does_not_rerun_model`, 78.511 vs 78.51 +- 5e-4, the known slider rounding issue). Kaggle environment was torch 2.11.0+cu130, Python 3.13, driver 580.178.04, `sm_120` in the arch list, T4 reports `sm_75`.
- **Decisions:** none. No code, config, seed, pin or SHA was changed. `--approve-gate` was not run; it waits for A's RD4AD half.
- **Problems / next:** (1) The "600-tile fit under 10 minutes" criterion was measured on 207 to 252 tiles (coreset at most 20.4 s); A to confirm that is acceptable. (2) The stated environment (PyTorch 2.7 to 2.x with cu128) differs from Kaggle's torch 2.11 cu130; A to confirm results are accepted. (3) Hand the five exports to A for hash verification and the paired review; fix the slider test before release.

## 2026-11-25: B1 Kaggle prep: MVTec dataset uploaded, export test passed
- **Phase:** 5 (study preparation), Member B step B1
- **Who / where:** Zain (Member B), laptop and Kaggle (CPU session, no GPU hours)
- **Done:** Uploaded the private Kaggle Dataset `loomguard-mvtec-gate` (five MVTec textures, 2,173 files, plus the gate `datasets.json` beside `mvtec_ad/`; v1 shows 2,174 files, 2.28 GB). Ran the export test: a CPU notebook wrote a 314,572,800-byte file, Save & Run All finished in 26 s, the file was downloaded from the saved version and its SHA-256 matched the notebook's printout. Wrote the private gate notebook (not run on real data).
- **Evidence:** export file SHA-256 `1f12f20b06c3c8eb4951a4a6b978fa330ab6b58856414eff79700259a59e81d7` (both sides). Dataset is Private. The tree-hash match against the pin is only proven by the first `--prepare` on Kaggle.
- **Decisions:** none. The upload was a single zip because Kaggle's folder drag did not work for the owner; Kaggle extracted it.
- **Problems / next:** Send the pin and SHA to A and C; C's tree-hash cross-check; set the dataset description and licence; import the gate notebook, read the weekly quota, then run `carpet` (B2) with `RUN_REAL=True`.

## 2026-11-25: Open team decisions resolved provisionally; work plan written
- **Phase:** 5 (study preparation)
- **Who / where:** Zain (Member B), laptop, on the owner's instruction; Members A and C to confirm
- **Done:** Resolved the five open questions of `LoomGuard_Team_Decisions_Needed.md` with the recommended options and recorded them in `LoomGuard_Decision_Resolutions.md`. Wrote `LoomGuard_Team_Work_Plan.md` (per-member training queues, non-training work, gates, rebalancing rule). Moved the handoff `data/` and `artifacts/` to the repository root (git-ignored).
- **Evidence:** the two documents; `main` at `67fb7f6` is the proposed freeze SHA.
- **Decisions:** R1 to R8 in the resolutions file (all provisional, reversible by A or C before the first real run). No scientific setting, ledger entry, pin or approval was changed.
- **Pin:** B created the MVTec gate-only pin on 2026-11-25 in a worktree at `67fb7f6` (validate: 33 jobs, 185 paths; gate dry runs A 5, B 5, C 0). `datasets.json` SHA-256 `702a0913c595aecfbdae73ab89be2f72ba2ccdead54f53b20f526b42c43981a0`, scope gate, MVTec tree is exactly the five textures (2,173 files). The file is not committed.
- **Problems / next:** C cross-checks the tree hash, A confirms the SHA, B uploads the private MVTec Kaggle Dataset and runs the CPU export test, then the MVTec gate runs. No real training has started.



## 2026-11-22: D-62 approves scoped RD4AD teacher precision for real-data validation

- Member B tested commit `f7330a75201388a5600e425237fbd49b933c2d3c` on the Kaggle T4. All 14 targeted tests passed. Two runs of the original synthetic overfit test finished at 0.04159279 with zero skipped updates and 2,415.6 MiB peak allocated VRAM. [Kaggle T4 regression results](LoomGuard_Member_B_RD4AD_T4_Regression_Results.md)
- With both the RTX 5060 Ti and T4 synthetic regressions passing, the team approved the D-60 precision amendment as D-62 for real-data validation. The MVTec accuracy gate is still pending and uses the existing pass criteria. [Decision D-62](DECISIONS.md)

## 2026-11-22: Scoped RD4AD teacher precision candidate

- CUDA training, validation and scoring share an FP32 frozen teacher forward with cuDNN TF32 disabled for that call. The prior setting is restored before the student forward. CUDA student training retains FP16 autocast and GradScaler. The cosine loss remains FP32. A-30 proposes an amendment to D-60 for team review.
- Two fresh processes on the RTX 5060 Ti ran the original slow test with PyTorch 2.11.0+cu128 and cuDNN 9.19.0. Both ran the actual implementation for 300 iterations on eight synthetic 256 × 256 weave tiles with seed 0. Only metric logging was added to the test.

| Run | Final loss (last 25 iterations) | Skipped updates | Iterations/s | Peak CUDA allocated (MiB) | Peak CUDA reserved (MiB) |
|---|---:|---:|---:|---:|---:|
| 1 | 0.04184032 | 0 | 10.14 | 2411.6 | 2486.0 |
| 2 | 0.04184032 | 0 | 10.57 | 2411.6 | 2486.0 |

- Both runs passed the 0.05 convergence gate and logged the same finite loss windows to the displayed precision. A final-review replay of the candidate matched all 12 saved windows from the earlier FP32-teacher/FP16-student control exactly (final loss 0.04184031814336777).
- An in-memory replay using `torch.backends.cudnn.flags(allow_tf32=False)` for the teacher alone matched all 12 windows of the earlier scoped diagnostic exactly (final loss 0.03414286583662033). In PyTorch 2.11.0+cu128 that context also defaults to `enabled=False`, so the teacher runs with cuDNN disabled. Both replays used the same seeded model initialization, synthetic inputs and production training loop. The original temporary code was unavailable. The exact replay identifies the backend difference, and the candidate retains cuDNN for the teacher.
- The targeted non-slow suite passed 31 tests with 1 deselected. It covered teacher-feature agreement, precision and TF32 restoration, GradScaler overflow recovery, checkpoint round trips and synthetic fit/evaluate integration. With CUDA hidden, the focused CPU check passed 3 tests with 3 CUDA cases skipped.
- The full non-slow suite finished with 184 passed, 1 failed and 1 deselected. The Streamlit test `test_sensitivity_slider_does_not_rerun_model` also failed on clean main. At sensitivity 0.5, the app displays 78.507 after scaling the full-precision threshold. The test scales the rounded display value from sensitivity 1.0 and expects 78.506. The app and slider test were left as found.

## 2026-11-22: RD4AD counts skipped fp16 steps; gate prep on laptop
- Phase: distributed MVTec gate preparation
- Who / where: laptop (CPU only)
- Done: `RD4AD.fit` returns `skipped_steps` (GradScaler overflow steps) and `scripts/fit.py` prints it. Skipped steps still count toward the fixed iteration budget (D-60); result JSON schema unchanged.
- Evidence: `--validate` gives 33 jobs, 185 canonical paths; gate dry-run selects the right five textures for A (RD4AD) and B (PatchCore); `test_rd4ad`, `test_fit_evaluate`, `test_distributed_study` give 39 passed.
- Not verified: CUDA/sm_120, overflow counter on a real GPU, GPU overfit test, MVTec gate results, VRAM. All are training-PC / Kaggle tasks.
- Next: A runs the RD4AD gate, B the PatchCore gate, then `--approve-gate`.

## 2026-11-21: Independent review of the distributed runner

- Reviewer verdict CHANGES REQUIRED, no blocker: sharding does not change the science, the gate cannot pass on a partial set, publish and resume are atomic, full suite 177 passed. Fixed: dataset hashing follows linked subdirectories; a missing staged attempt now retrains instead of blocking forever; `--pin-datasets` rejects a profile whose test pool lacks normal or defective tiles; `check_integration` reports AITEX as an intentional omission (D-61) and still checks the other cells; gate threshold gets a 1e-9 rounding tolerance; runbook lists no C ablation commands and says the `--approve-mini` review text is the full-ZJU go-ahead. Two tests added (25 pass). Not done (nits): clear error for a torch build without sm_75 kernels, no private path or host in committed results (use unsigned storage paths), cached dataset hash per start.

## 2026-11-21: D-61, three-person workflow approved with terms

- Owner approved the workflow. AITEX excluded; ZJU-mini first, full ZJU on a later go-ahead; each ablation on its baseline's machine (RAW-FABRID RD4AD `it1k`, `ov32` moved C to A; ZJU RD4AD `res256` moved C to B). Test expectation updated; dry runs: A 36, B 138 (+14 inactive AITEX), C 11 profiles; 23 runner tests pass.
- Load after the move (RD4AD iterations): A 302k, B 120k plus all PatchCore, C 120k. A is the longest queue; rebalance whole unstarted profiles after the first speed measurements.

## 2026-11-21: RD4AD precision changed to fp16 + GradScaler (D-60)

- Kaggle T4 benchmark (`scripts/bench_rd4ad_precision.py`, random input, no result): emulated bf16 0.78 it/s, fp16 + GradScaler 3.47 it/s, fp32 1.30 it/s; all losses finite, peak VRAM 3.2 / 3.2 / 4.2 GB. T4 reports `native_bf16: false` (sm_75, torch 2.11.0+cu130). Emulated bf16 would make a 20,000-iteration fit about 7 h.
- Owner approved fp16 autocast + `torch.amp.GradScaler` for all RD4AD runs on all machines. `loomguard/models/rd4ad.py` and the runner's hardware check (`fp16_tensor_cores`, compute capability 7.0+) were changed. No RD4AD had been trained, so nothing is rerun. Whether fp16 costs accuracy is shown by the MVTec RD4AD gate.

## 2026-11-21: Three-person training workflow implemented

- Added the proposed assignment ledger, complete-profile selection, and an execution runner with staged outputs, per-profile status and retry, data and source fingerprints, and shared MVTec and ZJU-mini approvals.
- The three dry runs validated 33 configs and 185 unique canonical output paths. Member A has 22 profiles, B has 119 plus 14 conditional AITEX profiles, and C has 30. The original experiment YAMLs, study lists, adapters, scoring and grading code were verified against the source commit.
- The CPU suite passed 177 tests with one slow test deselected. The 27 focused tests covered checkpoint sequencing, sharding, retry and integration, as well as the existing path helpers. The synthetic full matrix passed `summarize.py --check` and generated the reports. The suite also exercised the original fit, evaluate, study and Streamlit paths.
- The operating schedule and AITEX inclusion await recorded team approval. The next measurement is the paired MVTec gate after dataset verification and environment setup. The Kaggle RD4AD queues need tensor cores (T4 qualifies); the bfloat16 requirement was replaced by fp16 plus GradScaler, see D-60 and the 2026-11-21 precision entry.

## 2026-11-17: Training PC is an RTX 5060 Ti with 16 GB

- **Phase:** between 6 and 7
- **Who / where:** Zain, laptop
- **Done:** the teammate's GPU is an RTX 5060 Ti with 16 GB VRAM, not an RTX 5070 with 12 GB. Updated `CLAUDE.md`, `README.md`, `docs/ARCHITECTURE.md`, `docs/WORKFLOW.md`, `docs/TRAINING_PC_RUNBOOK.md`, `docs/IMPLEMENTATION_PLAN.md` (risk row), `docs/EVALUATION.md` (compute note) and `docs/DECISIONS.md`.
- **Evidence:** none to measure on the laptop. Same Blackwell generation, so the `sm_120` check and the CUDA 12.8 wheels still apply.
- **Decisions:** D-59 supersedes the hardware in D-03. No setting changes: batch 16 and the 11 GB peak-VRAM bar stay, so results hold on a 12 GB card too. The extra memory is headroom only.
- **Problems / next:** the compute estimates (about 38 h RD4AD) assumed a faster card. Expect about 50 h until Phase 3 measures the real speed.

## 2026-11-14: Decisions confirmed, training-PC runbook written

- **Phase:** between 6 and 7
- **Who / where:** Zain, laptop
- **Done:**
  - The owner confirmed decisions A-01 to A-29; they moved to "Accepted" in `DECISIONS.md`. O-01 (third method) stays open and is deferred by the owner until the core study is finished.
  - `docs/TRAINING_PC_RUNBOOK.md`: one ordered checklist for the training PC, built from section 8 of the six phase specs (environment, datasets and verification, MVTec gate, RAW-FABRID headline runs, cross-fabric study, app), with stop points and a pass-criteria table.
- **Evidence:** all laptop work of Phases 1 to 6 is committed and pushed; the full suite gives 154 passed, 1 slow test deselected.
- **Decisions:** none new.
- **Problems / next:**
  - Nothing has run on real data yet. The first training-PC run will probably need small adapter fixes.
  - Next: the operator follows the runbook from Part 0. Phase 7 (gate) and Phase 8 (report) need its results.

## 2026-11-14: Phase 6 laptop work, Streamlit application

- **Phase:** 6 (Code complete; real-bundle steps are training-PC / operator work)
- **Who / where:** developer session, laptop
- **Done:**
  - `app/streamlit_app.py`: thin UI over `Inspector` with three pages (Inspect image, Roll stream, Results), `read_image` upload validation, `unknown_fabric`, `scale_label`, `overlay`, `stream_roll`, bundle discovery and `urllib` download with cache (`ensure_bundle`), sample and roll pickers with attribution files. No tiling, scoring or grading code of its own.
  - `pyproject.toml` extra `app = ["streamlit>=1.45"]`, `.streamlit/config.toml`, `assets/profiles.json`, `assets/samples/` (one synthetic placeholder and `samples.json`), `deploy/requirements.txt` and `deploy/README_space.md`.
  - Spec `docs/specs/phase-6-streamlit-app.md` with the runbook (fresh clone, Hub upload, 10 s timing, equality with `evaluate.py`, demo roll against the Phase 4 number, wrong-fabric check, samples, Space deployment).
- **Evidence:** `pytest -q`: 154 passed, 0 failed, 1 slow test deselected (new: `tests/test_app.py`, 26). `fit.py` and `evaluate.py` run on the synthetic RAW-FABRID tree; the app's score, verdict and defect table for the test images equal the dumped `scores.csv` and the defects measured from the dumped masks, and its heatmap equals the hand-stitched one; the roll page's final points, points per 100 sq yd and grade equal the `rolls` entry of the result JSON; a smooth image on a noise-trained bundle triggers the unknown-fabric warning and hides the defect list, a fresh noise image does not; the sensitivity slider calls the model once for three settings; non-image, oversized, corrupt and GIF uploads each give one `st.error` and no result. Bundle download tested against a local HTTP server (three requests, then served from the cache; bad names, 404, refused connection and an oversize file give `BundleError` and leave no partial bundle). Headless `streamlit run app/streamlit_app.py --server.port 8599` answered `/_stcore/health` with `ok` and `/` with 200.
- **Decisions:** A-26 to A-29 in `DECISIONS.md` (urllib download instead of `huggingface_hub`, unknown-fabric thresholds, the `app` extra and upload limits, UI conventions). ARCHITECTURE 11 and 12, README and CLAUDE.md updated.
- **Review:** round 1 PASS, four minor findings (stale `list_profiles` docstring in the spec, A-20 not annotated as superseded, an incomplete bundle directory blocked the download, spec record) fixed; new test `test_ensure_bundle_replaces_incomplete_bundle`. Final suite: 154 passed.
- **Problems / next:** Not seen on real data: the time of a 1792x1024 image with the real PatchCore 1 percent bundle, equality with `evaluate.py` on a real sample, the real roll's final grade against the Phase 4 number, the unknown-fabric thresholds on real wrong-fabric images, the real Hub URLs and the Space build. The sample images are a synthetic placeholder: the operator adds CC BY 4.0 RAW-FABRID samples with credits. The tests need `pip install -e ".[dev,app]"`. Next: the operator runs the Phase 6 runbook after the Phase 4 one (needs bundles) and commits `phase 6: results`.

## 2026-11-14: Phase 5 laptop work, cross-fabric study

- **Phase:** 5 (Code complete; the study itself is training-PC work)
- **Who / where:** developer session, laptop
- **Done:**
  - Adapters `build_zju` and `build_aitex` (tolerant file lookup, strict error reports; AITEX border filter and united multi-mask), `load_mask`, `min_overlap` through manifests and `Inspector`, `ScoreCounts.merge`.
  - Regimes `per_group` and `held_out`, config `extends`, `dump_predictions`; `evaluate.py` adds `image_auroc`, `image_ap`, `by_fabric`, `tag`.
  - 27 new configs (regimes, ZJU-mini, held-out, ablations `res256 cr1 cr25 n50 n200 it1k ov32`), run lists `configs/studies/01..05`, `scripts/study.py` (resumable driver), `scripts/summarize.py` (markdown, CSV, stdlib SVG, `--check`), `scripts/failure_cases.py`.
  - Spec `docs/specs/phase-5-cross-fabric-study.md` with the runbook and GPU-time estimate (RD4AD about 38 h, above the 27 h of the compute table, mostly RAW-FABRID held-out at 20,000 iterations; A-22).
- **Evidence:** `pytest -q`: 128 passed, 0 failed, 1 slow test deselected (new: zju/aitex 7, study 9, summarize 2). End-to-end driver on a synthetic four-pattern ZJU tree runs four regimes, writes the expected result paths, skips on rerun and survives a failing config; held-out `fit` reads no train or val row of the held-out fabric; image metrics and `by_fabric` equal an independent recomputation; `test_every_cell_has_a_config` ties every EVALUATION 4.3 cell and ablation variant to a config; `test_headline_settings_fixed` pins D-23/D-24 values.
- **Decisions:** A-21 to A-25.
- **Review:** round 1 PASS (the first reviewer attempt could not run Bash and returned INCOMPLETE; the progress log entry it found missing was written before the second attempt). Three minor findings fixed: the AITEX `--check` wording, the review record, and the note that the plan's headline-settings tick covers the laptop part only (the operator logs any later change).
- **Problems / next:** ZJU-Leaper and AITEX layouts unverified; ZLEval glue and all runs, findings and failure-case images are training-PC tasks (runbook, spec section 8). Driver end-to-end test takes about 100 s.

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
