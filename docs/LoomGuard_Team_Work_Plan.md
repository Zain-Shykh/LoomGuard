# LoomGuard: Team Work Plan, From Here to Release

**Date:** 25 November 2026
**Audience:** Huzaima (Member A, local RTX 5060 Ti), Zain (Member B, Kaggle) and Hamza (Member C, Kaggle).
**Read with:** [Decision Resolutions](LoomGuard_Decision_Resolutions.md) (what was decided and why), [Team Decisions and Execution Plan](LoomGuard_Team_Decisions_Needed.md) (A's technical detail), [Distributed training runbook](DISTRIBUTED_TRAINING.md), [Implementation plan](IMPLEMENTATION_PLAN.md), [Evaluation](EVALUATION.md), [Decisions](DECISIONS.md).
**Rule:** this plan is the order of work. It changes no scientific setting. Progress is judged by acceptance criteria with evidence (D-54), not by dates, so the plan uses gates, not a calendar.

## 1. Where the project stands

| Area | State | Evidence |
|---|---|---|
| Phases 1 to 6 code (data layer, PatchCore, RD4AD, calibration and grading, study driver, Streamlit app) | Code complete, tested on synthetic data and CPU | `IMPLEMENTATION_PLAN.md` status table |
| Three-person workflow and runner | Implemented, reviewed and merged (D-61). Gate-only preparation merged (`09d708f`) | `DISTRIBUTED_TRAINING.md` |
| RD4AD fp16 precision (D-60) and scoped FP32 teacher (D-62) | Approved. Synthetic 300-iteration test passed twice on each GPU (loss 0.04184 on A's RTX 5060 Ti, 0.04159 on B's T4, 0 skipped steps) | `LoomGuard_Member_B_RD4AD_T4_Regression_Results.md` |
| Datasets | Originally prepared and verified by Hamza (C), independently checked and staged by Huzaima (A) at `data/` and `artifacts/` (git-ignored). RAW-FABRID 4.4 GB, ZJU-Leaper 7.0 GB, MVTec 2.2 GB, AITEX 0.17 GB. AITEX is excluded by D-61 | `artifacts/dataset_handoff_notes.md` (local) |
| Kaggle feasibility for B | Done: T4, CUDA, tests, persistence of small files verified | `LoomGuard_Member_B_Kaggle_Feasibility_Checklist.md` |
| Open decisions | Original gate pin verified against local MVTec. Paired gate approval and cloud permissions are pending | `LoomGuard_Decision_Resolutions.md` |
| **Real-data training** | Zain's five MVTec PatchCore results are on main. Huzaima verified both original pin files at receipt and all five PatchCore export pairs on 28 November. The RD4AD half and paired gate approval follow the Member A runbook. | [B1 to B4](PROGRESS_LOG.md), [A checklist](TEAM_TASKS_MEMBER_A.md) |
| Phases 7 and 8 | Not started (Phase 7 is a decision gate; Phase 8 is the report and release) | |

Use the runbook criteria and evidence from each completed batch. A reviews the five PatchCore results and both methods' exported bundles before recording the paired approval.

Git records Zain's initial Phase 1, PatchCore and later implementation work, including the MVTec PatchCore results in `85da445` and the full pin record in `9a314ff`. Huzaima authored the distributed workflow (`0740600`), scoped FP32 teacher correction (`f7330a7`), gate-only preparation (`09d708f`) and its integration record (`67fb7f6`). Local validation and independent verification are recorded in the [progress log](PROGRESS_LOG.md). Hamza's dataset work is evidenced by the original private handoff, four manifests and reports. The handoff describes verifier and AITEX adapter changes that need their source diff and authorship confirmed before integration.

## 2. The critical path (nothing here is optional, and the order matters)

```text
S0 freeze SHA 67fb7f6  ->  S1 gate pin (MVTec)  ->  S2 each runner: tests + readiness
        |                                              |
        |                                              v
        |            S3 MVTec gate: A = 5 RD4AD profiles, B = 5 PatchCore profiles (parallel)
        |                                              |
        |                                              v
        |            S4 team review + --approve-gate (all 10 results pass D-33 and VRAM < 11 GB)
        |                                              |
        v                                              v
 S5 full pin (all 3 datasets)   ->   S6 RAW-FABRID: A (RD4AD), B (PatchCore)    in parallel
                                     S7 ZJU-mini: B (both methods) -> --approve-mini = owner go-ahead
                                              |
                                              v
                      S8 full ZJU + ablations: B and C (A finishes RAW-FABRID ablations)
                                              |
                                              v
                      S9 integrate -> summarize -> FINDINGS -> app validation -> report -> release
```

If the gate misses a target (S4), stop, record it, and let the team decide. Do not tune on test results (D-35). A bug fix is allowed; a hyperparameter change is a decision.

## 3. Handoff and execution checklist

| # | Who | Step | Done when |
|---|---|---|---|
| 1 | A | The common SHA `67fb7f6...` is approved subject to original-pin verification. | A verifies the original gate pin |
| 2 | B | Provide the original gate and full pins generated from Hamza's verified handoff. Document permitted ZJU access with the affected runner. | A verifies the received pins |
| 3 | B | Create a clean worktree or clone at the frozen SHA. Run `python scripts/distributed_study.py --validate` and the dry runs. Create the gate-only pin (R2). | `gate-shared/datasets.json` exists, its SHA-256 is noted. **Done 2026-11-25**, SHA-256 `702a0913c595aecfbdae73ab89be2f72ba2ccdead54f53b20f526b42c43981a0` |
| 4 | B | Send `datasets.json` and its SHA-256 to A and C over a private channel. | A and C received it |
| 5 | A | Verify the original gate pin and compare its tree and config manifests to the local verified inputs. | Original file and local data agree |
| 6 | A | Use the prepared experimental checkout and record original-pin readiness. Launch the five `mvtec_rd4ad` profiles after it passes. | Fresh readiness and first profile running |
| 7 | B | Record the terms and source of the existing private MVTec input and give A the notebook version and retrieval evidence. | B1 reports the upload and 300 MiB export round-trip |
| 8 | B | Send A the five completed PatchCore exports, original checksum manifests and attempt logs. | A verifies the pairs and keeps a second private copy |
| 9 | B | Record minutes per profile, coreset seconds, VRAM/RAM, export size and remaining quota. | B2 and B4 contain reported measurements |
| 10 | C | Maintain the dataset verification record, prepare for assigned RD4AD jobs and plan ZLEval work. | see section 5 |

## 4. Training ownership (from the approved ledger, D-61)

The ledger is `configs/studies/distributed/assignments.yaml`. Counts below are taken from it. GPU-hour figures are rough estimates from measured synthetic speeds (A 10.1 to 10.6 it/s, T4 5.1 to 5.3 it/s) and the compute table in `EVALUATION.md`. They are not measurements of real profiles. Replace them after the first real profiles.

### Member A: Huzaima, RTX 5060 Ti, RD4AD (36 profiles, 302,000 iterations)

| Stage | Work | Profiles |
|---|---|---:|
| Gate | `mvtec_rd4ad`: carpet, grid, leather, tile, wood | 5 |
| Study (standard) | `raw_fabrid_rd4ad_s0` (unified, seed 0), `raw_fabrid_per_fabric_rd4ad` (7 rolls), `raw_fabrid_held_out_rd4ad` (7 leave-one-roll-out) | 15 |
| Ablations (standard) | RAW-FABRID RD4AD `it1k` and `ov32` per roll (14), unified seeds `s1` and `s2` | 16 |

A also: coordinates the source freeze and approvals (`--approve-gate`, `--approve-mini` as recorded by the owner), keeps the master archive of results and bundles, and records RD4AD checks (false alarms 5 to 20 per 1,000 on RAW-FABRID, three-seed mean and spread, seven-roll grading, peak VRAM under 11 GB).

### Member B: Kaggle T4 (Zain), PatchCore plus part of RD4AD (124 active profiles; 138 with AITEX, which is excluded)

| Stage | Work | Profiles |
|---|---|---:|
| Gate | `mvtec_patchcore`: five textures | 5 |
| ZJU-mini (study) | PatchCore and RD4AD headline, patterns 1, 4, 8, 13, 16 | 10 |
| Study (standard) | RAW-FABRID PatchCore: unified (1), per roll (7), leave-one-roll-out (7) | 15 |
| Full ZJU (after go-ahead) | PatchCore: 14 rest, 5 groups, unified, 5 held-out. RD4AD: 14 rest | 39 |
| Ablations | RAW-FABRID PatchCore `cr1`, `cr25`, `n50`, `n200`, `ov32` (35). ZJU-mini PatchCore `cr1`, `cr25`, `res256` (15) and RD4AD `res256` (5) | 55 |

### Member C: Kaggle T4 (Hamza), RD4AD on ZJU-Leaper (11 profiles, 120,000 iterations)

| Stage | Work | Profiles |
|---|---|---:|
| Full ZJU (after go-ahead) | `zju_per_group_rd4ad` (5 groups at 10k), `zju_unified_rd4ad` (1 at 20k), `zju_held_out_rd4ad` (5 at 10k) | 11 |

C has no training queue until the ZJU go-ahead. C can assess the 14 unstarted per-pattern ZJU RD4AD profiles if A requests a workload review at the ZJU go-ahead.

### Expected run order per member

- **A:** MVTec gate, then RAW-FABRID unified seed 0, per roll, leave-one-roll-out, then the ablation queue. The `it1k` and `ov32` ablations stay on A's machine with their baselines.
- **B:** MVTec gate, ZJU-mini PatchCore, ZJU-mini RD4AD, RAW-FABRID PatchCore (can run while the mini review is pending), RAW-FABRID PatchCore ablations, then after the go-ahead the full ZJU queue and the ZJU ablations.
- **C:** support work now; full ZJU RD4AD after the go-ahead.

### Kaggle quota rules (B and C)

- Free accounts get a weekly GPU allowance. B's account showed 29 of 30 hours on 22 November. C's account must be checked. Both read the live quota at the start of each week.
- Budget each session for the whole profile: readiness, fit, evaluation, export and a recovery margin. Stop idle sessions.
- Never leave a profile half exported. Export after each completed profile. A run that dies restarts the profile from the beginning.
- If a queue will not fit, tell the team and request an approved reassignment. Do not change budgets.

## 5. The rest of the project: ownership of non-training work

This is the work that remains after or alongside training, split so the load is fair given who is waiting on GPU time. Each owner is the first author; a different member reviews it.

| Work item | Plan reference | Owner | Reviewer | Starts when |
|---|---|---|---|---|
| Source freeze, gate approval, results integration (`--check-integration`, `summarize.py --check`, `summarize.py`) | Checkpoints 1 to 3 | A | B | after the first results |
| Phase 1 acceptance: verifier output pasted into `PROGRESS_LOG.md`, "unverified" items in `DATASETS.md` corrected, RAW-FABRID counts and roll verdict (SAME) recorded | Phase 1 criteria | C | A | now |
| Dataset and licence record: MVTec terms, ZJU-Leaper permission, citations as authors request | Phase 8 | B | affected runner | now |
| ZLEval glue and official ZJU metrics (skip if more than a few hours, say so) | Phase 5 task 7 | C | B | after ZJU-mini |
| Share the dataset handoff verifier/AITEX diff and authorship evidence for separate review | R1 | C | A | when supplied |
| FINDINGS: Q1, Q2 (RAW-FABRID), Q5 false alarms, Q6 grading | `EVALUATION.md` | A | B | after RAW-FABRID results |
| FINDINGS: Q3 (group, unified) and Q4 (held-out, leave-one-roll-out) | `EVALUATION.md` | C | A | after full ZJU |
| FINDINGS: Q7 ablations and ablation plots, PatchCore side of Q2 on ZJU | `EVALUATION.md` | B | C | after ablations |
| Failure-case panels: at least three per method (RAW-FABRID images only, no MVTec) | Phase 5 | B | A | after RAW-FABRID results |
| App validation on the CPU laptop (Part 5): under 10 s per 1792x1024 image, parity with `evaluate.py`, roll stream grade, wrong-fabric warning thresholds (A-27) | Phase 6 | B | C | after first RAW-FABRID bundles |
| Demo samples (RAW-FABRID CC BY, with credit) and Space deployment to the existing Space | Phase 6 task 6 | B (owner of the Hugging Face accounts) | A | after app validation |
| Real-bundle upload to the Hugging Face repo | A-20, A-26 | B | A | after the full study |
| Phase 7 decision gate (O-01 third method, O-02 stretch): default is none; record the decision | Phase 7 | all three | n/a | after Phase 5 results |
| Report: methods and MVTec benchmark, RAW-FABRID results, grading | Phase 8 | A | B | after results |
| Report: system, app, PatchCore, ablations, deployment, README reproduction steps | Phase 8 | B | C | after results |
| Report: datasets, protocol, ZJU results, limits (assumed scales, no hole rule, low-confidence calibration, licences), references | Phase 8 | C | A | after results |
| Final clean test run and one end-to-end reproduction by a member who did not write the code | Phase 8 | C reproduces one MVTec and one RAW-FABRID row from the README | n/a | last |

**Fairness check.** A carries the heaviest RD4AD and the integration. B carries the most profiles (cheap PatchCore ones), the app and the deployment. C carries the fewest training hours and so owns the data record, ZJU analysis and the independent reproduction. If measured times show an imbalance, use the rebalance rule in section 7.

## 6. Gates and what each one needs

| Gate | Needs | Pass criteria | If it fails |
|---|---|---|---|
| **G1 MVTec gate** (S4) | 10 verified result and bundle pairs on the gate pin | Mean over five textures. PatchCore 97.9 / 96.6 / 90.7. RD4AD 98.5 / 96.7 / 92.0 (image AUROC / pixel AUROC / AUPRO in %). RD4AD peak VRAM under 11 GB (D-33, D-59) | Stop. Record. Team review. Options listed in D-60 (RD4AD) and the risks table. Do not tune on test data |
| **G2 ZJU-mini** (S7) | 10 verified ZJU-mini pairs, adapter and split evidence, roll check (verdict SAME, so ZJU held-out carries the "unseen fabric" claim), first RAW-FABRID results | Owner records the go-ahead in `--approve-mini --review` | Fix adapter bugs only. No setting changes |
| **G3 RAW-FABRID checks** | A and B results | False alarms 5 to 20 per 1,000, unified PatchCore coreset under 600 s, RD4AD seed spread, grading for 7 rolls | Log, analyse, report as a finding (Q5, Q6) |
| **G4 Integration** | All three exports | `--check-integration` and `summarize.py --check` pass. The only accepted gaps are the 14 AITEX cells, reported as intentional omissions | Fix missing paths. Never claim a full study with gaps |
| **G5 App** | Real bundles on a CPU laptop | Phase 6 criteria | Fix, re-measure |
| **G6 Release** | Report, README, reproduction by someone else | Phase 8 criteria | Fix |

## 7. Rebalancing rule

Compare actual runtime and live quota after the first real profiles and again at the ZJU go-ahead. Keep the current ledger until a separate amendment is approved.

| Member | Work remaining | RD4AD fit-only estimate |
|---|---|---|
| A | 36 RD4AD profiles, 302,000 iterations, including 25,000 MVTec iterations | 7.9 to 8.3 h on the local RTX |
| B | 95 PatchCore and 24 RD4AD profiles, 120,000 RD4AD iterations | 6.3 to 6.6 h on a T4, plus PatchCore |
| C | 11 ZJU RD4AD profiles, 120,000 iterations | 6.3 to 6.6 h on a T4 |

The estimates use measured synthetic rates of 10.1 to 10.6 it/s locally and 5.1 to 5.3 it/s on T4. Add evaluation, preparation, export and session restarts. Published MVTec PatchCore bundles were 62 to 76 MB per texture. Budget about 12 GB for the 71 RD4AD weight files at 168 MB each, plus PatchCore bundles, logs, staging and backups. Measure later bundle sizes before scheduling exports. Check live Kaggle limits on each account.

B's RD4AD work is five ZJU-mini baseline profiles, 14 remaining per-pattern profiles and five resolution ablations, each at 5,000 iterations. If the team approves and C has a permitted T4 route and capacity, transfer the 14 unstarted remaining per-pattern profiles to C after ZJU-mini. This shifts 70,000 iterations. B would have 95 PatchCore and 10 RD4AD profiles remaining, with 50,000 RD4AD iterations (2.6 to 2.8 h fit-only). C would have 25 RD4AD profiles, with 190,000 iterations (10.0 to 10.4 h fit-only). Keep the five mini baseline and five resolution ablation profiles together on B's machine. A strict split could move that entire unstarted baseline and ablation pair to C before either runs, putting C at 240,000 iterations and about 12.6 to 13.1 h fit-only. A is another potential destination for the whole pair if its local capacity and access are confirmed. A transfer requires verification of the revised ledger, new readiness, stable canonical paths and the owner's explicit workload approval. Move whole unstarted profiles and preserve the machine for each baseline and its ablation.

## 8. Rules everyone follows

1. One frozen source SHA for training. Training checkouts never move to `main`. Docs go on `main`.
2. Never commit `data/`, `artifacts/`, bundles, MVTec images or secrets. Results JSON files in `results/` and documents are committed, from the results checkout, on separate branches and through reviewed merges.
3. Do not run `--pin-datasets` on someone else's behalf without the named owner agreeing. Do not edit `datasets.json`. Do not fake approvals. Do not skip gates.
4. Every completed profile is exported, hash-checked and retrieved by A before it counts as done.
5. Log measured numbers in `docs/PROGRESS_LOG.md`, newest first, one entry per finished batch. If a measurement contradicts a document, fix the document in the same commit.
6. MVTec AD is noncommercial and private. No MVTec images in Git, the demo or any public notebook or dataset.
7. Stop and ask when something does not match the pin (data, source, splits) instead of working around it.

## 9. Risks that matter now

| Risk | Effect | Response |
|---|---|---|
| Kaggle session cap or export loss on a failed run | Lost profile | One profile per saved run until timings are known. Export test before any GPU use (R4) |
| Account GPU quota runs out | B or C stalls | Budget per profile, stop idle sessions, request approved reassignment |
| MVTec terms question comes back "no" | B cannot use Kaggle for the gate | Fallback: A runs B's gate profiles (R3) |
| Gate misses a target | Study blocked | Stop, record, review. D-60 lists RD4AD options. No test-set tuning |
| Pin mismatch between machines | Run refused | Stop. Find the differing file or extraction. Do not edit the pin |
| ZJU licence unconfirmed | ZJU queues blocked on Kaggle | C asks the authors. Use authorized local or institutional GPUs |
| A is the single point for results integration | Delay | A keeps a second private copy. B keeps its downloaded exports until A confirms |
| Real-data bugs in adapters | Small fixes needed | Fix the bug, keep tests passing, log it. A fix after the freeze means a new SHA and a new pin. Prefer finding them on the gate and ZJU-mini first |

## 10. How to update this file

When a gate passes or an assignment changes, add a dated line to `docs/PROGRESS_LOG.md` and, if a decision changed, a new row in `LoomGuard_Decision_Resolutions.md` that supersedes the old one. This plan changes only when scope or ownership changes.
