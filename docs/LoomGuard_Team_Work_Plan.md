# LoomGuard: Team Work Plan, From Here to Release

**Date:** 25 November 2026
**Audience:** Members A (Hosama Malik, RTX 5060 Ti), B (Zain, Kaggle) and C (Hamza Atif, Kaggle), and their AI agents.
**Read with:** [Decision Resolutions](LoomGuard_Decision_Resolutions.md) (what was decided and why), [Team Decisions and Execution Plan](LoomGuard_Team_Decisions_Needed.md) (A's technical detail), [Distributed training runbook](DISTRIBUTED_TRAINING.md), [Implementation plan](IMPLEMENTATION_PLAN.md), [Evaluation](EVALUATION.md), [Decisions](DECISIONS.md).
**Rule:** this plan is the order of work. It changes no scientific setting. Progress is judged by acceptance criteria with evidence (D-54), not by dates, so the plan uses gates, not a calendar.

## 1. Where the project stands

| Area | State | Evidence |
|---|---|---|
| Phases 1 to 6 code (data layer, PatchCore, RD4AD, calibration and grading, study driver, Streamlit app) | Code complete, tested on synthetic data and CPU | `IMPLEMENTATION_PLAN.md` status table |
| Three-person workflow and runner | Implemented, reviewed and merged (D-61). Gate-only preparation merged (`09d708f`) | `DISTRIBUTED_TRAINING.md` |
| RD4AD fp16 precision (D-60) and scoped FP32 teacher (D-62) | Approved. Synthetic 300-iteration test passed twice on each GPU (loss 0.04184 on A's RTX 5060 Ti, 0.04159 on B's T4, 0 skipped steps) | `LoomGuard_Member_B_RD4AD_T4_Regression_Results.md` |
| Datasets | Verified by C, handed off, extracted and staged by B at `data/` and `artifacts/` (git-ignored). RAW-FABRID 4.4 GB, ZJU-Leaper 7.0 GB, MVTec 2.2 GB, AITEX 0.17 GB. AITEX is excluded by D-61 | `artifacts/dataset_handoff_notes.md` (local) |
| Kaggle feasibility for B | Done: T4, CUDA, tests, persistence of small files verified | `LoomGuard_Member_B_Kaggle_Feasibility_Checklist.md` |
| Open decisions | Resolved provisionally on 25 November | `LoomGuard_Decision_Resolutions.md` |
| **Real-data training** | **Not started.** No dataset pin, no gate result, no approval exists | |
| Phases 7 and 8 | Not started (Phase 7 is a decision gate; Phase 8 is the report and release) | |

Every acceptance criterion in `IMPLEMENTATION_PLAN.md` that says "on MVTec", "on RAW-FABRID" or "on the training PC" is still unticked. They are ticked only by the runs below.

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

## 3. Immediate steps (first 48 hours)

| # | Who | Step | Done when |
|---|---|---|---|
| 1 | A | Read `LoomGuard_Decision_Resolutions.md`. Confirm R1 (SHA `67fb7f6...`) or name another. Reply in `LoomGuard_Team_Decisions_Needed.md` or chat. | A's answer recorded |
| 2 | C | Read the same file. Confirm R2 (B makes the pin, C cross-checks the MVTec tree hash). Confirm which Kaggle account holds the ZJU copy later. | C's answer recorded |
| 3 | B | Create a clean worktree or clone at the frozen SHA. Run `python scripts/distributed_study.py --validate` and the dry runs. Create the gate-only pin (R2). | `gate-shared/datasets.json` exists, its SHA-256 is noted. **Done 2026-11-25**, SHA-256 `702a0913c595aecfbdae73ab89be2f72ba2ccdead54f53b20f526b42c43981a0` |
| 4 | B | Send `datasets.json` and its SHA-256 to A and C over a private channel. | A and C received it |
| 5 | C | Reproduce the MVTec tree digest on C's own copy and compare. | digests equal, or a stop-and-investigate note |
| 6 | A | On the RTX 5060 Ti: checkout the SHA, run the Part 0 environment check (`sm_120`, tests, slow overfit), then `--prepare --queue gate`, then start the five `mvtec_rd4ad` profiles. | A's readiness recorded; first profile running |
| 7 | B | Upload the private MVTec Kaggle Dataset (R3). Run the CPU-only export test (R4): about 300 MB file, saved version, download, hash. | upload visible, export test PASS |
| 8 | B | Open the gate notebook on the GPU T4 session. Run setup, tests, `--prepare --queue gate`, then `--job mvtec_patchcore --profile carpet`. | carpet result and bundle exported and hash-checked |
| 9 | B | Record minutes per profile, coreset seconds, VRAM/RAM, export size, remaining quota. Decide how many profiles per session. | numbers in `PROGRESS_LOG.md` |
| 10 | C | While waiting: items in section 5 (data and licence documentation, ZLEval glue). | see section 5 |

## 4. Training ownership (from the approved ledger, D-61)

The ledger is `configs/studies/distributed/assignments.yaml`. Counts below are taken from it. GPU-hour figures are rough estimates from measured synthetic speeds (A 10.1 to 10.6 it/s, T4 5.1 to 5.3 it/s) and the compute table in `EVALUATION.md`. They are not measurements of real profiles. Replace them after the first real profiles.

### Member A: RTX 5060 Ti, RD4AD (36 profiles, 302,000 iterations)

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

C has no training queue until the ZJU go-ahead. C has the cheapest training load and the most waiting time, so C takes more of the non-training work in section 5 and, if A approves at the ZJU go-ahead, the 14 per-pattern ZJU RD4AD profiles (resolution R8).

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
| Dataset and licence record: MVTec terms, ZJU-Leaper permission, citations as authors request | Phase 8 | C | B | now |
| ZLEval glue and official ZJU metrics (skip if more than a few hours, say so) | Phase 5 task 7 | C | B | after ZJU-mini |
| C's verifier/AITEX patch review and merge (post-study, separate commit) | R1 | C | A | after training |
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

Check at the first real profiles and again at the ZJU go-ahead.

1. Record minutes per PatchCore profile and RD4AD iterations per second per machine.
2. Estimate the remaining time of each member's unstarted profiles.
3. If one queue is clearly the longest, move only whole unstarted profiles to reduce the longest completion time. Candidate: the 14 ZJU per-pattern RD4AD profiles from B to C (see R8).
4. Rules for a move: A (coordinator) approves, the moved shard gets a new job ID, the plan is revalidated and passed with `--plan`, readiness is rerun, completed and running profiles stay where they are, and an ablation stays with its baseline. Seeds, budgets, configs, splits and metrics do not change.
5. A unified, per-group or held-out fit is one job and cannot be split.

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
| Weekly Kaggle quota (30 h) runs out | B or C stalls | Budget per profile, stop idle sessions, request approved reassignment |
| MVTec terms question comes back "no" | B cannot use Kaggle for the gate | Fallback: A runs B's gate profiles (R3) |
| Gate misses a target | Study blocked | Stop, record, review. D-60 lists RD4AD options. No test-set tuning |
| Pin mismatch between machines | Run refused | Stop. Find the differing file or extraction. Do not edit the pin |
| ZJU licence unconfirmed | ZJU queues blocked on Kaggle | C asks the authors. Use authorized local or institutional GPUs |
| A is the single point for results integration | Delay | A keeps a second private copy. B keeps its downloaded exports until A confirms |
| Real-data bugs in adapters | Small fixes needed | Fix the bug, keep tests passing, log it. A fix after the freeze means a new SHA and a new pin. Prefer finding them on the gate and ZJU-mini first |

## 10. How to update this file

When a gate passes or an assignment changes, add a dated line to `docs/PROGRESS_LOG.md` and, if a decision changed, a new row in `LoomGuard_Decision_Resolutions.md` that supersedes the old one. This plan changes only when scope or ownership changes.
