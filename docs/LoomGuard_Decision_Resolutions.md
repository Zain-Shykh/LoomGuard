# LoomGuard: Decision Resolutions (provisional, 25 November 2026)

**Why this file exists.** The five questions in [LoomGuard_Team_Decisions_Needed.md](LoomGuard_Team_Decisions_Needed.md) stayed open because Members A and C were busy. Member A's review there already settled the technical points (Decisions 2 and 5) and left Decisions 1, 3 and 4 waiting for an owner call. On 25 November the repository owner (Zain, Member B) asked for every open question to be resolved with the recommended option so training can start. This file records each answer, the reason, the fallback and who can reverse it.

**Status of every row below: decided by the owner on 25 November. Members A and C are informed, not asked to approve.** A or C can still raise an objection before the first real run. If either does, add a new row that supersedes the old one. Do not edit history.

**What this file does not do.** It does not change a scientific setting, a seed, a budget, a threshold, a split or the assignment ledger (`configs/studies/distributed/assignments.yaml`). It does not approve a checkpoint (`--approve-gate`, `--approve-mini`). It does not create `datasets.json`. It is not a legal opinion on any dataset licence.

## Summary

| # | Question | Resolution | Reversible by |
|---|---|---|---|
| R1 | Which source commit does everyone run? | Freeze `67fb7f6ca2dec0365b63a7454ffb57b3138b6875`. C's verifier/AITEX changes are not part of it. | A |
| R2 | Who creates the dataset pin? | Gate-only pin now, created by B from the handoff bytes it already holds. C cross-checks the MVTec tree hash. Full pin later, same way. | A, C |
| R3 | How does MVTec reach Kaggle? | B uploads one private Kaggle Dataset with the five textures, shared with nobody else. Fallback: A's RTX 5060 Ti runs B's gate profiles. | Owner |
| R4 | Where do bundles and results go? | P1: one profile per saved private Kaggle run, B downloads and hash-checks, then hands the files to A privately. P3 (tokens) is not used. | Owner |
| R5 | How does B prepare with only MVTec? | Already built and merged: gate-only pin plus `--prepare --queue gate`. No reduced plan file. | n/a (technical) |
| R6 | How is a Kaggle session run? | One T4, `CUDA_VISIBLE_DEVICES=0`, one profile per session until timings are measured. | B |
| R7 | RAW-FABRID and ZJU on Kaggle? | RAW-FABRID: private Kaggle Dataset when the standard queue starts. ZJU: deferred until the ZJU go-ahead, licence check first. | Owner, C |
| R8 | Fair workload split? | Keep the approved ledger now. Propose one move at the ZJU go-ahead (R8 below). Non-training work split in the plan file. | A (coordinator) |

## R1. Source commit

**Resolution.** Every real training run uses commit `67fb7f6ca2dec0365b63a7454ffb57b3138b6875` (head of `main` on 25 November). Its last code change is `09d708f794d20bfebdca30f84118debe8b9e6370` (gate-only preparation). It contains the RD4AD scoped-teacher-precision fix (D-62), the full ledger and the scoped runner. A's review of this code recorded a focused suite of 70 passed and a full non-slow suite of 187 passed with 1 known Streamlit rounding failure, which the owner accepted. B has not re-run the full suite on this SHA (the laptop has no GPU), so each runner records its own test result in its readiness evidence.

**How to use it.** Training checkouts check out the SHA directly (`git checkout 67fb7f6ca2dec0365b63a7454ffb57b3138b6875`, detached) in a separate clone or worktree. The runner requires the exact pinned SHA and rejects changes in `configs`, `loomguard`, `scripts` or `pyproject.toml`, including untracked files. Documentation commits on `main` after this SHA (including this file) do not affect training checkouts, but never run a training session from `main`.

**C's verifier and AITEX changes.** Not included. Reasons: the handoff contains no patch; AITEX is excluded by D-61; A found that the active manifests are reproduced by the runner's own `inspected_manifest` and match C's CSVs for the representative configs. The verifier is in `scripts/`, so merging it later changes the source SHA. Merge it **after the study** as a reporting fix on `main`. Result files keep the SHA they were produced with. If C finds a difference that changes an active split, target or setting, stop: that needs scientific approval and a new pin.

**Fallback.** If A wants a different SHA, A names it. Everything that follows (pin, readiness) is then redone on that SHA before any run.

## R2. Dataset pin

**Resolution.**
1. **Gate-only pin first.** It needs only the MVTec tree. Command (on the frozen checkout):
   `python scripts/distributed_study.py --shared artifacts/distributed/gate-shared --pin-datasets --queue gate --review "<evidence>"`.
2. **Who:** Member B, on the laptop, CPU only. Reason: B holds the full handoff bytes already, the pin is deterministic from file names and bytes, and waiting for C stalls both gate halves. C remains the dataset verifier of record.
3. **Cross-check:** C hashes the same five textures from C's copy by running the same command in a scratch directory and compares the `mvtec_ad` tree digest with B's `datasets.json` (`cmp` on the files, or compare the digest line). A mismatch stops everything. Possible causes are a different extraction, a stray file or a corrupt copy.
4. **Review reference:** the handoff's `artifacts/verify_datasets.txt` and `dataset_handoff_notes.md`, plus B's archive/tree hash note. It must name real evidence, not a placeholder.
5. **Full pin later:** after the gate halves are in, B creates the ordinary full pin the same way (B holds MVTec, RAW-FABRID and ZJU), C cross-checks, and A restores the ten gate results and compares the two approvals, as specified in the [advance-to-full-study steps](LoomGuard_Team_Decisions_Needed.md#advance-to-the-full-study).
6. **AITEX stays out** (D-61). No `--aitex-approval`.
7. `datasets.json` is small (hashes and the commit) and goes to A over a private channel. It is not committed.

## R3. MVTec to Kaggle

**Resolution.** B creates one **private** Kaggle Dataset containing only `mvtec_ad/{carpet,grid,leather,tile,wood}` with the original folder names (so the tree hash still matches). It is shared with no one else. A uses its local copy. C has no gate profile. The dataset description records the source, CC BY-NC-SA 4.0, attribution to MVTec and "academic, noncommercial, private". It is deleted when the project ends.

**Basis and risk.** The licence allows noncommercial sharing with attribution and ShareAlike. The project is a university assignment. Kaggle's terms require the uploader to hold the rights to what they upload, and a private setting does not by itself prove that. The owner accepts this risk for a private, access-limited, noncommercial copy. This is a project decision, not legal advice. If MVTec or the university says no, use the fallback.

**Fallback.** A's RTX 5060 Ti runs B's five `mvtec_patchcore` profiles in a separate time slot (PatchCore on five textures is short). That needs A's agreement, a note in the decision record and a recorded readiness for owner B on that machine. Nothing else about the ledger changes.

**Owner confirmation, 25 November.** The owner accepted the Kaggle upload for MVTec (private, noncommercial, no demo use, everything else git-ignored). If the Kaggle upload does not work, B's gate profiles run on A's RTX 5060 Ti.

**Not decided here.** Making any copy public. Never do that.

## R4. Bundles and results

**Resolution.**
- **Transport P1.** Each B profile runs inside a Kaggle notebook saved as a private version. The Output tab of that version holds only the approved export files. B downloads them, checks the SHA-256 of the result JSON and every bundle file against `provenance.json`, and copies them to a private location A can read (a private shared drive or an equivalent named-access folder). A retrieves, verifies and keeps the master archive plus a second private copy.
- **`--bundle-location`:** `kaggle-version://muhammadzain1x0/<notebook-slug>` for B's runs. It is only a text label recorded in provenance and contains no credentials. The version number goes into the `--review` text and the private handoff index.
- **P3 not used.** A Hugging Face or Kaggle write token in Kaggle Secrets is a credential decision. It is not needed to start. Revisit if P1 fails the retrieval test below. The repo's rule that final bundles go to the Hugging Face Hub still applies at release time (Phase 6), not for working copies.
- **One profile per saved run at first.** A failed or timed-out run may lose its output files, and this is untested at full size.

**Before spending GPU quota (CPU session, no GPU hours):** save a version that writes a harmless file of about 300 MB (more than RD4AD's 168 MB student weights), then download it in a fresh session and hash-check it. If this passes, go ahead. If it fails, stop and move to P3 or to A's machine.

**Demo bundles.** Pre-staged on the CPU laptop before the app starts (`LOOMGUARD_BUNDLES`). The app currently downloads public Hub files anonymously, so private bundles are copied locally first.

## R5. Preparation with only MVTec

Resolved by the merged runner (`09d708f`). Use the gate-only pin and `--prepare --queue gate`. Options O1 (attach all datasets) and O3 (wait) are not needed. O2 (a reduced plan file) is rejected by `load_plan` and A's change superseded it. B does not run `--pin-datasets` on Kaggle, `--approve-gate` or `--approve-mini`.

## R6. Kaggle session rules for B

- Accelerator: GPU T4 x2, but run on one device. Put `export CUDA_VISIBLE_DEVICES=0` in the shell that runs setup, readiness and training. Two T4s are separate devices. Do not enable multi-GPU.
- First profile: `--job mvtec_patchcore --profile carpet`, alone in its saved run. Record minutes per profile, coreset seconds, peak VRAM/RAM and export size. Run two or three profiles per session only if the measured time and remaining quota leave room and each result is exported immediately.
- Readiness goes stale per session. Re-run `--prepare --queue gate` in every new session. It needs a GPU session (CPU-only sessions fail).
- ImageNet weights (about 130 MB) download on first use. Keep Kaggle internet on and check once before a timed run.
- Check the weekly GPU quota at the start of each week. Stop idle sessions. If the account does not show a session cap, record it as unknown.
- B does not change precision, seeds, coreset ratio or budgets.

## R7. RAW-FABRID and ZJU on Kaggle

- **RAW-FABRID** (CC BY 4.0, 4.4 GB): B uploads a private Kaggle Dataset of `RAW_FABRID` when the standard queue is about to start, with credit, DOI and licence in the description. A uses its local copy.
- **ZJU-Leaper** (7 GB): the owner decided on 25 November that ZJU-Leaper may be used and uploaded privately to Kaggle for this noncommercial semester project, even though the authors state no licence. The data stays private, on team laptops and private Kaggle inputs only, and is never published. The ZJU go-ahead (D-61) for the full queues is still a separate step. C may still ask the authors for written permission, but it no longer blocks anything. If Kaggle upload proves impractical, B and C run ZJU on an authorized local GPU.
- Sizes are the extracted sizes from the handoff. B checks Kaggle's actual upload and storage limits before each upload.

## R8. Fair workload split

The ledger is unbalanced in profile count: A 36, B 124 active (AITEX excluded), C 11. Profile count is a poor measure, because RD4AD profiles are far more expensive than PatchCore ones. In iterations it is closer: A 302,000, B 120,000 plus about 100 PatchCore profiles, C 120,000. From the measured synthetic speeds (A 10.1 to 10.6 it/s, T4 5.1 to 5.3 it/s) that is roughly A 8 h, B 6.5 h of RD4AD plus PatchCore time, C 6.5 h, before evaluation, export and the large ZJU evaluations. Treat all of these as estimates until real profiles are timed.

**Decision now:** keep the approved ledger for the gate, the ZJU-mini headlines and the RAW-FABRID queues. Nobody is reassigned before measurements.

**Proposal for the ZJU go-ahead (needs A's approval, because it edits the ledger):** move the 14 `zju_per_fabric_rest_rd4ad` profiles (70,000 iterations) from B to C as a new job ID, and keep every ablation on its baseline's machine. After that, C is about 190,000 RD4AD iterations (about 10 h) and B is about 50,000 RD4AD iterations plus all the cheap PatchCore profiles. C already stages ZJU, so no extra data moves. Apply it only to unstarted profiles, with a revalidated plan file passed through `--plan`, fresh readiness and the same seeds, budgets and configs.

Until the ZJU go-ahead, C has no training queue. The plan file gives C other work in that time (data and licence checks, ZLEval glue, report sections).

## Open items that remain (not resolved here)

| Item | Who | Why it stays open |
|---|---|---|
| Written confirmation of MVTec terms from the actual download page | Owner | R3 is a risk acceptance, not a verification. Owner accepted it on 25 November. |
| Optional: written permission from the ZJU-Leaper authors | C | Owner decided use is acceptable without it (R7). |
| Telling A and C the SHA and the fallback | B | The owner decided approval from A and C is not required. They need the SHA to check out, and A's machine for the fallback. |
| Real Kaggle numbers: session cap, transfer size, quota | B | Measured by the first runs. |
| C's verifier/AITEX diff | C | Post-study reporting fix. |
| App fix for the `test_sensitivity_slider_does_not_rerun_model` rounding failure | Owner | Tracked separately; accepted for integration. |
