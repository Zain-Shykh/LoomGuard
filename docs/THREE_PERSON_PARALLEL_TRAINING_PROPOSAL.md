# LoomGuard — Three-Person Parallel Training Proposal

**Status:** Proposed team workflow; not yet an approved change to the repository's experiment protocol.  
**Repository:** https://github.com/Zain-Shykh/LoomGuard  
**Target:** Run independent experiments on one RTX 5060 Ti PC and two separate Kaggle GPU sessions, then combine results without changing the scientific configurations.

## 1. Proposed ownership

| Member | Execution environment | Primary responsibility | Assigned experiments |
|---|---|---|---|
| **Member A (local GPU PC)** | RTX 5060 Ti, 16 GB VRAM, 32 GB RAM | **RAW-FABRID RD4AD** and its benchmark | MVTec RD4AD; RAW-FABRID RD4AD per-roll, unified seeds 0/1/2, and leave-one-roll-out |
| **Member B (Kaggle 1)** | Own Kaggle GPU session | **All PatchCore**, plus shorter RD4AD jobs | MVTec PatchCore; PatchCore headline runs and ablations; ZJU-Leaper RD4AD per-pattern (mini and remaining patterns); optional AITEX RD4AD |
| **Member C (Kaggle 2)** | Separate Kaggle GPU session | **Remaining RD4AD cross-fabric work** | ZJU-Leaper RD4AD per-group, unified, held-out; RD4AD ablations except RAW-FABRID unified seeds 1/2 |

The three environments run independently. The team shares **one pinned source commit, one set of dataset/split definitions, and one result schema**.

### Starting workload allocation

These are **planning figures from the earlier repo estimate, scaled for the RTX 5060 Ti**, not measured Kaggle timings.

| Member | Approximate workload at the local 5060 Ti estimate | Notes |
|---|---:|---|
| A | ~22 h for RAW-FABRID RD4AD, plus MVTec RD4AD | Contains seven held-out jobs and three unified seeds |
| B | ~8–10 h PatchCore + ~9 h ZJU per-pattern RD4AD | Optional AITEX RD4AD adds ~3 h |
| C | ~11 h ZJU RD4AD group/unified/held-out + ~6 h RD4AD ablations | Includes full ZJU-mini prerequisite coordination |

**Equality rule:** Balance by **measured remaining wall-clock workload per person**, not by the number of config files. After initial measurements on both Kaggle and local hardware, reassign **unstarted whole experiments or complete profiles** to the least-busy member. Do not split an individual 5,000/10,000/20,000-iteration fit between machines. Account for Kaggle quota and session limits when scheduling.

## 2. Exact experiment allocation

Use these existing configurations. Do not duplicate a completed run or give the same result path to two people.

### A — local RTX 5060 Ti: RAW-FABRID RD4AD

**Benchmark gate**

- `configs/mvtec_rd4ad.yaml` — five MVTec texture profiles.

**Main RAW-FABRID work**

- `configs/raw_fabrid_per_fabric_rd4ad.yaml` — seven roll profiles.
- `configs/raw_fabrid_rd4ad_s0.yaml` — unified seed 0.
- `configs/raw_fabrid_rd4ad_s1.yaml` — unified seed 1.
- `configs/raw_fabrid_rd4ad_s2.yaml` — unified seed 2.
- `configs/raw_fabrid_held_out_rd4ad.yaml` — seven held-out roll profiles.

**A also owns:** RAW-FABRID RD4AD headline evaluation, false-alarm checks, and the related grading evidence. Each output should be available to the whole team for integration.

### B — Kaggle 1: PatchCore + ZJU per-pattern RD4AD

**Benchmark gate**

- `configs/mvtec_patchcore.yaml` — five MVTec texture profiles.

**PatchCore main study**

- `configs/zju_per_fabric_mini_patchcore.yaml`
- `configs/raw_fabrid_per_fabric_patchcore.yaml`
- `configs/raw_fabrid_patchcore.yaml`
- `configs/raw_fabrid_held_out_patchcore.yaml`
- `configs/zju_per_fabric_rest_patchcore.yaml`
- `configs/zju_per_group_patchcore.yaml`
- `configs/zju_unified_patchcore.yaml`
- `configs/zju_held_out_patchcore.yaml`
- `configs/aitex_per_fabric_patchcore.yaml` **only if AITEX is included**.

**PatchCore ablations** (existing `configs/studies/05_ablations.txt` PatchCore entries)

- `configs/zju_mini_patchcore_cr1.yaml`
- `configs/zju_mini_patchcore_cr25.yaml`
- `configs/zju_res256_patchcore.yaml`
- `configs/raw_fabrid_per_fabric_patchcore_cr1.yaml`
- `configs/raw_fabrid_per_fabric_patchcore_cr25.yaml`
- `configs/raw_fabrid_per_fabric_patchcore_n50.yaml`
- `configs/raw_fabrid_per_fabric_patchcore_n200.yaml`
- `configs/raw_fabrid_per_fabric_patchcore_ov32.yaml`

**RD4AD jobs assigned to B**

- `configs/zju_per_fabric_mini_rd4ad.yaml` — five ZJU-mini patterns.
- `configs/zju_per_fabric_rest_rd4ad.yaml` — remaining 14 ZJU patterns.
- `configs/aitex_per_fabric_rd4ad.yaml` **only if AITEX is included**.

### C — Kaggle 2: remaining RD4AD cross-fabric experiments

**Main cross-fabric RD4AD**

- `configs/zju_per_group_rd4ad.yaml` — five groups.
- `configs/zju_unified_rd4ad.yaml` — one unified model.
- `configs/zju_held_out_rd4ad.yaml` — five held-out patterns.

**RD4AD ablations**

- `configs/raw_fabrid_per_fabric_rd4ad_it1k.yaml`
- `configs/raw_fabrid_per_fabric_rd4ad_ov32.yaml`
- `configs/zju_res256_rd4ad.yaml`

**Not assigned to C:** RAW-FABRID RD4AD seeds 1 and 2. Those belong to A and must not be rerun just because they appear in the original ablation list.

### Optional scope after dataset verification

- **Always:** MVTec AD benchmark, RAW-FABRID primary runs.
- **Minimum ZJU first:** Both methods' `zju_per_fabric_mini_*` runs.
- **Full ZJU:** The per-pattern, per-group, unified and held-out runs above if the team retains the cross-fabric study.
- **AITEX:** Included only after an explicit team decision.
- **TSFabrics and Lusitano:** Not assigned.

The current repository's accepted decisions and `scripts/summarize.py --check` still expect full ZJU and AITEX coverage. If scope is reduced, the agent must **request a recorded team decision** and make omissions explicit in reporting. Do not quietly alter or bypass the existing full-study acceptance criteria.

## 3. Mandatory shared checkpoints

### Checkpoint 0 — prepare

1. Verify the real datasets and adapters once; provide identical verified datasets to the machines that need them.
2. Freeze the exact Git commit and record dataset archive/checksum, manifest/split hashes, package versions, and fixed configs.
3. Set up all three environments. On the RTX 5060 Ti, verify CUDA and `sm_120`; Kaggle T4 does **not** use `sm_120`.
4. Run the existing tests. Confirm storage and persistent export locations before long runs.
5. Create the distributed experiment assignment ledger described below.

### Checkpoint 1 — MVTec gate (A + B in parallel)

- A runs MVTec RD4AD.
- B runs MVTec PatchCore.
- C prepares the Kaggle environment and runs nontraining validation/dry-runs.
- Collect the five-category means for **both methods**. Keep the accepted criteria: PatchCore 97.9 / 96.6 / 90.7; RD4AD 98.5 / 96.7 / 92.0 (image AUROC / pixel AUROC / AUPRO, percent).
- **STOP if either model misses a target.** Record the failure; do not tune on test results. Proceed to main training only after the team's review and gate approval.

### Checkpoint 2 — parallel study

After the shared gate passes:

- A starts assigned RAW-FABRID RD4AD jobs.
- B starts PatchCore and the assigned ZJU per-pattern RD4AD jobs.
- C starts ZJU cross-fabric RD4AD jobs and RD4AD ablations.
- For the cross-fabric stage, run/check **ZJU-mini first** on both methods; do not use unverified adapters for full ZJU runs.
- Each job must write its own output files and status entry immediately after completion.

### Checkpoint 3 — integration

1. Merge all result JSON through reviewed pull requests.
2. Transfer model bundles separately through approved persistent storage / the team's Hugging Face model repository.
3. Run the original complete-study result-coverage check after every included experiment is present.
4. Generate consolidated figures and tables, then run CPU-only Streamlit checks on the intended demo laptop.

## 4. Required repository support for distributed execution

**Agent task:** Add a minimal distributed-execution layer. Preserve original study lists and scientific configs. Prefer selecting **existing complete model profiles**, not creating altered dataset subgroups.

### 4.1 Assignment ledger

Create a committed machine-readable assignment file, for example:

`configs/studies/distributed/assignments.yaml`

Each job must record:

- `job_id` (unique, stable)
- `owner` (`A`, `B`, or `C`)
- `config` (existing canonical config path)
- `profiles` (explicit profile names, or `all` for a genuinely single/whole config)
- `stage` (`gate`, `study`, or `ablation`)
- `expected_result_paths` (derived from existing `result_path`)
- `state` (`planned`, `running`, `passed`, `failed`, or `blocked`) in a **separate progress ledger**, not silently changed by code generation

Require **exactly one owner per expected result path** across the active plan. The scheduler must reject duplicate ownership and unknown profile names before running.

### 4.2 Profile-level sharding

Add the smallest practical CLI support to run only chosen existing profiles, for example:

```bash
python scripts/fit.py --config configs/raw_fabrid_held_out_rd4ad.yaml --profiles Rollo1A_held_out,Rollo2A_held_out
python scripts/evaluate.py --config configs/raw_fabrid_held_out_rd4ad.yaml --profiles Rollo1A_held_out,Rollo2A_held_out
```

These commands are **proposed new syntax**; they do not exist in the current repo. The implementation must:

1. Build the **normal full dataset rows and canonical profiles first**, then filter the *profile dictionary* by the requested exact names.
2. Not truncate the source training data for held-out, per-group or unified models.
3. Preserve existing filenames and result paths, seeds, thresholds, iteration counts, coreset ratios, tags, training/validation/test splits and calibration behavior.
4. Fail for absent/ambiguous profile names rather than silently running nothing.
5. Keep the old commands working unchanged when `--profiles` is omitted.
6. Never calculate the final five-texture MVTec benchmark pass result from a partial subset. Aggregate the five complete categories in a separate gate check.

**Critical:** Do not use `fabrics: [subset]` as a universal sharding method. For `per_group` and `unified`, that can change which data go into the trained model. Profile selection must happen *after* the unchanged profile is constructed.

### 4.3 Distributed runner

Provide a minimal command such as:

```bash
python scripts/distributed_study.py --plan configs/studies/distributed/assignments.yaml --owner B --dry-run
python scripts/distributed_study.py --plan configs/studies/distributed/assignments.yaml --owner B --stage study
```

The command syntax is a **requested interface**, not an existing command. Its implementation should:

- Validate all assigned configs, profiles, expected output paths and collisions before starting.
- Run fit then evaluation for the selected profiles; use the existing `fit.py` and `evaluate.py` implementations.
- Skip only **verified complete** jobs; a JSON file merely existing is insufficient evidence of success.
- Maintain a local status ledger, retain stdout/stderr logs, and support `--dry-run` and safe retries.
- Not mark a run complete unless the model bundle is available, calibration exists, result JSON parses correctly, and recorded config/seed/profile match the assignment.
- Avoid automatic experimental reruns when an existing result comes from a different source commit or dataset/split hash; flag it for review.
- Store approved result JSON at the **current canonical paths** so `scripts/summarize.py` can read them unchanged.
- Use distinct temporary filenames and atomic writes where practical; never overwrite someone else's result.

### 4.4 Tests to add

- Distributed plan has no duplicate canonical result paths.
- Every listed profile exists in its original config.
- Splitting one multi-profile config into shards produces the **same expected output-path set** as the original unsharded config.
- Held-out training retains all required non-held-out patterns/rolls.
- Per-group and unified runs are not accidentally converted into smaller training pools.
- CLI profile selection works for `fit.py` and `evaluate.py` on small synthetic fixtures.
- Existing no-selection behavior remains unchanged.
- Runner rejects bad/missing/inconsistent results rather than skipping them.
- Consolidation detects missing cells and duplicate jobs.

## 5. Running on three machines

### Preparation for every member

1. Checkout the **same pinned Git commit** containing the approved distributed plan.
2. Use the same verified dataset contents and deterministic split rules. Paths may differ locally, but the data must be equivalent.
3. Install the relevant PyTorch build; record GPU, CUDA, PyTorch and driver versions.
4. Run `pytest` and the runner's `--dry-run` before the first experiment.
5. Confirm the output destination is persistent. **Kaggle session disk is temporary**; export results and bundles immediately after every completed job/batch.

### Per-job sequence

1. Verify assigned job and GPU/session availability.
2. Verify no other member owns its result path.
3. Run fit; preserve the raw console output and measured GPU speed, RAM and VRAM.
4. Run evaluate on the **same saved bundle**.
5. Validate the JSON metrics, profile identity, checkpoint and calibration files.
6. Export the bundle and log to approved persistent storage, and commit only allowed results/documentation.
7. Update the shared ledger and `docs/PROGRESS_LOG.md` with job ID, machine, commit, config, dataset hash, elapsed time, peak memory, result path and status.
8. Mark the job complete only after its outputs can be retrieved on another machine.

**Kaggle constraints:** Use separate legitimately available Kaggle GPU sessions/accounts. Treat each T4 as an independent 16 GB device, not a pooled 32 GB device. Plan jobs around the current session and account quota limits. Do not upload or publish MVTec AD images in a public Kaggle dataset/notebook. Use private licensed inputs where permitted.

## 6. Balancing the workload after benchmark measurements

Keep the initial allocation in Section 2 until both GPU types have measured throughput. Then:

1. Record elapsed **minutes per completed PatchCore profile** and **RD4AD iterations per second** for each hardware environment.
2. Estimate the *remaining* duration of every unstarted profile on each eligible machine.
3. Sum estimated remaining wall-clock time for A, B and C, including Kaggle quota/session limits.
4. Reassign independent unstarted profiles to reduce the **largest** remaining completion time. Prefer moving a few RAW-FABRID held-out profiles from A if A is the bottleneck; or moving smaller ZJU RD4AD profiles if a Kaggle queue is long.
5. Change only the owner/profile allocation in the plan; do not change model parameters or trained-data membership.
6. Require a short team approval before changing assignments, and never reassign a running job without confirming whether it completed.

A single `unified` model remains a single job. It cannot be split into multiple shorter runs and combined into one equivalent model. The same applies to each individual held-out or per-group fit.

## 7. Combining results and models

### Git: results and documentation

- Each member develops and commits on a **separate branch**, e.g. `run/local-rd4ad`, `run/kaggle-patchcore`, `run/kaggle-rd4ad`.
- Each member adds only their assigned `results/**/*.json`, small permitted logs, and clearly attributed progress notes.
- Keep all result filenames in the existing structure:

```text
results/<dataset>/<regime>/<method_or_method_tag>/<profile>.json
```

- Merge through pull requests in a coordinated order. Resolve `docs/PROGRESS_LOG.md` conflicts without deleting either person's entries; keep newest entries first.
- Re-run result-coverage validation after each merge. Do not force-replace a result produced by another member.
- The integration owner generates `results/report/` once from the consolidated `results/` directory.

### Persistent model artifacts

- `artifacts/`, `data/`, trained `model.pt`, calibration and bundles **must not** be committed to GitHub.
- Transfer model bundles using the approved Hugging Face model repo or private durable storage, while respecting licensing and project access rules.
- Record a bundle manifest containing original config, dataset/split hash, profile, model method, seed, git commit, bundle checksum and location.
- Preserve canonical bundle folder names. Avoid adding machine identifiers to scientific method tags or production bundle paths.
- Do not share Hugging Face write tokens in code, GitHub, notebooks, logs or agent prompts.

### One final integration machine

After merging all result branches:

```bash
python scripts/summarize.py --check
python scripts/summarize.py
```

`--check` currently expects the **full** RAW-FABRID, ZJU-Leaper and AITEX result matrix for both methods. If an explicitly approved reduced scope is used, report missing study cells as intentional; do not claim full-study completion from an incomplete result set.

Complete the final model upload/registry update, run Streamlit with the approved bundles on a CPU-only laptop, verify prediction parity, and confirm the offline demo.

## 8. Deliverables the coding agent should produce

- [ ] Approved, documented three-person assignment file with **unique result-path ownership**.
- [ ] Method/profile-safe distributed execution CLI, with dry-run and validation.
- [ ] Profile-level fit/evaluate selection that preserves the canonical datasets and configuration semantics.
- [ ] Kaggle-compatible setup/run instructions for Members B and C.
- [ ] Local RTX 5060 Ti instructions for Member A.
- [ ] Safe export and merge procedure for model bundles and results.
- [ ] Tests verifying sharding equivalence, collision prevention and resumability at the **completed-job** level.
- [ ] Updated runbook, progress-log procedure, and any necessary recorded decisions.
- [ ] Original single-machine runbook commands remain functional.
- [ ] No experiments executed automatically while making these documentation/orchestration changes.

### Exact instruction for the coding agent

> Read `CLAUDE.md`, `docs/TRAINING_PC_RUNBOOK.md`, `docs/DECISIONS.md`, `docs/EVALUATION.md`, `scripts/fit.py`, `scripts/evaluate.py`, `scripts/study.py`, `scripts/summarize.py`, and all files in `configs/studies/`. Implement the distributed execution support described in this proposal with minimal changes, preserving the existing science and canonical output paths. First produce a concrete change plan and identify any decisions that require team approval. Add tests and documentation, then run CPU tests and dry-run validation. Do not start actual model training, change hyperparameters, weaken benchmark or coverage gates, alter dataset splits, or upload private datasets or credentials. Provide the exact commands and job ownership lists for all three machines, and stop for approval on any deviation from accepted project decisions.

---

**Reference paths:** [`TRAINING_PC_RUNBOOK.md`](https://github.com/Zain-Shykh/LoomGuard/blob/main/docs/TRAINING_PC_RUNBOOK.md) · [`DECISIONS.md`](https://github.com/Zain-Shykh/LoomGuard/blob/main/docs/DECISIONS.md) · [`scripts/study.py`](https://github.com/Zain-Shykh/LoomGuard/blob/main/scripts/study.py) · [`configs/studies/`](https://github.com/Zain-Shykh/LoomGuard/tree/main/configs/studies)
