# Three-person training runbook

The [proposal](THREE_PERSON_PARALLEL_TRAINING_PROPOSAL.md) and [assignment ledger](../configs/studies/distributed/assignments.yaml) describe the execution schedule, approved by the owner on 2026-11-21 (D-61). [Evaluation](EVALUATION.md), [accepted decisions](DECISIONS.md), and the [single-machine runbook](TRAINING_PC_RUNBOOK.md) govern the experiments and acceptance criteria.

| Member | Environment | Config-level assignment | Profile results |
|---|---|---|---:|
| A — Huzaima | RTX 5060 Ti, 16 GB VRAM, Ryzen 7 7700, 32 GB DDR5, Windows/WSL Ubuntu | MVTec RD4AD. RAW-FABRID RD4AD per-roll, unified seeds 0–2, and leave-one-roll-out. RD4AD false-alarm and grading evidence. RAW-FABRID RD4AD 1k-iteration and 32 px overlap ablations (moved from C by D-61, so each ablation sits on its baseline's machine). | 36 |
| B — Zain | Kaggle GPU 1 | MVTec PatchCore. RAW-FABRID PatchCore per-roll, unified and held-out. ZJU PatchCore mini, remaining patterns, groups, unified and held-out. All eight PatchCore ablation configs. ZJU RD4AD mini and remaining patterns. ZJU RD4AD resolution ablation (moved from C by D-61). | 124 active (+14 inactive AITEX) |
| C — Hamza | Kaggle GPU 2 | ZJU RD4AD groups, unified and held-out. | 11 |

Hamza prepared and verified the original dataset handoff. Huzaima independently checked and staged it on the training PC. Zain generated and distributed the original shared pins from the verified data.

Member B owns both AITEX methods, one model per fabric for seven fabrics per method. D-61 (2026-11-21) excludes AITEX: do not pass `--aitex-approval`, so these 14 profiles stay inactive. Keep using the original full-study check; its only expected failure is the missing AITEX cells, to be reported as intentional. ZJU: run ZJU-mini first; the full-ZJU queues start only after the owner records the go-ahead (D-61).

## Checkpoint 0. Prepare and verify

### Independent MVTec gate

Zain supplies the original gate pin he generated from Hamza's verified handoff. Huzaima checks it against the five mounted textures using the [local handoff](TEAM_TASKS_MEMBER_A.md). A and B each mount the verified tree at `data/mvtec_ad` and use the original `datasets.json` at `artifacts/distributed/gate-shared/`. Verify the release SHA, authorized data access and export destination first. Each review variable refers to the actual evidence for that operation.

```bash
# Each trainer sets OWNER to A (RD4AD) or B (PatchCore)
python scripts/distributed_study.py --shared artifacts/distributed/gate-shared --owner "$OWNER" --prepare --queue gate --review "$ENV_REVIEW_REF" --bundle-location "$BUNDLE_ROOT"
python scripts/distributed_study.py --shared artifacts/distributed/gate-shared --owner "$OWNER" --stage gate
```

A verifies all ten exported result/bundle pairs and uses the gate shared directory for checkpoint 1. When all active data is ready, A checks B's original full pin against B's verified dataset evidence on the same SHA. A re-verifies the ten outputs and compares the approval files before distributing the full pin and checkpoint. The full-study path also supports beginning with all active datasets. [Exact transition commands](LoomGuard_Team_Decisions_Needed.md#advance-to-the-full-study)

### Full-study pin

1. Agree on one source commit containing the reviewed plan, code, and unchanged configs. Work at the repository root. Start the three work branches at that commit, for example `git switch -c run/local-rd4ad "$SOURCE_COMMIT"`, `run/kaggle-patchcore`, and `run/kaggle-rd4ad`. Keep training checkouts pinned. Use a separate results checkout or worktree if committing results before the queue finishes.
2. Follow [Part 1 of the training-PC runbook](TRAINING_PC_RUNBOOK.md#part-1-datasets-and-verification-phase-1). Record source archive SHA-256 values, verifier output, the RAW-FABRID patch-count and metadata checks, and the ZJU adapter findings. Read the verifier output even when it exits successfully. Stage identical extracted trees at the configured `data/` paths (copies or directory links; links to subdirectories are followed when hashing). A needs MVTec and RAW-FABRID. B needs MVTec, RAW-FABRID and ZJU, plus AITEX if included. C's training assignments need only ZJU. Zain keeps the original full pin for all three active datasets. Hamza keeps the original verification evidence and Huzaima retains the independent local verification and receipt. Keep data immutable over the runs and leave unrelated files outside those roots. Use private licensed inputs where permitted and keep restricted data, including MVTec images, out of public notebooks, datasets and Git.
3. Preserve and distribute the existing full pin. For a future study with new verified inputs, create the new pin from the coordinator checkout. AITEX is excluded under D-61. The pin hashes the extracted file contents and each config's ordered split manifest. Changes to data or source need a new pin, review of prior outputs and new checkpoint approvals.

```bash
python scripts/distributed_study.py --validate
python scripts/distributed_study.py --pin-datasets --review "link or reference to verified dataset evidence"
```

### Member A setup

Use [Part 0 of the training-PC runbook](TRAINING_PC_RUNBOOK.md#part-0-environment) for the CUDA 12.8 wheels and the slow overfit check. Activate the virtual environment and install `.[dev,app]`. Check the RTX 5060 Ti and `sm_120` using the runbook command. On Windows, activate `.venv\Scripts\Activate.ps1` in PowerShell and use a local NTFS checkout. Keep the same unmodified batch, precision and iteration settings.

### Members B and C setup

Create separate private Kaggle notebooks/sessions and enable an accelerator in **Settings → Accelerator**. Use a writable session scratch checkout. Put the curated export under `/kaggle/working` for the saved version. Keep the datasets in private versioned inputs under `/kaggle/input` where their terms allow it, then link or copy their verified trees to the configured relative paths. Set `RAW_FABRID_INPUT` to the authorized mounted tree for this example.

```bash
mkdir -p data
ln -s "$RAW_FABRID_INPUT" data/RAW_FABRID
python -m pip install -e '.[dev,app]'
nvidia-smi
CUDA_VISIBLE_DEVICES=0 python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(), torch.cuda.get_arch_list(), torch.cuda.get_device_capability())"
```

Use the preinstalled matching PyTorch and torchvision if their CUDA smoke check works. If installing a different build, use a matched pair from the PyTorch selector that supports the session driver and the specific GPU. Keep `CUDA_VISIBLE_DEVICES=0` consistent for readiness and execution. A dual-T4 allocation exposes two separate devices, each with 16 GB VRAM. The existing training code uses a single device. T4 and P100 can serve PatchCore when CUDA works. RD4AD trains in fp16 with a GradScaler on every machine (D-60), so B and C only need tensor cores (compute capability 7.0 or newer), which the Kaggle T4 has. The P100 (6.0) does not qualify for RD4AD. If B changes accelerators, repeat readiness in the new session. If the assigned GPU is unavailable, record the blocked profiles and ask the team to approve a compatible machine or reassignment. [Kaggle notebooks](https://www.kaggle.com/docs/notebooks) · [PyTorch installation selector](https://pytorch.org/get-started/locally/)

Use the remaining session time and account quota displayed by Kaggle when scheduling. Budget training plus evaluation, model export and transfer verification for each profile. Reserve scratch space for the input tree, the current attempt and its output. Keep durable copies outside the live session. Use the agreed private storage or saved private output versions, and confirm that a fresh session can retrieve the files. [Kaggle GPU usage and quotas](https://www.kaggle.com/docs/efficient-gpu-usage)

### Readiness on each machine

For the B3 full pin, use `artifacts/distributed/full-shared/` for `datasets.json` and pass `--shared artifacts/distributed/full-shared` consistently on preparation, execution, approval and integration commands. The generic examples below assume the CLI default `artifacts/distributed/shared/`. Run `python -m pytest -q`, record the test and environment evidence, then use your member letter in the commands below. The review value points to that evidence. Use a durable bundle location prefix that contains no credentials. You can set `--shared <directory>` consistently to use a different checkpoint metadata directory.

```bash
python scripts/distributed_study.py --owner A --dry-run
python scripts/distributed_study.py --owner B --dry-run
python scripts/distributed_study.py --owner C --dry-run

python scripts/distributed_study.py --owner A --prepare --review "<A setup evidence>" --bundle-location "<private storage root for A>"
python scripts/distributed_study.py --owner B --prepare --review "<B setup evidence>" --bundle-location "<private storage root for B>"
python scripts/distributed_study.py --owner C --prepare --review "<C setup evidence>" --bundle-location "<private storage root for C>"
```

For full-pin queue preparation, append `--queue standard`, `--queue mini` or `--queue zju_full` to the member's preparation command. Each successful preparation replaces that member's local readiness record. The execution selection must fit its verified config set. Dry runs validate every assignment, including conditional paths, without reading datasets or writing files. Preparation matches actual data and split digests to the pin and records hardware and package versions. Keep one active runner per machine. When moving jobs between sessions, restore canonical results and bundles first and repeat `--prepare` for the current environment.

## Checkpoint 1. Run both MVTec benchmarks

The commands below use the default full pin. When using the gate-only pin, add `--shared artifacts/distributed/gate-shared` to each execution and approval command. B has completed its PatchCore profiles. A runs RD4AD after original-pin readiness. C prepares for its assigned ZJU work.

```bash
# A
python scripts/distributed_study.py --owner A --stage gate
# B
python scripts/distributed_study.py --owner B --stage gate
```

Export all five results and their corresponding bundles from each member. Place them at the canonical paths on the coordinator's checkout. Review the loss traces and measured time and memory before recording the gate approval.

```bash
python scripts/distributed_study.py --approve-gate --review "<recorded team benchmark approval>"
```

The approval verifies ten results and their bundles, averages all five categories per method, and enforces PatchCore 97.9 / 96.6 / 90.7 and RD4AD 98.5 / 96.7 / 92.0 (image AUROC / pixel AUROC / AUPRO in percent), plus the RD4AD peak-VRAM bar under 11 GB. If any target is missed, stop and record the results for team review. Distribute `gate.json` once both methods pass and the team approves. Main and ablation jobs require this file.

## Checkpoint 2. Run the independent queues

After the gate, A runs RAW-FABRID. B runs RAW-FABRID and, once the ZJU execution route is authorized, both ZJU-mini headline jobs with PatchCore first. B can run RAW-FABRID while ZJU-mini is being reviewed. C prepares for its ZJU queue.

```bash
# A
python scripts/distributed_study.py --owner A --stage study
python scripts/distributed_study.py --owner A --stage ablation

# B
python scripts/distributed_study.py --owner B --stage study --queue mini --job zju_per_fabric_mini_patchcore
python scripts/distributed_study.py --owner B --stage study --queue mini --job zju_per_fabric_mini_rd4ad
python scripts/distributed_study.py --owner B --stage study --queue standard
python scripts/distributed_study.py --owner B --stage ablation --queue standard

# C
```

The coordinator verifies the ten ZJU-mini bundles and results, reviews the adapter and split evidence and records the team's approval.

```bash
python scripts/distributed_study.py --approve-mini --review "<recorded ZJU-mini verification>"
```

Distribute `mini.json` before launching the larger ZJU work and the ZJU ablations. The `--review` text of `--approve-mini` is the owner's recorded go-ahead for full ZJU (D-61): state in it that the Part 1 roll check and the first RAW-FABRID results were reviewed.

```bash
# B
python scripts/distributed_study.py --owner B --queue zju_full
python scripts/distributed_study.py --owner B --stage ablation --queue mini

# C
python scripts/distributed_study.py --owner C --queue zju_full
```

For a timed session, select one complete profile by job ID and name.

```bash
python scripts/distributed_study.py --owner C --job zju_per_group_rd4ad --profile group1
python scripts/distributed_study.py --owner B --job zju_per_fabric_rest_rd4ad --profile pattern2
```

A records the three RAW-FABRID unified RD4AD seeds' mean and spread, seed 0 false alarms, and all seven rolls' predicted vs ground-truth grading. B records matching PatchCore headline evidence, including the coreset duration. Use [Part 3 of the original runbook](TRAINING_PC_RUNBOOK.md#part-3-raw-fabrid-headline-runs-phases-2-3-4) for the checks. Report deviations against the existing criteria and investigate without tuning on the test set.

### Export, retry and rebalance

- After each completed profile, copy its `results/<dataset>/<regime>/<method-tag>/<profile>.json` and its directory under `artifacts/<dataset>/<profile>/<method-tag>` to durable storage. Use the bundle location in the result's `distributed` field. Keep all four bundle files and `provenance.json`. Keep `status.json` under `artifacts/distributed/profiles/` and the corresponding `fit.log`, `evaluate.log` and manifest CSV under `artifacts/distributed/attempts/` in the private export, alongside the small shared metadata files. A profile without a status file is planned. Runner status `passed` means the local outputs were verified. Check retrieval before marking the shared progress entry complete.
- You can monitor the current attempt log with `tail -f` or PowerShell `Get-Content -Wait`. If an attempt fails, check GPU capability, memory, data paths and remaining quota. Wait for any prior child process to exit, then rerun the same profile. Its attempt has a fresh staging directory. An interrupted training or evaluation attempt restarts the profile from the beginning. If both stages completed and only publishing was interrupted, the runner can finish publishing the verified outputs. A restored valid bundle and result skip training. Existing incomplete or conflicting canonical outputs block the profile until the team reviews and reconciles them.
- Record RD4AD iterations per second, PatchCore minutes per profile, evaluation/export time, VRAM/RAM, and quota time. Estimate the remaining duration on each eligible machine. If balancing would reduce the longest queue, obtain team approval and move only unstarted complete profiles or jobs. Keep every completed or running profile's job ID, owner and result path. Create a new job ID for a moved shard. Share the reviewed revised plan separately from the pinned source checkout (for example under `artifacts/distributed/shared/`) and pass `--plan` on all invocations. Revalidate and repeat readiness using that plan. The original config and its full training populations stay fixed. Confirm the destination has the verified dataset and compatible hardware.

For a completed MVTec profile, set `METHOD` to `rd4ad` or `patchcore`, `TEXTURE` to its texture and `EXPORT_ROOT` to a new private directory. On Kaggle choose a directory under `/kaggle/working`, run inside the saved private version, then download that version's output. Check from the download root before handing it to A.

```bash
status="artifacts/distributed/profiles/mvtec/per_fabric/$METHOD/$TEXTURE/status.json"
attempt="$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["attempt"])' "$status")"
logs="artifacts/distributed/attempts/$attempt"
mkdir -p "$EXPORT_ROOT"
cp -a --parents "results/mvtec/per_fabric/$METHOD/$TEXTURE.json" \
  "artifacts/mvtec/$TEXTURE/$METHOD" "$status" "$logs/fit.log" "$logs/evaluate.log" \
  "$logs/artifacts/manifests/mvtec.csv" "$EXPORT_ROOT/"
(
  cd "$EXPORT_ROOT" || exit
  find results artifacts -type f -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256
  sha256sum --check --strict MANIFEST.sha256
)
```

Use the original attempt logs for a restored profile whose local status lacks an attempt ID. Keep any original saved-version checksum manifest. A verifies the download checksum and the runner verifies the bundle file hashes against the result provenance. The app uses `LOOMGUARD_BUNDLES` to read pre-staged bundles locally on the CPU laptop.

## Checkpoint 3. Integrate results and validate the CPU app

1. Make one isolated export root per member, each with their canonical `results/` and `artifacts/` hierarchies. Take incremental inventory and check transfer hashes as exports arrive. Once both checkpoint files exist, run the verifier before copying or merging results. It flags duplicate paths, checks output provenance and bundle hashes, compares approved result checksums and reports missing paths. A full integration check is expected to fail on a partial matrix. Set `EXPORT_A`, `EXPORT_B` and `EXPORT_C` to the isolated export roots.

```bash
python scripts/distributed_study.py --check-integration --incoming "$EXPORT_A" --incoming "$EXPORT_B" --incoming "$EXPORT_C"
```

2. On the separate results branches, stage only each member's assigned JSONs and relevant progress notes. Merge through reviewed PRs. When merging log entries, keep everyone’s entries with newest first. Keep datasets, model bundles, and private logs outside Git. Restore the verified bundles to their original paths from durable storage. Run the checks on the consolidated checkout using the same shared checkpoint metadata.

```bash
python scripts/distributed_study.py --check-integration
python scripts/summarize.py --check
python scripts/summarize.py
```

3. Generate figures and `docs/FINDINGS.md` using [Part 4 of the training-PC runbook](TRAINING_PC_RUNBOOK.md#part-4-the-cross-fabric-study-phase-5), including failure cases and all ablations. If the team has formally approved a reduced scope, include the exact omissions and the full-study check outcome in the report.
4. Validate on the intended CPU-only laptop with the approved bundles. Run `streamlit run app/streamlit_app.py` and complete [Part 5 of the training-PC runbook](TRAINING_PC_RUNBOOK.md#part-5-the-app-phase-6) and the [app spec](specs/phase-6-streamlit-app.md#8-training-pc-runbook). Record the under-10-second timing, prediction parity, roll grade, unknown-fabric behavior and offline demo. Bundle registry, Hub transfer and Space deployment follow the project's existing approval and release process.

## Progress record

Record the job ID, owner, profile, hardware and driver, source commit, resolved config and manifest hashes, dataset and archive hashes, split and fit seeds, wall time, peak RAM/VRAM, result path, durable bundle location and retrieval status. Use `docs/PROGRESS_LOG.md` for shared checkpoint decisions and completed batches. The runner's per-profile status and provenance files supply the machine records.
