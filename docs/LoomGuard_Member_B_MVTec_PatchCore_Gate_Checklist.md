# LoomGuard: Member B MVTec PatchCore Gate Checklist (prepared, not executed)

**Status:** prepared on 2026-11-22. **Nothing in sections 3 to 7 has been run.** Every real-data command is marked `DO NOT RUN UNTIL HANDOFF`.

**Sources read for this checklist:** `docs/DISTRIBUTED_TRAINING.md` (Checkpoints 0 and 1, "Members B and C setup", "Export, retry and rebalance"), `configs/mvtec_patchcore.yaml`, `docs/DATASETS.md` (MVTec section), `scripts/distributed_study.py --help`, and `configs/studies/distributed/assignments.yaml`. Argument names below are the ones the script defines. Where the repo does not say something, this file says so.

## 1. What this gate is

- Member B fits and evaluates **PatchCore** on the five MVTec AD textures: carpet, grid, leather, tile, wood. Config: `configs/mvtec_patchcore.yaml` (root `data/mvtec_ad`, `coreset_ratio` 0.10, `max_tiles` 600, seed 0, images resized to 256 as the one documented MVTec exception).
- Pass bar (mean over the five textures, image AUROC / pixel AUROC / AUPRO in percent): **97.9 / 96.6 / 90.7**.
- Member A runs the RD4AD half of the gate on the RTX 5060 Ti. The shared gate is approved only when both methods pass and their results and bundles are integrated.
- Do not change the settings above, the seed, or the budgets. A failed bar is logged and reviewed, not tuned away.

## 2. Preconditions (all must be true before section 3)

| # | Precondition | Owner | Status when written |
|---|---|---|---|
| 1 | The five MVTec texture folders are downloaded, extracted and verified (`scripts/verify_datasets.py` output read, not just the exit code) | Member C | pending |
| 2 | The team has agreed one source commit and everyone checks out exactly that commit | team | pending |
| 3 | `artifacts/distributed/shared/datasets.json` exists, created by `--pin-datasets` on the coordinator checkout | coordinator | pending |
| 4 | MVTec reaches Kaggle as a **private** input, and the team confirms that is allowed under CC BY-NC-SA 4.0 (non-commercial, no redistribution, no public notebook, dataset or Git copy) | team | pending, not decided |
| 5 | Member B's export storage for bundles is chosen and retrieval from a fresh session is proven (see section 8) | Member B | persistence proof recorded separately |

The repository does not name the private storage. `--bundle-location` takes "a private durable storage prefix", so the team must pick one. This file does not invent it.

## 3. Notebook setup (`DO NOT RUN UNTIL HANDOFF`)

Private notebook, Accelerator **GPU T4 x2** (the runner uses one device), Internet on for the clone and pip.

```bash
git clone https://github.com/Zain-Shykh/LoomGuard.git /kaggle/working/LoomGuard
cd /kaggle/working/LoomGuard
git checkout <PINNED_SOURCE_COMMIT>        # the commit the team agreed; fill in
git rev-parse HEAD && git status --short   # must match the pin and be clean
mkdir -p data
ln -s /kaggle/input/<PRIVATE_MVTEC_INPUT>/mvtec_ad data/mvtec_ad   # config root is data/mvtec_ad; fill in the input name
python -m pip install -e ".[dev,app]"
nvidia-smi
CUDA_VISIBLE_DEVICES=0 python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(), torch.cuda.get_arch_list(), torch.cuda.get_device_capability())"
mkdir -p artifacts/distributed/shared
cp /kaggle/input/<PRIVATE_SHARED_INPUT>/datasets.json artifacts/distributed/shared/   # the pin from the coordinator; fill in
```

- Keep `CUDA_VISIBLE_DEVICES=0` for every command below. Do not enable multi-GPU.
- The checkout must be at the pinned commit with a clean tree. If it is not, stop.
- The folder layout inside the private input must match what the pin hashed (`train/good`, `test/<defect>`, `ground_truth/<defect>` per texture). The `verify_datasets.py` output from Member C is the evidence. If the layout differs, stop and ask.

## 4. Readiness (`DO NOT RUN UNTIL HANDOFF`)

```bash
python scripts/distributed_study.py --validate                        # expect 33 jobs, 185 paths
python scripts/distributed_study.py --owner B --stage gate --dry-run  # expect 5 PatchCore profiles
python -m pytest -q                                                    # record the result as setup evidence
python scripts/distributed_study.py --owner B --prepare \
  --review "<B setup evidence: commit, GPU, pytest result, dataset check, link>" \
  --bundle-location "<PRIVATE_STORAGE_ROOT_FOR_B>"
```

- `--prepare` requires `--owner` and a non-empty `--bundle-location` (the script rejects it otherwise).
- `--review` must point to real evidence. Do not write a placeholder or an unverified statement.
- `--prepare` matches the real data and split digests against `datasets.json`, and records hardware and package versions. A mismatch means the data, source or split differs from the pin: stop and tell the team. Do not edit `datasets.json`.
- The full `pytest -q` is slow on a Kaggle CPU (the targeted three-file set took 98 s). The runbook says to run it, so budget for it.
- Do not run `--pin-datasets`, `--approve-gate` or `--approve-mini`. Those belong to the coordinator.

## 5. The gate run (`DO NOT RUN UNTIL HANDOFF`)

```bash
python scripts/distributed_study.py --owner B --stage gate
```

- Runs the five profiles in order: carpet, grid, leather, tile, wood. To run one, add `--job mvtec_patchcore --profile <texture>`.
- Keep one active runner per session.
- If an attempt fails, check GPU capability, memory, data paths and remaining quota, wait for the old child process to exit, then rerun the same profile. A failed training or evaluation attempt restarts from the beginning. A restored valid result and bundle skips training.
- Record, per texture: minutes for fit, minutes for evaluation and export, coreset duration, peak VRAM and RAM, and the quota used. These feed the schedule for the large PatchCore queues. None of these numbers exist yet.

## 6. Outputs to keep after every texture (not only at the end)

Canonical result for each texture (five in total):

```
results/mvtec/per_fabric/patchcore/<texture>.json
```

Per profile, from `docs/DISTRIBUTED_TRAINING.md` "Export, retry and rebalance":

- Its directory under `artifacts/<dataset>/<profile>/<method-tag>` with all four bundle files and `provenance.json`.
- `status.json` under `artifacts/distributed/profiles/`.
- The matching `fit.log`, `evaluate.log` and manifest CSV under `artifacts/distributed/attempts/`.
- The small shared metadata files (`datasets.json`, the ready/prepare record).

Confirm the exact bundle directory name from the first finished profile before relying on any path pattern. The `distributed` field inside each result JSON names its bundle location.

Copy these to the agreed private storage right after each texture, then confirm a fresh session can retrieve them before marking anything complete. Never put results into a public dataset. MVTec images must not appear in any output.

## 7. After the five textures

1. Check the five result files exist and read the loss traces, times and memory.
2. Send the export to the coordinator for `python scripts/distributed_study.py --approve-gate --review "..."`. Member B does not run the approval.
3. Report: per-texture and mean image AUROC / pixel AUROC / AUPRO against 97.9 / 96.6 / 90.7, and the timings from section 5.
4. If the bar is missed: stop, log the miss, and report. Do not tune on the test set.
5. Next for Member B once the gate is approved: the ZJU-mini queue (`--owner B --stage study --queue mini`), then the standard queues. Member C waits for B's ZJU-mini approval. Those need ZJU data and are out of scope here.

## 8. Kaggle session rules that affect this run

- Free GPU quota is limited (30 hours per week shown on the account). A session cap is **not confirmed** by anything the repo or Kaggle page showed us. Plan per profile and save after each one.
- Files in `/kaggle/working` live only as long as the session unless a version is saved or they are copied out. Use the persistence result recorded in the feasibility checklist section 10.
- Stop the session when idle to stop the quota clock.

## 9. Open questions for the team

1. Which private storage holds Member B's bundles (the value for `--bundle-location`)?
2. Is attaching MVTec as a private Kaggle input acceptable under its license, and who uploads it?
3. What is the pinned source commit?
4. Who runs `--pin-datasets`, and how is `datasets.json` handed to Kaggle?
