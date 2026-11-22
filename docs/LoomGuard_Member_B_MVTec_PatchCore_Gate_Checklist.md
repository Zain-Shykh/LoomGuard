# LoomGuard: Member B MVTec PatchCore Gate Checklist (prepared, not executed)

**Status:** prepared on 2026-11-22, reviewed against the code the same day (see section 10). **Nothing in sections 3 to 7 has been run.** Every real-data command is marked `DO NOT RUN UNTIL HANDOFF`.

**Read section 10 first.** The code review found that `--prepare` is not MVTec-only: it checks every dataset that Member B's whole job list uses. This decides whether B can start the PatchCore gate on MVTec alone, and the team has to settle it before handoff.

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
| 5 | Member B's export storage for bundles is chosen (section 9, proposal P1). Retrieval from a saved Kaggle version is proven for small files (feasibility checklist section 10 and 12) | team | proposal pending approval |
| 6 | The `--prepare` scope question in section 10 is decided | team | open |

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
- Layout the adapter requires (`loomguard/data/manifest.py`, `build_mvtec`): `data/mvtec_ad/<texture>/train/good/*`, `data/mvtec_ad/<texture>/test/<defect or good>/*`, and for each defective test image `data/mvtec_ad/<texture>/ground_truth/<defect>/<image stem>_mask.png`. A missing mask or an empty `train/good` stops the run.
- The pin hashes **every file under `data/mvtec_ad`** (links to subdirectories are followed). That folder must hold exactly the five texture folders that Member C verified and nothing else, byte for byte. Extra categories or stray files change the hash and `--prepare` will refuse.
- The checkout must be a real clone with history, not a shallow one: the script runs `git rev-parse --verify <pinned commit>`.

## 4. Readiness (`DO NOT RUN UNTIL HANDOFF`)

```bash
python scripts/distributed_study.py --validate                        # expect 33 jobs, 185 paths
python scripts/distributed_study.py --owner B --stage gate --dry-run  # expect 5 PatchCore profiles
python -m pytest -q                                                    # record the result as setup evidence
python scripts/distributed_study.py --owner B --prepare \
  --review "<B setup evidence: commit, GPU, pytest result, dataset check, link>" \
  --bundle-location "<PRIVATE_STORAGE_ROOT_FOR_B>"
```

- `--prepare` requires `--owner` and a non-empty `--bundle-location` (the script rejects it otherwise). It also requires `--review` and refuses if the checkout is not at the pinned commit.
- **Scope (see section 10):** `--prepare --owner B` hashes the datasets and rebuilds the split manifests for **all of Member B's jobs**, not only the MVTec gate job.
- **Staleness:** the readiness record `artifacts/distributed/ready-B.json` stores the machine name, GPU, driver and package versions. The gate command compares them again and stops with "checkpoint 0 readiness is incomplete or stale" if any differ. A new Kaggle session usually has a new machine name, so **run `--prepare` again in every new session** before the gate.
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

Bundle directory per texture, from `loomguard/runs.py` (`bundle_dir`): `artifacts/mvtec/<texture>/patchcore`. It must contain `config.yaml`, `calibration.json`, `metrics.json`, `model.pt` and `provenance.json`; the verifier rejects missing or empty files. The `distributed` field inside each result JSON records `<bundle-location>/artifacts/mvtec/<texture>/patchcore`.

Bundle size is unknown until the first texture finishes. My estimate (not measured): the PatchCore memory bank at 10 % coreset of at most 600 tiles with a 128-dimensional projection is tens to a few hundred MB per texture. Check the real size and re-check it against the Kaggle output limit.

Copy these to the agreed private storage right after each texture, then confirm a fresh session can retrieve them before marking anything complete. Never put results into a public dataset. MVTec images must not appear in any output.

## 7. After the five textures

1. Check the five result files exist and read the loss traces, times and memory.
2. Send the export to the coordinator for `python scripts/distributed_study.py --approve-gate --review "..."`. Member B does not run the approval.
3. Report: per-texture and mean image AUROC / pixel AUROC / AUPRO against 97.9 / 96.6 / 90.7, and the timings from section 5.
4. If the bar is missed: stop, log the miss, and report. Do not tune on the test set.
5. Next for Member B once the gate is approved: the ZJU-mini queue (`--owner B --stage study --queue mini`), then the standard queues. Member C waits for B's ZJU-mini approval. Those need ZJU data and are out of scope here.

## 8. Kaggle session rules that affect this run

- Free GPU quota is limited (30 hours per week shown on the account). A session cap is **not confirmed** by anything the repo or Kaggle page showed us. Plan per profile and save after each one.
- Files in `/kaggle/working` live only as long as the session unless a version is saved or they are copied out. The persistence proof is in the feasibility checklist sections 10 and 12. It covers small files from a **Save & Run All (Commit)** version. It does not cover large files, interactive sessions, or runs that fail or hit a time cap.
- A Commit version re-runs the notebook from the top in a fresh session. That is why proposal P1 below runs the whole gate inside the committed notebook.
- Stop the session when idle to stop the quota clock.

## 9. Open decisions for the team (with proposals)

**D1. Source commit.** The repo's latest commit on `main` when this was written is the one after `07963a7`. The team must name one commit that holds the reviewed plan, code and configs, and every machine checks it out. Any later change to code, configs or data needs a new pin. Proposal: pin the commit that includes this file set and nothing else changed afterwards. The team picks the SHA.

**D2. Dataset pin.** `--pin-datasets` runs once on a coordinator checkout that holds every dataset the active plan uses (see section 10 for why that means all of them). It writes `artifacts/distributed/shared/datasets.json`, which is then copied to each machine. Proposal: the coordinator is whoever finishes Member C's verification. They should run `python scripts/distributed_study.py --validate`, then `--pin-datasets --review "<link to the verifier output>"`, and send `datasets.json` through a private channel. It is a small file with only hashes and the commit. It contains no images and no secrets.

**D3. Private MVTec transfer.** MVTec AD is CC BY-NC-SA 4.0: non-commercial, no redistribution. Proposal P2: Member C (or whoever holds the licensed download) creates a **private** Kaggle Dataset containing only `mvtec_ad/<five textures>`, shared only with the two Kaggle accounts that need it, never public, never attached to a public notebook. In the notebook it appears under `/kaggle/input/<dataset>/mvtec_ad` and is linked to `data/mvtec_ad`. The team must confirm this private use is acceptable under the licence terms. This file does not decide that. A private upload that keeps the original folder names does not change the hash, so the pin still matches.

**D4. Where bundles and results go (`--bundle-location`).** The value is only a text prefix recorded in each result's provenance. Nothing uploads for you. It must contain no credentials. Proposal P1:
- Run the whole gate (setup, tests, `--prepare`, five textures, copy to an `export/` folder) inside **one notebook** and save it as **Save & Run All (Commit)**. The committed version's Output tab then holds the exports, as proven for small files. Download them from there to the coordinator. This needs no API token in the notebook.
- Prefix to record: `kaggle-version://muhammadzain1x0/<notebook-slug>` followed by the version number once known. Because the number is only known after the run, the team may prefer a fixed notebook name and write the version into the `--review` text.
- Risk: a Commit run that fails or hits a time cap may not keep its output files. This has **not** been tested. Run the notebook per texture or keep a second copy the same way if the first texture shows a long run time.
- Alternative P3, if the team wants durable copies that outlive a notebook: a private Kaggle Dataset or private Hugging Face repo that the notebook uploads to after each texture. That needs an API token in a Kaggle Secret and is a credential decision for the owner, not for this file. The repo's own rule is that bundles go to the Hugging Face Hub (`CLAUDE.md` rule 6), so a private repo there fits the project.

Decision needed from the team: P1, P3, or both.

## 10. Code review findings (2026-11-22)

Checked `scripts/distributed_study.py` and `loomguard/distributed.py` line by line for what the commands in sections 3 to 7 really do.

1. **`--prepare` covers all of Member B's jobs.** `prepare([j for j in active if j.owner == "B"], ...)` calls `check_inputs` and rebuilds a split manifest for every dataset and config of B's list. That includes MVTec, RAW-FABRID and ZJU, and AITEX only if it is approved. Separately, `load_pin` requires `datasets.json` to match the whole active plan: its datasets and manifests must equal the plan's. So with the plan as it stands, B cannot run `--prepare`, and so cannot run the gate, with only MVTec attached.
2. **Options for the team (decide before handoff):**
   - **O1:** Stage all three datasets (MVTec, RAW-FABRID, ZJU) as private Kaggle inputs in B's gate notebook, so `--prepare` passes. Cost: hashing all of them in every new session; large inputs; and RAW-FABRID and ZJU must already be verified.
   - **O2:** Agree a **reduced plan file** that holds only the MVTec gate jobs, pass it with `--plan` on every command, and create its own pin with `--pin-datasets --plan <file>`. `docs/DISTRIBUTED_TRAINING.md` allows a reviewed revised plan this way. Cost: the reduced plan and its pin are a second set of files, the later full plan needs a new pin, and the team must review and approve the reduced plan. I did not create it, because it changes the agreed workflow.
   - **O3:** Wait until all datasets are verified and pinned, then run as written. Slowest, but no change to the agreed workflow.
   - Recommendation: O1 if RAW-FABRID and ZJU will be verified within the next day or two, otherwise O2. This is a team decision.
3. **Readiness goes stale per session** (machine name, GPU, driver, packages). Repeat `--prepare` in each new Kaggle session. If `--prepare` re-hashes large datasets, that time counts against the quota.
4. **`hardware()` needs `nvidia-smi` and CUDA.** On Kaggle this works with the GPU accelerator on, as the feasibility run showed. `--prepare` fails on a CPU-only session, so do not try it in the CPU persistence notebook.
5. **The gate itself needs only MVTec** (`check_inputs([r.job for r in runs], pin)` hashes the datasets of the selected jobs). The extra datasets in point 1 matter only for `--prepare`.
6. **Gate stage has no prerequisite checkpoint.** The gate stage does not require an earlier approval. Later stages require the approved gate, and full ZJU also requires the approved ZJU-mini.
7. **Bundle path, result path and bundle files** are confirmed from `loomguard/runs.py` and `loomguard/distributed.py`. See section 6.
8. **Not changed:** no code, config, seed, budget or threshold was touched.
9. **Still unverified:** real data handling on Kaggle, the dataset hashing time, the real PatchCore run time and bundle size, and whether a failed Commit run keeps its output.

## 11. Ready-to-execute check

| Item | State |
|---|---|
| Commands match the script's arguments | Yes (`--owner`, `--prepare`, `--review`, `--bundle-location`, `--stage gate`, `--job`, `--profile`) |
| Paths match the code | Yes (`data/mvtec_ad`, `results/mvtec/per_fabric/patchcore/<texture>.json`, `artifacts/mvtec/<texture>/patchcore`) |
| Can it run today | No. It needs D1 to D4 and the `--prepare` scope decision (section 10, points 1 and 2) |
| Real-data commands run | None |
