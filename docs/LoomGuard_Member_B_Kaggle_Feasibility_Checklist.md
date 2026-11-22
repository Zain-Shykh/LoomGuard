# LoomGuard — Member B Kaggle Feasibility Checklist (Before Dataset Handoff)

**Purpose:** Verify that Member B can run the assigned LoomGuard workloads on a free Kaggle GPU **without waiting for Member C's dataset verification**. This is a hardware, software, and execution-path check, **not** a real MVTec benchmark or permission to begin headline training.

**Repository:** https://github.com/Zain-Shykh/LoomGuard  
**Source of truth:** Current `main`, `CLAUDE.md`, `docs/DECISIONS.md` (D-60 and D-61), `docs/DISTRIBUTED_TRAINING.md`, and `configs/studies/distributed/assignments.yaml`.  
**Scope:** Member B only. Member A separately tests the RTX 5060 Ti. Member C continues downloading and verifying real datasets.

## 1. What Member B needs to verify now

- [ ] Kaggle notebook can run a working CUDA-enabled PyTorch + torchvision installation.
- [ ] The active GPU model, compute capability, GPU memory, driver, CUDA and PyTorch versions are recorded.
- [ ] The GPU runs PatchCore fitting and inference on **synthetic** images, with finite scores.
- [ ] If a T4 or another supported Tensor Core GPU is available (compute capability **7.0 or greater**), RD4AD runs in **FP16 + GradScaler** on synthetic images with finite loss and acceptable VRAM.
- [ ] The repository's targeted tests pass and Member B's distributed plan validates/dry-runs.
- [ ] Timing, logs, synthetic test results, and environment details can be exported **out of the Kaggle session** and retrieved later.
- [ ] Known blockers (GPU type, RAM, quota, timeouts, failures) are documented.

**Do not** download the real datasets again, fabricate manifests or `datasets.json`, run `--prepare`, run `--stage gate` without the verified dataset pin, approve gates, modify approved training settings, or start RAW-FABRID/ZJU training.

## 2. Set up a private Kaggle notebook

1. Create a **private** Kaggle notebook (not a public dataset/notebook).
2. Enable **GPU** under notebook settings and enable internet temporarily if cloning or installing dependencies requires it.
3. Use the currently available GPU; **record its actual name**. A dual-T4 allocation provides **two separate 16 GB GPUs**, not one 32 GB GPU. The existing training runner uses one GPU.
4. Clone the repository into `/kaggle/working`. Use the exact reviewed Git commit for any later coordinated training; for feasibility testing, record the checkout commit and report if it differs from the team's intended pin.

**Notebook cells** (run one at a time):

```python
!git clone https://github.com/Zain-Shykh/LoomGuard.git /kaggle/working/LoomGuard
%cd /kaggle/working/LoomGuard
!git rev-parse HEAD
!git status --short
!nvidia-smi
```

Check the preinstalled PyTorch/torchvision pair before reinstalling it:

```python
import torch, torchvision
print('torch:', torch.__version__, 'torch CUDA:', torch.version.cuda)
print('torchvision:', torchvision.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('Compute capability:', torch.cuda.get_device_capability(0))
    print('GPU properties:', torch.cuda.get_device_properties(0))
    print('Architectures:', torch.cuda.get_arch_list())
    print('CUDA smoke:', torch.ones(1, device='cuda').add_(1).item())
```

If the PyTorch import or CUDA smoke test fails, install a **matched PyTorch and torchvision build compatible with Kaggle's GPU/driver** using https://pytorch.org/get-started/locally/ and rerun this check. Do not blindly install the RTX 5060 Ti's CUDA 12.8 wheel instructions on Kaggle.

Install LoomGuard dependencies:

```python
!python -m pip install -e ".[dev,app]"
```

## 3. Validate the distributed assignment without datasets

Run from the repository root:

```python
!python scripts/distributed_study.py --validate
!python scripts/distributed_study.py --owner B --dry-run
!python scripts/distributed_study.py --owner B --stage gate --dry-run
!python scripts/distributed_study.py --owner B --stage study --queue mini --dry-run
```

**Expected:** The plan validator reports **33 jobs and 185 unique canonical result paths** (for the reviewed plan). The MVTec gate dry-run shows **five PatchCore textures**: `carpet`, `grid`, `leather`, `tile`, `wood`. No actual dataset is read and no training runs. AITEX remains excluded under D-61.

If counts or owners differ because `main` changed, **report the discrepancy** and consult the latest approved assignment ledger rather than editing it to force the old count.

## 4. Run targeted unit tests

```python
!python -m pytest -q tests/test_patchcore.py tests/test_rd4ad.py tests/test_distributed_study.py
```

The project's pytest defaults exclude slow tests. Report the actual pass/fail counts and any deselections. If dependencies or test collection fail, record the error and fix only environment/install issues within the approved workflow. Do not claim the complete test suite passed unless it was actually run.

## 5. Synthetic PatchCore GPU smoke test

This uses **the real PatchCore implementation** with a randomly initialized backbone and tiny synthetic data. It checks feasibility, not anomaly-detection accuracy. Paste into a Kaggle Python cell while the working directory is the repository root:

```python
import time
import torch
from loomguard.models.backbone import Backbone
from loomguard.models.patchcore import PatchCore

assert torch.cuda.is_available(), 'A CUDA GPU is required'
torch.manual_seed(0)
model = PatchCore(Backbone(weights=None), coreset_ratio=0.01, seed=0, device='cuda')
train_x = torch.randn(2, 3, 256, 256)   # Synthetic; no dataset downloads
start = time.perf_counter()
fit = model.fit([train_x])
score, heat = model.score(torch.randn(1, 3, 256, 256))
torch.cuda.synchronize()
print('PatchCore fit:', fit)
print('Output:', 'score', tuple(score.shape), 'heat', tuple(heat.shape))
print('Finite:', bool(torch.isfinite(score).all() and torch.isfinite(heat).all()))
print('Elapsed seconds:', round(time.perf_counter() - start, 2))
assert fit['bank_size'] > 0 and tuple(score.shape) == (1,) and tuple(heat.shape) == (1, 1, 256, 256)
assert torch.isfinite(score).all() and torch.isfinite(heat).all()
```

**Success:** No CUDA error or out-of-memory failure, coreset is produced, both score and heatmap are finite, and the outputs have the expected shapes. **Do not extrapolate full-study run time from this tiny fit.**

## 6. Synthetic RD4AD FP16 feasibility test

Member B's eventual queue includes ZJU-mini and other RD4AD work, so PatchCore-only testing is insufficient to establish Member B's hardware readiness.

1. Verify the GPU compute capability. For the approved RD4AD FP16 Tensor Core setup, use **7.0 or greater**; T4 is supported, P100 does **not** meet the current runbook's RD4AD requirement.
2. Run the repository's synthetic throughput probe in **FP16 mode only** to avoid the very slow emulated-BF16 path on T4:

```python
!python scripts/bench_rd4ad_precision.py --modes fp16 --warmup 20 --iters 100 --batch 16
```

The script prints GPU information and JSON metrics including iterations/second, final loss, a finite-loss flag and peak allocated VRAM. This probe uses **random teacher weights and synthetic images**; it is not a real benchmark accuracy result.

3. Exercise the production RD4AD `.fit()` path on a very short synthetic sequence:

```python
import torch
from loomguard.models.backbone import Backbone
from loomguard.models.rd4ad import RD4AD

assert torch.cuda.is_available() and torch.cuda.get_device_capability(0) >= (7, 0)
rd = RD4AD(Backbone(weights=None), device='cuda', seed=0)
tiles = torch.randn(2, 3, 256, 256)
summary = rd.fit((tiles for _ in range(4)), iterations=4, log_every=1)
print('RD4AD fit summary:', summary)
assert summary['iterations'] == 4
assert torch.isfinite(torch.tensor(summary['final_train_loss']))
```

If the latest source includes `skipped_steps`, record that number; if not, report which source commit was tested. A few scaler adjustments at startup may be acceptable; sustained overflow or non-finite loss is a blocker. **Do not change precision, batch size, iteration budgets or the paper-based configuration to make a test pass.**

## 7. Check Kaggle runtime, quota and export

- Record `nvidia-smi` output, GPU model, GPU memory, PyTorch/CUDA versions, source commit, test summary, PatchCore smoke result, RD4AD FP16 metrics, peak VRAM and any errors.
- Note the **actual remaining GPU quota** and session time displayed for the account. Do not assume another member's quota is available.
- Test saving a small, non-sensitive report under `/kaggle/working/loomguard-feasibility/`, then save the notebook version/download the output. Confirm the file can be retrieved after a restart or from outside the live notebook.
- Do **not** publish pretrained/licensed datasets or model bundles to a public Kaggle dataset. No real datasets are needed for this feasibility stage.

An example evidence file creation cell:

```python
from pathlib import Path
import json, torch, subprocess
out = Path('/kaggle/working/loomguard-feasibility')
out.mkdir(parents=True, exist_ok=True)
report = {
    'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'torch': str(torch.__version__),
    'cuda': str(torch.version.cuda),
    'gpu': torch.cuda.get_device_name(0),
    'compute_capability': list(torch.cuda.get_device_capability(0)),
    'note': 'Synthetic environment feasibility only; no real MVTec results',
}
(out / 'environment.json').write_text(json.dumps(report, indent=2))
print(out / 'environment.json')
```

Attach or copy logs/test output separately into the same directory. **Creating an output file does not itself prove persistence**; check that it can be downloaded or restored.

## 8. Report back to the team

Share one compact report containing:

| Item | Required evidence |
|---|---|
| Repository version | Commit SHA and any local modifications |
| Kaggle hardware | GPU model, per-GPU VRAM, compute capability, remaining quota/session limit |
| Torch environment | Python, torch, torchvision, CUDA, driver and CUDA smoke output |
| Assignments | `--validate` output and Member B gate dry-run showing five PatchCore profiles |
| Tests | Number passed/failed/skipped and any failure output |
| PatchCore feasibility | Coreset size, elapsed time, finite scores and heatmap shape |
| RD4AD FP16 feasibility | Iterations/sec, peak VRAM, finite loss, `.fit()` result, skipped updates if available |
| Persistence | Location of exported feasibility report and confirmation of retrieval |
| Blockers | GPU incompatibility, OOM, quota, package mismatch or unexpected changes |

### Decision after these checks

- **Ready for Member B's dataset handoff:** CUDA works, targeted tests and both synthetic model smoke tests pass, distributed dry-run is correct, and evidence is retrievable.
- **Not yet cleared for real MVTec:** That needs Member C's verified MVTec files, the same approved source commit on both machines, the shared `artifacts/distributed/shared/datasets.json` pin, and each member's real `--prepare` checkpoint.
- **Not yet cleared for main studies:** Both MVTec methods must finish and pass the shared gate. Full ZJU still requires its separate decision/checkpoint.
- **Member C's Kaggle RD4AD environment needs its own checks** before declaring all three execution machines ready.

**Stop condition for this assignment:** Provide the results and any blockers. Do not start real dataset fitting or modify/commit/push the repository as part of this feasibility task.

---

## 9. Results of the feasibility run (2026-11-22)

**Verdict:** Kaggle Free can run Member B's PatchCore and RD4AD (FP16) workloads on synthetic data. One checklist item, export persistence, was not verified (see 9.6). No real dataset was used, no MVTec or other training was started, and nothing was committed or pushed.

Run in a private Kaggle notebook (`notebook7a70a2f5d5`) by a coding agent following sections 2 to 7. The checklist code was run as `exec(...)` one-liners with the same behaviour.

### 9.1 Environment

| Item | Value |
|---|---|
| Source commit | `826b962e624dd2dc427da555515781ad1b109c8e` (current `main`, includes the `skipped_steps` counter); `git status --short` was empty |
| GPU | 2x Tesla T4, 14.56 GiB each, compute capability 7.5 (two separate GPUs; the runner uses one) |
| Driver / CUDA | 580.178.04 / CUDA 13.0 |
| Python | 3.13 |
| torch / torchvision | 2.11.0+cu130 / 0.26.0+cu130 (preinstalled, no reinstall needed) |
| Arch list | sm_75, sm_80, sm_86, sm_90, sm_100, sm_120 |
| CUDA smoke | `ones(1).add_(1)` returned 2.0 |
| GPU quota at start | 29 of 30 hours remaining |

### 9.2 Assignment validation (no datasets)

- `--validate`: 33 jobs and 185 unique canonical result paths, as expected.
- `--owner B --stage gate --dry-run`: five PatchCore profiles (carpet, grid, leather, tile, wood), 0 conditional.
- `--owner B --dry-run`: 138 profiles, 14 conditional. The 14 are the inactive AITEX profiles (D-61).
- `--owner B --stage study --queue mini --dry-run`: 10 profiles, 0 conditional.

### 9.3 Tests

`pytest -q tests/test_patchcore.py tests/test_rd4ad.py tests/test_distributed_study.py`: **41 passed, 1 deselected (slow), 98 s.** Only these targeted tests were run, not the full suite.

### 9.4 PatchCore on GPU (real implementation, random backbone, 2 synthetic tiles)

- Bank size 20 from 2,048 patches. Coreset 0.22 s. Fit plus score 5.39 s in total.
- Score shape `[1]`, heatmap `[1, 1, 256, 256]`, all values finite. No CUDA error and no OOM.
- This is a tiny fit. It says nothing about run time on real data.

### 9.5 RD4AD FP16 on a T4 (random teacher, synthetic images)

| Test | Result |
|---|---|
| `bench_rd4ad_precision.py --modes fp16 --warmup 20 --iters 100 --batch 16` | 3.48 it/s, peak VRAM 3.15 GB, final loss 0.156, finite. `native_bf16` is false, as expected on a T4. |
| Production `RD4AD.fit()`, 4 iterations | 3.87 it/s, final loss 1.137 (finite), `skipped_steps` = 0, peak VRAM 1.82 GB |

This matches the earlier Kaggle T4 measurement of 3.47 it/s. Rough time at about 3.5 it/s, ignoring data loading (not measured):

| Iterations | Time |
|---|---|
| 10,000 | about 48 min |
| 20,000 | about 1.6 h |
| 120,000 | about 9.5 h |

The 120,000 figure is Member B's RD4AD budget in the plan. It would only fit in one Kaggle session if the session cap is above about 10 h. The cap (assumed about 12 h) is unconfirmed, so plan the work per profile with a result saved after each one.

### 9.6 Gaps and risks (read before assigning the next task)

1. **Export persistence is not verified.** `report.json` was written to `/kaggle/working/loomguard-feasibility/`, but no notebook version was saved and nothing was downloaded. The session was stopped afterwards, so the file is probably gone. The real runs must use Save Version (Commit), or copy results and bundles into a private Kaggle dataset, before the session ends. Test this once before relying on it for long runs.
2. **Remaining quota and the session-time cap were not recorded** after the run. Only the starting 29 of 30 hours is known.
3. **The export folder is incomplete.** `nvidia-smi`, test and dry-run logs are in the notebook output only, not in the export folder. The report file is `report.json`, not the `environment.json` of the example cell, and has no driver version.
4. **All speeds come from random data.** Real image loading, disk I/O and dataset-mount speed are untested. The real PatchCore fit and evaluation time per texture is unknown.
5. No dataset-mount paths were checked, because no dataset was attached.

### 9.7 Decision for the team

- **Ready for Member B's dataset handoff:** CUDA works, targeted tests and both synthetic smoke tests pass, the dry-runs are correct, and FP16 RD4AD runs on a T4 with finite loss and no skipped steps.
- **Before real runs:** close gap 1 (about 10 min of GPU quota: re-run, copy logs into the export folder, Save Version, download the output).
- **Still blocked on others, unchanged:** Member C's verified datasets and the shared `datasets.json` pin, the same source commit on all machines, and the MVTec gate approval (Member A's RD4AD result plus Member B's PatchCore result).
- **Not yet checked:** Member C's Kaggle environment needs its own check.

---

## 10. Export persistence proof and quota (2026-11-22)

Follow-up to the gap in 9.6, run as described in `docs/LoomGuard_Member_B_Next_Steps_Awaiting_Gate.md` section 2. No GPU was used (accelerator None).

### 10.1 Persistence test

| Item | Result |
|---|---|
| Notebook | `muhammadzain1x0/notebookc05cde1fff` (private), only one harmless cell, no training code |
| Saved version | Version 1 of 1, type "Save & Run All (Commit)", visibility Private, run time 22 s, "successful" |
| Output of the saved version | `loomguard-export-test/manifest.json` (240 B) and `loomguard-export-test/marker.txt` (42 B), listed under the notebook's Output tab |
| Created by | A fresh run of the saved version, not the live session (the saved manifest timestamp is `2026-11-22T12:54:12Z`, later than the interactive run's `12:53:27Z`) |
| `marker.txt` content | `LOOMGUARD_MEMBER_B_EXPORT_TEST_2026_11_22`, 42 bytes, which is the text plus one newline |
| `manifest.json` sha256 | `8214aa99dbbbf61a91acaad924842638f1e293a3b15ad647bc05766dbab200de` |
| Independent hash | `sha256sum` of the same marker text on another machine gives the same value |

**Result: PASS (with one deviation).** The files are retrievable from the saved version outside the live session, and the hash in the manifest matches an independent hash of the marker text.

**Deviation:** I did not download the two files to a disk and re-hash the downloaded copy, as step 4 and 5 of the next-steps file ask. I read both files in the version's Output viewer (name, size, content) and compared the manifest hash with a hash computed on another machine. A human can do the literal download check in about a minute: open the version's Output tab, download the two files, and run the Python snippet from step 5. The `kaggle kernels output <owner>/<notebook> -p <dest>` command shown on the Output page also works if the Kaggle CLI is set up.

**Not tested:** the optional stronger proof (a fresh session restoring the files), large files, and bundle-sized outputs. A real bundle directory has four bundle files plus `provenance.json`, so test one real-size export once during the gate run.

### 10.2 Quota and session limits

- The Accelerator dialog showed: "Availability is limited to 30 hours per week. You have 29 hours remaining." It shows whole hours, so the exact figure is unknown.
- **No maximum session duration is shown** by the account or the dialog. It stays **not confirmed**. Do not assume 12 hours.
- The CPU session used for this test was stopped afterwards. The notebook accelerator is None.

### 10.3 Still open

- Real-data checks (dataset mount, disk I/O, MVTec accuracy, full PatchCore timing) need the verified datasets and the pinned commit.
- Earlier logs printed only in the stopped notebook (`nvidia-smi` full text, per-command output) are not saved anywhere. Nothing was reconstructed. Only the figures already recorded in section 9 are reported.
- The prepared but not executed gate workflow is in `docs/LoomGuard_Member_B_MVTec_PatchCore_Gate_Checklist.md`.

---

## 11. Verdict for Member B (2026-11-22)

**Member B is feasible and ready to wait for the dataset handoff.** Kaggle Free can run the assigned PatchCore and RD4AD (FP16) work. The export path is proven, with one literal step left for a human.

| Question | Answer | Evidence |
|---|---|---|
| Does CUDA PyTorch work on Kaggle Free? | Yes | 2x Tesla T4, cc 7.5, torch 2.11.0+cu130, `sm_75` in the arch list (9.1) |
| Does the plan validate and dry-run for B? | Yes | 33 jobs, 185 paths; five PatchCore gate profiles (9.2) |
| Do the tests pass there? | Yes | 41 passed, 1 deselected (9.3) |
| Does PatchCore run on the GPU? | Yes | Finite score and heatmap, correct shapes, no OOM (9.4) |
| Does RD4AD FP16 run on a T4? | Yes | 3.48 it/s, 3.15 GB peak VRAM, finite loss, 0 skipped steps (9.5) |
| Can results be saved outside the live session? | Yes | Saved version, files listed in its Output, manifest hash matches an independent hash (10.1) |
| Is the quota enough to start? | Yes | 29 of 30 weekly hours shown (10.2) |

**Open items (none blocks waiting, all matter before real runs):**
1. The literal download-and-rehash of the two test files was not done (10.1). One minute of human work.
2. Session length cap is not confirmed (10.2). Run per profile and save after each one.
3. All speeds are from random data. Real PatchCore and RD4AD timing, disk I/O and dataset mount are unmeasured.
4. Bundle-sized export from Kaggle has not been tried.
5. The gate workflow needs four team decisions: bundle storage for `--bundle-location`, whether MVTec may be a private Kaggle input, the pinned source commit, and who runs `--pin-datasets` and hands over `datasets.json`. They are listed in `docs/LoomGuard_Member_B_MVTec_PatchCore_Gate_Checklist.md`.

**Not claimed:** no MVTec accuracy, no real training time, no gate result. This verdict covers feasibility only.

**Next for Member B:** hold GPU work until Member C finishes the verified dataset handoff and the team pins the source commit. Then run the five-texture PatchCore gate from the prepared checklist. The shared gate is approved only after Member A's RD4AD results and Member B's PatchCore results both pass and are integrated.
