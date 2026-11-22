# LoomGuard: Member B, RD4AD Scoped Teacher-Precision Regression on Kaggle T4

**Date:** 22 November 2026
**Task:** validate the corrected RD4AD implementation on a Kaggle Tesla T4 (no real datasets, no PatchCore repeat).
**Branch tested:** `fix/rd4ad-scoped-teacher-precision`
**Commit tested:** `f7330a75201388a5600e425237fbd49b933c2d3c` (`fix(rd4ad): scope teacher precision for stable FP16 training`)
**Note:** the task text called the branch `fix/rd4ad-teacher-precision`. The pushed branch is `fix/rd4ad-scoped-teacher-precision`. No SHA was sent in chat, so the tip of this branch was used. Member A should confirm it is the intended commit.

## 1. Environment (observed)

| Item | Value |
|---|---|
| Checkout | `git checkout f7330a75...`, `git rev-parse HEAD` printed that SHA, `git status --short` empty |
| GPU | Tesla T4 (two devices listed; tests use device 0), compute capability 7.5 |
| Driver / CUDA (nvidia-smi) | 580.178.04 / 13.0 |
| PyTorch | 2.11.0+cu130 (CUDA 13.0), cuDNN 92700 |
| Python / pytest | 3.13.16 / 8.4.2 |
| cuDNN flags at start | `cudnn.enabled=True`, `cudnn.allow_tf32=True` (PyTorch default) |
| Install | `pip install -e .[dev]` on the existing Kaggle image |

## 2. Targeted RD4AD tests: PASS

Command: `pytest -v -s tests/test_rd4ad.py --deselect tests/test_rd4ad.py::test_overfit_eight_tiles`

Result: **14 passed, 1 deselected in 72.88 s.** All passed:
`test_shapes`, `test_parameter_count`, `test_teacher_frozen_through_training`,
`test_teacher_and_student_precision[cpu-True]`, `[cuda-True]`, `[cuda-False]`,
`test_teacher_restores_tf32_after_error`, `test_cosine_loss_hand_values`, `test_anomaly_map_and_score`,
`test_fit_budget_and_logging`, `test_loss_falls_quickly`, `test_fp16_grad_scaler_skips_overflow_and_recovers`,
`test_save_load_float16[cpu]`, `[cuda]`.

## 3. 300-iteration GPU overfit test (`test_overfit_eight_tiles`): PASS, twice

Command (two fresh processes): `pytest -v -s tests/test_rd4ad.py -m slow`
Setup: eight synthetic 256 x 256 weave tiles, seed 0, 300 iterations, unmodified test body.

| Run | Final loss (last 25 it.) | Skipped optimizer steps | Iterations/s | Peak allocated (MiB) | Peak reserved (MiB) | Wall time |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.04159279 | 0 | 5.34 | 2415.6 | 2464.0 | 61.51 s |
| 2 | 0.04159279 | 0 | 5.08 | 2415.6 | 2464.0 | 64.56 s |

Both runs gave the same loss curve (12 logged points, 0.6057 down to 0.0416), asserting `final_train_loss < 0.05`.

## 4. Comparison with Member A's RTX 5060 Ti numbers (from `docs/PROGRESS_LOG.md` on the branch)

| Metric | RTX 5060 Ti (A) | Kaggle T4 (B) | Reading |
|---|---:|---:|---|
| Final loss | 0.04184032 | 0.04159279 | Both below 0.05; differ by 0.0025 absolute (0.6 %). Different GPU and CUDA build, so not expected to be bit-identical. |
| Skipped updates | 0 | 0 | Same. |
| Iterations/s | 10.14 / 10.57 | 5.34 / 5.08 | T4 about half the speed. Expected for the GPU class. |
| Peak allocated (MiB) | 2411.6 | 2415.6 | Same within 0.2 %. |
| Peak reserved (MiB) | 2486.0 | 2464.0 | Same within 1 %. |

Peak VRAM is far under the 11 GB RD4AD limit from the gate.

## 5. TF32 restoration

- The three CUDA precision tests and `test_teacher_restores_tf32_after_error` passed. They check that `torch.backends.cudnn.allow_tf32` is restored after the teacher forward, including after an error, with the flag set both True and False.
- Direct check in the notebook: set `allow_tf32 = True`, build `RD4AD`, call `_teacher_features` on a CUDA tensor, then read the flag. Result: **`True` (restored).**
- **Limit of this evidence:** a T4 (Turing, sm_75) has no TF32 hardware. The restoration logic ran, but the T4 cannot show the TF32 rounding that the fix removes for the FP32 teacher. That effect can only be seen on Ampere or newer (Member A's 5060 Ti). This run checks the code path and that nothing regresses on a non-TF32 GPU.

## 6. Verdict for the merge decision

**On the T4, the correction passes:** 14 of 14 targeted tests, two 300-iteration overfit runs under the 0.05 loss bound with zero skipped steps, VRAM unchanged from Member A's run, TF32 flag restored. No regression found on T4.

Not shown by this run, and not claimed:
- TF32 numerical behaviour (needs Ampere or newer; covered by Member A's tests).
- Real MVTec accuracy or RD4AD gate numbers. No dataset training was done.
- Full test suite; only `tests/test_rd4ad.py` was run, as asked.

## 7. Caveats and housekeeping

- **Logs:** the three logs were written to `/kaggle/working/rd4ad-regression/` (`targeted_tests.log`, `overfit_run1.log`, `overfit_run2.log`) in a temporary session. The notebook was **not saved as a version**, so the session files were not persisted. The numbers above were read from the notebook's cell output. If the raw logs must be kept, rerun with Save Version (about 5 GPU minutes).
- **Accelerator:** the notebook started on CPU. I switched it to GPU T4 x2 (the first attempt on CPU was cancelled after the `nvidia-smi: not found` message, before any test ran). The session was stopped after the run.
- **GPU use:** about 5 minutes of the weekly 30 h quota (29 h shown before the run). Exact remaining quota was not re-read after the run.
- The old export-test cell is still in the notebook above the new cell. It does not run at startup.
