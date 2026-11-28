# Huzaima Malik · Member A

Huzaima set up and tested the local CUDA/PyTorch environment, investigated RD4AD convergence and FP16 instability, implemented the scoped FP32 frozen teacher and gate-only preparation, integrated and verified the distributed runner, independently checked Hamza's dataset handoff, and verified Zain's artifact transfer. Huzaima owns local RD4AD, result integration, scientific checkpoint review and RD4AD reporting. The [team work plan](../LoomGuard_Team_Work_Plan.md) and [evaluation protocol](../EVALUATION.md) govern the assignments and metrics.

## START NOW

Gate readiness passed on the frozen checkout at `67fb7f6ca2dec0365b63a7454ffb57b3138b6875`. The RTX 5060 Ti exposes 16,311 MiB, compute capability 12.0 and CUDA 12.8 through PyTorch 2.11.0+cu128. The WSL environment has Python 3.12.3, about 14 GiB available of its 15 GiB visible RAM, 4 GiB swap and about 897 GiB free disk. The host has a Ryzen 7 7700 and 32 GB DDR5. The approved ImageNet backbone is in the private Torch cache and its SHA-256 was checked. Use the mounted `data/mvtec_ad` tree and original gate pin (SHA-256 `702a0913c595aecfbdae73ab89be2f72ba2ccdead54f53b20f526b42c43981a0`).

```bash
cd "$HOME/projects/LoomGuard"
source artifacts/private-training-archive/tools/activate-rd4ad.sh
python "$LOOMGUARD_ARCHIVE/tools/receive_pin.py" gate
python scripts/distributed_study.py --shared artifacts/distributed/gate-shared \
  --owner A --prepare --queue gate --review "$LOOMGUARD_ARCHIVE/logs/frozen-environment.txt" \
  --bundle-location "$LOOMGUARD_ARCHIVE/verified/member-a"
cd "$LOOMGUARD_PROJECT"
python3 scripts/run_local_mvtec_gate.py check
```

`mvtec_rd4ad` assigns carpet, grid, leather, tile and wood. Carpet and grid are complete and checksum-verified. Leather was interrupted during fitting and needs a fresh 5,000-iteration attempt. Tile and wood are unstarted. The completed pair used about 13.6–13.7 minutes each through evaluation, 6.42–6.43 iterations/s and 3.20 GB peak allocated VRAM. They had finite losses and zero skipped updates, with low-confidence calibration flagged for review. Budget about 42 minutes for the three remaining profiles at the observed rate, or 45–55 minutes with repeated input checks and export. Keep batch 16, 5,000 iterations, FP16 student with GradScaler and the scoped FP32 teacher. The five-texture RD4AD mean targets are 98.5% image AUROC, 96.7% pixel AUROC and 92.0% AUPRO@0.3. Peak allocated VRAM must stay below 11 GB.

The next command is for a separately authorized launch. It detaches in WSL and returns the worker PID. The wrapper reports the current texture, re-verifies and skips completed profiles, exports each completed result and bundle with its checksum and logs, and exits if an unexpected error occurs. The development checkout supplies the wrapper and the frozen checkout supplies the runner.

```bash
cd "$HOME/projects/LoomGuard"
python3 scripts/run_local_mvtec_gate.py check
python3 scripts/run_local_mvtec_gate.py start
```

From another WSL terminal use these commands for status, live output and safe stop. The stop command validates the recorded Linux process identity, then signals the worker. Wait for the status to show `INTERRUPTED` and exit status `130` before retrying.

```bash
cd "$HOME/projects/LoomGuard"
python3 scripts/run_local_mvtec_gate.py status
tail -F artifacts/private-training-archive/runtime/mvtec-gate/training.log
# In another terminal to stop:
python3 scripts/run_local_mvtec_gate.py stop
```

The same runtime directory holds `training.pid`, `status.txt`, `exit-code.txt` and `gpu.csv`. During fitting, the frozen runner writes attempt logs under `artifacts/distributed/attempts/` in the frozen checkout. The fit script prints loss-curve iteration lines when fitting returns. Use live GPU samples and the current texture to monitor progress during the fit. An interrupted fit or evaluation starts a fresh attempt. The runner can recover verified publication when both stages finished. Resolve provenance conflicts before retrying.

Gate completion requires five verified RD4AD pairs plus Zain's five accepted PatchCore pairs. Confirm each exported texture with `sha256sum --check --strict checksums/TEXTURE.sha256` from `verified/member-a`. Review metric means, individual losses, calibration, overflow count and VRAM. Once the team explicitly approves the combined scientific evidence, the coordinator can use the gate and full pin directories in turn and compare the generated approvals before distributing the full-pin copy. These commands require that review:

```bash
source "$HOME/projects/LoomGuard/artifacts/private-training-archive/tools/activate-rd4ad.sh"
python scripts/distributed_study.py --shared artifacts/distributed/gate-shared \
  --approve-gate --review "$GATE_REVIEW"
python scripts/distributed_study.py --shared artifacts/distributed/full-shared \
  --approve-gate --review "$GATE_REVIEW"
cmp artifacts/distributed/gate-shared/gate.json artifacts/distributed/full-shared/gate.json
```

## START AFTER COMBINED MVTec GATE

Use `data/RAW_FABRID` and the original full-study pin (SHA-256 `144a80d3f518a46b0183df1c3b5707627e58e128c89413a07907ca247374d40b`). The full pin is installed and verified. Complete the physical-scale and roll check in the [training-PC runbook](../TRAINING_PC_RUNBOOK.md#part-1-datasets-and-verification-phase-1). Keep RAW-FABRID baselines and ablations on this GPU and use the approved high-resolution adapter.

| Experiments | Profiles |
|---|---:|
| `raw_fabrid_rd4ad_s0`, unified seed 0 | 1 |
| `raw_fabrid_per_fabric_rd4ad`, Rollo1A through Rollo7A | 7 |
| `raw_fabrid_held_out_rd4ad`, Rollo1A_held_out through Rollo7A_held_out | 7 |
| `raw_fabrid_per_fabric_rd4ad_it1k` and `raw_fabrid_per_fabric_rd4ad_ov32`, Rollo1A through Rollo7A each | 14 |
| `raw_fabrid_rd4ad_s1` and `raw_fabrid_rd4ad_s2`, unified seeds 1 and 2 | 2 |

```bash
source "$HOME/projects/LoomGuard/artifacts/private-training-archive/tools/activate-rd4ad.sh"
python "$LOOMGUARD_ARCHIVE/tools/receive_pin.py" full
shared=artifacts/distributed/full-shared
study=(python scripts/distributed_study.py --shared "$shared" --owner A)
"${study[@]}" --prepare --queue standard --review "$LOOMGUARD_ARCHIVE/logs/frozen-environment.txt" \
  --bundle-location "$LOOMGUARD_ARCHIVE/verified/member-a"
"${study[@]}" --stage study --queue standard
"${study[@]}" --stage ablation --queue standard
```

For a short window, append `--job JOB --profile PROFILE` to the applicable stage command. There are 31 RAW-FABRID profiles and 277,000 remaining iterations after the gate. Extrapolating from MVTec gives about 12 fit hours, with roughly 12–19 hours including evaluation, writes and export as a planning range. Verify with the first real RAW-FABRID runs. Budget private storage for roughly 160 MiB of weights per RD4AD profile plus attempts, evaluation outputs and copies.

Export using [the shared per-profile procedure](../DISTRIBUTED_TRAINING.md#export-retry-and-rebalance). Pass the result path, bundle path, status, fit/evaluation logs and attempt manifest to the export step and verify `MANIFEST.sha256` from the private copy. Record timing and memory, unified three-seed mean and spread, ablation outcomes, false alarms per 1,000, seven-roll grading and held-out roll performance. Integrate result JSONs and retain bundles and operational evidence privately.

## START AFTER ZJU-MINI APPROVAL

Huzaima has no ZJU training allocation. Review both five-profile mini sets and the first RAW-FABRID/roll evidence with Zain and Hamza. After explicit owner approval, use the full pin and record the review.

```bash
python scripts/distributed_study.py --shared artifacts/distributed/full-shared \
  --approve-mini --review "$MINI_REVIEW"
```

Distribute the full-study `gate.json` and `mini.json` privately. Verify each member's incoming results and bundles during integration.

## DEFERRED OR OPTIONAL

AITEX is inactive under D-61. The shipped RAW-FABRID patch benchmark is a separate possible study requiring protocol and implementation review. The [assignment ledger](../../configs/studies/distributed/assignments.yaml) is binding. The local wrapper is limited to the MVTec gate.
