# Zain · Member B

Zain owns the repository and led initial architecture, model, pipeline and app implementation. He prepared Kaggle, tested PatchCore feasibility, generated both original shared pins from Hamza's verified data, and ran and exported all five MVTec PatchCore gate profiles. Huzaima verified the original transfer and accepted all five result/bundle pairs. Their means are 99.1794% image AUROC, 97.2354% pixel AUROC and 90.8848% AUPRO@0.3. The combined gate still awaits Huzaima's remaining RD4AD profiles and scientific review. Reuse the accepted PatchCore outputs.

## START NOW

Use CPU sessions to confirm authorized dataset access, private storage and retrieval, and the handoff of the original full pin to Hamza. Validate the ledger and stage commands. The frozen source for all experiments is `67fb7f6ca2dec0365b63a7454ffb57b3138b6875`. Development main has the handoffs and published results. MVTec PatchCore `mvtec_patchcore` is complete for carpet, grid, leather, tile and wood. The original gate pin starts `702a0913` and is archived with those exports. Remaining allocation is 95 PatchCore and 24 RD4AD profiles (120,000 iterations). AITEX is inactive.

In a private Kaggle notebook, run the following in a single Bash cell per session, or run it as a setup script sourced by each later cell. Set `RAW_FABRID_INPUT`, `ZJU_INPUT` and `FULL_PIN` to the exact authorized input paths. ZJU may be linked when its route and account access are confirmed. Use Kaggle repository access if prompted. Keep pin, model and checkpoint inputs private.

```bash
set -euo pipefail
src=67fb7f6ca2dec0365b63a7454ffb57b3138b6875
work=/kaggle/temp/loomguard-frozen
mkdir -p "$(dirname "$work")"
if [[ ! -d "$work/.git" ]]; then
  git clone https://github.com/Zain-Shykh/LoomGuard.git "$work"
  git -C "$work" switch --detach "$src"
fi
cd "$work"
test "$(git rev-parse HEAD)" = "$src"
test -z "$(git status --porcelain -- configs loomguard scripts pyproject.toml)"
python -m pip install -e '.[dev,app]'
shared=artifacts/distributed/full-shared
mkdir -p data "$shared"
# Mount only the datasets permitted for this session.
if [[ -n "${RAW_FABRID_INPUT:-}" && ! -e data/RAW_FABRID ]]; then
  ln -sT -- "$RAW_FABRID_INPUT" data/RAW_FABRID
fi
if [[ -n "${ZJU_INPUT:-}" && ! -e data/ZJU-Leaper ]]; then
  ln -sT -- "$ZJU_INPUT" data/ZJU-Leaper
fi
printf '%s  %s\n' 144a80d3f518a46b0183df1c3b5707627e58e128c89413a07907ca247374d40b "$FULL_PIN" | sha256sum --check --strict
if [[ ! -e "$shared/datasets.json" ]]; then cp -- "$FULL_PIN" "$shared/datasets.json"; fi
cmp -- "$FULL_PIN" "$shared/datasets.json"
python scripts/distributed_study.py --validate
python scripts/distributed_study.py --owner B --dry-run --queue standard
python scripts/distributed_study.py --owner B --dry-run --queue mini
python scripts/distributed_study.py --owner B --dry-run --queue zju_full
```

For actual training after approvals, use the T4, `CUDA_VISIBLE_DEVICES=0`, matching CUDA-enabled PyTorch/torchvision and approved ImageNet weights. Check `nvidia-smi`, `df -h /kaggle/working /kaggle/temp` and `free -h`. Check the live account quota and session time in the Kaggle editor before enabling the accelerator. The previous RAW-FABRID `--prepare` took about 19 minutes and the ZJU tree hash took 4.9 hours on CPU. Recheck elapsed preparation time with the real GPU mount and group complete profiles within the remaining time. Save verified versions before the session expires. Keep comparisons on the same approved machine, grouping baselines and ablations together when the checkpoint schedule allows it. Escalate any machine change that affects those comparisons. [Kaggle notebook settings](https://www.kaggle.com/docs/notebooks)

Before live training, obtain the approved Wide ResNet-50-2 `IMAGENET1K_V1` weight file from the official PyTorch source or the team's verified cache. Put it under `$(python -c 'import torch; print(torch.hub.get_dir())')/checkpoints/wide_resnet50_2-95faca4d.pth` and confirm SHA-256 `95faca4d11227dddf8633dbb5ff6c8a9003c1aa5b8945c73834b8007b10950b8`. Keep access to the cache across saved sessions if the session plan allows it.

## START AFTER COMBINED MVTec GATE

Use `gate.json` promoted by Huzaima against the full-study pin. Set `GATE_FILE` to that received file and set `REVIEW` to the saved environment and dataset evidence reference. Keep the same variables and working directory throughout the session. Use `RAW_FABRID_INPUT` in standard sessions and `ZJU_INPUT` in authorized ZJU sessions. Example uses separate private export roots per saved batch.

```bash
shared=artifacts/distributed/full-shared
cmp -- "$FULL_PIN" "$shared/datasets.json"
cp -- "$GATE_FILE" "$shared/gate.json"
export CUDA_VISIBLE_DEVICES=0
nvidia-smi
python -c 'import torch; assert torch.cuda.is_available() and torch.cuda.get_device_capability() >= (7,0); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())'
study=(python scripts/distributed_study.py --shared "$shared" --owner B)
export_root=/kaggle/working/loomguard-b-export
"${study[@]}" --prepare --queue standard --review "$REVIEW" --bundle-location "$export_root"
"${study[@]}" --stage study --queue standard
"${study[@]}" --stage ablation --queue standard
```

RAW-FABRID standard uses the pinned high-resolution representation. Study jobs are `raw_fabrid_patchcore` (unified), `raw_fabrid_per_fabric_patchcore` (Rollo1A through Rollo7A) and `raw_fabrid_held_out_patchcore` (Rollo1A_held_out through Rollo7A_held_out), 15 profiles total. Ablations are `raw_fabrid_per_fabric_patchcore_cr1`, `raw_fabrid_per_fabric_patchcore_cr25`, `raw_fabrid_per_fabric_patchcore_n50`, `raw_fabrid_per_fabric_patchcore_n200` and `raw_fabrid_per_fabric_patchcore_ov32`, seven rolls each, 35 profiles total. Run and export a whole profile at a time when the study command is too large. Append `--job JOB --profile PROFILE` to its stage command.

When ZJU account and route access are confirmed, attach `data/ZJU-Leaper` as above and run the five patterns (1, 4, 8, 13, 16) for each mini headline job. Use the same GPU environment as the corresponding future ablations. Export both five-profile mini sets for owner review before larger ZJU work.

```bash
"${study[@]}" --prepare --queue mini --review "$REVIEW" --bundle-location "$export_root"
"${study[@]}" --stage study --queue mini --job zju_per_fabric_mini_patchcore
"${study[@]}" --stage study --queue mini --job zju_per_fabric_mini_rd4ad
```

## START AFTER ZJU-MINI APPROVAL

Obtain the full-study `mini.json` plus the owner go-ahead after review of the roll check and first RAW-FABRID results. Set `MINI_FILE` to that received file. Use per-profile selection for short sessions and repeat queue readiness in each new environment.

```bash
cp -- "$MINI_FILE" "$shared/mini.json"
"${study[@]}" --prepare --queue zju_full --review "$REVIEW" --bundle-location "$export_root"
"${study[@]}" --stage study --queue zju_full
"${study[@]}" --prepare --queue mini --review "$REVIEW" --bundle-location "$export_root"
"${study[@]}" --stage ablation --queue mini
```

| Jobs | Profiles |
|---|---|
| `zju_per_fabric_rest_patchcore` and `zju_per_fabric_rest_rd4ad` | 14 each, patterns 2, 3, 5, 6, 7, 9, 10, 11, 12, 14, 15, 17, 18, 19 |
| `zju_per_group_patchcore` and `zju_unified_patchcore` | group1 through group5, plus unified |
| `zju_held_out_patchcore` | pattern3_held_out, pattern7_held_out, pattern12_held_out, pattern15_held_out, pattern19_held_out |
| `zju_mini_patchcore_cr1`, `zju_mini_patchcore_cr25`, `zju_res256_patchcore` and `zju_res256_rd4ad` | 5 each, patterns 1, 4, 8, 13, 16 |

Plan 6.3–6.6 hours of RD4AD fit at the measured synthetic T4 rate of 5.1–5.3 iterations/s. The five MVTec PatchCore runs took 62–75 seconds each through fit and evaluation. Budget approximately 3–8 hours for the 95 remaining PatchCore profiles as a planning estimate, and reserve separate time for repeated readiness, large evaluations and transfers. ZJU I/O can dominate and can require several sessions. Allow roughly 160 MiB of weights per RD4AD profile, variable PatchCore bundle sizes and headroom for attempts and staging.

For each completed profile, set `JOB` and `PROFILE` to its ledger values. Use the export command below before the next profile when practical. Preserve each downloaded checksum manifest and saved private notebook version reference. Verify the copy on a clean consumer session before marking the profile delivered.

```bash
mapfile -t paths < <(python - "$JOB" "$PROFILE" <<'PY'
from pathlib import Path
import sys
from loomguard.distributed import load_plan
job, profile = sys.argv[1:]
run, = (r for j in load_plan(Path("configs/studies/distributed/assignments.yaml"))
          if j.job_id == job and j.owner == "B" for r in j.runs() if r.profile == profile)
print(run.result)
print(run.bundle)
PY
)
result="${paths[0]}"
bundle="${paths[1]}"
status="artifacts/distributed/profiles/${result#results/}"
status="${status%.json}/status.json"
attempt=$(python -c 'import json,sys; s=json.load(open(sys.argv[1])); assert s["state"]=="passed"; print(s["attempt"])' "$status")
[[ "$attempt" == attempt-* && "$attempt" != */* ]]
logs="artifacts/distributed/attempts/$attempt"
mkdir -p "$export_root"
cp -a --parents "$result" "$bundle" "$status" "$logs/fit.log" "$logs/evaluate.log" "$logs/artifacts/manifests" "$export_root/"
(cd "$export_root" && find results artifacts -type f -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256 &&
 sha256sum --check --strict MANIFEST.sha256)
```

Save the private version, download the export, then run `sha256sum --check --strict MANIFEST.sha256` there and send it to Huzaima privately. Keep the original run status and attempt logs. Restore accepted canonical outputs before restarting a session. The runner skips verified outputs and recovers staged publication when available. Interrupted fitting or evaluation restarts from a fresh attempt. Resolve conflicts before re-running and keep the failed logs. Completion requires matching result and bundle hashes for every assigned profile, timings/quota, transfer receipts and PatchCore/ZJU analysis. Validate the app on the CPU workflow, then handle the approved deployment and report acceptance evidence using the [deployment runbook](../../deploy/README_space.md).

## DEFERRED OR OPTIONAL

AITEX accounts for 14 inactive profiles. The possible redistribution of the unstarted ZJU per-pattern RD4AD job awaits separate approval. D-61 keeps the current ledger binding.
