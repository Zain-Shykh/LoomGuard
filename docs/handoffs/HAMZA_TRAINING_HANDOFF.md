# Hamza · Member C

Hamza acquired, prepared and originally verified the datasets. He produced the private archive, dataset manifests and reports, compared the RAW-FABRID patch and high-resolution representations, verified ZJU-Leaper Setting 1 and investigated the AITEX mask exclusions. Huzaima independently reproduced the relevant local evidence and verified Zain's original pins. All five MVTec PatchCore artifacts are accepted, and the combined gate still awaits the remaining RD4AD work and review. Hamza owns the assigned full-ZJU RD4AD profiles, dataset support, cross-fabric evaluation and reporting.

## START NOW

Confirm authorized access to the exact ZJU tree for your Kaggle account or other approved machine. The mounted archive includes CC BY-NC-SA 4.0 text and its README restricts the dataset to academic/research use and asks that the source and paper be cited. Zain verified the technical private Kaggle upload against the full pin. Confirm that your account can attach it and that the team has recorded the applicable terms for this execution route. Keep the study inputs and derived artifacts in the agreed private locations. [ZJU project and citation](https://nico-zck.github.io/zju-leaper/)

Send Huzaima the existing verifier and AITEX adapter diff, its base commit, authorship and test evidence if available. The team can review it on development main. AITEX is inactive under D-61.

Run the following as one Bash setup cell or source it in later cells. Use CPU while checking access. Set `ZJU_INPUT` to the exact authorized dataset root and `FULL_PIN` to Zain's original full pin. The common experimental source is `67fb7f6ca2dec0365b63a7454ffb57b3138b6875` and the full pin hashes all three active datasets. Your queue needs only ZJU mounted. The MVTec gate pin remains with the coordinator.

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
if [[ ! -e data/ZJU-Leaper ]]; then ln -sT -- "$ZJU_INPUT" data/ZJU-Leaper; fi
printf '%s  %s\n' 144a80d3f518a46b0183df1c3b5707627e58e128c89413a07907ca247374d40b "$FULL_PIN" | sha256sum --check --strict
if [[ ! -e "$shared/datasets.json" ]]; then cp -- "$FULL_PIN" "$shared/datasets.json"; fi
cmp -- "$FULL_PIN" "$shared/datasets.json"
python scripts/distributed_study.py --validate
python scripts/distributed_study.py --owner C --dry-run --queue gate
python scripts/distributed_study.py --owner C --dry-run --queue zju_full
```

Before live training, obtain the approved Wide ResNet-50-2 `IMAGENET1K_V1` weight file from the official PyTorch source or the team's verified cache. Put it under `$(python -c 'import torch; print(torch.hub.get_dir())')/checkpoints/wide_resnet50_2-95faca4d.pth` and confirm SHA-256 `95faca4d11227dddf8633dbb5ff6c8a9003c1aa5b8945c73834b8007b10950b8`. Keep access to the cache across saved sessions if the session plan allows it.

## START AFTER COMBINED MVTec GATE

Receive `gate.json` for the full-study pin. Review ZJU-mini access, Setting 1 splits and artifact evidence with Zain. Zain runs the two mini jobs. Hamza's training queue requires the mini approval and the full-ZJU go-ahead.

## START AFTER ZJU-MINI APPROVAL

Receive `mini.json` for the same full pin after the owner approves the two mini benchmarks and reviews the roll check and first RAW-FABRID results. Set `GATE_FILE` and `MINI_FILE` to the received files and `REVIEW` to your environment and dataset evidence reference. Use a private Kaggle T4 session, `CUDA_VISIBLE_DEVICES=0`, a matching CUDA PyTorch/torchvision pair and approved ImageNet weights. Check the live GPU quota and session time in the editor. Use the same approved protocol and resolve any machine change with the team. [Kaggle notebook settings](https://www.kaggle.com/docs/notebooks)

```bash
shared=artifacts/distributed/full-shared
cmp -- "$FULL_PIN" "$shared/datasets.json"
cp -- "$GATE_FILE" "$shared/gate.json"
cp -- "$MINI_FILE" "$shared/mini.json"
export CUDA_VISIBLE_DEVICES=0
nvidia-smi
df -h /kaggle/working /kaggle/temp
free -h
python -c 'import torch; assert torch.cuda.is_available() and torch.cuda.get_device_capability() >= (7,0); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())'
study=(python scripts/distributed_study.py --shared "$shared" --owner C)
export_root=/kaggle/working/loomguard-c-export
"${study[@]}" --prepare --queue zju_full --review "$REVIEW" --bundle-location "$export_root"
"${study[@]}" --stage study --queue zju_full --job zju_per_group_rd4ad
"${study[@]}" --stage study --queue zju_full --job zju_unified_rd4ad
"${study[@]}" --stage study --queue zju_full --job zju_held_out_rd4ad
```

| Job | Profiles | Iterations |
|---|---|---:|
| `zju_per_group_rd4ad` | group1 through group5 | 50,000 |
| `zju_unified_rd4ad` | unified | 20,000 |
| `zju_held_out_rd4ad` | pattern3_held_out, pattern7_held_out, pattern12_held_out, pattern15_held_out, pattern19_held_out | 50,000 |

Total is 11 profiles and 120,000 iterations. At the measured synthetic T4 rate of 5.1–5.3 iterations/s, fitting alone is about 6.3–6.6 hours. Real ZJU evaluation time remains to be measured. Budget separate time for evaluation, exports and readiness. Zain's private-mount tree hash took about 4.9 hours on CPU. Plan to split complete profiles across sessions if the live clock requires it. Allow about 160 MiB for weights per profile plus staging and evaluation scratch. Add `--profile PROFILE` to a job command to run and export exactly one.

For each completed profile set `JOB` and `PROFILE`, then export from the frozen checkout. Use one private export root per saved batch.

```bash
mapfile -t paths < <(python - "$JOB" "$PROFILE" <<'PY'
from pathlib import Path
import sys
from loomguard.distributed import load_plan
job, profile = sys.argv[1:]
run, = (r for j in load_plan(Path("configs/studies/distributed/assignments.yaml"))
          if j.job_id == job and j.owner == "C" for r in j.runs() if r.profile == profile)
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

Save the private version, download it, and run `sha256sum --check --strict MANIFEST.sha256` from the downloaded export. Send the intact version references, results, bundles, checksums and evidence privately to Huzaima. Restore verified outputs first in a fresh session. Repeat readiness when the environment changes. If interrupted during fitting or evaluation, preserve the attempt and retry the same whole profile. The runner can recover verified staged publication.

Completion requires all 11 verified pairs, timing and resource records, a review of ZJU cross-fabric and held-out outcomes against both methods, and results reported for the pinned split. Document any access or adapter issue and share the verification evidence.

## DEFERRED OR OPTIONAL

The proposed transfer of Zain's 14 unstarted per-pattern RD4AD profiles would add 70,000 iterations and requires a separate allocation amendment. Keep the 11 currently assigned profiles. AITEX source review, if the patch arrives, is a separate activity.
