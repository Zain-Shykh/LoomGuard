# Member A local RD4AD handoff

A approved the common experimental source `67fb7f6ca2dec0365b63a7454ffb57b3138b6875` subject to verification of the original pins. B has completed the five PatchCore gate profiles. The five RD4AD profiles and paired checkpoint are pending.

## Verify the original gate pin and prepare

Use the detached local training checkout and its activation helper. The helper activates the existing environment, resolves imports from that checkout and selects the local verified ImageNet cache.

```bash
source "$HOME/projects/LoomGuard/artifacts/private-training-archive/tools/activate-rd4ad.sh"
git rev-parse HEAD
git status --short
python scripts/distributed_study.py --owner A --stage gate --dry-run
```

B sends the original gate file privately. Place the received bytes at `$LOOMGUARD_ARCHIVE/incoming/member-b/pins/gate/datasets.json`. The reported file SHA-256 is `702a0913c595aecfbdae73ab89be2f72ba2ccdead54f53b20f526b42c43981a0`. The helper checks the file checksum, source, config digests, local tree, both canonical gate manifests and the five tracked PatchCore results. It installs the original bytes at `artifacts/distributed/gate-shared/datasets.json` under the training checkout and checks existing bytes before reuse. Stop on any mismatch.

```bash
python "$LOOMGUARD_ARCHIVE/tools/receive_pin.py" gate &&
  python scripts/distributed_study.py --shared artifacts/distributed/gate-shared \
    --owner A --prepare --queue gate --review "$LOOMGUARD_ARCHIVE/logs/frozen-environment.txt" \
    --bundle-location "$LOOMGUARD_ARCHIVE/verified/member-a"
```

The comparison receipt is historical evidence. The command above creates the fresh `artifacts/distributed/ready-A.json` for the original pin. Run it again after an environment or hardware change. The experimental SHA predates the Streamlit test-only fix. Development main carries the passing non-slow test evidence.

## Launch one profile at a time after readiness

Set `TEXTURE` to `carpet`, `grid`, `leather`, `tile` or `wood`. One profile uses 5,000 iterations, batch 16, and the approved D-60/D-62 precision settings. The measured local synthetic rate of 10.1 to 10.6 iterations/s gives about 39 to 41 minutes of fit compute over all five. Budget 60 to 90 minutes total for actual fitting, evaluation, model writes and export. Record real time on the first profile.

```bash
TEXTURE=carpet
LOG_STEM="$LOOMGUARD_ARCHIVE/logs/${TEXTURE}-$(date +%Y%m%dT%H%M%S)"
(
  set -o pipefail
  nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv -l 2 \
    > "$LOG_STEM.gpu.csv" &
  gpu_monitor=$!
  trap 'kill "$gpu_monitor" 2>/dev/null || true' EXIT
  free -h > "$LOG_STEM.memory.txt"
  /usr/bin/time -v python scripts/distributed_study.py --shared artifacts/distributed/gate-shared \
    --owner A --stage gate --job mvtec_rd4ad --profile "$TEXTURE" 2>&1 | tee "$LOG_STEM.run.log"
)
```

Review fit and evaluation logs for loss traces, skipped updates, throughput and resource usage. The gate uses the recorded RD4AD peak VRAM below 11 GB. GPU monitoring shows device-wide consumption and can include other workloads. Keep one runner at a time. After an interruption, wait for the old worker to stop and repeat the same command. The runner verifies and skips existing complete pairs and can finish interrupted publication from the staged attempt. Incomplete fit or evaluation starts a fresh attempt. Preserve conflicting outputs for review.

## Export after each successful profile

From the training checkout, copy the result, bundle and matching attempt logs into the private A archive with the canonical layout. Use the original attempt logs if a restored status lacks an attempt ID. Record the launch-time resource logs separately under `$LOOMGUARD_ARCHIVE/logs/`.

```bash
(
  set -euo pipefail
  dest="$LOOMGUARD_ARCHIVE/verified/member-a"
  status="artifacts/distributed/profiles/mvtec/per_fabric/rd4ad/$TEXTURE/status.json"
  attempt=$(python -c 'import json, sys; entry=json.load(open(sys.argv[1])); assert entry["state"]=="passed"; print(entry["attempt"])' "$status")
  [[ "$attempt" == attempt-* && "$attempt" != */* ]]
  logs="artifacts/distributed/attempts/$attempt"
  result="results/mvtec/per_fabric/rd4ad/$TEXTURE.json"
  bundle="artifacts/mvtec/$TEXTURE/rd4ad"
  files=("$result" "$bundle/"{config.yaml,calibration.json,metrics.json,model.pt,provenance.json}
    "$status" "$logs/fit.log" "$logs/evaluate.log" "$logs/artifacts/manifests/mvtec.csv")
  mkdir -p "$dest/checksums"
  cp -a --no-clobber --parents "${files[@]}" "$dest/"
  sum="$dest/checksums/$TEXTURE.sha256"
  if [[ ! -e "$sum" ]]; then (set -o noclobber; sha256sum "${files[@]}" > "$sum"); fi
  sha256sum --check --strict "$sum"
  (cd "$dest" && sha256sum --check --strict "$sum")
)
```

## Verify both methods and promote the checkpoint

Receive B's original export batches with their checksum manifests and saved-version references, then follow [the receipt workflow](LoomGuard_Member_B_MVTec_PatchCore_Gate_Checklist.md#26-november-handoff). Restore the five verified B JSON and bundle pairs into the frozen checkout at their canonical paths. Archive and verify both methods before approval. B's mean image AUROC, pixel AUROC and AUPRO@0.3 are 99.1794%, 97.2354% and 90.8848%. The combined gate also requires RD4AD 98.5%, 96.7% and 92.0% and peak VRAM below 11 GB.

Use B's full pin from `$LOOMGUARD_ARCHIVE/incoming/member-b/pins/full/datasets.json`. Its reported file SHA-256 is `144a80d3f518a46b0183df1c3b5707627e58e128c89413a07907ca247374d40b`. The full pin is needed for promotion and later queues. The local helper verifies it against all three active trees and the recorded 31 canonical manifests.

After both methods and the paired review have been approved, set `GATE_REVIEW_REF` to the actual review record and run:

```bash
python scripts/distributed_study.py --shared artifacts/distributed/gate-shared \
  --approve-gate --review "$GATE_REVIEW_REF"
python "$LOOMGUARD_ARCHIVE/tools/receive_pin.py" full &&
  python scripts/distributed_study.py --shared artifacts/distributed/full-shared \
    --approve-gate --review "$GATE_REVIEW_REF" &&
  cmp artifacts/distributed/gate-shared/gate.json artifacts/distributed/full-shared/gate.json
```

After promotion, distribute the verified full pin and its `gate.json` privately. Record full-pin readiness per assigned queue before running it. Full ZJU additionally requires the permitted execution route, ZJU-mini results, Part 1 roll review, first RAW-FABRID results and the recorded owner go-ahead. [Distributed checkpoints](DISTRIBUTED_TRAINING.md#checkpoint-2-run-the-independent-queues)
