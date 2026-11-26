# Member C RD4AD handoff

C owns 11 full-ZJU RD4AD profiles under D-61. The assigned results cover five groups, one unified profile and five held-out patterns, for 120,000 configured iterations. B leads dataset preparation and verification. A coordinates checkpoints and archive receipt.

C has zero MVTec gate profiles. Obtain the verified full pin and checkpoint files from A after the paired MVTec gate and ZJU-mini approval. Confirm the permitted ZJU execution route, ingest B's verified data and prepare using the original full pin at the approved experimental SHA. The current assignments and same-machine ablation rule apply unless a separate workload amendment is approved.

```bash
python scripts/distributed_study.py --shared artifacts/distributed/full-shared \
  --owner C --prepare --queue zju_full --review "$ENV_REVIEW_REF" --bundle-location "$BUNDLE_ROOT"
python scripts/distributed_study.py --shared artifacts/distributed/full-shared --owner C --queue zju_full --dry-run
```

C exports completed profiles and their original checksums to A. Review the live T4 quota, session time and private artifact capacity before accepting any additional unstarted profiles. [Workload review](LoomGuard_Team_Work_Plan.md#7-rebalancing-rule) and [checkpoint sequence](DISTRIBUTED_TRAINING.md#checkpoint-2-run-the-independent-queues).
