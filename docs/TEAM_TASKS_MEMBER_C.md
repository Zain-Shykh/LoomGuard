# Hamza (Member C) dataset and RD4AD handoff

**Superseded operational instructions.** Follow [Hamza’s training handoff](handoffs/HAMZA_TRAINING_HANDOFF.md) for the current assignments, status and commands. This file preserves the earlier checklist and evidence.

Hamza acquired, prepared and originally verified the four core datasets. He produced the four manifests, original handoff archive and verification reports, investigated the two RAW-FABRID representations, reproduced ZJU Setting 1 and investigated the AITEX mask exclusions. The handoff describes verifier and AITEX adapter changes whose source diff and authorship still need review. Under D-61, Hamza owns 11 full-ZJU RD4AD profiles across five groups, one unified profile and five held-out patterns for 120,000 iterations. Huzaima independently verified and staged the handoff and coordinates checkpoints and archive receipt. Zain generated and distributed the original shared pins.

C has zero MVTec gate profiles. Obtain the verified full pin and checkpoint files from A after the paired MVTec gate and ZJU-mini approval. Confirm the permitted ZJU execution route, mount the authorized ZJU copy from the verified release and prepare using the original full pin at the approved experimental SHA. The current assignments and same-machine ablation rule apply unless a separate workload amendment is approved.

```bash
python scripts/distributed_study.py --shared artifacts/distributed/full-shared \
  --owner C --prepare --queue zju_full --review "$ENV_REVIEW_REF" --bundle-location "$BUNDLE_ROOT"
python scripts/distributed_study.py --shared artifacts/distributed/full-shared --owner C --queue zju_full --dry-run
```

C exports completed profiles and their original checksums to A. Review the live T4 quota, session time and private artifact capacity before accepting any additional unstarted profiles. [Workload review](LoomGuard_Team_Work_Plan.md#7-rebalancing-rule) and [checkpoint sequence](DISTRIBUTED_TRAINING.md#checkpoint-2-run-the-independent-queues).
