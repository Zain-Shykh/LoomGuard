# Tasks for Member C (Hamza): unblock Member B

C has no training blocker for B, but B needs two independent checks and C's queue needs the same approvals as B's.

Reference values
- Training SHA: `67fb7f6ca2dec0365b63a7454ffb57b3138b6875`
- Full pin `datasets.json` SHA-256: `144a80d3f518a46b0183df1c3b5707627e58e128c89413a07907ca247374d40b`
- Tree digests in the full pin:
  - mvtec: `f1842104...7920` (full value is in the pin file)
  - raw_fabrid: `ce7cc3bf6a6e8bb2f713a4db2f73c4bc576276e7414df2b43ace4471433cce9e`
  - zju_leaper: `c2d7b294f2d4c32ccdc0c4d59b5377ac23c3bc69357e5bdf3ea7de73e72995d0`

## 1. Get the pin and set it up
B sends you `datasets.json` (full pin). Put it at `artifacts/distributed/full-shared/datasets.json` on a checkout at the training SHA. Check its SHA-256 against the value above. Do not overwrite the gate pin in `artifacts/distributed/gate-shared/`.

## 2. Cross-check the tree digests on your own copies
Run the pin command in a scratch directory (not in the shared directory), against your own copies of MVTec, RAW-FABRID and ZJU-Leaper. Compare each digest with the list above. If any digest differs: stop and tell B. This is a second check of what B uploaded to Kaggle.

## 3. Later: your ZJU queue
Your 11 RD4AD profiles use `--queue zju_full` and need `mini.json` from A (after B's ZJU-mini jobs and A's `--approve-mini`). Run `--prepare --queue zju_full` for your environment before then, so you are ready.
