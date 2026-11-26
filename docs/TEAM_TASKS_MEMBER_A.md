# Tasks for Member A (Hosama): unblock Member B

Member B has finished B1 (MVTec on Kaggle), B2 (MVTec PatchCore gate) and B3 (full pin). B4 (RAW-FABRID PatchCore) and everything after it are blocked on items below. Do them in order. Items 1 to 4 are the real blockers.

Reference values
- Training SHA: `67fb7f6ca2dec0365b63a7454ffb57b3138b6875`
- Full pin `datasets.json` SHA-256: `144a80d3f518a46b0183df1c3b5707627e58e128c89413a07907ca247374d40b`
- Gate pin `datasets.json` SHA-256: `702a0913c595aecfbdae73ab89be2f72ba2ccdead54f53b20f526b42c43981a0`
- Decisions are in `docs/LoomGuard_Decision_Resolutions.md` (R1 to R8, D-61 and D-62 approved).

## 1. Confirm the SHA
Check out `67fb7f6...` and confirm it is the frozen training SHA. If you want a different SHA, tell B now: B2 would have to be rerun.

## 2. Get B's five PatchCore gate results and bundles
B sends them privately (not in Git): `results/mvtec/per_fabric/patchcore/{carpet,grid,leather,tile,wood}.json` are already in the repo (commit `85da445`). The five bundles come from B's export folders. Put them at the canonical paths on your checkout. Check each `MANIFEST.sha256`.
B's means against the bars (image AUROC / pixel AUROC / AUPRO at FPR 0.3): 99.18 / 97.24 / 90.88 against 97.9 / 96.6 / 90.7. All met.

## 3. Accept two deviations (reply to B in chat, or note in the review text)
- Coreset time was measured on 207 to 252 tiles per texture, not 600. Largest coreset time was 20.4 s, so the "600-tile fit under 10 min" bar is met with a wide margin.
- Kaggle ran torch 2.11.0+cu130 on Python 3.13, not the documented 2.7+ with cu128. Results are valid for the gate.

## 4. Run the second `--approve-gate` against the full pin (the blocker)
B cannot run this. Place the full pin `datasets.json` at `artifacts/distributed/full-shared/datasets.json` (B sends the file; check the SHA-256 above). Then, on a clean checkout at the training SHA:
```bash
python scripts/distributed_study.py --approve-gate --review "<recorded team benchmark approval>"
```
It verifies ten results and bundles, averages per method and enforces PatchCore 97.9 / 96.6 / 90.7 and RD4AD 98.5 / 96.7 / 92.0 plus RD4AD peak VRAM under 11 GB. Send the resulting `artifacts/distributed/full-shared/gate.json` to B. B puts it beside `datasets.json` in a private Kaggle dataset. Without this file the standard queue stops at "waiting for shared checkpoint approval".
If any bar is missed, do not force it: tell B and the team.

## 5. Later: ZJU-mini approval
After B finishes both ZJU-mini jobs (PatchCore first), B hands you the ten bundles and results. You run:
```bash
python scripts/distributed_study.py --approve-mini --review "<recorded ZJU-mini verification>"
```
The review text is the owner's go-ahead for full ZJU (D-61): state that the Part 1 roll check and the first RAW-FABRID results were reviewed. Send `mini.json` to B and C.

## 6. Your own queue
Your 36 profiles (RD4AD and integration) follow the same gate. Tell B when your RAW-FABRID RD4AD results are committed so B can run the combined study checks.
