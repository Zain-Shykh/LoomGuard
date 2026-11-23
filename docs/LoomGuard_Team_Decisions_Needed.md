# LoomGuard: Decisions Needed Before Member B Can Start the MVTec PatchCore Gate

**From:** Member B (Kaggle account, T4 x2)
**To:** Member A and Member C
**Date:** 24 November 2026
**Purpose:** Member B is ready to run the MVTec PatchCore gate, but five team decisions block it. Please read each one, pick an option (or propose another) and reply in the "Answer" line. Nothing has been trained and nothing has been uploaded.

## Where things stand

| Item | Status |
|---|---|
| RD4AD scoped teacher-precision fix | Tested on Kaggle T4 by B (14/14 tests, two 300-iteration overfit runs, loss 0.04159, 0 skipped steps, TF32 flag restored). A has merged it. Results: `docs/LoomGuard_Member_B_RD4AD_T4_Regression_Results.md` |
| Member C dataset handoff | Received and extracted by B. Read `artifacts/dataset_handoff_notes.md`. B has not re-verified the data. |
| Kaggle feasibility, persistence and quota | Done earlier. See `docs/LoomGuard_Member_B_Kaggle_Feasibility_Checklist.md` |
| B's MVTec PatchCore gate (five textures) | Prepared, **not run**. Procedure: `docs/LoomGuard_Member_B_MVTec_PatchCore_Gate_Checklist.md` |
| Source commit pin, dataset pin, bundle storage, MVTec transfer | **Open. This file.** |

## Decision 1. Which source commit does everyone run? (owner: A and C to agree, A as coordinator proposed)

Every machine must check out one agreed commit, and any later change to code, configs or data needs a new pin.

Two things make this harder than "use latest main":
- Member C's handoff notes say the datasets were validated with the baseline commit `e24d8a4679c1df130460722e69091c9a7b9a6915` **plus verifier and AITEX policy changes that came with the handoff**. Those changes are **not in the repository** that B can see (`manifest.py` has no AITEX exclusion list and `main` has no verifier changes from C).
- Member A's merge of the RD4AD fix is already on `main` (`8e73e77` and later).

**Ask:**
1. Member C: please push the verifier and AITEX adapter changes (or send a patch) so the pinned commit contains the code that produced your manifests.
2. Team: name one commit SHA that has the reviewed plan, the RD4AD fix and C's changes, with nothing else changed afterwards.

Answer (SHA, and who pushes what): ______

## Decision 2. Who creates the dataset pin, and when? (owner: Member C proposed)

`--pin-datasets` writes `artifacts/distributed/shared/datasets.json`: hashes of the datasets, the manifests and the commit. It contains no images or secrets and is copied to each machine. It must be run once on a checkout that holds every dataset the active plan uses (MVTec, RAW-FABRID and ZJU, plus AITEX only if approved).

Proposal:
1. Coordinator (the person holding all verified data, so Member C) runs `python scripts/distributed_study.py --validate`.
2. Then `--pin-datasets --review "<link to the verifier output>"`.
3. Coordinator sends `datasets.json` privately to A and B.

Note: the pin hashes **every file** under `data/mvtec_ad`. That folder must hold exactly the five texture folders (carpet, grid, leather, tile, wood) and nothing else, byte for byte, on every machine.

Also confirm AITEX: decision D-61 excludes it, so `--aitex-approval` is not passed and the 14 AITEX profiles stay inactive. Member C's handoff still ships an AITEX manifest. Please confirm that AITEX stays out of the pin.

Answer (who, when, AITEX in or out of pin): ______

## Decision 3. How does the MVTec data reach Kaggle? (owner: Member C and the project owner)

MVTec AD is CC BY-NC-SA 4.0: noncommercial, no public redistribution. Kaggle notebooks cannot read the laptop disk, so the data must be uploaded.

Proposal P2: a **private** Kaggle Dataset containing only `mvtec_ad/<five textures>`, shared only with the Kaggle accounts that need it (B, and A if needed), never public and never attached to a public notebook. It appears in the notebook under `/kaggle/input/<dataset>/mvtec_ad` and is linked to `data/mvtec_ad`. Keeping the original folder names keeps the hash, so the pin still matches. The size is about 2.2 GB.

Questions for the team:
1. Is a private Kaggle upload acceptable under the MVTec licence for this project? (This is the licence reading of the team, not something B can decide.)
2. Who uploads? Proposal: Member C created the verified copy, so C uploads once and shares it with B. This avoids three different copies. B can upload from the laptop instead if C prefers.
3. B has not checked Kaggle's dataset size or private-sharing limits against these sizes. B will check before uploading.

Sizes for later stages (not needed for the gate): RAW-FABRID 4.4 GB, ZJU-Leaper 7.0 GB.

Answer (licence OK? who uploads?): ______

## Decision 4. Where do bundles and results go? (`--bundle-location`) (owner: project owner)

The value is a text prefix recorded in each result's provenance. Nothing uploads automatically, and it must not contain credentials.

- **P1 (proposed default):** run the whole gate in one Kaggle notebook (setup, tests, `--prepare`, five textures, copy to `export/`) and save it with **Save & Run All (Commit)**. The version's Output tab holds the exports and B downloads them to the coordinator. No token needed. Prefix: `kaggle-version://muhammadzain1x0/<notebook-slug>` plus the version number. Retrieval is proven for small files only. Risk: a failed or time-capped Commit run may not keep its outputs. This has not been tested for large bundles.
- **P3 (durable copies):** the notebook uploads to a private Kaggle Dataset or a private Hugging Face repo after each texture. This needs an API token in a Kaggle Secret, which is a credential decision for the owner. The repo rule (`CLAUDE.md`, rule 6) says bundles go to the Hugging Face Hub, so a private HF repo fits.

Answer (P1, P3 or both): ______

## Decision 5. How does B run `--prepare` with only MVTec? (owner: team)

`--prepare` rebuilds a split manifest and checks inputs for **every** dataset on B's job list, and `load_pin` requires `datasets.json` to match the whole active plan. With the plan as written, B cannot pass `--prepare` with only MVTec attached. The gate itself needs only MVTec.

- **O1:** attach all three datasets (MVTec, RAW-FABRID, ZJU) as private Kaggle inputs so `--prepare` passes. Costs: about 14 GB of inputs and hashing time in every new session.
- **O2:** agree a **reduced plan file** with only the MVTec gate jobs, use `--plan <file>` on every command, and make its own pin. Costs: a second set of files, and a new pin later for the full plan. Needs team review.
- **O3:** wait until everything is verified and pinned, then run as written. Slowest, no change to the agreed workflow.

Now that Member C's verification is delivered, O1 and O3 are more realistic than when this was first written. B's recommendation: **O3 if the pin (Decisions 1 and 2) can be done within a day**, otherwise O2. O1 only if Kaggle accepts 14 GB of private inputs.

Answer (O1, O2 or O3): ______

## Things from C's handoff that affect all three of us

- RAW-FABRID: use the shipped patch labels for the published patch benchmark. Use the high-resolution masks for localization and grading. Always record which representation a result used (25 tile labels and 532 defect areas differ between them).
- The seven rolls look visually similar (verdict SAME). Leave-one-roll-out only checks roll generalization. Use ZJU-Leaper held-out patterns for any claim about unseen fabrics.
- AITEX segmentation manifest excludes two defective images (`0100_025_08`, no source mask; `0106_010_03`, empty mask). AITEX is out of scope under D-61.
- Keep the data archive restricted to the team and use MVTec AD for noncommercial work only.

## Suggested order once answered

1. C pushes the verifier and AITEX changes. Team names the commit SHA (Decision 1).
2. Coordinator runs `--validate` and `--pin-datasets` and sends `datasets.json` (Decision 2).
3. MVTec goes to a private Kaggle Dataset (Decision 3).
4. B runs the MVTec PatchCore gate in one committed notebook using the answers to Decisions 4 and 5. B does not run `--pin-datasets`, `--approve-gate` or `--approve-mini`.
5. A's RD4AD gate can run at the same time. The team approves the shared gate only after both methods pass with real results.

Until these are answered, B starts no real-data run, uploads nothing and spends no Kaggle GPU time.
