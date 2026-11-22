# LoomGuard — Member B Next Steps (While Members A and C Finish Their Tasks)

**Date:** 22 November 2026  
**Scope:** Member B, Kaggle Free; **no real datasets or training yet**.  
**Reference:** `LoomGuard_Member_B_Kaggle_Feasibility_Checklist (1).md`, Section 9.  
**Project:** https://github.com/Zain-Shykh/LoomGuard

## 1. Current position — do not repeat completed tests

Member B's 22 November synthetic checks reported:

| Check | Recorded result |
|---|---|
| Repository commit | `826b962e624dd2dc427da555515781ad1b109c8e` |
| Kaggle hardware | Tesla T4 ×2; ~14.56 GiB available per GPU; compute capability 7.5 |
| CUDA / packages | CUDA smoke passed; PyTorch 2.11.0+cu130, torchvision 0.26.0+cu130 |
| Assignment plan | 33 jobs; 185 unique canonical results; Member B gate = five PatchCore textures |
| Targeted tests | 41 passed, 1 slow test deselected |
| PatchCore synthetic fit | Bank size 20, finite score/heatmap, no GPU error |
| RD4AD FP16 synthetic benchmark | 3.48 iterations/s; 3.15 GB peak VRAM; finite loss |
| Production RD4AD `.fit()` smoke | 4 iterations; finite loss; 0 skipped optimizer steps |
| Export/retrieval | **Not verified** |

These results establish **synthetic feasibility**, not MVTec benchmark accuracy or full-study runtime. Do not run the same 100-iteration performance benchmark again unless a source/environment change makes it necessary.

## 2. Immediate required task: prove Kaggle export persistence

This task can be performed with the Kaggle notebook accelerator set to **None / CPU** to conserve GPU hours.

1. Open or create a **private** Kaggle notebook. Confirm that outputs are not being shared publicly.
2. Run the following small Python cell. It creates harmless test evidence under `/kaggle/working`:

```python
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

out = Path('/kaggle/working/loomguard-export-test')
out.mkdir(parents=True, exist_ok=True)
marker = b'LOOMGUARD_MEMBER_B_EXPORT_TEST_2026_11_22\n'
(out / 'marker.txt').write_bytes(marker)
manifest = {
    'purpose': 'Test Kaggle output persistence; no datasets or model bundles',
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'filename': 'marker.txt',
    'sha256': hashlib.sha256(marker).hexdigest(),
}
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
```

3. Use **Save Version** with output saving enabled. For a short, self-contained notebook, use **Save & Run All** if needed, but first ensure the notebook contains *only* these harmless cells and cannot accidentally start training. Kaggle's exact control labels may vary.
4. Wait for the saved version to finish. Open that version's **Output** files in the notebook viewer, then download `marker.txt` and `manifest.json` to a computer outside the live Kaggle session.
5. Verify that the downloaded marker has the SHA-256 shown in the downloaded manifest:

```python
from pathlib import Path
import hashlib
import json

p = Path('marker.txt')
m = json.loads(Path('manifest.json').read_text())
assert hashlib.sha256(p.read_bytes()).hexdigest() == m['sha256']
print('PASS: downloaded Kaggle output matches manifest')
```

6. Optional stronger proof: end the notebook session and confirm the files can still be downloaded from the **saved version**. Do **not** rely on files left only in a temporary live session.

**Acceptance criterion:** `marker.txt` and `manifest.json` are retrievable from a saved Kaggle version outside the live session and hashes match. Record the notebook/version link or ID (keep it private), time, and PASS/FAIL. Creating files in `/kaggle/working` without retrieving them is **not** sufficient.

## 3. Complete the feasibility evidence record

Prepare a short Markdown or JSON report, ideally using existing notebook output. Include:

- Source commit `826b962e...`, whether working tree was clean, and any newer repository commit detected.
- GPU model and per-GPU VRAM; Python, PyTorch, torchvision, CUDA and NVIDIA driver versions.
- 41 passed / 1 deselected; assignment validation counts and five gate profiles.
- PatchCore bank size, output shapes, finite checks, elapsed time.
- RD4AD FP16 iterations/s, peak VRAM, final loss and skipped steps.
- Output persistence result from Section 2, including saved version and verified download.
- Remaining Kaggle GPU quota and any session-duration limit **actually shown** by the account. If the maximum session time is not displayed, mark it **not confirmed**; do not assume 12 hours.
- Missing real-data checks (dataset mount, I/O, MVTec accuracy, full training timing).

Earlier logs that were only printed in a terminated notebook may be unavailable. **Do not fabricate or reconstruct outputs as if observed.** Mark them missing; rerun only a cheap command if the information is essential.

## 4. Prepare the MVTec PatchCore handoff — without running it

Member B can prepare the workflow for when Member C finishes verifying data:

- Read the latest `CLAUDE.md`, `docs/DECISIONS.md`, `docs/DISTRIBUTED_TRAINING.md`, `docs/TRAINING_PC_RUNBOOK.md` and the approved assignment YAML.
- Check `python scripts/distributed_study.py --help` and the runbook to identify the **exact** preparation and export arguments. Do not invent a `--prepare` invocation or fill review evidence with an unverified statement.
- Create a **private, non-running** Kaggle notebook or copy-paste execution checklist for Member B's gate. Include installing dependencies, confirming GPU/CUDA, verifying a matching pinned source commit, mounting **only the verified MVTec textures**, validating shared dataset metadata, running the approved `--prepare` step, the five PatchCore gate jobs, and exporting their canonical results/bundles.
- Mark every real-data command as **DO NOT RUN UNTIL HANDOFF**. Do not attach or publish MVTec images before confirming the dataset license restrictions and approved private distribution method.
- Plan output recovery after **each completed texture/job**, not only at the end of a long session. Confirm paths and required output files from the actual code before finalizing the notebook.
- Verify with `--validate` and `--owner B --stage gate --dry-run` if the checkout changes. Otherwise, reuse the already recorded successful results.

**Important:** Kaggle's free T4 allocation is two separate devices. The existing runner uses one device; do not silently enable multi-GPU training or change scientific settings.

## 5. What NOT to do while waiting

- Do not download/verify datasets already assigned to Member C.
- Do not create a fake `datasets.json` or approve checkpoint 0.
- Do not run `--prepare` without the verified dataset pin and genuine review evidence.
- Do not run real MVTec, RAW-FABRID, ZJU-Leaper, or AITEX jobs.
- Do not change model architectures, precision, random seeds, coreset ratios, training budgets, or evaluation criteria.
- Do not commit/push or upload model bundles without coordination and approval.
- Do not spend remaining free GPU quota repeating synthetic measurements unnecessarily.

## 6. Handoff back to the team

Send one message containing:

1. **Persistence:** PASS/FAIL, saved Kaggle notebook version, proof the files were downloaded and hash-checked.
2. **Feasibility summary:** T4 hardware, 41 passed / 1 deselected, PatchCore finite smoke, RD4AD FP16 3.48 it/s and 3.15 GB VRAM.
3. **Quota:** Actual remaining GPU hours and any verified session cap.
4. **Gate notebook:** Location of the prepared, *not executed*, Member B MVTec PatchCore workflow.
5. **Blockers:** Anything still missing, including verified MVTec files, shared dataset pin and Member A's benchmark readiness.

**Stop here until Member C completes the verified dataset handoff and the team pins the source commit.** Then run the original Member B MVTec gate (five PatchCore textures) using the approved distributed workflow. Member A's RD4AD gate can run concurrently. The team approves the shared gate only after both methods' actual results pass.
