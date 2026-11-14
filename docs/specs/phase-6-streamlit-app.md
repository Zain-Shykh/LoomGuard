# Phase 6: Streamlit application

| | |
|---|---|
| Phase | 6 |
| Status | Code complete |
| Spec written | 2026-11-14 |
| Written by | developer session, phase 6 |
| Depends on | Phases 1 to 5 (Code complete, runbooks queued, D-58). Needs `Inspector` and the bundle format of Phase 4 and the results layout of Phase 5. |
| Plan entry | `docs/IMPLEMENTATION_PLAN.md`, Phase 6 |

## 1. Objective

A Streamlit app, a thin UI over `loomguard/pipeline.py` `Inspector`, with three pages (Inspect image, Roll stream, Results), safe upload handling, bundle download and caching, a sample picker with attribution, and the deployment files for a Hugging Face Space. Everything that does not need a real bundle is built and tested here on synthetic bundles; the real-bundle measurements are in the runbook (section 8).

## 2. Background

- `docs/ARCHITECTURE.md` sections 3 (tiler), 8 (calibration, one `tau`), 9 (grading, scale labels, grade is a roll result), 10 (bundle), 11 (the app), 12 (layout).
- `docs/EVALUATION.md` section 5 (result file layout read by the Results page).
- `docs/IMPLEMENTATION_PLAN.md` Phase 6.
- Decisions: D-05, D-06, D-07, D-40 to D-43, D-52, D-53, D-58, A-15 (operating score), A-17 (a "roll" in `evaluate.py` is the test-pool frames of a fabric), A-18, A-20 (bundle upload is a runbook step; this phase decides how the app downloads), A-26 to A-29 (this phase).
- CLAUDE.md rule 5: tiling, scoring and grading live in `Inspector` only. Rule 6: bundles are loaded with `torch.load(..., weights_only=True)` (done inside `Inspector.load`). Rule 7: dependencies are added in the phase that needs them.

## 3. Scope

**In scope**

- `app/streamlit_app.py` with the pages Inspect image, Roll stream, Results.
- Upload validation (type by content, size, pixel count, decode safety) and clear error messages.
- Bundle discovery (local `artifacts/` plus a registry file), download from a Hugging Face Hub model repository with the standard library, cache on disk, `st.cache_resource` for loaded inspectors.
- Sample picker and demo-roll picker driven by files in `assets/`, with an attribution file format.
- The "unknown fabric" check, the "assumed scale" label, the "extrapolation" label for single-image grades.
- `[app]` optional dependency, `.streamlit/config.toml`, Space deployment files in `deploy/`.
- Tests with `streamlit.testing.v1.AppTest` and plain function tests.

**Out of scope**

- Deploying to Hugging Face, uploading bundles, adding real CC BY sample images (runbook).
- Automatic fabric-profile selection (open decision O-02), TSFabrics, defect-type classification.
- Any new tiling, scoring, thresholding or grading code; a new model; changes to `Inspector` (none are needed).
- Overriding `px_per_mm` in the UI (the scale is shown and labelled, not editable).

## 4. Design

### 4.1 Files

| File | New or changed | Purpose |
|---|---|---|
| `app/streamlit_app.py` | new | The whole app: pure helpers (validation, bundle lookup, unknown-fabric rule, rendering) and the three pages. `main()` runs only when `__name__ == "__main__"` (Streamlit and `AppTest` both run the script as `__main__`), so tests can import the helpers from the file path without drawing any UI. |
| `assets/profiles.json` | new | Registry: Hub repository name and the profiles and methods the Space offers. |
| `assets/samples/samples.json` | new | Attribution file for the sample images. |
| `assets/samples/synthetic_placeholder.png` | new | One tiny synthetic image, labelled synthetic in `samples.json`. |
| `.streamlit/config.toml` | new | Upload limit above the app's own limit, usage stats off. |
| `deploy/requirements.txt`, `deploy/README_space.md` | new | Files the Space repository needs (CPU PyTorch, app extra; Space front matter). |
| `pyproject.toml` | changed | Optional extra `app = ["streamlit>=1.45"]`. |
| `tests/test_app.py` | new | See section 7. |
| `docs/*`, `README.md`, `CLAUDE.md` | changed | Standard sync (section 9). |

No file under `loomguard/` or `scripts/` changes. `scripts/summarize.py` is imported by the Results page (loaded from `ROOT / "scripts"`), not copied.

### 4.2 Interfaces

All in `app/streamlit_app.py`. `Inspection`, `Scored`, `Defect`, `RollGrade` are the existing classes.

```python
MAX_BYTES = 25 * 1024 * 1024        # upload size limit
MAX_PIXELS = 16_000_000             # width x height limit (a 1792x1024 frame is 1.8 MP)
MIN_SIDE = 256                      # one model tile
FORMATS = ("PNG", "JPEG", "TIFF", "BMP")
UNKNOWN_MASK_FRACTION = 0.30        # share of image pixels in the cleaned mask
UNKNOWN_TILE_FRACTION = 0.50        # share of flagged tiles, only for images with at least UNKNOWN_MIN_TILES tiles
UNKNOWN_MIN_TILES = 4

class UploadError(ValueError): ...  # message is shown to the user as is

def read_image(data: bytes) -> Image.Image
    """Decode untrusted bytes. Raises UploadError with a clear message for: too many bytes, a format not in FORMATS
    (decided by Pillow from the content, never from a file name), too many pixels, a side under MIN_SIDE, a corrupt or
    truncated file. Returns the fully loaded first frame, mode unchanged."""

def unknown_fabric(ins: Inspection) -> bool
    """True if the image does not look like the profile's fabric: the cleaned mask covers at least UNKNOWN_MASK_FRACTION of
    the pixels, or (at least UNKNOWN_MIN_TILES tiles and at least UNKNOWN_TILE_FRACTION of them flagged)."""

def scale_label(px_per_mm: float, scale_assumed: bool, from_upload: bool) -> str
    """"4.0 px/mm (assumed scale)" if scale_assumed or from_upload, else "4.0 px/mm (stated by the dataset)"."""

def overlay(img: Image.Image, heat: Tensor, tau: float) -> Image.Image
    """img (any mode, H x W) blended with the heatmap (H x W float32): transparent at 0, yellow to red up to 2 * tau, clipped."""

def defect_table(defects: Iterable[Defect]) -> pandas.DataFrame
    """Columns: #, length mm, area mm2, x, y, points (centre of the defect in image pixels)."""

def stream_roll(insp: Inspector, frames: Iterable[Image.Image]) -> Iterator[tuple[int, Inspection, RollGrade]]
    """Per frame: index, its Inspection, the running grade (points and area summed so far, same cut-offs as
    Inspector.inspect_roll). The last item's RollGrade equals Inspector.inspect_roll(frames)[1]."""

@dataclass(frozen=True)
class Locations:                    # from environment variables, read at call time
    bundles: Path                   # LOOMGUARD_BUNDLES, default <repo>/artifacts
    assets: Path                    # LOOMGUARD_ASSETS, default <repo>/assets
    results: Path                   # LOOMGUARD_RESULTS, default <repo>/results
    hub_url: str                    # LOOMGUARD_HUB_URL, default https://huggingface.co
    hub_repo: str                   # LOOMGUARD_HUB_REPO, else "hub_repo" of assets/profiles.json, else ""
def locations() -> Locations

def list_profiles(loc: Locations) -> dict[str, dict]
    """id "<dataset>/<fabric>" -> {"label": str, "methods": {display name: method directory}}. Union of assets/profiles.json
    (only when a Hub repository is configured, A-26) and the bundles found under loc.bundles (<dataset>/<fabric>/<method dir>/calibration.json)."""

def ensure_bundle(loc: Locations, profile_id: str, method_dir: str) -> Path
    """<loc.bundles>/<profile_id>/<method_dir> if it holds model.pt, calibration.json and config.yaml. Otherwise download
    those three files from <hub_url>/<hub_repo>/resolve/main/<profile_id>/<method_dir>/ into a temporary directory, then rename
    it into place. Raises BundleError (clear message) if nothing is configured or the download fails."""

def list_samples(loc: Locations) -> list[dict]
    """Image files in assets/samples/ (png, jpg, jpeg, tif, tiff, bmp), sorted by name, each with the keys
    file, path, title, credit, profile (str or None), synthetic (bool), from samples.json when it has an entry for the file."""

def list_rolls(loc: Locations) -> list[dict]
    """assets/rolls/<name>/roll.json -> {"name", "label", "profile", "dir", "frames": [file names], "credit"}."""
```

Streamlit-facing functions (not unit-tested by signature, covered through `AppTest`): `main()`, `page_inspect(loc)`, `page_roll(loc)`, `page_results(loc)`, `load_inspector(bundle_path: str) -> Inspector` (decorated with `st.cache_resource`; calls `Inspector.load(path, "cpu")`).

#### File formats

`assets/profiles.json`:

```json
{"hub_repo": "", "profiles": [
  {"id": "raw_fabrid/Rollo1A", "label": "RAW-FABRID roll 1",
   "methods": {"PatchCore (1% coreset)": "patchcore_cr1", "Reverse Distillation": "rd4ad"}}]}
```

`assets/samples/samples.json` (attribution; every shipped CC BY image needs an entry):

```json
[{"file": "synthetic_placeholder.png", "title": "...", "credit": "synthetic, generated for tests", "profile": null, "synthetic": true}]
```

`profile` names the profile id the sample belongs to (a sample of that dataset, so its scale is the dataset's own); a sample without it is treated like an upload for the scale label. `credit` is shown under the image (author, source, licence). A file with no entry is still offered, with the caption "no attribution entry".

`assets/rolls/<name>/roll.json`: `{"label": str, "profile": "raw_fabrid/Rollo1A", "frames": ["f000.png", ...], "credit": str}`; frames sit in the same directory and are played in the order listed. A frame name must be a plain file name (no `/`), checked.

#### Pages

- **Inspect image.** Sidebar: fabric profile selector, model selector (the profile's methods), sensitivity slider, scale text, upload and sample picker. Main: three images (original, heatmap overlay, defect mask), verdict (`DEFECTIVE` / `OK`), score and threshold, processing time, the defect table with points, and the extrapolated grade. The model runs once per (image, bundle); the slider calls `Inspector.decide(scored, tau)` with `tau = insp.tau / sensitivity` (slider 0.5 to 2.0, default 1.0, higher is more sensitive), so it never re-runs the model (ARCHITECTURE 11).
- **Roll stream.** Select a demo roll (it fixes the profile) and a model; button "Play roll". Frames are inspected in order with `stream_roll`; after each frame the page updates the current frame with overlay, a defect log (frame, length, points), running points, running area in square yards, running points per 100 sq yd and grade. After the last frame the final grade is shown with the frame count.
- **Results.** Reads `results/` with `scripts/summarize.py` (`load_results`, `headline`, `det_metric`, `missing_cells`, `cross_table`). Dataset and regime selectors; a table of fabrics by method (detection metric, AUPRO 0.3, tile recall, false alarms per 1,000) with a mean row, a bar chart of the detection metric, the same-fabric-under-every-regime table, and the count of cells without a result file. With no result files it shows an info message.

### 4.3 Behaviour and edge cases

- **Upload.** `read_image` order: size, then `Image.open` (any `UnidentifiedImageError`, `OSError` from open, `DecompressionBombError` becomes "not a supported image"), format in `FORMATS`, pixel count, minimum side, then `load()` (an `OSError`, `SyntaxError`, `ValueError`, `EOFError` or `struct.error` here becomes "corrupt"). The uploaded file name is never used. `st.file_uploader(type=["png","jpg","jpeg","tif","tiff","bmp"])` filters in the browser; the check on content is the one that counts. Errors appear with `st.error` and the page stops; no traceback is shown.
- **Unknown fabric.** If `unknown_fabric(ins)` is true, the page shows a warning ("this does not look like <profile>: x% of tiles flagged, x% of pixels in the mask; choose the right fabric profile") and the heatmap, and does **not** show the verdict `DEFECTIVE`, the defect table, points or grade. The thresholds are heuristics (A-27) to be checked on real wrong-fabric images (runbook step 7). In the roll stream a frame that triggers it still counts toward the roll (as in `evaluate.py`) and a warning with the number of such frames is shown.
- **Scale.** Every result shows `scale_label(...)`. Uploads and samples without a `profile` entry are `from_upload=True`, so they are labelled "assumed scale" even for a RAW-FABRID profile.
- **Grade.** On the Inspect page the grade line reads "Grade for this single frame (extrapolation, a grade is a roll-level result): Reject, 730 points per 100 sq yd". A caption says holes are scored by length (D-43). Roll page grades are labelled as the roll's grade.
- **Low confidence.** If `calibration.json` has `low_confidence`, a warning says the threshold rests on few validation tiles.
- **No bundle.** If the chosen bundle is missing locally and no Hub repository is set, or the download fails, the page shows `BundleError`'s message and stops. If no profile exists at all the sidebar says so. The app starts with no dataset and no bundle.
- **Download.** HTTPS URL built from `hub_url`, a repository name matching `^[\w.-]+/[\w.-]+$` and path parts matching `^[\w.-]+$`; 600 MB cap per file; written to a temporary directory next to the target, renamed on success, removed on failure. Only `model.pt`, `calibration.json`, `config.yaml` are fetched (the bundle is loaded with `weights_only=True` by `Inspector.load`).
- **Caching.** `st.cache_resource` on `load_inspector`; the scored image of the Inspect page is kept in `st.session_state` keyed by (bundle path, SHA-256 of the image bytes). The model runs on CPU.

### 4.4 Configuration

Environment variables only (no new YAML keys): `LOOMGUARD_BUNDLES`, `LOOMGUARD_ASSETS`, `LOOMGUARD_RESULTS`, `LOOMGUARD_HUB_REPO`, `LOOMGUARD_HUB_URL`. `.streamlit/config.toml`: `server.maxUploadSize = 32` (MB; above `MAX_BYTES` so the app's message is the one shown), `browser.gatherUsageStats = false`.

### 4.5 Dependencies

`streamlit>=1.45` in the optional extra `app` of `pyproject.toml` (pandas and the other Streamlit requirements come with it). Installed already by the orchestrator (1.67.0). Not added: `huggingface_hub` (A-26: the download is `urllib` from the standard library, bundles are public files). `deploy/requirements.txt` installs CPU PyTorch then `.[app]` for the Space.

## 5. Tasks

### Laptop (developer session)

1. Write this spec; commit `phase 6: spec`.
2. `pyproject.toml` extra, `.streamlit/config.toml`, `assets/profiles.json`, `assets/samples/` placeholder and `samples.json`, `deploy/` files.
3. Pure helpers: `read_image`, `unknown_fabric`, `scale_label`, `overlay`, `defect_table`, `stream_roll`, `locations`, `list_profiles`, `ensure_bundle`, `list_samples`, `list_rolls`; tests for each.
4. Pages and `main()`; `AppTest` tests for the three pages and each validation path.
5. End-to-end test: `fit.py` and `evaluate.py` on the synthetic RAW-FABRID tree, then the app's output on the same frames equals the evaluation output; roll stream equals the per-roll result.
6. Run the app headless for a few seconds (`timeout`), full test suite, commit `phase 6: implementation`.
7. Review loop; documents; commit `phase 6: review fixes and doc sync`.

### Training PC / operator (not done here; see section 8)

1. Fresh-clone install and start on a CPU-only machine.
2. Upload bundles to the Hub; measure the 1792x1024 time on the laptop CPU with the PatchCore 1 percent bundle.
3. Equality of the app with `scripts/evaluate.py` on a real sample.
4. Add CC BY samples and a demo roll; the roll stream's final grade against the Phase 4 number.
5. Wrong-fabric check on real images; adjust the heuristic thresholds if needed (new decision).
6. Deploy to the Space.

## 6. Acceptance criteria

The first seven are the plan's, in its order. "Laptop part" is what this session verifies; the rest is the runbook and keeps the criterion open until the operator ticks it.

| # | Criterion | Where | Verified by | Met |
|---|---|---|---|---|
| A1 | From a clean clone, `pip install` and `streamlit run app/streamlit_app.py` work on a CPU-only machine with no dataset. | Laptop part: the extra is declared and the app starts and renders every page with no dataset and no bundle. Fresh clone: runbook 1 | `pytest tests/test_app.py::test_starts_without_bundles_or_data`, `test_pyproject_declares_app_extra`; headless `streamlit run` (section 8, step 0); runbook 1 | [ ] |
| A2 | A 1792x1024 image is processed in under 10 seconds on the laptop CPU with the PatchCore 1 percent bundle. Measured time recorded. | Laptop part: the page reports the measured time of the inspection. Measurement: runbook 2 | `test_inspect_sample_reports_time_and_matches_evaluate`; runbook 2 (value in the progress log) | [ ] |
| A3 | The app's heatmap and defect list for a sample image equal the output of `scripts/evaluate.py` for the same image and bundle. | Laptop part: synthetic bundle, equal score, flag, defect table and heatmap. Real sample: runbook 3 | `test_inspect_sample_reports_time_and_matches_evaluate`; runbook 3 | [ ] |
| A4 | A non-image file, an oversized file and a corrupt image each produce a clear error message and no crash. | Laptop | `test_read_image_rejects`, `test_upload_errors_in_app` (non-image, oversized, corrupt, wrong format, too many pixels, too small) | [x] |
| A5 | An image of the wrong fabric produces the "unknown fabric" warning instead of a meaningless defect list. | Laptop part: synthetic wrong fabric, rule unit test. Real images: runbook 5 | `test_unknown_fabric_rule`, `test_wrong_fabric_warning_hides_defects`; runbook 5 | [ ] |
| A6 | Results for uploaded images of unknown origin are labelled "assumed scale"; single-image grades are labelled as extrapolations. | Laptop | `test_scale_label`, `test_upload_is_labelled_assumed_scale_and_grade_extrapolation` | [x] |
| A7 | The roll stream runs to the end on a RAW-FABRID demo roll and its final grade matches the Phase 4 number. | Laptop part: synthetic roll runs to the end and equals `evaluate.py`'s per-roll result. Real roll: runbook 4 | `test_roll_stream_matches_evaluate`, `test_stream_roll_equals_inspect_roll`; runbook 4 | [ ] |
| A8 | The Results page shows tables and plots from `results/` and does not crash on an empty folder. (Plan task 3.) | Laptop | `test_results_page` | [x] |
| A9 | A bundle that is not on disk is downloaded from the weights host once and then served from the cache; bad names and failed downloads give a clear error. (Plan task 5.) | Laptop, against a local HTTP server | `test_ensure_bundle_downloads_once`, `test_ensure_bundle_errors` | [x] |
| A10 | The sample picker and the roll picker work from whatever is in `assets/`, show the attribution, and handle a missing attribution entry. (Plan task 5.) | Laptop | `test_list_samples_and_rolls`, `test_shipped_samples_are_listed_with_attribution`, `test_upload_is_labelled_assumed_scale_and_grade_extrapolation` | [x] |
| A11 | Deployment files for a Hugging Face Space exist (CPU PyTorch requirements, Space front matter); the actual deployment is the runbook. (Plan task 6.) | Laptop part; deploy: runbook 6 | `test_deploy_files`; runbook 6 | [ ] |
| A12 | The app contains no tiling, scoring or grading code of its own (CLAUDE.md rule 5). | Laptop | `test_app_has_no_pipeline_code` (the source imports `Inspector` and does not import `loomguard.data.tiling`, `loomguard.models`, `loomguard.inspection.postprocess`), reviewer | [x] |

## 7. Test plan

All on CPU, no dataset. Bundles are built by the real `fit.py` on the synthetic RAW-FABRID tree of `conftest.py` with a seeded random backbone (as `test_fit_evaluate.py`); `loomguard.models.backbone.Backbone` is monkeypatched to the seeded one so the app's `Inspector.load` sees the same network. Environment variables point the app at `tmp_path`.

| Test | Proves |
|---|---|
| `test_read_image_rejects` | Not an image, text with a `.png` name, an oversized byte string, a truncated PNG, GIF (unsupported), a 5000x5000 image over the pixel limit, a 100x100 image: each raises `UploadError` with the expected message; valid PNG, JPEG, TIFF, BMP decode. |
| `test_scale_label` | Truth table of the three inputs. |
| `test_unknown_fabric_rule` | Hand-built `Inspection` objects: small mask and few flagged tiles is false; mask over 30% is true; half the tiles flagged with 4 or more tiles is true; a single flagged tile of one is false. |
| `test_stream_roll_equals_inspect_roll` | With a deterministic fake model, the running grade after each frame equals `grade_roll` of the partial sums, and the last equals `Inspector.inspect_roll`. |
| `test_overlay_and_defect_table` | Zero heat leaves the image unchanged; heat at `2 * tau` is red-yellow; the table has the columns and values of known `Defect`s. |
| `test_list_samples_and_rolls`, `test_shipped_samples_are_listed_with_attribution` | Files without entries, entries without files, extension filtering, a roll with a `/` or a missing frame left out; the shipped placeholder is listed with its credit. The credit text of a sample appears on the page: checked in `test_upload_is_labelled_assumed_scale_and_grade_extrapolation`. |
| `test_ensure_bundle_downloads_once`, `test_ensure_bundle_errors` | A threaded `http.server` serves a bundle; first call downloads three files, second call makes no request; hub unset, bad repo name, `..` in a path part, HTTP 404 and an oversize file give `BundleError` and leave no partial directory. |
| `test_starts_without_bundles_or_data` | `AppTest` with empty directories: no exception on any of the three pages; the app says no bundle is available. |
| `test_inspect_sample_reports_time_and_matches_evaluate` | End to end: `fit.py`, `evaluate.py` (with `dump_predictions`) on the synthetic tree; the app on a test image shows the score, the verdict, the defect table and the processing time; they equal the dumped `scores.csv` and the defects measured from the dumped mask; `Inspector.score_image(...).heat` equals the heat stitched by hand. |
| `test_upload_errors_in_app` | Through `file_uploader.set_value`: text file, oversized, corrupt PNG, GIF: `st.error` with the message, no exception element, no result shown. |
| `test_upload_is_labelled_assumed_scale_and_grade_extrapolation` | Upload to a RAW-FABRID profile (`scale_assumed: false`): "assumed scale" and "extrapolation" appear; a sample with `profile` set shows "stated by the dataset". |
| `test_sensitivity_slider_does_not_rerun_model` | Changing the slider changes the defect list and leaves the model call count at 1 (counting wrapper on `Inspector.score_image`). |
| `test_wrong_fabric_warning_hides_defects` | A bank built from one smooth texture; a noise image of the same size triggers the warning and shows no defect table, no `DEFECTIVE` verdict; the normal texture does not. |
| `test_roll_stream_matches_evaluate` | The roll page plays the test-pool frames of `RolloA` to the end; frame count, points, points per 100 sq yd and grade equal the `rolls["RolloA"]` entry of the result JSON. |
| `test_results_page` | Synthetic result files: table and chart appear, the numbers equal the files; empty folder: info message. |
| `test_pyproject_declares_app_extra`, `test_deploy_files` | `streamlit` in the `app` extra with a lower bound; `deploy/requirements.txt` has the CPU index and `.[app]`; `deploy/README_space.md` has `sdk: streamlit` and `app_file: app/streamlit_app.py`; `.streamlit/config.toml` upload limit above `MAX_BYTES`. |
| `test_app_has_no_pipeline_code` | Rule 5 by import inspection. |

## 8. Training-PC runbook

Run from the repository root with the project environment active. Steps 1 to 6 need real bundles (steps 1 and 6 can be done on any machine). Record every value in `PROGRESS_LOG.md` and tick the criteria in this spec.

```bash
# 0. (done on the laptop) headless start, stopped after 8 seconds
timeout 8 python3 -m streamlit run app/streamlit_app.py --server.headless true --server.port 8599

# 1. fresh clone, CPU-only machine, no dataset (criterion A1)
git clone <repo-url> loomguard-clean && cd loomguard-clean
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[app]"
streamlit run app/streamlit_app.py        # open the three pages; with no bundle the app says so

# 2. upload the demo bundles to a Hugging Face model repository (A-20), then set the repository
#    (training PC; keep the dataset/fabric/method layout of artifacts/)
huggingface-cli upload <user>/loomguard-bundles artifacts/raw_fabrid raw_fabrid \
    --include "*/patchcore_cr1/model.pt" "*/patchcore_cr1/calibration.json" "*/patchcore_cr1/config.yaml" \
              "*/rd4ad/model.pt" "*/rd4ad/calibration.json" "*/rd4ad/config.yaml"
# edit assets/profiles.json: "hub_repo": "<user>/loomguard-bundles", one profile per roll

# 2b. timing (A2) on the laptop CPU: first use downloads the bundle
export LOOMGUARD_HUB_REPO=<user>/loomguard-bundles
python3 - <<'EOF'
import sys, time; sys.path.insert(0, "app")
from pathlib import Path
from PIL import Image
import streamlit_app as app
from loomguard.pipeline import Inspector
loc = app.locations()
insp = Inspector.load(app.ensure_bundle(loc, "raw_fabrid/Rollo1A", "patchcore_cr1"), "cpu")
img = Image.open("<a 1792x1024 RAW-FABRID image>")
insp.inspect(img)                                            # warm-up
ts = []
for _ in range(5):
    t = time.perf_counter(); insp.inspect(img); ts.append(time.perf_counter() - t)
print("seconds per image:", sorted(ts))                      # criterion: median under 10
EOF

# 3. equality with evaluate.py (A3): evaluate with dump_predictions, then compare with the app
#    (config = the per-roll PatchCore 1 percent config with `dump_predictions: true` added to a copy)
python3 scripts/evaluate.py --config <copy of configs/raw_fabrid_per_fabric_patchcore_cr1.yaml with dump_predictions: true>
#    predictions are in artifacts/predictions/raw_fabrid/per_fabric/patchcore_cr1/Rollo1A/ (scores.csv and one PNG per image)
python3 - <<'EOF'
# the app's Inspect page (AppTest) on one test image: its score and defect table against scores.csv and the dumped mask
import csv, json, os, shutil
from pathlib import Path
import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest
from loomguard.pipeline import Inspector
rel = "<an image path as written in the first column of scores.csv, e.g. images/Rollo1A_....png>"
root, name = Path("data/RAW_FABRID"), Path(rel).name
pred = Path("artifacts/predictions/raw_fabrid/per_fabric/patchcore_cr1/Rollo1A")
a = Path("artifacts/demo_assets"); (a / "samples").mkdir(parents=True, exist_ok=True)
shutil.copy(root / rel, a / "samples" / name)
(a / "samples" / "samples.json").write_text(json.dumps([{"file": name, "profile": "raw_fabrid/Rollo1A"}]))
os.environ.update(LOOMGUARD_ASSETS=str(a), LOOMGUARD_BUNDLES="artifacts")
at = AppTest.from_file("app/streamlit_app.py", default_timeout=600).run()
at.sidebar.selectbox(key="profile").select("raw_fabrid/Rollo1A")
at.sidebar.selectbox(key="model").select("patchcore_cr1")   # a bundle found on disk is listed under its directory name
at.sidebar.selectbox(key="pick").select(name).run()
row = next(r for r in csv.DictReader(open(pred / "scores.csv")) if r["parent"] == rel)
m = {x.label: x.value for x in at.metric}
print("score app:", m["Score"], "evaluate:", row["score"], "flagged evaluate:", row["flagged"], "verdict app:", at.subheader[0].value)
mask = np.asarray(Image.open(pred / f"{Path(rel).stem}.png")) > 0
print("defects app:     ", at.dataframe[0].value[["length mm", "points"]].values.tolist() if at.dataframe else [])
print("defects evaluate:", [(round(d.length_mm, 1), d.points) for d in Inspector.load("artifacts/raw_fabrid/Rollo1A/patchcore_cr1").measure(mask)[1]])
EOF

# 4. demo roll and Phase 4 number (A7): build a local roll from the test pool of one roll (not committed)
python3 - <<'EOF'
import json, shutil
from pathlib import Path
from loomguard.runs import build_rows, load_config
cfg = load_config("configs/raw_fabrid_per_fabric_patchcore_cr1.yaml")
rows = [r for r in build_rows(cfg) if r.fabric == "Rollo1A" and r.split == "test"]
frames = sorted({r.image for r in rows})
out = Path("artifacts/demo_assets/rolls/Rollo1A"); out.mkdir(parents=True, exist_ok=True)
for f in frames: shutil.copy(Path(cfg["root"]) / f, out / Path(f).name)
(out / "roll.json").write_text(json.dumps({"label": "RAW-FABRID roll 1, test pool", "profile": "raw_fabrid/Rollo1A",
    "frames": [Path(f).name for f in frames], "credit": "RAW-FABRID, CC BY 4.0, <citation>"}))
EOF
cp assets/profiles.json artifacts/demo_assets/
LOOMGUARD_ASSETS=artifacts/demo_assets streamlit run app/streamlit_app.py   # Roll stream, play to the end
#    compare the final points per 100 sq yd and grade with results/raw_fabrid/per_fabric/patchcore_cr1/Rollo1A.json["rolls"]["Rollo1A"]

# 5. wrong fabric (A5): on the Inspect page choose the Rollo1A profile and upload an image of another roll; record the
#    share of flagged tiles and mask pixels for 5 images per other roll and for 5 normal images of Rollo1A. If a normal image
#    triggers the warning, or most wrong-fabric images do not, change the three UNKNOWN_* constants and record a new decision.

# 6. samples and Space (A11)
#    copy 3 to 5 RAW-FABRID images (CC BY 4.0) into assets/samples/ and add one entry per file to assets/samples/samples.json
#    (title, credit with author, source and licence, "profile": "raw_fabrid/Rollo1A"); a short roll can go to assets/rolls/<name>/.
#    On huggingface.co create a Streamlit Space, then:
git clone https://huggingface.co/spaces/<user>/loomguard-space && cd loomguard-space
git --git-dir=../.git archive main | tar -x          # or copy the working tree
cp deploy/requirements.txt requirements.txt && cp deploy/README_space.md README.md
#    set the Space variable LOOMGUARD_HUB_REPO if assets/profiles.json has no hub_repo; commit and push; check the three pages
```

| Step | Expected | Record in |
|---|---|---|
| 1 | Installs without CUDA wheels; app starts; no crash without data | `PROGRESS_LOG.md`, tick A1 |
| 2b | Median seconds per 1792x1024 image under 10 | `PROGRESS_LOG.md`, tick A2 |
| 3 | Same score, flag and defect list as `scores.csv` and the dumped mask | tick A3 |
| 4 | Roll runs to the last frame; final points per 100 sq yd and grade equal the result JSON | tick A7 |
| 5 | Warning on wrong-fabric images, none on normal ones | tick A5, decision if constants change |
| 6 | Space builds on the free CPU tier and the three pages work | tick A11, record the Space URL |

## 9. Documents to update

- `docs/ARCHITECTURE.md` section 11 (what was built: download, registry, unknown-fabric rule, file names) and 12.
- `docs/DECISIONS.md` A-26 to A-29; `docs/IMPLEMENTATION_PLAN.md`; `docs/PROGRESS_LOG.md`.
- `README.md` (state, setup with the `app` extra, how to run the app) and `CLAUDE.md` (setup line, layout: `assets/`, `deploy/`).

## 10. Risks and open points

| Risk or open point | Handling |
|---|---|
| The unknown-fabric thresholds (30% of pixels, 50% of at least 4 tiles) are guesses until real wrong-fabric images are seen | A-27; runbook step 5; constants are named, one place to change |
| The Hugging Face resolve URL layout and rate limits are not verified from the laptop (no network) | `hub_url` is configurable; the test uses a local server with the same path layout; runbook step 2b exercises the real host |
| RAW-FABRID rolls may look alike, so a "wrong fabric" image of another roll may not trigger the warning | Not a defect of the app; reported by runbook step 5; Q4 of the study is the real answer |
| CPU time of 28 tiles with RD4AD may exceed 10 s | The criterion names the PatchCore 1 percent bundle; the RD4AD time is recorded too, and the page shows the measured time |
| `AppTest` bypasses the browser, so the real upload widget and rendering are not exercised | Headless start is run; runbook step 1 opens the pages in a browser |
| Free Space disk is not persistent: bundles are downloaded again after a restart | Bundles are small (PatchCore 1 percent), acceptable; documented |
| Nothing depends on an open decision (O-01, O-02) | |

## 11. Completion record

| | |
|---|---|
| Spec commit | 29e2e21 |
| Implementation commit | 0649118 |
| Review commit | d4f4ee4 (docs, pre-review) and the final `phase 6: review fixes and doc sync` |
| Review rounds and verdict | 1 round, PASS; four minor findings fixed (spec docstring of `list_profiles`, A-20 annotation, incomplete-bundle retry with a new test, this record) |
| Tests | `pytest -q`: 154 passed, 0 failed, 1 slow deselected (`tests/test_app.py`: 26) |
| Results commit (training PC) | pending: runbook steps 1 to 6 |
| Decisions made | A-26 to A-29 |
| Deviations from this spec | `use_container_width` dropped (deprecated in the installed Streamlit); Streamlit enforces the file extension too, so tests send a GIF named `.png` for the content check; A1, A2, A3, A5, A7 and A11 stay unticked: only their laptop part is met, the rest is the runbook. |
| Closed by owner | |
