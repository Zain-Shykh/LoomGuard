"""LoomGuard Streamlit app: a thin UI over loomguard.pipeline.Inspector (CLAUDE.md rule 5, docs/ARCHITECTURE.md section 11).

    streamlit run app/streamlit_app.py

Tiling, scoring, thresholding and grading all happen inside Inspector; this file validates uploads, finds and downloads
bundles, draws pages. Spec: docs/specs/phase-6-streamlit-app.md. The helpers above main() have no UI and are unit-tested.
"""
import hashlib
import io
import json
import os
import re
import shutil
import struct
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from torch import Tensor

from loomguard.inspection.grading import RollGrade, grade_roll
from loomguard.pipeline import Inspection, Inspector

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 25 * 1024 * 1024        # upload size limit
MAX_PIXELS = 16_000_000             # width x height limit (a 1792x1024 frame is 1.8 MP)
MIN_SIDE = 256                      # one model tile
FORMATS = ("PNG", "JPEG", "TIFF", "BMP")
EXTENSIONS = ("png", "jpg", "jpeg", "tif", "tiff", "bmp")
# Heuristic for "this is not the fabric of the profile" (A-27); check on real images (spec runbook step 5).
UNKNOWN_MASK_FRACTION = 0.30        # share of image pixels in the cleaned mask
UNKNOWN_TILE_FRACTION = 0.50        # share of flagged tiles, only for images with at least UNKNOWN_MIN_TILES tiles
UNKNOWN_MIN_TILES = 4
BUNDLE_FILES = ("model.pt", "calibration.json", "config.yaml")
MAX_BUNDLE_FILE = 600 * 1024 * 1024
NAME = re.compile(r"[\w.-]+")
PAGES = ("Inspect image", "Roll stream", "Results")
HOLE_NOTE = "Holes are scored by length like any other defect: LoomGuard has no defect-type labels (D-43)."


class UploadError(ValueError):
    """The message is shown to the user as is."""


class BundleError(RuntimeError):
    """The message is shown to the user as is."""


# ----------------------------------------------------------------------------- upload validation and small helpers

def read_image(data: bytes) -> Image.Image:
    """Decode untrusted bytes. The format is decided by Pillow from the content, never from a file name."""
    if len(data) > MAX_BYTES:
        raise UploadError(f"The file is {len(data) / 2**20:.1f} MB; the limit is {MAX_BYTES // 2**20} MB.")
    try:
        im = Image.open(io.BytesIO(data))
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):  # UnidentifiedImageError is an OSError
        raise UploadError("This file is not an image the app can read. Upload a PNG, JPEG, TIFF or BMP file.") from None
    if im.format not in FORMATS:
        raise UploadError(f"{im.format} images are not supported. Upload a PNG, JPEG, TIFF or BMP file.")
    w, h = im.size
    if w * h > MAX_PIXELS:
        raise UploadError(f"The image is {w}x{h} ({w * h / 1e6:.1f} megapixels); the limit is {MAX_PIXELS / 1e6:.0f} megapixels.")
    if min(w, h) < MIN_SIDE:
        raise UploadError(f"The image is {w}x{h}; both sides must be at least {MIN_SIDE} px (one model tile).")
    try:
        im.load()
    except (OSError, SyntaxError, ValueError, EOFError, struct.error):
        raise UploadError("The image is corrupt or truncated and could not be decoded.") from None
    return im


def unknown_fabric(ins: Inspection) -> bool:
    """True if the image does not look like the profile's fabric: a large share of the pixels or of the tiles is flagged."""
    n = len(ins.flagged_tiles)
    return bool(ins.mask.mean() >= UNKNOWN_MASK_FRACTION
                or (n >= UNKNOWN_MIN_TILES and ins.flagged_tiles.mean() >= UNKNOWN_TILE_FRACTION))


def scale_label(px_per_mm: float, scale_assumed: bool, from_upload: bool) -> str:
    return f"{px_per_mm:g} px/mm (" + ("assumed scale" if scale_assumed or from_upload else "stated by the dataset") + ")"


def overlay(img: Image.Image, heat: Tensor, tau: float) -> Image.Image:
    """Heatmap over the image: transparent at 0, yellow to red at 2 * tau and above."""
    h, w = heat.shape
    base = np.asarray(img.convert("RGB").resize((w, h)) if img.size != (w, h) else img.convert("RGB"), np.float32)
    t = np.clip(heat.numpy() / (2 * max(tau, 1e-9)), 0, 1)[..., None]
    colour = np.concatenate([np.full_like(t, 255), 255 * (1 - t), np.zeros_like(t)], -1)
    return Image.fromarray((base * (1 - 0.6 * t) + colour * 0.6 * t).round().astype(np.uint8))


def defect_table(defects: Iterable) -> pd.DataFrame:
    return pd.DataFrame([{"#": i + 1, "length mm": round(d.length_mm, 1), "area mm2": round(d.area_mm2, 1),
                          "x": round(d.centre[0]), "y": round(d.centre[1]), "points": d.points} for i, d in enumerate(defects)],
                        columns=["#", "length mm", "area mm2", "x", "y", "points"])


def stream_roll(insp: Inspector, frames: Iterable[Image.Image]) -> Iterator[tuple[int, Inspection, RollGrade]]:
    """Per frame: its index, its Inspection and the grade of the roll so far. The last grade equals Inspector.inspect_roll's."""
    pts = px = 0
    for i, f in enumerate(frames):
        ins = insp.inspect(f)
        pts, px = pts + ins.points, px + ins.heat.numel()
        yield i, ins, grade_roll(pts, px / insp.px_per_mm ** 2, insp.grading["cutoffs"])


# ----------------------------------------------------------------------------- places: bundles, samples, rolls

@dataclass(frozen=True)
class Locations:
    bundles: Path
    assets: Path
    results: Path
    hub_url: str
    hub_repo: str


def _json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def locations() -> Locations:
    env = os.environ.get
    assets = Path(env("LOOMGUARD_ASSETS") or ROOT / "assets")
    repo = env("LOOMGUARD_HUB_REPO") or _json(assets / "profiles.json", {}).get("hub_repo", "")
    return Locations(Path(env("LOOMGUARD_BUNDLES") or ROOT / "artifacts"), assets, Path(env("LOOMGUARD_RESULTS") or ROOT / "results"),
                     env("LOOMGUARD_HUB_URL") or "https://huggingface.co", repo)


def list_profiles(loc: Locations) -> dict[str, dict]:
    """id "<dataset>/<fabric>" -> {"label", "methods": {display name: method directory}}: the bundles on disk, plus the
    registry of assets/profiles.json when a Hub repository is configured (they can be downloaded)."""
    out: dict[str, dict] = {}
    if loc.hub_repo:
        for p in _json(loc.assets / "profiles.json", {}).get("profiles", []):
            out[p["id"]] = {"label": p.get("label", p["id"]), "methods": dict(p["methods"])}
    for cal in sorted(loc.bundles.glob("*/*/*/calibration.json")):
        d = cal.parent
        if d.name.startswith(".") or not (d / "model.pt").is_file() or not (d / "config.yaml").is_file():
            continue
        e = out.setdefault(f"{d.parent.parent.name}/{d.parent.name}", {"label": f"{d.parent.parent.name}/{d.parent.name}", "methods": {}})
        if d.name not in e["methods"].values():
            e["methods"][d.name] = d.name
    return out


def ensure_bundle(loc: Locations, profile_id: str, method_dir: str) -> Path:
    """The bundle directory <bundles>/<dataset>/<fabric>/<method dir>; downloaded from the Hub repository first if absent."""
    parts = [*profile_id.split("/"), method_dir]
    if len(parts) != 3 or not all(NAME.fullmatch(p) and p not in (".", "..") for p in parts):
        raise BundleError(f"Invalid bundle name {profile_id!r} / {method_dir!r}.")
    rel = "/".join(parts)
    dest = loc.bundles.joinpath(*parts)
    if all((dest / f).is_file() for f in BUNDLE_FILES):
        return dest
    if not loc.hub_repo:
        raise BundleError(f"Bundle {rel} is not on disk ({dest}) and no Hub repository is configured (LOOMGUARD_HUB_REPO).")
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", loc.hub_repo):
        raise BundleError(f"Invalid Hub repository name {loc.hub_repo!r}.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".download-", dir=dest.parent))
    try:
        for f in BUNDLE_FILES:
            with urllib.request.urlopen(f"{loc.hub_url.rstrip('/')}/{loc.hub_repo}/resolve/main/{rel}/{f}", timeout=60) as r, \
                    open(tmp / f, "wb") as out:
                size = 0
                while chunk := r.read(1 << 20):
                    size += len(chunk)
                    if size > MAX_BUNDLE_FILE:
                        raise BundleError(f"{rel}/{f} is larger than {MAX_BUNDLE_FILE // 2**20} MB; download stopped.")
                    out.write(chunk)
        if dest.exists() and not all((dest / f).is_file() for f in BUNDLE_FILES):
            shutil.rmtree(dest)  # an incomplete bundle must not block the download
        if not dest.exists():  # another session may have finished first
            tmp.rename(dest)
    except BundleError:
        raise
    except (OSError, ValueError) as e:  # URLError and HTTPError are OSErrors
        raise BundleError(f"Could not download bundle {rel} from {loc.hub_repo}: {e}") from e
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return dest


def list_samples(loc: Locations) -> list[dict]:
    d = loc.assets / "samples"
    meta = {e["file"]: e for e in _json(d / "samples.json", []) if isinstance(e, dict) and "file" in e}
    return [{"file": p.name, "path": p, "title": meta.get(p.name, {}).get("title", p.name),
             "credit": meta.get(p.name, {}).get("credit") or "no attribution entry", "profile": meta.get(p.name, {}).get("profile"),
             "synthetic": bool(meta.get(p.name, {}).get("synthetic"))}
            for p in (sorted(d.iterdir()) if d.is_dir() else []) if p.is_file() and p.suffix.lower().lstrip(".") in EXTENSIONS]


def list_rolls(loc: Locations) -> list[dict]:
    """assets/rolls/<name>/roll.json; a roll with a bad or missing frame is left out (a silent gap would change its grade)."""
    out = []
    for f in sorted((loc.assets / "rolls").glob("*/roll.json")):
        m = _json(f, None)
        if not isinstance(m, dict) or not m.get("profile") or not m.get("frames"):
            continue
        if all(isinstance(n, str) and Path(n).name == n and (f.parent / n).is_file() for n in m["frames"]):
            out.append({"name": f.parent.name, "label": m.get("label", f.parent.name), "profile": m["profile"], "dir": f.parent,
                        "frames": list(m["frames"]), "credit": m.get("credit") or "no attribution entry"})
    return out


@st.cache_resource(show_spinner=False)
def load_inspector(bundle: str) -> Inspector:
    return Inspector.load(bundle, "cpu")  # torch.load(weights_only=True) inside


# ----------------------------------------------------------------------------- pages

def _calibration_notes(bundle: Path) -> None:
    if _json(bundle / "calibration.json", {}).get("low_confidence"):
        st.warning("The threshold of this bundle rests on few validation tiles (low confidence).")


def _open_bundle(loc: Locations, profile_id: str, method_dir: str) -> tuple[Path, Inspector] | None:
    try:
        with st.spinner("Loading the model bundle (downloaded on first use)..."):
            bundle = ensure_bundle(loc, profile_id, method_dir)
            return bundle, load_inspector(str(bundle))
    except BundleError as e:
        st.error(str(e))
    except Exception as e:  # a damaged bundle must not crash the app
        st.error(f"The bundle could not be loaded: {e}")
    return None


def _unknown_text(ins: Inspection, label: str) -> str:
    return (f"Unknown fabric: this does not look like {label}. {ins.flagged_tiles.mean():.0%} of the tiles and "
            f"{ins.mask.mean():.0%} of the pixels are flagged, so the defect list would be meaningless. "
            "Choose the fabric profile that matches the image.")


def page_inspect(loc: Locations) -> None:
    profiles = list_profiles(loc)
    if not profiles:
        st.info("No model bundle is available: none under the bundles folder and no Hub repository configured. "
                "Fit one with scripts/fit.py or set LOOMGUARD_HUB_REPO (see docs/specs/phase-6-streamlit-app.md).")
        return
    sb = st.sidebar
    pid = sb.selectbox("Fabric profile", list(profiles), format_func=lambda i: profiles[i]["label"], key="profile")
    method = sb.selectbox("Model", list(profiles[pid]["methods"]), key="model")
    sens = sb.slider("Sensitivity", 0.5, 2.0, 1.0, 0.05, key="sens", help="Moves the threshold without re-running the model. "
                     "1.0 is the calibrated threshold (about 1% false alarms on normal tiles); higher flags more.")
    up = sb.file_uploader("Upload an image", type=list(EXTENSIONS), key="upload")
    samples = {s["file"]: s for s in list_samples(loc)}
    pick = sb.selectbox("or pick a sample", ["none", *samples], format_func=lambda f: f if f == "none" else samples[f]["title"], key="pick")
    if up is not None:
        data, from_upload, credit = up.getvalue(), True, None
    elif pick != "none":
        s = samples[pick]
        data, from_upload, credit = s["path"].read_bytes(), s["profile"] is None, s["credit"]
    else:
        st.info("Upload an image or pick a sample in the sidebar.")
        return
    try:
        img = read_image(data)
    except UploadError as e:
        st.error(str(e))
        return
    opened = _open_bundle(loc, pid, profiles[pid]["methods"][method])
    if opened is None:
        return
    bundle, insp = opened
    _calibration_notes(bundle)
    key = (str(bundle), hashlib.sha256(data).hexdigest())
    if st.session_state.get("scored_key") != key:  # the model runs once per image and bundle
        with st.spinner("Inspecting..."):
            t0 = time.perf_counter()
            scored = insp.score_image(img)
            st.session_state.update(scored_key=key, scored=scored, seconds=time.perf_counter() - t0)
    scored = st.session_state["scored"]
    ins = insp.decide(scored, insp.tau / sens)
    sb.caption(f"Scale: {scale_label(ins.px_per_mm, ins.scale_assumed, from_upload)}")
    if credit:
        st.caption(f"Sample credit: {credit}")
    rgb = img.convert("RGB")
    cols = st.columns(3)
    cols[0].image(rgb, caption="Original")
    cols[1].image(overlay(rgb, ins.heat, ins.tau), caption="Heatmap overlay")
    unknown = unknown_fabric(ins)
    if unknown:
        st.warning(_unknown_text(ins, profiles[pid]["label"]))
    else:
        cols[2].image(Image.fromarray(ins.mask.astype(np.uint8) * 255), caption="Defect mask")
    m = st.columns(4)
    m[0].metric("Score", f"{ins.score:.3f}")
    m[1].metric("Threshold", f"{ins.tau:.3f}")
    m[2].metric("Tiles flagged", f"{int(ins.flagged_tiles.sum())} of {len(ins.flagged_tiles)}")
    m[3].metric("Inspection time", f"{st.session_state['seconds']:.1f} s")
    if unknown:
        return
    st.subheader("Verdict: DEFECTIVE" if ins.flagged else "Verdict: OK")
    if ins.defects:
        st.dataframe(defect_table(ins.defects), hide_index=True)
    st.write(f"Points for this image: **{ins.points}**. Lengths use the {scale_label(ins.px_per_mm, ins.scale_assumed, from_upload)}.")
    g = ins.grade
    st.write(f"Grade for this single frame (an extrapolation: a grade is a roll-level result): **{g.grade}**, "
             f"{g.points_per_100:.0f} points per 100 square yards.")
    st.caption(HOLE_NOTE)


def page_roll(loc: Locations) -> None:
    rolls = list_rolls(loc)
    if not rolls:
        st.info("No demo roll is installed. A roll is a folder assets/rolls/<name>/ with roll.json and its frames "
                "(see docs/specs/phase-6-streamlit-app.md).")
        return
    profiles = list_profiles(loc)
    by_name = {r["name"]: r for r in rolls}
    sb = st.sidebar
    roll = by_name[sb.selectbox("Demo roll", list(by_name), format_func=lambda n: by_name[n]["label"], key="roll")]
    prof = profiles.get(roll["profile"])
    if prof is None:
        st.info(f"No model bundle is available for {roll['profile']}.")
        return
    method = sb.selectbox("Model", list(prof["methods"]), key="roll_model")
    delay = sb.slider("Pause between frames (s)", 0.0, 1.0, 0.1, 0.05, key="delay")
    st.caption(f"{len(roll['frames'])} frames. Credit: {roll['credit']}")
    if not st.button("Play roll", key="play"):
        return
    opened = _open_bundle(loc, roll["profile"], prof["methods"][method])
    if opened is None:
        return
    bundle, insp = opened
    _calibration_notes(bundle)
    st.caption(f"Scale: {scale_label(insp.px_per_mm, insp.scale_assumed, False)}")
    bar, frame_ph, stats, log_ph = st.progress(0.0), st.empty(), st.empty(), st.empty()
    log: list[dict] = []
    unknown_frames, g = 0, None
    try:
        frames = (read_image((roll["dir"] / n).read_bytes()) for n in roll["frames"])
        for i, ins, g in stream_roll(insp, frames):
            unknown_frames += unknown_fabric(ins)
            for d in ins.defects:
                log.append({"frame": roll["frames"][i], "length mm": round(d.length_mm, 1), "x": round(d.centre[0]),
                            "y": round(d.centre[1]), "points": d.points})
            bar.progress((i + 1) / len(roll["frames"]))
            frame_ph.image(overlay(Image.open(roll["dir"] / roll["frames"][i]), ins.heat, ins.tau),
                           caption=f"Frame {i + 1} of {len(roll['frames'])}: {roll['frames'][i]}")
            stats.write(f"Running: **{g.points}** points over {g.area_sq_yd:.2f} sq yd = **{g.points_per_100:.1f}** points per 100 sq yd, "
                        f"grade **{g.grade}**")
            log_ph.dataframe(pd.DataFrame(log, columns=["frame", "length mm", "x", "y", "points"]), hide_index=True)
            time.sleep(delay)
    except UploadError as e:
        st.error(f"A frame of the roll could not be read: {e}")
        return
    if unknown_frames:
        st.warning(f"{unknown_frames} frame(s) look like a different fabric than the profile; they are counted in the roll anyway.")
    st.subheader(f"Final grade: {g.grade}")
    st.write(f"{len(roll['frames'])} frames, {g.points} points, {g.points_per_100:.1f} points per 100 square yards "
             f"(roll grade, not an extrapolation).")
    st.caption(HOLE_NOTE)


def page_results(loc: Locations) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import summarize  # noqa: E402  reused, not copied: the layout of results/ is defined there
    R = summarize.load_results(loc.results)
    if not R:
        st.info(f"No result files under {loc.results} yet. They are written by scripts/evaluate.py on the training PC.")
        return
    ds, regime = st.selectbox("Dataset and regime", sorted({k[:2] for k in R}), format_func=lambda k: f"{k[0]}, {k[1]}", key="regime")
    det = summarize.det_metric(ds)
    dirs = {m: summarize.headline(ds, regime, m) for m in summarize.METHODS}
    names = sorted({k[3] for k in R if k[:3] in {(ds, regime, d) for d in dirs.values()}})
    metrics = [(det, det), ("aupro_30", "aupro_30"), ("tile_recall", "recall"), ("false_alarms_per_1000", "FA/1000")]
    rows = [{"fabric": n, **{f"{m} {label}": (R.get((ds, regime, dirs[m], n)) or {}).get(k) for m in summarize.METHODS
                             for k, label in metrics}} for n in names]
    df = pd.DataFrame(rows)
    st.subheader(f"{ds}, {regime}")
    st.dataframe(pd.concat([df, df.mean(numeric_only=True).to_frame().T.assign(fabric="mean")]), hide_index=True)
    st.bar_chart(df.set_index("fabric")[[f"{m} {det}" for m in summarize.METHODS]], y_label=det)
    with st.expander("The same fabric under every regime (Q2, Q3, Q4)"):
        st.markdown(summarize.cross_table(R))
    missing = summarize.missing_cells(R)
    st.caption(f"{len(missing)} cell(s) of the regime table have no result file yet." if missing
               else "Every cell of the regime table has a result file for both methods.")


def main() -> None:
    st.set_page_config(page_title="LoomGuard", layout="wide")
    st.title("LoomGuard")
    page = st.radio("Page", PAGES, horizontal=True, label_visibility="collapsed", key="page")
    {"Inspect image": page_inspect, "Roll stream": page_roll, "Results": page_results}[page](locations())


if __name__ == "__main__":
    main()
