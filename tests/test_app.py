"""The Streamlit app (app/streamlit_app.py): helpers, upload validation, bundle download, and the three pages through AppTest.
Bundles come from the real fit.py and evaluate.py on the synthetic RAW-FABRID tree (random backbone, seeded). No dataset."""
import ast
import importlib.util
import io
import json
import sys
import threading
import tomllib
import types
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from streamlit.testing.v1 import AppTest

import conftest
from loomguard.data.tiling import stitch, tile
from loomguard.data.dataset import preprocess
from loomguard.inspection.grading import grade_roll
from loomguard.inspection.postprocess import Defect
from loomguard.models.backbone import Backbone
from loomguard.pipeline import Inspector, Scored

ROOT = Path(__file__).parents[1]
APP = ROOT / "app" / "streamlit_app.py"
_spec = importlib.util.spec_from_file_location("streamlit_app_under_test", APP)
app = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = app
_spec.loader.exec_module(app)  # __name__ is not "__main__": defines the helpers, draws nothing


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fit, evaluate = load_script("fit"), load_script("evaluate")


def seeded_backbone(weights=None):
    torch.manual_seed(0)
    return Backbone(weights)


@pytest.fixture(autouse=True)
def same_random_backbone(monkeypatch):
    """fit, evaluate and the app's Inspector.load all build the same random network."""
    for mod in (fit, evaluate):
        monkeypatch.setattr(mod, "Backbone", seeded_backbone)
    monkeypatch.setattr("loomguard.models.backbone.Backbone", seeded_backbone)


def png(arr, fmt="PNG") -> bytes:
    b = io.BytesIO()
    Image.fromarray(arr).save(b, fmt)
    return b.getvalue()


# ----------------------------------------------------------------------------- helpers

@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "TIFF", "BMP"])
def test_read_image_accepts(fmt):
    im = app.read_image(png(np.random.default_rng(0).integers(0, 256, (256, 300), dtype=np.uint8), fmt))
    assert im.size == (300, 256)


def test_read_image_rejects(monkeypatch):
    noise = np.random.default_rng(0).integers(0, 256, (300, 300), dtype=np.uint8)
    good = png(noise)
    cases = {
        "not an image the app can read": b"just some text, even if called x.png",
        "not an image the app can read ": b"",
        "corrupt or truncated": good[: len(good) // 2],
        "GIF images are not supported": png(noise, "GIF"),
        "limit is 16 megapixels": png(np.zeros((5000, 5000), np.uint8)),
        "at least 256 px": png(np.zeros((100, 400), np.uint8)),
    }
    for expect, data in cases.items():
        with pytest.raises(app.UploadError, match=expect.strip()):
            app.read_image(data)
    assert app.read_image(good).size == (300, 300)
    monkeypatch.setattr(app, "MAX_BYTES", len(good) - 1)
    with pytest.raises(app.UploadError, match=r"The file is .* MB; the limit is"):
        app.read_image(good)


def test_scale_label():
    assert app.scale_label(4.0, False, False) == "4 px/mm (stated by the dataset)"
    assert app.scale_label(4.0, True, False) == "4 px/mm (assumed scale)"
    assert app.scale_label(4.0, False, True) == "4 px/mm (assumed scale)"
    assert app.scale_label(5.0, True, True) == "5 px/mm (assumed scale)"


def fake_ins(mask_frac, flagged):
    mask = np.zeros((256, 512), bool)
    mask[:, : int(512 * mask_frac)] = True
    return types.SimpleNamespace(mask=mask, flagged_tiles=np.array(flagged, bool))


def test_unknown_fabric_rule():
    assert not app.unknown_fabric(fake_ins(0.01, [True, False, False, False]))          # a real defect
    assert not app.unknown_fabric(fake_ins(0.01, [True]))                                # one tile, flagged: not enough evidence
    assert app.unknown_fabric(fake_ins(0.31, [True]))                                    # the mask covers a third of the image
    assert not app.unknown_fabric(fake_ins(0.29, [True]))
    assert app.unknown_fabric(fake_ins(0.02, [True, True, False, False]))                # half of 4 tiles
    assert not app.unknown_fabric(fake_ins(0.02, [True, True, False, False, False]))    # 40 percent of 5
    assert not app.unknown_fabric(fake_ins(0.02, [True, True, True]))                    # 3 tiles are too few for the tile rule


class FakeModel:
    """heatmap = first channel of the tile (background about -2.1, bright pixels about +2.2); tile score = half its maximum."""
    device = torch.device("cpu")

    def score(self, x):
        heat = x[:, :1].clone()
        return heat.amax((1, 2, 3)) * 0.5, heat


def line_frame(boxes, h=256, w=512):
    a = np.zeros((h, w), np.uint8)
    for y0, y1, x0, x1 in boxes:
        a[y0:y1, x0:x1] = 255
    return Image.fromarray(a)


def test_stream_roll_equals_inspect_roll():
    insp = Inspector(FakeModel(), 0.0, px_per_mm=4.0)
    frames = [line_frame([(50, 53, 10, 210)]), line_frame([(50, 53, 10, 410), (150, 153, 10, 90)]), line_frame([])]
    seen = list(app.stream_roll(insp, frames))
    assert [i for i, _, _ in seen] == [0, 1, 2] and [ins.points for _, ins, _ in seen] == [1, 3, 0]
    for i, (_, _, g) in enumerate(seen):  # the running grade is the roll grade of the frames so far
        assert g == insp.inspect_roll(frames[: i + 1])[1]
    assert seen[-1][2].points == 4 and seen[-1][2] == grade_roll(4, 3 * 256 * 512 / 16)


def test_overlay_and_defect_table():
    img = Image.fromarray(np.full((256, 256), 100, np.uint8))
    assert np.array_equal(np.asarray(app.overlay(img, torch.zeros(256, 256), 1.0)), np.full((256, 256, 3), 100))
    hot = np.asarray(app.overlay(img, torch.full((256, 256), 2.0), 1.0))[0, 0]
    assert hot.tolist() == [round(100 * 0.4 + 255 * 0.6), 40, 40]                      # red at 2 * tau
    mid = np.asarray(app.overlay(img, torch.full((256, 256), 1.0), 1.0))[0, 0]
    assert mid[1] == round(100 * 0.7 + 127.5 * 0.3) and mid[2] == 70                   # half way: orange
    assert app.overlay(img, torch.zeros(512, 512), 1.0).size == (512, 512)             # resized to the heatmap (MVTec profiles)
    d = Defect((10, 20, 110, 30), (59.5, 24.5), 500, 100.0, 25.0, 31.25, 1)
    t = app.defect_table([d, d])
    assert list(t.columns) == ["#", "length mm", "area mm2", "x", "y", "points"] and len(t) == 2
    assert t.iloc[0].tolist() == [1, 25.0, 31.2, 60, 24, 1]
    assert len(app.defect_table([])) == 0


def test_list_samples_and_rolls(tmp_path):
    a = tmp_path / "assets"
    (a / "samples").mkdir(parents=True)
    for n in ("b.png", "a.JPG", "c.tiff"):
        (a / "samples" / n).write_bytes(b"x")
    (a / "samples" / "notes.txt").write_text("x")
    (a / "samples" / "samples.json").write_text(json.dumps([
        {"file": "a.JPG", "title": "A", "credit": "Author, CC BY 4.0", "profile": "raw_fabrid/Rollo1A"},
        {"file": "gone.png", "title": "no file"}, {"file": "b.png", "synthetic": True}]))
    loc = app.Locations(tmp_path / "b", a, tmp_path / "r", "http://x", "")
    s = {e["file"]: e for e in app.list_samples(loc)}
    assert list(s) == ["a.JPG", "b.png", "c.tiff"]
    assert s["a.JPG"]["credit"] == "Author, CC BY 4.0" and s["a.JPG"]["profile"] == "raw_fabrid/Rollo1A" and s["a.JPG"]["title"] == "A"
    assert s["c.tiff"]["credit"] == "no attribution entry" and s["c.tiff"]["profile"] is None and s["b.png"]["synthetic"]
    assert app.list_samples(app.Locations(tmp_path, tmp_path / "nowhere", tmp_path, "", "")) == []

    def roll(name, frames, make=("f0.png", "f1.png")):
        d = a / "rolls" / name
        d.mkdir(parents=True)
        for f in make:
            (d / f).write_bytes(b"x")
        (d / "roll.json").write_text(json.dumps({"label": name, "profile": "raw_fabrid/Rollo1A", "frames": frames, "credit": "c"}))
    roll("ok", ["f1.png", "f0.png"])
    roll("slash", ["../samples/b.png"])
    roll("missing", ["f0.png", "f9.png"])
    (a / "rolls" / "norollfile").mkdir()
    rolls = app.list_rolls(loc)
    assert [r["name"] for r in rolls] == ["ok"] and rolls[0]["frames"] == ["f1.png", "f0.png"] and rolls[0]["profile"] == "raw_fabrid/Rollo1A"


def test_shipped_samples_are_listed_with_attribution():
    s = app.list_samples(app.locations())
    assert [e["file"] for e in s] == ["synthetic_placeholder.png"] and s[0]["synthetic"] and "ynthetic" in s[0]["credit"]
    assert app.read_image(s[0]["path"].read_bytes()).size == (512, 256)


# ----------------------------------------------------------------------------- bundle download

@pytest.fixture
def hub(tmp_path):
    """A local HTTP server with the layout of the Hugging Face resolve URLs."""
    served = tmp_path / "hub"
    b = served / "user" / "repo" / "resolve" / "main" / "raw_fabrid" / "RolloA" / "patchcore"
    b.mkdir(parents=True)
    for f in app.BUNDLE_FILES:
        (b / f).write_bytes(f"content of {f}".encode())
    hits = []

    class H(SimpleHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            super().do_GET()

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), partial(H, directory=str(served)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield types.SimpleNamespace(url=f"http://127.0.0.1:{srv.server_port}", hits=hits)
    srv.shutdown()


def test_ensure_bundle_downloads_once(tmp_path, hub):
    loc = app.Locations(tmp_path / "bundles", tmp_path, tmp_path, hub.url, "user/repo")
    d = app.ensure_bundle(loc, "raw_fabrid/RolloA", "patchcore")
    assert d == tmp_path / "bundles" / "raw_fabrid" / "RolloA" / "patchcore" and len(hub.hits) == 3
    assert (d / "model.pt").read_bytes() == b"content of model.pt"
    assert app.ensure_bundle(loc, "raw_fabrid/RolloA", "patchcore") == d and len(hub.hits) == 3   # served from the cache
    assert list(d.parent.glob(".download-*")) == []


def test_ensure_bundle_replaces_incomplete_bundle(tmp_path, hub):
    loc = app.Locations(tmp_path / "bundles", tmp_path, tmp_path, hub.url, "user/repo")
    broken = tmp_path / "bundles" / "raw_fabrid" / "RolloA" / "patchcore"
    broken.mkdir(parents=True)
    (broken / "model.pt").write_bytes(b"half a bundle")
    d = app.ensure_bundle(loc, "raw_fabrid/RolloA", "patchcore")
    assert all((d / f).is_file() for f in app.BUNDLE_FILES) and (d / "model.pt").read_bytes() == b"content of model.pt"


def test_ensure_bundle_errors(tmp_path, hub, monkeypatch):
    mk = lambda repo, url=hub.url: app.Locations(tmp_path / "bundles", tmp_path, tmp_path, url, repo)  # noqa: E731
    for loc, pid, m, msg in [(mk(""), "raw_fabrid/RolloA", "patchcore", "no Hub repository"),
                             (mk("bad repo name"), "raw_fabrid/RolloA", "patchcore", "Invalid Hub repository"),
                             (mk("user/repo"), "raw_fabrid/../x", "patchcore", "Invalid bundle name"),
                             (mk("user/repo"), "raw_fabrid/RolloA", "a/b", "Invalid bundle name"),
                             (mk("user/repo"), "raw_fabrid", "patchcore", "Invalid bundle name"),
                             (mk("user/repo"), "raw_fabrid/RolloZ", "patchcore", "Could not download"),   # 404
                             (mk("user/repo", "http://127.0.0.1:9"), "raw_fabrid/RolloA", "patchcore", "Could not download")]:  # refused
        with pytest.raises(app.BundleError, match=msg):
            app.ensure_bundle(loc, pid, m)
    monkeypatch.setattr(app, "MAX_BUNDLE_FILE", 5)
    with pytest.raises(app.BundleError, match="larger than"):
        app.ensure_bundle(mk("user/repo"), "raw_fabrid/RolloA", "patchcore")
    b = tmp_path / "bundles"
    assert not list(b.rglob("*.pt")) and not any(p.name.startswith(".download-") for p in b.rglob("*"))  # no partial bundle left


# ----------------------------------------------------------------------------- pages (AppTest)

def run_app(monkeypatch, bundles, assets, results=None, **env):
    for k, v in {"LOOMGUARD_BUNDLES": bundles, "LOOMGUARD_ASSETS": assets, "LOOMGUARD_RESULTS": results or assets / "no_results",
                 "LOOMGUARD_HUB_REPO": "", **env}.items():
        monkeypatch.setenv(k, str(v))
    return AppTest.from_file(str(APP), default_timeout=240).run()


def test_starts_without_bundles_or_data(tmp_path, monkeypatch):
    at = run_app(monkeypatch, tmp_path / "b", tmp_path / "a")
    assert not at.exception and "No model bundle is available" in at.info[0].value
    for page, text in (("Roll stream", "No demo roll"), ("Results", "No result files")):
        at.radio(key="page").set_value(page).run()
        assert not at.exception and text in at.info[0].value
    at.radio(key="page").set_value("Inspect image").run()
    assert not at.exception and len(at.title) == 1


def test_shipped_assets_start_the_app(monkeypatch, tmp_path):
    """The repository's own assets/ (placeholder sample, profile registry without a Hub repository) and no bundles."""
    monkeypatch.setenv("LOOMGUARD_BUNDLES", str(tmp_path))
    monkeypatch.delenv("LOOMGUARD_ASSETS", raising=False)
    monkeypatch.delenv("LOOMGUARD_HUB_REPO", raising=False)
    at = AppTest.from_file(str(APP)).run()
    assert not at.exception and "No model bundle is available" in at.info[0].value


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """fit.py, evaluate.py on the synthetic RAW-FABRID tree for RolloA: a real bundle, the result JSON, dumped predictions."""
    root = tmp_path_factory.mktemp("trained")
    raw = conftest.raw_root.__wrapped__(root)
    cfg = {"dataset": "raw_fabrid", "root": str(raw), "regime": "per_fabric", "fabrics": ["RolloA"], "method": "patchcore", "seed": 0,
           "backbone_weights": None, "device": "cpu", "hist_bins": 2000, "grade_rolls": True, "px_per_mm": 4.0,
           "scale_assumed": False, "dump_predictions": True, "artifacts": str(root / "artifacts"), "results": str(root / "results"),
           "patchcore": {"max_tiles": 8, "coreset_ratio": 0.05, "proj_dim": 32, "batch_size": 4}}
    p = root / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    seeded = (fit, evaluate)
    saved = [m.Backbone for m in seeded]
    for m in seeded:
        m.Backbone = seeded_backbone
    try:
        fit.main(["--config", str(p)])
        res = evaluate.main(["--config", str(p)])["RolloA"]
    finally:
        for m, b in zip(seeded, saved):
            m.Backbone = b
    pred = root / "artifacts" / "predictions" / "raw_fabrid" / "per_fabric" / "patchcore" / "RolloA"
    scores = {Path(r.split(",")[0]).stem: r.split(",") for r in (pred / "scores.csv").read_text().strip().split("\n")[1:]}
    return types.SimpleNamespace(root=root, raw=raw, bundles=root / "artifacts", results=root / "results", res=res, pred=pred, scores=scores)


def assets_with(tmp_path, trained, samples=("RolloA_d000.png",), roll=None, profile="raw_fabrid/RolloA"):
    a = tmp_path / "assets"
    (a / "samples").mkdir(parents=True)
    meta = []
    for n in samples:
        (a / "samples" / n).write_bytes((trained.raw / "images" / n).read_bytes())
        meta.append({"file": n, "title": f"Frame {n}", "credit": "Synthetic test frame, no licence", "profile": profile})
    (a / "samples" / "samples.json").write_text(json.dumps(meta))
    if roll:
        d = a / "rolls" / "demo"
        d.mkdir(parents=True)
        for n in roll:
            (d / n).write_bytes((trained.raw / "images" / n).read_bytes())
        (d / "roll.json").write_text(json.dumps({"label": "Demo roll", "profile": "raw_fabrid/RolloA", "frames": list(roll), "credit": "synthetic"}))
    return a


def text_of(at):
    return " ".join([*(m.value for m in at.markdown), *(c.value for c in at.caption), *(s.value for s in at.subheader)])


def test_inspect_sample_reports_time_and_matches_evaluate(tmp_path, monkeypatch, trained):
    a = assets_with(tmp_path, trained, samples=("RolloA_d000.png", "RolloA_d001.png"))
    at = run_app(monkeypatch, trained.bundles, a)
    assert not at.exception and at.sidebar.selectbox(key="profile").value == "raw_fabrid/RolloA"
    for name in ("RolloA_d000.png", "RolloA_d001.png"):
        at.sidebar.selectbox(key="pick").select(name).run()
        assert not at.exception and not at.error
        m = {x.label: x.value for x in at.metric}
        score, flagged = trained.scores[Path(name).stem][3:5]
        assert float(m["Score"]) == pytest.approx(float(score), abs=5e-4) and m["Inspection time"].endswith(" s")
        assert at.subheader[0].value == ("Verdict: DEFECTIVE" if flagged == "1" else "Verdict: OK")
        # defect table = the defects measured from the mask that evaluate.py dumped for the same image
        insp = Inspector.load(trained.bundles / "raw_fabrid" / "RolloA" / "patchcore")
        mask = np.asarray(Image.open(trained.pred / f"{Path(name).stem}.png")) > 0
        want = [(round(d.length_mm, 1), d.points) for d in insp.measure(mask)[1]]
        got = [(r["length mm"], r["points"]) for _, r in at.dataframe[0].value.iterrows()] if at.dataframe else []
        assert got == want
        # the heatmap the app draws from is the stitched tile heatmaps, computed by hand
        scored = insp.score_image(Image.open(trained.raw / "images" / name))
        tiles, origins = tile(preprocess(Image.open(trained.raw / "images" / name)))
        assert torch.equal(scored.heat, stitch(insp.model.score(tiles)[1], origins, (256, 512))[0])


def test_upload_errors_in_app(tmp_path, monkeypatch, trained):
    at = run_app(monkeypatch, trained.bundles, assets_with(tmp_path, trained))
    good = png(np.random.default_rng(1).integers(0, 256, (300, 300), dtype=np.uint8))
    bad = {"notes.png": (b"this is text, not an image", "not an image the app can read"),
           "huge.png": (good + b"0" * app.MAX_BYTES, "the limit is 25 MB"),
           "broken.png": (good[: len(good) // 2], "corrupt or truncated"),
           "anim.png": (png(np.zeros((256, 256), np.uint8), "GIF"), "GIF images are not supported")}
    for name, (data, msg) in bad.items():
        at.sidebar.file_uploader(key="upload").set_value((name, data, "image/png")).run()
        assert not at.exception, name
        assert len(at.error) == 1 and msg in at.error[0].value, (name, [e.value for e in at.error])
        assert not at.metric and not at.dataframe and not at.image  # nothing shown for a rejected file


def test_upload_is_labelled_assumed_scale_and_grade_extrapolation(tmp_path, monkeypatch, trained):
    a = assets_with(tmp_path, trained)
    at = run_app(monkeypatch, trained.bundles, a)
    # a sample of the profile's own dataset: the bundle says the scale is stated by the dataset
    at.sidebar.selectbox(key="pick").select("RolloA_d000.png").run()
    assert any("stated by the dataset" in c.value for c in at.sidebar.caption)
    assert "Synthetic test frame" in text_of(at) and "extrapolation" in text_of(at)
    # an upload of unknown origin: the same bundle, but the scale is an assumption
    at.sidebar.file_uploader(key="upload").set_value(("mine.png", (trained.raw / "images" / "RolloA_d000.png").read_bytes(), "image/png")).run()
    assert not at.exception and any("assumed scale" in c.value for c in at.sidebar.caption)
    assert "assumed scale" in text_of(at) and "extrapolation" in text_of(at) and "roll-level" in text_of(at)


def test_sensitivity_slider_does_not_rerun_model(tmp_path, monkeypatch, trained):
    calls = []
    real = Inspector.score_image
    monkeypatch.setattr(Inspector, "score_image", lambda self, img: calls.append(1) or real(self, img))
    at = run_app(monkeypatch, trained.bundles, assets_with(tmp_path, trained))
    at.sidebar.selectbox(key="pick").select("RolloA_d000.png").run()
    tau = float(next(m.value for m in at.metric if m.label == "Threshold"))
    flagged = {}
    for s in (0.5, 1.0, 2.0):
        at.sidebar.slider(key="sens").set_value(s).run()
        m = {x.label: x.value for x in at.metric}
        assert float(m["Threshold"]) == pytest.approx(tau / s, abs=5e-4 * (1 + 1 / s))  # tau is read back rounded to 3 dp, so its error is scaled by 1/s
        flagged[s] = int(m["Tiles flagged"].split()[0])
    assert len(calls) == 1 and flagged[0.5] <= flagged[1.0] <= flagged[2.0]


def test_wrong_fabric_warning_hides_defects(tmp_path, monkeypatch, trained):
    """The bundle knows white noise. A smooth image is nothing like it, and so lights up everywhere."""
    at = run_app(monkeypatch, trained.bundles, assets_with(tmp_path, trained))
    y, x = np.mgrid[0:512, 0:1024]
    smooth = png((128 + 100 * np.sin(x / 90.0) * np.cos(y / 70.0)).astype(np.uint8))
    at.sidebar.file_uploader(key="upload").set_value(("other_fabric.png", smooth, "image/png")).run()
    assert not at.exception
    assert [w.value for w in at.warning if "Unknown fabric" in w.value] and "RAW-FABRID" not in " ".join(w.value for w in at.warning)
    assert not at.dataframe and not any(s.value.startswith("Verdict") for s in at.subheader)
    assert all("Points for this image" not in m.value and "Grade for this single frame" not in m.value for m in at.markdown)
    # an image like the training fabric does not trigger it
    noise = png(np.random.default_rng(9).integers(0, 256, (512, 1024), dtype=np.uint8))
    at.sidebar.file_uploader(key="upload").set_value(("normal.png", noise, "image/png")).run()
    assert not at.exception and not any("Unknown fabric" in w.value for w in at.warning)
    assert any(s.value.startswith("Verdict") for s in at.subheader)


def test_roll_stream_matches_evaluate(tmp_path, monkeypatch, trained):
    frames = sorted(p.name for p in (trained.raw / "masks").glob("RolloA_d*.png"))
    a = assets_with(tmp_path, trained, roll=frames)
    at = run_app(monkeypatch, trained.bundles, a)
    at.radio(key="page").set_value("Roll stream").run()
    assert not at.exception and at.sidebar.selectbox(key="roll").value == "demo"
    at.sidebar.slider(key="delay").set_value(0.0)
    at.button(key="play").click().run()
    assert not at.exception and not at.error
    want = trained.res["rolls"]["RolloA"]
    assert at.subheader[-1].value == f"Final grade: {want['grade_pred']}"
    final = [m.value for m in at.markdown if "roll grade, not an extrapolation" in m.value]
    assert len(final) == 1 and f"{len(frames)} frames, {want['points_pred']} points, {want['points_per_100_pred']:.1f} points per 100" in final[0]
    assert want["frames"] == len(frames)
    # the running line after the last frame is the roll's grade too
    assert any(f"**{want['points_per_100_pred']:.1f}** points per 100 sq yd" in m.value for m in at.markdown)
    assert len(at.dataframe) == 1 and set(at.dataframe[0].value.columns) == {"frame", "length mm", "x", "y", "points"}
    assert len(at.dataframe[0].value) == want["defects_pred"]


def test_results_page(tmp_path, monkeypatch):
    res = tmp_path / "results"
    for method, vals in (("patchcore", (0.9, 0.8)), ("rd4ad", (0.95, 0.7))):
        for fabric, d in (("Rollo1A", 0.0), ("Rollo2A", 0.02)):
            p = res / "raw_fabrid" / "per_fabric" / method / f"{fabric}.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"dataset": "raw_fabrid", "regime": "per_fabric", "method": method, "fabric": fabric,
                                     "tile_auroc": vals[0] - d, "aupro_30": vals[1] - d, "tile_recall": 0.5, "false_alarms_per_1000": 12.0}))
    at = run_app(monkeypatch, tmp_path / "b", tmp_path / "a", results=res)
    at.radio(key="page").set_value("Results").run()
    assert not at.exception and at.subheader[0].value == "raw_fabrid, per_fabric"
    df = at.dataframe[0].value.set_index("fabric")
    assert df.loc["Rollo1A", "patchcore tile_auroc"] == 0.9 and df.loc["Rollo2A", "rd4ad aupro_30"] == pytest.approx(0.68)
    assert df.loc["mean", "patchcore tile_auroc"] == pytest.approx(0.89) and df.loc["mean", "rd4ad FA/1000"] == 12.0
    assert len(at.get("vega_lite_chart")) == 1
    assert any("have no result file yet" in c.value for c in at.caption)


# ----------------------------------------------------------------------------- repository files

def test_pyproject_declares_app_extra():
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    assert any(d.startswith("streamlit>=") for d in extras["app"])
    assert "streamlit" not in " ".join(tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"])


def test_deploy_files():
    req = (ROOT / "deploy" / "requirements.txt").read_text().split("\n")
    assert "--extra-index-url https://download.pytorch.org/whl/cpu" in req and "torch" in req and ".[app]" in req
    front = (ROOT / "deploy" / "README_space.md").read_text().split("---")[1]
    meta = yaml.safe_load(front)
    assert meta["sdk"] == "streamlit" and meta["app_file"] == "app/streamlit_app.py"
    assert (ROOT / meta["app_file"]).is_file()
    cfg = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text())
    assert cfg["server"]["maxUploadSize"] * 2**20 > app.MAX_BYTES
    reg = json.loads((ROOT / "assets" / "profiles.json").read_text())
    assert reg["profiles"] and all(set(p) >= {"id", "label", "methods"} for p in reg["profiles"])


def test_app_has_no_pipeline_code():
    """CLAUDE.md rule 5: tiling, scoring and post-processing stay in the Inspector."""
    tree = ast.parse(APP.read_text())
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {a.name for n in ast.walk(tree)
                                                                                     if isinstance(n, ast.Import) for a in n.names}
    assert "loomguard.pipeline" in imported
    forbidden = {"loomguard.data.tiling", "loomguard.models", "loomguard.models.backbone", "loomguard.models.patchcore",
                 "loomguard.models.rd4ad", "loomguard.inspection.postprocess", "loomguard.eval.calibrate", "loomguard.eval.metrics"}
    assert not imported & forbidden
