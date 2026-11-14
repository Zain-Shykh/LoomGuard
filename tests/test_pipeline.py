"""Inspector on a deterministic fake model, and a bundle round trip with a tiny random PatchCore."""
import json

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from loomguard.data.dataset import preprocess
from loomguard.data.tiling import stitch, tile
from loomguard.inspection.grading import grade_roll
from loomguard.inspection.postprocess import clean_mask, measure
from loomguard.models.backbone import Backbone
from loomguard.models.patchcore import PatchCore
from loomguard.pipeline import Inspector, Scored

TAU = 0.0  # the fake heatmap is the normalised first channel: background about -2.1, bright pixels about +2.2


class FakeModel:
    """Deterministic: heatmap = first channel of the tile, tile score = half its maximum."""
    device = torch.device("cpu")

    def score(self, x):
        heat = x[:, :1].clone()
        return heat.amax((1, 2, 3)) * 0.5, heat


def frame(h, w, boxes):
    a = np.zeros((h, w), np.uint8)
    for y0, y1, x0, x1 in boxes:
        a[y0:y1, x0:x1] = 255
    return Image.fromarray(a)


def test_inspector_matches_manual_pipeline():
    img = frame(1024, 1792, [(300, 308, 250, 350), (700, 703, 100, 1100)])  # first crosses a tile border; second spans 5 tiles
    insp = Inspector(FakeModel(), TAU, px_per_mm=4.0, batch_size=5)
    got = insp.inspect(img)
    # by hand: tile, score each tile, stitch, threshold, clean, measure
    tiles, origins = tile(preprocess(img))
    assert len(origins) == 28
    heat = stitch(torch.cat([FakeModel().score(t[None])[1] for t in tiles]), origins, (1024, 1792))[0]
    assert torch.equal(got.heat, heat)
    manual = measure(clean_mask((heat > TAU).numpy()), 4.0)
    assert [(d.bbox, d.area_px, d.length_px) for d in got.defects] == [(d.bbox, d.area_px, d.length_px) for d in manual]
    # and the numbers the synthetic image was built with
    assert [d.bbox for d in got.defects] == [(250, 300, 350, 308), (100, 700, 1100, 703)]
    assert [d.length_mm for d in got.defects] == [25.0, 250.0] and [d.points for d in got.defects] == [1, 4] and got.points == 5
    assert got.flagged and got.flagged_tiles.sum() == 2 + 5 and got.score == pytest.approx(float(heat.max()))
    assert got.grade.points == 5 and got.grade.grade == "Reject"  # one frame treated as a roll: an extrapolation


def test_tile_flag_matches_mask():
    """Decision and mask share one map: a tile is flagged iff one of its pixels is above tau."""
    torch.manual_seed(0)

    class Noise(FakeModel):
        def score(self, x):
            heat = torch.rand(len(x), 1, 256, 256)
            return heat.amax((1, 2, 3)), heat

    insp = Inspector(Noise(), 0.9999)
    s = insp.score_image(frame(512, 768, []))
    for tau in (0.9999, 0.99999, 2.0, 0.0):
        d = insp.decide(s, tau)
        for i, (y, x) in enumerate(s.origins):
            assert d.flagged_tiles[i] == bool((s.heat[y:y + 256, x:x + 256] > tau).any())
    assert not insp.decide(s, 2.0).flagged


def test_patchcore_peaks_are_blurred_max_and_score_unchanged():
    torch.manual_seed(0)
    pc = PatchCore(Backbone(None), seed=0)
    pc.bank = torch.randn(40, 1536).half()
    insp = Inspector(pc, 0.0)
    img = Image.fromarray(np.random.default_rng(0).integers(0, 256, (256, 512), dtype=np.uint8))
    s = insp.score_image(img)
    tiles, _ = tile(preprocess(img))
    score, heat = pc.score(tiles)
    assert torch.allclose(s.scores, score)                # the published score (unblurred grid max) is untouched
    assert torch.allclose(s.peaks, heat.amax((1, 2, 3)))  # the operating score is the heatmap max
    tau = float(s.peaks.median())
    d = insp.decide(s, tau)
    assert [bool(f) for f in d.flagged_tiles] == [bool((heat[i] > tau).any()) for i in range(2)]


def test_tau_monotone():
    img = frame(256, 512, [(10, 20, 10, 100), (100, 103, 300, 420)])
    insp = Inspector(FakeModel(), TAU)
    s = insp.score_image(img)
    prev = None
    for tau in (-3.0, 0.0, 2.3):
        d = insp.decide(s, tau)
        assert prev is None or not (d.mask & ~prev.mask).any()  # raising tau never adds a pixel
        assert prev is None or d.flagged_tiles.sum() <= prev.flagged_tiles.sum()
        prev = d
    assert not prev.flagged and prev.defects == [] and prev.points == 0 and prev.grade.grade == "A"


def test_decide_without_model():
    heat = torch.zeros(64, 64)
    heat[10:20, 10:40] = 1.0
    s = Scored(heat, torch.tensor([0.5]), torch.tensor([1.0]), [(0, 0)], (64, 64))
    d = Inspector(None, 0.5, px_per_mm=1.0).decide(s)
    assert len(d.defects) == 1 and d.defects[0].length_px == 30 and d.flagged and d.score == 1.0
    assert Inspector(None, 0.5).decide(s, 1.5).flagged is False  # the sensitivity slider hook


def test_resize_policy():
    insp = Inspector(FakeModel(), TAU, resize=256)
    s = insp.score_image(Image.fromarray(np.zeros((100, 80, 3), np.uint8)))
    assert s.heat.shape == (256, 256) and s.origins == [(0, 0)] and s.size == (256, 256)


def test_roll_grade():
    insp = Inspector(FakeModel(), TAU, px_per_mm=4.0)
    frames = [frame(256, 512, [(50, 53, 10, 210)]), frame(256, 512, [(50, 53, 10, 410), (150, 153, 10, 90)]), frame(256, 512, [])]
    results, roll = insp.inspect_roll(frames)
    # 200 px = 50 mm: 1 point; 400 px = 100 mm: 2 points; 80 px = 20 mm: 1 point
    assert [r.points for r in results] == [1, 3, 0] and roll.points == 4
    assert roll == grade_roll(4, 3 * 256 * 512 / 16)
    assert roll.area_sq_yd == pytest.approx(3 * 256 * 512 / 16 / 914.4 ** 2)


def test_bundle_round_trip(tmp_path, monkeypatch):
    def backbone():
        torch.manual_seed(0)
        return Backbone(None)

    pc = PatchCore(backbone(), seed=0)
    pc.bank = torch.randn(40, 1536, generator=torch.Generator().manual_seed(1)).half()
    b = tmp_path / "bundle"
    b.mkdir()
    pc.save(b / "model.pt")
    (b / "config.yaml").write_text(yaml.safe_dump({"method": "patchcore", "backbone_weights": None, "px_per_mm": 4.0,
                                                    "scale_assumed": False, "resize": None, "patchcore": {"batch_size": 1}}))
    (b / "calibration.json").write_text(json.dumps({"tau": 1.25}))
    calls, real = [], torch.load
    monkeypatch.setattr(torch, "load", lambda *a, **k: calls.append(k) or real(*a, **k))
    loaded = Inspector.load(b, backbone=backbone())
    assert calls and all(k.get("weights_only") is True for k in calls)
    assert (loaded.tau, loaded.px_per_mm, loaded.scale_assumed, loaded.batch_size) == (1.25, 4.0, False, 1)
    img = Image.fromarray(np.random.default_rng(0).integers(0, 256, (256, 512), dtype=np.uint8))
    ref = Inspector(pc, 1.25, 4.0, False, batch_size=1).inspect(img)
    got = loaded.inspect(img)
    assert torch.equal(got.heat, ref.heat) and got.score == ref.score and got.flagged == ref.flagged
