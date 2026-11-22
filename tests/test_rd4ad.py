"""RD4AD with random-weight teachers. CUDA precision tests run when a GPU is available.

The 300-iteration overfit test (256 x 256) is marked slow (pytest -m slow).
"""
import copy
import math

import pytest
import torch

from loomguard.models.backbone import Backbone
from loomguard.models.rd4ad import RD4AD, Student, anomaly_map, cosine_loss


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def tiles(n, size=256, seed=0):
    return torch.randn(n, 3, size, size, generator=torch.Generator().manual_seed(seed)) * 0.5


def millions(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6


def test_shapes():
    bb, s = Backbone(None), Student().eval()
    enc = bb(tiles(1))
    assert [tuple(e.shape) for e in enc] == [(1, 256, 64, 64), (1, 512, 32, 32), (1, 1024, 16, 16)]
    with torch.no_grad():
        assert s.bottleneck(*enc).shape == (1, 2048, 8, 8)
        dec = s(*enc)
    assert [tuple(d.shape) for d in dec] == [(1, 256, 64, 64), (1, 512, 32, 32), (1, 1024, 16, 16)]


def test_parameter_count():
    s = Student()
    b = s.bottleneck
    assert millions(s) == pytest.approx(83.80, rel=0.01)
    assert millions(b.fuse1) + millions(b.fuse2) == pytest.approx(10.62, rel=0.01)
    assert millions(b.embed) == pytest.approx(48.26, rel=0.01)
    assert millions(s.decoder) == pytest.approx(24.92, rel=0.01)
    assert millions(Backbone(None)) == 0


def test_teacher_frozen_through_training():
    rd = RD4AD(Backbone(None))
    before_t, before_s = copy.deepcopy(rd.backbone.state_dict()), copy.deepcopy(rd.student.state_dict())
    rd.student.train()
    rd.backbone.train()  # Backbone keeps eval() even when asked otherwise
    rd.fit(iter([tiles(2, 32)] * 3), 3, log_every=3)
    assert not rd.backbone.training and all(not m.training for m in rd.backbone.modules())
    assert not any(p.requires_grad for p in rd.backbone.parameters())
    assert all(torch.equal(v, before_t[k]) for k, v in rd.backbone.state_dict().items())  # weights and BN statistics
    assert any(not torch.equal(v, before_s[k]) for k, v in rd.student.state_dict().items())


@pytest.mark.parametrize("device, allow_tf32", [
    ("cpu", True),
    pytest.param("cuda", True, marks=requires_cuda),
    pytest.param("cuda", False, marks=requires_cuda),
])
def test_teacher_and_student_precision(device, allow_tf32):
    rd = RD4AD(Backbone(None), device)
    x = tiles(2, 32)
    teacher_features, student_dtypes = [], []
    cuda = device == "cuda"

    def record_teacher(module, args, features):
        assert not torch.is_autocast_enabled(device)
        assert torch.backends.cudnn.allow_tf32 == (allow_tf32 and not cuda)
        assert args[0].dtype == next(module.parameters()).dtype == torch.float32
        assert all(f.dtype == torch.float32 and not f.requires_grad for f in features)
        teacher_features.append(features)

    def record_student(module, args, features):
        assert torch.backends.cudnn.allow_tf32 == allow_tf32
        assert torch.is_autocast_enabled(device) == (cuda and module.training)
        student_dtypes.append(tuple(f.dtype for f in features))

    rd.backbone.register_forward_hook(record_teacher)
    rd.student.register_forward_hook(record_student)
    with torch.backends.cudnn.flags(enabled=True, allow_tf32=allow_tf32):
        with torch.autocast(device, dtype=torch.float16 if cuda else torch.bfloat16):
            loss = rd._loss(x, amp=cuda)
            assert torch.is_autocast_enabled(device)
        assert loss.dtype == torch.float32 and torch.isfinite(loss)
        scores, heat = rd.score(x)
        assert torch.backends.cudnn.allow_tf32 == allow_tf32

    assert student_dtypes == [(torch.float16 if cuda else torch.float32,) * 3, (torch.float32,) * 3]
    assert len(teacher_features) == 2
    for train_feature, score_feature in zip(*teacher_features):
        torch.testing.assert_close(train_feature, score_feature)
    assert scores.shape == (2,) and heat.shape == (2, 1, 32, 32)
    assert scores.dtype == heat.dtype == torch.float32
    torch.testing.assert_close(scores, heat.amax((1, 2, 3)))


@requires_cuda
def test_teacher_restores_tf32_after_error():
    rd = RD4AD(Backbone(None), "cuda")

    def fail_forward(*args):
        assert not torch.backends.cudnn.allow_tf32
        raise RuntimeError("test teacher failure")

    rd.backbone.register_forward_pre_hook(fail_forward)
    with torch.backends.cudnn.flags(enabled=True, allow_tf32=True):
        with pytest.raises(RuntimeError, match="test teacher failure"):
            rd.score(tiles(1, 32))
        assert torch.backends.cudnn.allow_tf32


def test_cosine_loss_hand_values():
    a = [torch.randn(3, c, h, h) for c, h in ((4, 8), (6, 4), (8, 2))]
    assert float(cosine_loss(a, a)) == pytest.approx(0, abs=1e-6)
    assert float(cosine_loss(a, [-x for x in a])) == pytest.approx(6, abs=1e-5)  # 2 per scale
    e = [torch.tensor([[[[1.0]], [[0.0]]]]).expand(2, 2, 1, 1)] * 3
    d = [torch.tensor([[[[0.0]], [[1.0]]]]).expand(2, 2, 1, 1)] * 3
    assert float(cosine_loss(e, d)) == pytest.approx(3.0)  # orthogonal: 1 per scale
    # flattened per image: image 0 identical, image 1 opposite, over a 2 x 2 map -> mean (0 + 2) / 2 = 1 per scale
    x = torch.randn(2, 3, 2, 2)
    y = torch.stack([x[0], -x[1]])
    assert float(cosine_loss([x] * 3, [y] * 3)) == pytest.approx(3.0, abs=1e-5)
    # one vector per image, not per position: the maps differ in sign at one position only
    z = x.clone()
    z[0, :, 0, 0] *= -1
    flat = 1 - torch.nn.functional.cosine_similarity(x.flatten(1), z.flatten(1), dim=1).mean()
    assert float(cosine_loss([x], [z])) == pytest.approx(float(flat), abs=1e-6)


def test_anomaly_map_and_score():
    g = torch.Generator().manual_seed(1)
    enc = [torch.randn(2, c, h, h, generator=g) for c, h in ((256, 64), (512, 32), (1024, 16))]
    dec = [e.clone() for e in enc]
    m = anomaly_map(enc, dec)
    assert m.shape == (2, 1, 256, 256) and m.dtype == torch.float32 and float(m.abs().max()) < 1e-5
    for e, d in zip(enc, dec):  # break the match at one position (image 0, top-left quarter) at every scale
        d[0, :, : e.shape[-2] // 4, : e.shape[-1] // 4] *= -1
    m = anomaly_map(enc, dec)
    peak = divmod(int(m[0, 0].argmax()), 256)
    assert peak[0] < 64 and peak[1] < 64 and float(m[0].max()) > 10 * float(m[1].max()) + 0.5
    # the tile score of RD4AD.score is the max of that heatmap
    rd = RD4AD(Backbone(None))
    score, heat = rd.score(tiles(1, 64))
    assert heat.shape == (1, 1, 64, 64) and float(score[0]) == float(heat.max())


def test_fit_budget_and_logging():
    rd = RD4AD(Backbone(None))
    used = []

    def stream(n):
        for i in range(n):
            used.append(i)
            yield tiles(2, 32, seed=i)

    val = [tiles(2, 32, seed=99)]
    out = rd.fit(stream(10), 4, val=val, val_every=2, log_every=2)  # more batches than the budget: the rest is never consumed
    assert len(used) == 4 and out["iterations"] == 4
    assert [c[0] for c in out["loss_curve"]] == [2, 4] and all(c[2] is not None for c in out["loss_curve"])
    assert out["final_train_loss"] == out["loss_curve"][-1][1] and out["final_val_loss"] == out["loss_curve"][-1][2]
    assert out["iters_per_second"] > 0 and not rd.student.training  # eval mode afterwards, last state kept
    assert out["skipped_steps"] == 0  # CPU: scaler disabled, no step is ever skipped
    with pytest.raises(ValueError):
        rd.fit(stream(1), 2)
    # no validation tiles: nothing is logged for them
    assert RD4AD(Backbone(None)).fit(iter([tiles(2, 32)] * 2), 2, log_every=1)["final_val_loss"] is None


def test_loss_falls_quickly():
    rd = RD4AD(Backbone(None))
    x = tiles(4, 32)
    out = rd.fit(iter([x] * 15), 15, log_every=5)
    first = out["loss_curve"][0][1]
    assert out["final_train_loss"] < 0.5 * first


@requires_cuda
def test_fp16_grad_scaler_skips_overflow_and_recovers():
    torch.manual_seed(0)
    rd = RD4AD(Backbone(None), "cuda")
    param = next(rd.student.parameters())
    initial = param.detach().clone()
    x = weave(8, 256)  # the overfit fixture has finite scaled gradients without the injected overflow

    def batches():
        with param.register_hook(lambda grad: torch.full_like(grad, float("inf"))):
            yield x
        torch.testing.assert_close(param, initial, rtol=0, atol=0)
        yield x

    out = rd.fit(batches(), 2, log_every=1)
    assert out["iterations"] == 2 and out["skipped_steps"] == 1
    assert all(math.isfinite(loss) for _, loss, _ in out["loss_curve"])
    assert torch.isfinite(param).all() and not torch.equal(param, initial)


def weave(n, size, seed=0):
    """Fabric-like normal tiles: a 4-pixel weave with random phases, a slow drift and light noise."""
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(size).float()
    yy, xx = torch.meshgrid(t, t, indexing="ij")
    out = []
    for i in range(n):
        ph = torch.rand(2, generator=g) * 6.28
        w = torch.sin(2 * math.pi * xx / 4 + ph[0]) * torch.sin(2 * math.pi * yy / 4 + ph[1]) + 0.3 * torch.sin(2 * math.pi * xx / 16)
        out.append(torch.stack([w, 0.8 * w, 0.6 * w + 0.1 * i]) + 0.05 * torch.randn(3, size, size, generator=g))
    return torch.stack(out)


@pytest.mark.slow
def test_overfit_eight_tiles():
    """A3. About 40 minutes on the laptop CPU (loss 0.048 at iteration 250, 0.037 at 300); seconds on the GPU."""
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    rd = RD4AD(Backbone(None), device)
    out = rd.fit(iter([weave(8, 256)] * 300), 300, log_every=25)
    print([round(c[1], 4) for c in out["loss_curve"]])
    print(f'final_train_loss={out["final_train_loss"]:.8f}, skipped_steps={out["skipped_steps"]}, '
          f'iters_per_second={out["iters_per_second"]:.2f}')
    if device == "cuda":
        print(f'peak_allocated_mib={torch.cuda.max_memory_allocated() / 2**20:.1f}, '
              f'peak_reserved_mib={torch.cuda.max_memory_reserved() / 2**20:.1f}')
    assert out["final_train_loss"] < 0.05


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=requires_cuda)])
def test_save_load_float16(tmp_path, device):
    rd = RD4AD(Backbone(None), device, seed=3)
    rd.fit(iter([tiles(2, 32)] * 2), 2, log_every=2)  # moves BN running statistics away from their initial values
    p = tmp_path / "model.pt"
    rd.save(p)
    sd = torch.load(p, weights_only=True)
    assert set(sd) == set(rd.student.state_dict())
    assert all(v.dtype == torch.float16 for v in sd.values() if v.is_floating_point())
    assert any(v.dtype == torch.int64 for v in sd.values())  # num_batches_tracked is kept as an integer
    rd2 = RD4AD.load(p, rd.backbone, device)  # same teacher: random-weight backbones differ between calls
    assert not rd2.student.training and all(q.dtype == torch.float32 for q in rd2.student.parameters())
    for q in rd.student.parameters():  # the reference: the same student rounded to float16 and back
        q.data = q.data.half().float()
    for k, b in rd.student.named_buffers():
        if b.is_floating_point():
            b.data = b.data.half().float()
    x = tiles(2, 64, seed=5)
    s1, h1 = rd.score(x)
    s2, h2 = rd2.score(x)
    assert torch.allclose(h1, h2, atol=1e-5) and torch.allclose(s1, s2, atol=1e-5)

