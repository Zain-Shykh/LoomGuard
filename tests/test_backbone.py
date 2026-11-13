import pytest
import torch

from loomguard.models.backbone import Backbone, gaussian_blur


def test_shapes_and_frozen():
    b = Backbone(weights=None)
    outs = b(torch.randn(2, 3, 256, 256))
    assert [tuple(o.shape) for o in outs] == [(2, 256, 64, 64), (2, 512, 32, 32), (2, 1024, 16, 16)]
    assert not any(p.requires_grad for p in b.parameters())
    assert not any(o.requires_grad for o in outs)
    assert sum(p.numel() for p in b.parameters()) / 1e6 == pytest.approx(24.86, abs=0.01)
    b.train()
    assert not any(m.training for m in b.modules())  # BatchNorm stays in eval mode


def test_unknown_weights_name():
    with pytest.raises(KeyError):
        Backbone(weights="NOT_A_WEIGHT")


def test_gaussian_blur():
    assert torch.allclose(gaussian_blur(torch.full((1, 1, 40, 40), 3.0)), torch.full((1, 1, 40, 40), 3.0), atol=1e-5)
    x = torch.zeros(1, 1, 61, 61)
    x[0, 0, 30, 30] = 1.0
    y = gaussian_blur(x)
    assert y.shape == x.shape and y.sum().item() == pytest.approx(1.0, abs=1e-5)  # kernel sums to one
    assert y.argmax() == 30 * 61 + 30 and torch.allclose(y, y.flip(2), atol=1e-7) and torch.allclose(y, y.flip(3), atol=1e-7)
    # sigma 4: the centre value of a 2-D Gaussian is 1 / (2 pi sigma^2)
    assert y[0, 0, 30, 30].item() == pytest.approx(1 / (2 * torch.pi * 16), rel=0.02)
