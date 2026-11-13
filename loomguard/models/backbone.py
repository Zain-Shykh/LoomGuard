"""Frozen WideResNet-50-2 up to layer3 (docs/ARCHITECTURE.md section 4), shared by PatchCore and Reverse Distillation."""
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2


class Backbone(nn.Module):
    def __init__(self, weights: str | None = "IMAGENET1K_V1"):
        super().__init__()
        w = Wide_ResNet50_2_Weights[weights] if weights else None  # KeyError on an unknown name
        m = wide_resnet50_2(weights=w)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.layer1, self.layer2, self.layer3 = m.layer1, m.layer2, m.layer3  # layer4, avgpool, fc are dropped
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def train(self, mode: bool = True):
        return super().train(False)  # BatchNorm statistics stay fixed (CLAUDE.md rule 4)

    @torch.no_grad()
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        l1 = self.layer1(self.stem(x))
        l2 = self.layer2(l1)
        return l1, l2, self.layer3(l2)


def gaussian_blur(x: Tensor, sigma: float = 4.0) -> Tensor:
    """B x 1 x H x W -> same shape. Separable Gaussian, radius int(4 sigma + 0.5), reflect padding."""
    r = int(4 * sigma + 0.5)
    t = torch.arange(-r, r + 1, dtype=x.dtype, device=x.device)
    k = torch.exp(-t ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))
