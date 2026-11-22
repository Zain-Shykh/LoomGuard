"""Reverse Distillation (docs/ARCHITECTURE.md section 6), from the paper: Deng and Li, arXiv 2201.10703.

Teacher: the frozen Backbone. Student: bottleneck (OCBE) + decoder, trained to rebuild the teacher's three feature maps
from a 2048 x 8 x 8 embedding. Same fit/score/save/load surface as PatchCore, so scripts/evaluate.py treats both alike.
"""
import itertools
import time
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from loomguard.models.backbone import Backbone, gaussian_blur


def _conv_bn_relu(cin: int, cout: int, stride: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class _Block(nn.Module):
    """ResNet bottleneck block. up=False: encoder-style, a 3x3 conv (stride `stride`) in the middle. up=True: decoder-style,
    a 2x2 stride-2 transposed conv in the middle when stride == 2 (also on the shortcut), else a 3x3 conv."""

    def __init__(self, cin: int, width: int, cout: int, stride: int = 1, up: bool = False):
        super().__init__()
        if stride == 2 and up:
            mid, short = nn.ConvTranspose2d(width, width, 2, 2, bias=False), nn.ConvTranspose2d(cin, cout, 2, 2, bias=False)
        else:
            mid = nn.Conv2d(width, width, 3, stride, 1, bias=False)
            short = nn.Conv2d(cin, cout, 1, stride, bias=False)
        self.body = nn.Sequential(nn.Conv2d(cin, width, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
                                  mid, nn.BatchNorm2d(width), nn.ReLU(inplace=True),
                                  nn.Conv2d(width, cout, 1, bias=False), nn.BatchNorm2d(cout))
        self.short = nn.Sequential(short, nn.BatchNorm2d(cout)) if (stride != 1 or cin != cout) else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(self.body(x) + self.short(x))


class Bottleneck(nn.Module):
    """OCBE. (E1 B x 256 x 64 x 64, E2 B x 512 x 32 x 32, E3 B x 1024 x 16 x 16) -> embedding B x 2048 x 8 x 8."""

    def __init__(self):
        super().__init__()
        self.fuse1 = nn.Sequential(_conv_bn_relu(256, 512, 2), _conv_bn_relu(512, 1024, 2))  # E1 -> 1024 x 16 x 16
        self.fuse2 = _conv_bn_relu(512, 1024, 2)                                              # E2 -> 1024 x 16 x 16
        self.embed = nn.Sequential(_Block(3072, 1024, 2048, 2), _Block(2048, 1024, 2048), _Block(2048, 1024, 2048))

    def forward(self, e1: Tensor, e2: Tensor, e3: Tensor) -> Tensor:
        return self.embed(torch.cat([self.fuse1(e1), self.fuse2(e2), e3], 1))


class Decoder(nn.Module):
    """Mirror of layer3, layer2, layer1. B x 2048 x 8 x 8 -> (D1 B x 256 x 64 x 64, D2 B x 512 x 32 x 32, D3 B x 1024 x 16 x 16)."""

    def __init__(self):
        super().__init__()

        def stage(cin: int, width: int, cout: int, n: int) -> nn.Sequential:
            return nn.Sequential(_Block(cin, width, cout, 2, up=True), *[_Block(cout, width, cout, up=True) for _ in range(n - 1)])

        self.d3, self.d2, self.d1 = stage(2048, 512, 1024, 3), stage(1024, 256, 512, 4), stage(512, 128, 256, 6)

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        d3 = self.d3(z)
        d2 = self.d2(d3)
        return self.d1(d2), d2, d3


class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.bottleneck, self.decoder = Bottleneck(), Decoder()
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, e1: Tensor, e2: Tensor, e3: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.decoder(self.bottleneck(e1, e2, e3))


def cosine_loss(enc: Sequence[Tensor], dec: Sequence[Tensor]) -> Tensor:
    """Sum over the 3 scales of mean over the batch of 1 - cos(E_k, D_k), each map flattened to one vector. Scalar float32."""
    return sum((1 - F.cosine_similarity(e.flatten(1).float(), d.flatten(1).float(), dim=1)).mean() for e, d in zip(enc, dec))


def anomaly_map(enc: Sequence[Tensor], dec: Sequence[Tensor], size: int = 256) -> Tensor:
    """Per-position cosine distance at each scale, bilinearly upsampled to size x size and added, then Gaussian blur
    (sigma 4). Returns B x 1 x size x size float32."""
    m = sum(F.interpolate(1 - F.cosine_similarity(e.float(), d.float(), dim=1, eps=1e-8).unsqueeze(1), size=(size, size),
                          mode="bilinear", align_corners=False) for e, d in zip(enc, dec))
    return gaussian_blur(m, 4.0)


class RD4AD:
    def __init__(self, backbone: Backbone, device: str | torch.device = "cpu", seed: int = 0):
        self.device = torch.device(device)
        self.backbone = backbone.to(device=self.device, dtype=torch.float32).eval()  # frozen teacher, always in eval()
        torch.manual_seed(seed)
        self.student = Student().to(self.device)

    def _teacher_features(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x = x.to(device=self.device, dtype=torch.float32)
        with torch.autocast(self.device.type, enabled=False):
            if self.device.type != "cuda":
                return self.backbone(x)
            # Avoid TF32 rounding of the frozen targets while preserving the caller's cuDNN settings.
            previous_tf32 = torch.backends.cudnn.allow_tf32
            torch.backends.cudnn.allow_tf32 = False
            try:
                return self.backbone(x)
            finally:
                torch.backends.cudnn.allow_tf32 = previous_tf32

    def _loss(self, x: Tensor, amp: bool) -> Tensor:
        enc = self._teacher_features(x)
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=amp):  # student AMP on CUDA (D-60)
            dec = self.student(*enc)
        return cosine_loss(enc, dec)

    @torch.no_grad()
    def val_loss(self, batches: Sequence[Tensor]) -> float:
        self.student.eval()
        n = sum(len(b) for b in batches)
        out = sum(float(self._loss(b, False)) * len(b) for b in batches) / n
        self.student.train()
        return out

    def fit(self, batches: Iterable[Tensor], iterations: int, lr: float = 0.005, betas: tuple[float, float] = (0.5, 0.999),
            val: Sequence[Tensor] = (), val_every: int = 500, log_every: int = 100) -> dict:
        """Fixed budget of `iterations` Adam steps on batches (B x 3 x 256 x 256 normal crops); the last state is kept.
        `val` (optional) is a list of normal-tile batches whose loss is logged every `val_every` steps, only to spot
        divergence. Returns {iterations, seconds, iters_per_second, final_train_loss, final_val_loss, loss_curve, skipped_steps}.
        skipped_steps counts fp16 overflow steps the GradScaler skipped; they still count toward `iterations` (D-60)."""
        amp = self.device.type == "cuda"
        opt = torch.optim.Adam(self.student.parameters(), lr=lr, betas=betas)  # student only: the teacher is not in it
        scaler = torch.amp.GradScaler(self.device.type, enabled=amp)  # fp16 needs loss scaling (D-60); a no-op off CUDA
        self.student.train()
        curve, window, spent, n, skipped = [], [], 0.0, 0, 0
        for x in itertools.islice(batches, iterations):
            t0 = time.perf_counter()
            self.backbone.eval()
            loss = self._loss(x, amp)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scale = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            skipped += scaler.get_scale() < scale  # scale shrinks only when inf/NaN grads made the scaler skip opt.step()
            window.append(float(loss.detach()))  # float() syncs the device, so the timing below is honest
            spent += time.perf_counter() - t0
            n += 1
            if n % log_every == 0 or n == iterations:
                v = self.val_loss(val) if val and (n % val_every == 0 or n == iterations) else None
                curve.append([n, sum(window) / len(window), v])
                window = []
        if n != iterations:
            raise ValueError(f"batches ran out after {n} of {iterations} iterations")
        self.student.eval()
        vals = [c[2] for c in curve if c[2] is not None]
        return {"iterations": n, "seconds": spent, "iters_per_second": n / spent, "final_train_loss": curve[-1][1],
                "final_val_loss": vals[-1] if vals else None, "loss_curve": curve, "skipped_steps": int(skipped)}

    @torch.no_grad()
    def score(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """x: B x 3 x 256 x 256. Returns (tile score B = max of the heatmap, heatmap B x 1 x 256 x 256), float32."""
        self.student.eval()
        enc = self._teacher_features(x)
        heat = anomaly_map(enc, self.student(*enc), x.shape[-1])
        return heat.amax((1, 2, 3)), heat

    def save(self, path: Path) -> None:
        """Student state dict, floating-point tensors in float16 (168 MB)."""
        torch.save({k: v.half() if v.is_floating_point() else v for k, v in self.student.state_dict().items()}, path)

    @classmethod
    def load(cls, path: Path, backbone: Backbone, device: str | torch.device = "cpu", **kw) -> "RD4AD":
        rd = cls(backbone, device, **kw)
        rd.student.load_state_dict(torch.load(path, weights_only=True))  # copies float16 into the float32 parameters
        rd.student.eval()
        return rd
