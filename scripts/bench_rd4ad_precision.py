"""Throughput probe: RD4AD training speed under bf16 autocast, fp16 autocast + GradScaler, and fp32.

Random input, untrained teacher weights, no dataset, produces NO result: speed, peak VRAM and a finite-loss check only.
Used to choose a precision for GPUs without native bfloat16 (T4, P100). Run:  python scripts/bench_rd4ad_precision.py
"""
import argparse
import json
import time

import torch

from loomguard.models.backbone import Backbone
from loomguard.models.rd4ad import RD4AD, cosine_loss


def bench(mode: str, iters: int, warmup: int, batch: int) -> dict:
    torch.backends.cudnn.benchmark = False
    rd = RD4AD(Backbone(weights=None), "cuda", seed=0)
    opt = torch.optim.Adam(rd.student.parameters(), lr=0.005, betas=(0.5, 0.999))
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[mode]
    scaler = torch.amp.GradScaler("cuda", enabled=mode == "fp16")
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(batch, 3, 256, 256, generator=gen).cuda()
    torch.cuda.reset_peak_memory_stats()
    rd.student.train()
    for i in range(warmup + iters):
        if i == warmup:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=dtype or torch.float32, enabled=dtype is not None):
            enc = rd.backbone(x)
            dec = rd.student(*enc)
        loss = cosine_loss(enc, dec)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()
    return {"mode": mode, "it_per_s": iters / (time.perf_counter() - t0), "final_loss": float(loss),
            "finite": bool(torch.isfinite(loss)), "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--modes", default="bf16,fp16,fp32")
    a = ap.parse_args()
    print(json.dumps({"gpu": torch.cuda.get_device_name(), "arch": torch.cuda.get_device_capability(),
                      "torch": torch.__version__, "native_bf16": torch.cuda.is_bf16_supported(including_emulation=False)}))
    for m in a.modes.split(","):
        try:
            print(json.dumps(bench(m, a.iters, a.warmup, a.batch)), flush=True)
        except Exception as e:  # a mode may be unsupported on old GPUs
            print(json.dumps({"mode": m, "error": f"{type(e).__name__}: {e}"}), flush=True)
        torch.cuda.empty_cache()
