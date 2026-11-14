"""Failure-case panels of one fitted bundle: the defective test images it detects worst and the images with the most
false-alarm pixels. Each PNG shows image | ground truth | heatmap | predicted mask. Selection is for the report only.

    python scripts/failure_cases.py --config configs/raw_fabrid_per_fabric_patchcore.yaml --profile Rollo4A [--n 3] [--out DIR]

Commit only panels of RAW-FABRID (CC BY 4.0); MVTec, ZJU-Leaper and AITEX images must not be committed.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))  # evaluate.py sits next to this file
import evaluate  # noqa: E402

from loomguard.data.manifest import select  # noqa: E402
from loomguard.models.backbone import Backbone  # noqa: E402
from loomguard.pipeline import Inspector  # noqa: E402
from loomguard.runs import build_rows, bundle_dir, load_config, pick_device, profiles  # noqa: E402

PANEL = 448  # panel width in pixels


def render(img: Image.Image, gt: np.ndarray, heat: np.ndarray, mask: np.ndarray, tau: float) -> Image.Image:
    """Four panels side by side; the heatmap is red above 2 tau, blue at 0."""
    v = np.clip(heat / (2 * tau), 0, 1)[..., None]
    panels = [np.asarray(img.convert("RGB")), np.repeat(gt[..., None] * 255, 3, 2).astype(np.uint8),
              (np.concatenate([v, 0 * v, 1 - v], 2) * 255).astype(np.uint8), np.repeat(mask[..., None] * 255, 3, 2).astype(np.uint8)]
    scale = min(1.0, PANEL / img.width)
    size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
    out = Image.new("RGB", (4 * size[0] + 12, size[1]), "white")
    for i, p in enumerate(panels):
        out.paste(Image.fromarray(p).resize(size, Image.NEAREST), (i * (size[0] + 4), 0))
    return out


def main(argv: list[str] | None = None) -> list[Path]:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out", default="artifacts/failure_cases")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    prows = profiles(build_rows(cfg), cfg)[a.profile]
    inspector = Inspector.load(bundle_dir(cfg, a.profile), pick_device(cfg), Backbone(cfg["backbone_weights"]))
    cases = []  # (parent, dice, false-alarm pixels, image path, gt, heat, mask)
    for g, scored, gt in evaluate._images(inspector, select(prows, split="test"), cfg):
        dec = inspector.decide(scored)
        m = dec.mask
        dice = 2 * (m & gt).sum() / (m.sum() + gt.sum()) if gt.any() else None
        cases.append((g[0].parent, dice, int((m & ~gt).sum()), g[0].image, gt, scored.heat.numpy(), m))
    picks = {"miss": sorted((c for c in cases if c[1] is not None), key=lambda c: (c[1], c[0]))[:a.n],
             "false_alarm": sorted((c for c in cases if c[2] > 0), key=lambda c: (-c[2], c[0]))[:a.n]}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for kind, chosen in picks.items():
        for k, (parent, _, _, image, gt, heat, mask) in enumerate(chosen):
            with Image.open(Path(cfg["root"]) / image) as im:
                if cfg["resize"]:
                    im = im.resize((cfg["resize"],) * 2, Image.BILINEAR)
                panel = render(im, gt, heat, mask, inspector.tau)
            path = out / f"{cfg['dataset']}_{cfg['regime']}_{cfg['method']}{'_' + cfg['tag'] if cfg['tag'] else ''}_{a.profile}_{kind}{k}.png"
            panel.save(path)
            written.append(path)
            print(f"{kind} {k}: {parent} -> {path}")
    return written


if __name__ == "__main__":
    main()
