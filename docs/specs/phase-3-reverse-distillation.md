# Phase 3: Reverse Distillation

| | |
|---|---|
| Phase | 3 |
| Status | Written |
| Spec written | 2026-11-13 |
| Written by | developer session, phase 3 |
| Depends on | Phases 1 and 2 (Code complete; their training-PC runbooks are queued and do not block this phase, decision D-58) |
| Plan entry | `docs/IMPLEMENTATION_PLAN.md`, Phase 3 |

## 1. Objective

An RD4AD implementation (bottleneck, mirrored decoder, cosine loss, anomaly map, training loop with a fixed iteration budget and last-checkpoint export in float16) wired into `scripts/fit.py` and `scripts/evaluate.py` through the config's `method` key, tested end to end on CPU with synthetic data. At the end the training-PC operator has configs and a runbook to reproduce the published MVTec AD texture numbers, measure VRAM and speed, and produce the first RAW-FABRID unified RD4AD results over 3 seeds.

## 2. Background

RD4AD (Deng and Li, arXiv 2201.10703) trains only a student: a bottleneck that squeezes the frozen teacher's three feature maps into a 2048 x 8 x 8 embedding, and a decoder that mirrors the teacher backwards and must rebuild the three maps. It sees normal fabric only; on a defect the teacher and the student disagree. Phase 2 supplied the frozen `Backbone`, `gaussian_blur`, the metrics, `runs.py` and the two scripts; Phase 1 the tiler, manifests and `TileDataset`.

Read:

- `docs/ARCHITECTURE.md` section 4 (backbone), section 6 (RD4AD: 6.1 network and parameter table, 6.2 training, 6.3 scoring), section 10 (bundle)
- `docs/EVALUATION.md` section 2 (metrics), section 3 (pass criteria), section 5 (result files), section 6 (compute table)
- Decisions: D-02 (own implementation), D-20 (backbone), D-21 (native tiles, MVTec resized), D-22 (random crops only, no flips or rotations), D-24 (Adam 0.005, betas 0.5/0.999, batch 16, mixed precision, 5,000 / 10,000 / 20,000 iterations), D-25 (last checkpoint), D-30 (splits), D-33 (targets 98.5 / 96.7 / 92.0), D-35, D-58, A-01 to A-09 (unchanged)

## 3. Scope

**In scope**

- `loomguard/models/rd4ad.py`: bottleneck, decoder, student, cosine loss, anomaly map, `RD4AD` (fit, score, save, load).
- A random-crop training dataset in `loomguard/data/dataset.py` (D-22).
- `scripts/fit.py` and `scripts/evaluate.py` take `method: rd4ad`; `loomguard/runs.py` gains the RD4AD config block, method dispatch for loading, and a `tag` key that separates runs of one method (seeds).
- Configs `configs/mvtec_rd4ad.yaml` and `configs/raw_fabrid_rd4ad_s{0,1,2}.yaml`.
- CPU tests, including a synthetic `fit.py` then `evaluate.py` run for RD4AD, and the overfit test.
- The training-PC runbook (section 8).

**Out of scope**

- Calibration, `tau`, operating-point metrics, grading, `Inspector`, `calibration.json` (Phase 4). Hyperparameter tuning of any kind (D-35).
- Group and held-out regimes, ZJU-Leaper and AITEX (Phase 5). Dinomaly (open, O-01).
- Any dataset download, training run, GPU use or pretrained-weight download on the laptop.

## 4. Design

### 4.1 Files

| File | New or changed | Purpose |
|---|---|---|
| `loomguard/models/rd4ad.py` | new | `Bottleneck`, `Decoder`, `Student`, `cosine_loss`, `anomaly_map`, `RD4AD` |
| `loomguard/data/dataset.py` | changed | `TileDataset.get(row)` (the body of `__getitem__`), new `RandomCropDataset` |
| `loomguard/runs.py` | changed | `method` accepts `rd4ad`; `rd4ad` config block; `tag`; `load_model` |
| `scripts/fit.py`, `scripts/evaluate.py` | changed | method dispatch (`fit_patchcore`, `fit_rd4ad`); evaluation is method-agnostic |
| `configs/mvtec_rd4ad.yaml`, `configs/raw_fabrid_rd4ad_s0.yaml`, `_s1`, `_s2` | new | the training-PC runs |
| `pyproject.toml` | changed | `slow` pytest marker, deselected by default (no new dependency) |
| `tests/test_rd4ad.py` | new | model, loss, training-loop tests |
| `tests/test_dataset.py`, `tests/test_runs.py`, `tests/test_fit_evaluate.py` | changed | random crops, config, end to end for RD4AD |
| `docs/specs/phase-3-reverse-distillation.md` | new | this spec |
| `docs/IMPLEMENTATION_PLAN.md`, `PROGRESS_LOG.md`, `DECISIONS.md`, `EVALUATION.md`, `ARCHITECTURE.md`, `README.md`, `CLAUDE.md` | changed | status, evidence, decisions, corrected text |

### 4.2 Interfaces

Image batches are `B x C x H x W` float32, ImageNet-normalised.

```python
# loomguard/models/rd4ad.py
class Bottleneck(nn.Module):
    """OCBE. forward(e1 B x 256 x 64 x 64, e2 B x 512 x 32 x 32, e3 B x 1024 x 16 x 16) -> B x 2048 x 8 x 8.
    fuse1: conv3x3 s2 256->512, BN, ReLU, conv3x3 s2 512->1024, BN, ReLU. fuse2: conv3x3 s2 512->1024, BN, ReLU.
    embed: cat(fuse1(e1), fuse2(e2), e3) = 3072 x 16 x 16, then 3 bottleneck blocks (width 1024, out 2048, first stride 2)."""

class Decoder(nn.Module):
    """forward(z B x 2048 x 8 x 8) -> (d1 B x 256 x 64 x 64, d2 B x 512 x 32 x 32, d3 B x 1024 x 16 x 16).
    d3: 3 blocks, d2: 4 blocks, d1: 6 blocks; the first block of each stage upsamples with a 2x2 stride-2 transposed conv
    (also on its shortcut), the others use 3x3 convs."""

class Student(nn.Module):
    bottleneck: Bottleneck; decoder: Decoder
    def forward(self, e1, e2, e3) -> tuple[Tensor, Tensor, Tensor]:   # (d1, d2, d3)

def cosine_loss(enc: Sequence[Tensor], dec: Sequence[Tensor]) -> Tensor:
    """Scalar float32. Sum over the 3 scales of mean over the batch of 1 - cos(E_k, D_k), each map flattened to one vector per image."""

def anomaly_map(enc: Sequence[Tensor], dec: Sequence[Tensor], size: int = 256) -> Tensor:
    """B x 1 x size x size float32. Per position 1 - cosine over channels at each scale, bilinear upsample to size, add, gaussian_blur(sigma 4)."""

class RD4AD:
    def __init__(self, backbone: Backbone, device: str | torch.device = "cpu", seed: int = 0):
        """The backbone is the teacher: moved to `device`, kept in eval(), never in the optimiser. `seed` seeds the student's initialisation."""
    student: Student
    def fit(self, batches: Iterable[Tensor], iterations: int, lr: float = 0.005, betas: tuple[float, float] = (0.5, 0.999),
            val: Sequence[Tensor] = (), val_every: int = 500, log_every: int = 100) -> dict:
        """Exactly `iterations` Adam steps on `batches` (B x 3 x 256 x 256 normal crops), student parameters only; the last state is kept.
        bfloat16 autocast on CUDA only. `val`: list of normal-tile batches; their loss (student in eval()) is logged every `val_every`
        steps and at the end, only to spot divergence. Raises ValueError if `batches` has fewer than `iterations` items; extra items
        are never consumed. Returns {iterations, seconds, iters_per_second, final_train_loss, final_val_loss | None,
        loss_curve: [[iteration, mean train loss since the last log, val loss | None], ...]}."""
    def val_loss(self, batches: Sequence[Tensor]) -> float: ...
    def score(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """x: B x 3 x 256 x 256. Returns (tile score B float32 = max of the heatmap, heatmap B x 1 x 256 x 256 float32). Full precision, student in eval()."""
    def save(self, path: Path) -> None:    # student state dict, floating tensors in float16
    @classmethod
    def load(cls, path: Path, backbone: Backbone, device="cpu", **kw) -> "RD4AD":    # torch.load(weights_only=True)
```

`fit`, `score`, `save` and `load` have the same signatures and meaning as `PatchCore`'s, so `evaluate.py` treats both alike.

```python
# loomguard/data/dataset.py
class TileDataset:
    def get(self, r: TileRow) -> tuple[Tensor, Tensor, int]:   # what __getitem__ did for rows[i]; __getitem__(i) = get(rows[i])

class RandomCropDataset(Dataset):
    def __init__(self, rows: Sequence[TileRow], root: Path | str, length: int, seed: int): ...
    def __len__(self) -> int:                                   # length
    def __getitem__(self, i: int) -> tuple[Tensor, Tensor, int]:
        """3 x 256 x 256 crop, mask 1 x 256 x 256, label. Parent image uniform at random, crop origin uniform over the parent
        (extent = union of its rows' crop boxes). Deterministic in (seed, i). No flip, no rotation. A row with `resize` (MVTec)
        is returned whole."""
```

```python
# loomguard/runs.py
RD4AD_DEFAULTS = {"iterations": 5000, "batch_size": 16, "lr": 0.005, "betas": [0.5, 0.999], "seed": None,
                  "val_tiles": 64, "val_every": 500, "log_every": 100}
def load_model(cfg: dict, path: Path, backbone: Backbone, device: torch.device):   # PatchCore.load or RD4AD.load by cfg["method"]
```

`scripts/fit.py main(argv)` and `scripts/evaluate.py main(argv) -> dict[str, dict]` keep their signatures.

### 4.3 Behaviour and edge cases

**Network.** Layer sizes, block counts and widths follow the official implementation (ARCHITECTURE 6.1). Convolutions are initialised Kaiming-normal (fan_out), BatchNorm to weight 1 and bias 0. Trainable parameters: 83.80 M, of which fusion convs 10.62 M, embedding blocks 48.26 M, decoder 24.92 M (ARCHITECTURE table). The student's BatchNorm layers are in training mode while fitting and in `eval()` for validation loss and scoring.

**Teacher frozen (CLAUDE.md rule 4).** `Backbone` already has `requires_grad=False` and overrides `train()` to stay in eval. `RD4AD` additionally calls `backbone.eval()` at construction and before every step, never puts backbone parameters in the optimiser, and the teacher forward is already under `no_grad`. A test checks that the teacher's BatchNorm running statistics and weights are bit-identical after training steps.

**Loss.** ARCHITECTURE 6.2: per scale, flatten the teacher map and the student map to one vector per image, `1 - cos`, mean over the batch, sum over three scales. Computed in float32 even under autocast.

**Training loop.** Adam(lr 0.005, betas (0.5, 0.999)), fixed `iterations` (D-24; the config sets 5,000 per fabric profile and 20,000 unified), the last state is the checkpoint (D-25). No scheduler, no early stopping, no best-checkpoint selection. Mixed precision (D-24) is `torch.autocast(dtype=torch.bfloat16)` on CUDA only, no loss scaler (bfloat16 has float32's range), and disabled on CPU so the same code runs in tests (decision A-10). The loss is logged every `log_every` steps (mean since the last log) and the validation loss every `val_every` steps and at the last step. `iters_per_second` is training steps over the time spent in those steps (validation excluded); the timing includes a device sync each step through `float(loss)`.

**Scoring.** ARCHITECTURE 6.3: tile score is the maximum of the blurred heatmap. (PatchCore's tile score is the max of the unblurred grid, A-09; for RD4AD the heatmap is the only map, so tile score and mask share one source, which is what D-32's single `tau` needs.)

**Checkpoint.** `save` writes the student state dict only (the teacher is rebuilt from torchvision weights at load time), floating tensors cast to float16 (168 MB), integer buffers (`num_batches_tracked`) unchanged. `load` reads with `weights_only=True`; `load_state_dict` copies float16 into the float32 parameters.

**Random crops (D-22).** Training batches come from `RandomCropDataset` over the profile's `train` rows only: a random parent image, a random 256 x 256 window inside it, nothing else. No flip, rotation, colour or scale augmentation. The dataset is a map-style dataset of length `iterations * batch_size` whose item `i` uses `random.Random(f"{seed}/{i}")`, so it is reproducible and safe with DataLoader workers; `DataLoader(batch_size=B, shuffle=False)` then yields exactly `iterations` batches. For MVTec the manifest row is the whole resized 256 x 256 image, so a "crop" is that image (the documented exception, D-21). Parent images are sampled uniformly, so a roll with fewer images is not under-weighted by tile count.

**`fit.py` for RD4AD.** Per profile: manifest as before; `train` rows only for the crop stream; the validation batches are the profile's `val` rows (normal by construction, D-30) capped to `rd4ad.val_tiles` with `subsample` and loaded once; `test` rows are never read (CLAUDE.md rule 2). The bundle is `model.pt` (float16 student), `config.yaml` (resolved config plus `profile`, `git_commit`, `tile_size`, `layers: [layer1, layer2, layer3]`) and `metrics.json` with `n_train_tiles` (train rows in the profile), `iterations`, `iters_per_second`, `final_train_loss`, `final_val_loss`, `loss_curve`, `fit_minutes`, `peak_vram_gb`, `peak_ram_gb`, `bundle_mb`. Per-log-interval losses are also printed. `calibration.json` is Phase 4.

**`evaluate.py` for RD4AD.** Identical flow to PatchCore (two passes, stitching, histograms); the model comes from `load_model`; the batch size is `cfg[method]["batch_size"]`. The result JSON has the Phase 2 keys, with the PatchCore-only `coreset_ratio`, `bank_size`, `coreset_seconds` replaced for RD4AD by `iterations`, `iters_per_second`, `final_train_loss`, `final_val_loss` (copied from the bundle's `metrics.json`). PatchCore's result files are unchanged.

**Seeds and tags.** `rd4ad.seed` (default: the config `seed`) seeds the student initialisation and the crop stream. The config `seed` still drives the split (A-03), so the three RAW-FABRID seeds share one split and measure training variance only. `tag` (default empty) is appended to the method directory so runs do not overwrite each other: `artifacts/raw_fabrid/unified/rd4ad_s1/`, `results/raw_fabrid/unified/rd4ad_s1/unified.json` (decision A-12). The `method` field in the result JSON stays `rd4ad`.

### 4.4 Configuration

```yaml
method: rd4ad                  # patchcore | rd4ad
tag: ""                        # optional suffix of the method directory (seeds)
rd4ad:
  iterations: 5000             # D-24: 5,000 per fabric profile, 10,000 per group, 20,000 unified
  batch_size: 16
  lr: 0.005
  betas: [0.5, 0.999]
  seed: null                   # null = the config seed
  val_tiles: 64                # normal validation tiles for the divergence check
  val_every: 500
  log_every: 100
num_workers: 0                 # the shipped RD4AD configs use 4 (MVTec) and 8 (RAW-FABRID): random crops decode full parent images
```

`configs/mvtec_rd4ad.yaml`: the five textures, per fabric, 5,000 iterations, `targets` 98.5 / 96.7 / 92.0. `configs/raw_fabrid_rd4ad_s{0,1,2}.yaml`: unified, 20,000 iterations, `tag` and `rd4ad.seed` equal to 0, 1 and 2, no targets.

### 4.5 Dependencies

None added (torch, torchvision, numpy, pillow, pyyaml are present). The `slow` marker is registered in `pyproject.toml`.

## 5. Tasks

### Laptop (developer session)

1. `models/rd4ad.py`; shape, parameter-count, frozen-teacher, loss and anomaly-map tests (plan task 1).
2. Training loop in `RD4AD.fit` with validation loss and float16 export; loop, overfit and round-trip tests (plan task 2).
3. `RandomCropDataset`; `runs.py` and `fit.py` / `evaluate.py` method dispatch; configs; end-to-end tests on synthetic MVTec-like and RAW-FABRID-like trees.
4. Run the slow overfit test at 32 x 32 and at 256 x 256 on the CPU (hours of CPU at 256, run once in the background) and record the result.
5. Full `pytest`; commit `phase 3: implementation`.
6. Review loop, doc sync, commit `phase 3: review fixes and doc sync`.

### Training PC (operator; not done by the developer session)

7. Environment as in the Phase 1 runbook; `pytest` and `pytest -m slow` pass (the slow tests run at full tile size on the GPU).
8. MVTec textures RD4AD, five categories (plan task 3); compare with D-33 targets.
9. Peak VRAM at batch 16 and iterations per second from the bundles' `metrics.json`; update the EVALUATION compute table.
10. RAW-FABRID unified, three seeds (plan task 4); mean and spread.

## 6. Acceptance criteria

A1 to A7 are the plan's criteria with unchanged meaning (A2 is made precise: the zero-trainable-parameter and eval-mode claims are separate checks; A3 is run at two tile sizes, see below). A8 onward are added.

| # | Criterion | Where | Verified by | Met |
|---|---|---|---|---|
| A1 | Shape unit test: embedding is 2048x8x8; decoder outputs are 256x64x64, 512x32x32, 1024x16x16 (for a 256x256 input through the real backbone architecture) | Laptop | `pytest tests/test_rd4ad.py::test_shapes` | [ ] |
| A2 | Trainable parameter count is 83.8 M (within 1 percent). The teacher has zero trainable parameters and stays in `eval()` mode during training (unit test) | Laptop | `pytest tests/test_rd4ad.py::test_parameter_count tests/test_rd4ad.py::test_teacher_frozen_through_training` | [ ] |
| A3 | Overfit test: on 8 tiles the loss falls below 0.05 within 300 iterations | Laptop (32 x 32 tiles and 256 x 256 tiles on CPU, `slow`); Training PC (256 x 256 on the GPU) | `pytest -m slow tests/test_rd4ad.py::test_overfit_eight_tiles` | [ ] |
| A4 | RD4AD on MVTec textures, mean over five categories: image AUROC 98.5 or more, pixel AUROC 96.7 or more, AUPRO 92.0 or more | Training PC (code, config and targets check tested on the laptop) | Laptop: `pytest tests/test_fit_evaluate.py tests/test_runs.py::test_shipped_configs_load`. Training PC: runbook step 3, the mean table printed by `evaluate.py` shows PASS for all three | [ ] |
| A5 | Peak VRAM at batch 16 is under 11 GB. Measured iterations per second recorded, and the compute table in EVALUATION.md updated | Training PC | runbook step 4: `peak_vram_gb`, `iters_per_second` in the bundles' `metrics.json` | [ ] |
| A6 | No code path reads test data during training (code review item) | Laptop | `pytest tests/test_fit_evaluate.py::test_rd4ad_fit_reads_no_test_rows`; reviewer reads `fit_rd4ad` and `RD4AD.fit` | [ ] |
| A7 | RAW-FABRID unified results for 3 seeds are in `results/`, with mean and spread in `PROGRESS_LOG.md` | Training PC | runbook steps 5 to 6: `results/raw_fabrid/unified/rd4ad_s{0,1,2}/unified.json` | [ ] |
| A8 | The cosine loss equals hand values (identical maps 0, opposite maps 6, orthogonal maps 3, flattened per image, mean over the batch) and the anomaly map is B x 1 x 256 x 256, peaks where the student disagrees and its max is the tile score | Laptop | `pytest tests/test_rd4ad.py::test_cosine_loss_hand_values tests/test_rd4ad.py::test_anomaly_map_and_score` | [ ] |
| A9 | The training loop does exactly `iterations` steps, keeps the last state, uses autocast only on CUDA, logs validation loss, and the loss falls on a learnable case within the default suite | Laptop | `pytest tests/test_rd4ad.py::test_fit_budget_and_logging tests/test_rd4ad.py::test_loss_falls_quickly` | [ ] |
| A10 | A saved checkpoint has float16 floating tensors, loads with `weights_only=True` and reproduces the scores of the float16-rounded student | Laptop | `pytest tests/test_rd4ad.py::test_save_load_float16` | [ ] |
| A11 | Training crops are 256 x 256 windows of the train parents' pixels (no flip, no rotation), deterministic per seed, drawn only from the given rows; an MVTec row is returned whole | Laptop | `pytest tests/test_dataset.py::test_random_crops` | [ ] |
| A12 | `fit.py` then `evaluate.py` run end to end for `method: rd4ad` on tiny synthetic MVTec-like and RAW-FABRID-like datasets on CPU, produce the bundle files and a result JSON with all keys, and the tile and pixel metrics equal an independent recomputation (tile, score, stitch, exact metrics) | Laptop | `pytest tests/test_fit_evaluate.py` | [ ] |
| A13 | The config schema: `rd4ad` defaults, seed fallback, `tag` in bundle and result paths, shipped configs load with the D-24 and D-33 values | Laptop | `pytest tests/test_runs.py` | [ ] |
| A14 | The Phase 1 and 2 tests still pass; the whole default suite passes on the laptop with no dataset present | Laptop | `pytest` in the project venv | [ ] |

## 7. Test plan

All CPU, synthetic, no dataset, no downloaded weights (`Backbone(weights=None)`). The CPU cost is dominated by the 84 M-parameter Adam step (about 1 s per iteration even at 32 x 32), so the default suite only runs a handful of iterations; the long ones carry `@pytest.mark.slow` and are deselected by default (`-m "not slow"` in `pyproject.toml`).

| Test | Proves |
|---|---|
| `test_rd4ad.py::test_shapes` | A1: embedding B x 2048 x 8 x 8 and the three decoder outputs for a 256 x 256 input, teacher maps of matching shapes |
| `test_rd4ad.py::test_parameter_count` | A2: 83.80 M trainable student parameters within 1 percent, and the three sub-totals of the ARCHITECTURE table within 1 percent; the backbone has none |
| `test_rd4ad.py::test_teacher_frozen_through_training` | A2: after `student.train()`, `backbone.train()` and several `fit` steps the backbone is in eval mode, none of its parameters requires grad, its state dict is bit-identical, the student's changed, and the optimiser holds student parameters only |
| `test_rd4ad.py::test_cosine_loss_hand_values` | A8 |
| `test_rd4ad.py::test_anomaly_map_and_score` | A8 |
| `test_rd4ad.py::test_fit_budget_and_logging` | A9: a counting generator is consumed exactly `iterations` times; a short generator raises ValueError; the curve has the expected entries and the val loss is logged at `val_every` and at the end; the model is in eval mode afterwards |
| `test_rd4ad.py::test_loss_falls_quickly` | A9: 15 iterations on 4 tiles of 32 x 32 at lr 0.005 cut the loss by at least half |
| `test_rd4ad.py::test_overfit_eight_tiles[32, 256]` (slow) | A3: 8 fixed tiles, 300 iterations, final train loss below 0.05 (CPU or CUDA, whichever the machine has) |
| `test_rd4ad.py::test_save_load_float16` | A10 |
| `test_dataset.py::test_random_crops` | A11 |
| `test_runs.py::test_load_config`, `test_profiles_and_paths`, `test_shipped_configs_load` | A13 |
| `test_fit_evaluate.py::test_mvtec_end_to_end[patchcore, rd4ad]`, `test_raw_fabrid_unified_end_to_end[patchcore, rd4ad]` | A12 (the Phase 2 tests, now parametrised over the method; the RAW-FABRID tree has defects straddling a tile border) |
| `test_fit_evaluate.py::test_rd4ad_fit_reads_no_test_rows` | A6: spies on every dataset `fit.py` builds; only `train` rows reach the crop stream, only `val` rows the validation batches |

## 8. Training-PC runbook

Prerequisite: the Phase 1 and 2 environment (PyTorch 2.7 or newer with CUDA 12.8 wheels, `sm_120`), the datasets as in the Phase 1 runbook (`data/mvtec_ad`, `data/RAW_FABRID`), real ImageNet weights (downloaded on first use). The Phase 2 PatchCore runs are independent of this runbook.

```bash
# 0. Update and test
git pull
source .venv/bin/activate
pytest                                               # default suite must pass
pytest -m slow tests/test_rd4ad.py                   # overfit test at 32 and 256 on the GPU: 8 tiles, loss < 0.05 in 300 iterations

# 1. MVTec AD textures, RD4AD, 5,000 iterations per category (about 20 to 25 minutes each at 4 it/s)
python scripts/fit.py      --config configs/mvtec_rd4ad.yaml | tee artifacts/phase3_mvtec_fit.txt
python scripts/evaluate.py --config configs/mvtec_rd4ad.yaml | tee artifacts/phase3_mvtec_eval.txt
# A4: the end of phase3_mvtec_eval.txt prints the mean over the five categories and PASS/FAIL against 98.5 / 96.7 / 92.0.
# Per-category files: results/mvtec/per_fabric/rd4ad/*.json. Look at the train/val loss lines for divergence.

# 2. A5: speed and memory (batch 16 is the config value)
python - <<'EOF'
import json, glob
for f in sorted(glob.glob("artifacts/mvtec/*/rd4ad/metrics.json")):
    m = json.load(open(f)); print(f.split("/")[2], "peak_vram_gb", round(m["peak_vram_gb"], 2), "it/s", round(m["iters_per_second"], 2),
                                  "fit_minutes", round(m["fit_minutes"], 1), "bundle_mb", round(m["bundle_mb"]))
EOF
# peak_vram_gb must be under 11. If it/s is far below the GPU's speed, the crop loader is the bottleneck: raise num_workers
# in the config (RAW-FABRID decodes 1792 x 1024 parents) and note it. Update the compute table in docs/EVALUATION.md section 6.

# 3. RAW-FABRID unified, 3 seeds, 20,000 iterations each (about 80 minutes each at 4 it/s)
for s in 0 1 2; do
  python scripts/fit.py      --config configs/raw_fabrid_rd4ad_s$s.yaml | tee artifacts/phase3_raw_s${s}_fit.txt
  python scripts/evaluate.py --config configs/raw_fabrid_rd4ad_s$s.yaml | tee artifacts/phase3_raw_s${s}_eval.txt
done

# 4. A7: mean and spread over the three seeds
python - <<'EOF'
import json, statistics as st
keys = ["tile_auroc", "tile_ap", "aupro_30", "pixel_auroc", "pixel_ap", "ms_per_tile_cuda", "peak_vram_gb", "final_val_loss"]
res = [json.load(open(f"results/raw_fabrid/unified/rd4ad_s{s}/unified.json")) for s in range(3)]
for k in keys:
    v = [r[k] for r in res]; print(f"{k}: mean {st.mean(v):.4f}  sd {st.stdev(v):.4f}  min {min(v):.4f}  max {max(v):.4f}")
EOF

# 5. Commit (phase 3: results): results/**.json, the PROGRESS_LOG.md entry with the tables, the EVALUATION.md compute table,
#    ticked criteria in this spec and in IMPLEMENTATION_PLAN.md. If a target is missed: do not tune on these results (D-35);
#    log the miss and the loss curves and raise it as a decision.
```

| Step | Expected | Record in |
|---|---|---|
| 0 | all tests pass, overfit loss below 0.05 at both sizes | `PROGRESS_LOG.md` |
| 1 | mean over 5 textures: image AUROC at least 98.5, pixel AUROC at least 96.7, AUPRO at least 92.0 (percent) | `PROGRESS_LOG.md`, `results/mvtec/per_fabric/rd4ad/` |
| 2 | peak VRAM under 11 GB; iterations per second noted; compute table updated | `PROGRESS_LOG.md`, `docs/EVALUATION.md` |
| 3 to 4 | three result files; mean and spread of the metrics in the table | `PROGRESS_LOG.md`, `results/raw_fabrid/unified/rd4ad_s{0,1,2}/` |

## 9. Documents to update

- `docs/ARCHITECTURE.md`: layout (RD4AD and `RandomCropDataset` homes, `tag` in the bundle path, bundle metrics for RD4AD); section 6.2 states the `bfloat16` autocast.
- `docs/EVALUATION.md`: section 5 result keys for RD4AD and the tagged result path; section 6 compute table (after the measurement).
- `docs/DECISIONS.md`: "Awaiting confirmation" rows A-10 onward (section 10).
- `README.md` "Current state"; `CLAUDE.md` layout notes if inaccurate.

## 10. Risks and open points

| Risk or open point | Handling |
|---|---|
| A-10: mixed precision is bfloat16 autocast on CUDA only, no GradScaler; the loss is computed in float32 | D-24 says "mixed precision" without a dtype. bfloat16 needs no scaler and the RTX 5070 supports it. Easy to switch |
| A-11: crops are drawn from the train rows' parent images (uniform over parents, uniform origin over the parent extent), not from the fixed tile grid | D-22 says random crops; the manifest only holds grid tiles. Crops come only from train parents, so rule 1 and rule 2 are not touched |
| A-12: `rd4ad.seed` and the `tag` config key; the 3 RAW-FABRID seeds share the split (config seed 0) and differ in initialisation and crops; `tag` is appended to the method directory | EVALUATION says "3 seeds for RD4AD" without defining what varies. A fixed split keeps the seeds comparable with PatchCore and measures training variance. Easy to reverse |
| A-13: validation loss uses up to 64 `val` tiles (config `val_tiles`), with the student in `eval()`; the two overfit tests are `slow` and deselected by default | D-25 only says "validation loss on held-out normal tiles, logged only to spot divergence". The CPU cost of an 84 M-parameter Adam step makes a 300-iteration test too slow for the default suite |
| The 5,000-iteration budget (D-24) is not the paper's 200 epochs on MVTec categories with 280 to 320 images (about 3,500 iterations at batch 16, so close), but the targets may still be missed | Report and analyse; no tuning on test results (D-35) |
| The real backbone and real data are not available on the laptop, so numbers, VRAM and speed cannot be seen here. The overfit test uses random-weight teachers | Runbook with exact commands; integration tests with the real network architecture at random weights |
| Random-crop loading of 1792 x 1024 PNGs may bound the iterations per second | `num_workers` in the configs; the runbook says to record and raise it if it/s is far below the GPU's speed |
| bfloat16 may change the loss slightly against the paper's float32 | Training-PC check A4; the choice is reversible (A-10) |

Nothing in this spec depends on an item under "Open" in `DECISIONS.md` (O-01, O-02).

## 11. Completion record

| | |
|---|---|
| Spec commit | |
| Implementation commit | |
| Review commit | |
| Review rounds and verdict | |
| Tests | |
| Results commit (training PC) | |
| Decisions made | |
| Deviations from this spec | |
| Closed by owner | |
