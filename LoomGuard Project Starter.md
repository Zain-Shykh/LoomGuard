# LoomGuard: Project Starter

## What the project is

LoomGuard is a fabric inspection system built with deep learning. It learns what normal fabric looks like from defect-free images only, then flags defects it has never seen, shows where they are, and turns that into a quality grade.

The goal is learning. The scope is deliberately large enough to teach real skills: implementing methods from papers, building a fair benchmark, and handling real production data.

## What the system does

1. **Anomaly detection:** decide whether a fabric image or tile is defective, using a model trained on normal fabric only.
2. **Localization:** produce a per-pixel heatmap and a defect mask showing where the problem is.
3. **Inspection and quality grading:** turn the masks into defect size, count and position, then into a grade such as A, B, C or reject. The idea is adapted from the textile four-point system.

Terminology used in this project: anomaly detection plus localization is the model's job. Inspection and grading is the application layer on top. We are not doing box-style object detection with defect-type labels.

## Constraints

- One GPU with 12 GB of VRAM for training.
- Public datasets only, freely accessible.
- Training uses defect-free images only.
- No physical scale exists in the public datasets, so defect sizes are measured in pixels. For grading we will state an assumed pixels-per-centimeter value openly.

## Approach: the shared idea

Both methods use a frozen CNN pretrained on ImageNet (WideResNet-50) as a description machine. It turns each small patch of a tile into a list of numbers. We take feature maps from intermediate layers because they capture texture detail. The two methods differ in how they decide that a patch is abnormal.

Both use the same backbone so the comparison is fair.

### Method 1: PatchCore (baseline, no gradient training)

- Run normal tiles through the frozen backbone and take layer 2 and layer 3 maps.
- Average over small neighborhoods, match sizes, and combine into one vector per patch (about 1,024 patches per 256x256 tile, each about 1024 numbers).
- Store patch vectors from all normal tiles, then shrink them with coreset sampling (keep 1 to 10 percent). This is the memory bank.
- At test time, each patch is scored by its distance to the nearest stored normal patch. Scores form a grid, which is upsampled and blurred into the heatmap. The image score is the heatmap maximum.
- Knowledge of normal lives in the stored notebook.
- Why it is here: fast to get working, strong on textures, validates the data pipeline and metrics early, and gives a baseline to beat or lose to.
- Known issues: memory bank grows with data, nearest-neighbor search is slow on wide images, normal variation not in the bank causes false alarms, and contaminated training data is stored as normal.

### Method 2: Reverse Distillation (the trained model)

- Teacher: the frozen backbone, producing three feature maps (fine, medium, coarse) for each tile.
- Student: a trainable bottleneck plus a decoder. The bottleneck fuses the three teacher maps into one compact embedding. The decoder mirrors the teacher in reverse and tries to rebuild the three teacher maps from that embedding.
- Loss: one minus cosine similarity between teacher and decoder maps at each scale, summed over scales. Only the student gets gradients.
- Training uses normal fabric only. The squeeze through the bottleneck forces the student to learn the general pattern of normal fabric, so it cannot reproduce anything unusual.
- At test time, the mismatch between teacher and rebuilt features at each position is the anomaly score. Scales are upsampled, combined and blurred into the heatmap.
- Knowledge of normal lives in the student's trained weights.
- Why it is here: it is the genuinely trained model, so it teaches loss design, tuning and overfitting, and it is a good base for extensions.

### How they relate

They are two independent solutions to the same problem, not two stages of one pipeline. PatchCore stores what normal looks like. Reverse Distillation learns it. Comparing them across fabrics, and finding where each fails, is the research result. If PatchCore wins everywhere, that is still a valid finding.

## Datasets

- **ZJU-Leaper (core benchmark):** 19 fabric patterns in 5 groups, a normal-only training setting and test masks. The groups enable the cross-fabric study. Download via the official page. Check the file host and size first.
- **Lusitano:** real factory line-scan images at 4096x1024, with a normal training pack and a large test set. Used for realism, tiling and the demo. Verify whether pixel masks ship with the test set.
- **AITEX:** 245 images at 4096x256 across 7 fabrics, with masks. Small but widely used, for comparison with published numbers.
- **RAW-FABRID:** 709 grayscale images with 204 masks, CC BY 4.0, MVTec-style folders. Used to get the pipeline running first.
- **MVTec AD textures (carpet, grid, leather):** sanity check that our implementations reproduce published numbers. Non-commercial license.
- **TSFabrics (optional):** sequential knitting-machine images for the simulated live stream. Check resolution and mask format first.

Skipped: Tianchi (restricted access), Roboflow sets (mostly box-only and augmented), TILDA (old, grayscale).

## Evaluation plan

### Metrics

- **Image AUROC:** can the model rank defective tiles above normal ones.
- **Pixel AUROC:** does the heatmap rank defect pixels above normal pixels. It can look good even when small defects are missed.
- **PRO (per-region overlap):** coverage of each defect region, averaged, up to a low false-positive rate. The key localization metric because small defects count equally.
- **ZJU-Leaper metrics:** pixel Dice, region F1 and sample F1, for comparison with the dataset's published baselines.

### Splits

- Split by whole image (or fabric roll) before tiling, never by tile, to avoid leakage.
- Three pools: normal training images, held-out normal validation images used to set the threshold, and defect test images never used for tuning.
- Three training regimes on ZJU-Leaper groups: one model per fabric, one per group, and one unified model.
- A held-out-fabric test: the model never sees that weave. Expect it to flag everything. That failure is a finding.

### Inspection layer

- Threshold: choose a score so that about 99 percent of held-out normal tiles fall below it. Report false alarms per 1,000 tiles.
- Heatmap to mask: smooth, apply the threshold, remove tiny blobs.
- Measure each connected defect: size, bounding box, position and count. Stitch tiles back into the full line-scan image.
- Grade by penalty points per area, adapted from the four-point system into A, B, C or reject.

## Planned pipeline modules

1. Tiler and dataset loader (with stitching)
2. Feature extractor (shared frozen backbone)
3. PatchCore: patch aggregator, coreset sampler, memory bank and scorer
4. Reverse Distillation: bottleneck, decoder, training loop
5. Heatmap post-processing
6. Metrics and splits
7. Threshold calibration and grading
8. Streamlit demo simulating a scrolling fabric stream with heatmap overlay and defect log

## Open decisions

1. **Third method (not yet decided):** options are EfficientAD (small student-teacher built for very low latency, suits the streaming demo) or Dinomaly (frozen DINOv2 transformer features with a reconstruction decoder, higher accuracy, needs transformer knowledge and more memory). Default is EfficientAD. Decide after baseline results.
2. **Optional extension (not yet decided):** classify the defect type (hole, stain, yarn break) with a few labeled examples after anomaly detection finds it. Adds scope. Decide later.

## Next steps

1. Draft the system architecture: how raw line-scan images flow through tiling, models, calibration, grading and the demo.
2. Draft a week-by-week plan of roughly ten weeks, starting with the data pipeline, metrics and PatchCore, then Reverse Distillation, the cross-fabric study and calibration, grading and the demo, and finally the write-up.
3. Before relying on any dataset, download it and verify that masks exist and the folder structure matches what the plan assumes.
