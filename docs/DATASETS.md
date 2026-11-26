# LoomGuard: Datasets

Verified on 2026-11-12 without downloading any dataset. Methods used: official pages, HTTP `HEAD`/range requests for file sizes, reading zip central directories remotely, repository APIs (Mendeley, Figshare, GitHub), and the dataset papers.

Every row below says what was actually checked. Anything marked **unverified** must be confirmed by the first person who downloads it (Phase 1 task).

## Summary

| Dataset | Role in LoomGuard | Size | Masks | License | Physical scale | Status |
|---|---|---|---|---|---|---|
| MVTec AD (5 textures) | Sanity check: reproduce published numbers | About 2.2 GB across five source archives | Yes | CC BY-NC-SA 4.0 | none | Mounted copy verified. Download needs a form. |
| **RAW-FABRID** | **Primary**: pipeline, localization, grading, demo | 4.13 GB zip | Yes (204 high-res, 687 patch) | CC BY 4.0 | **4 px/mm** | **Fully verified**, including file listing |
| **ZJU-Leaper** | **Core cross-fabric benchmark** | 5.8 GB 7z | Yes | none stated on the project site | none | Full Setting 1 manifest reproduced from the handoff. Execution permissions are pending for the chosen route. |
| AITEX | Optional small-data stress test | 176 MB | Yes | none stated (cite paper) | none | Excluded from the approved study by D-61. |
| TSFabrics | Optional: time-series stream demo | 9.19 GB zip | Yes (defects and cutlines) | CC BY 4.0 | 5 px/mm (127 PPI) | Verified via Figshare API and paper |
| Lusitano | **Dropped** (decision D-14) | 5.96 GB + 13.3 GB | **No evidence of masks** | none stated | none | Files exist; host TLS certificate expired |

Core storage needed on the training PC: about 15 GB compressed, roughly 30 GB with extraction. The optional TSFabrics set adds 9 GB compressed.

The November handoff contains extracted copies. Keep dataset bytes on approved private machines or storage. The audited copy was mounted locally.

## Changes from the starter file

| Starter assumption | What verification found | Decision |
|---|---|---|
| RAW-FABRID is only for "getting the pipeline running first" | It is the only core dataset with a permissive license, verified masks, roll IDs, sequential line-scan frames and a known physical scale | Promoted to the primary dataset for localization, grading and the demo |
| "No physical scale exists in the public datasets" | RAW-FABRID states 4 px/mm. TSFabrics states 127 PPI (5 px/mm). | Grading uses real millimetres on these two. Other datasets use an assumed scale, flagged in the UI. |
| Lusitano for "realism, tiling and the demo"; masks to be verified | The paper reports only image-level AUC and mAP. No mask folder is documented. 19 GB. Host certificate expired. | Dropped by the team on 2026-11-12 (decision D-14) |
| RAW-FABRID has 204 masks | 204 mask files, but the metadata CSV lists 202 defective images | Count masks at ingest; treat the CSV as the source of truth for labels |
| TSFabrics: "check resolution and mask format" | 800x640 grayscale JPEG, pixel masks for defects and for cutlines, 30 fps sequences | Optional. RAW-FABRID rolls already give a stream for the demo. |
| AITEX for "comparison with published numbers" | No official train/test split exists, so published numbers use different protocols | Used as a small-data stress test, not for head-to-head comparison |

## 1. RAW-FABRID (primary)

- **Source:** Mendeley Data, DOI `10.17632/db6g85xsyg.1`, published 2026-04-01. <https://data.mendeley.com/datasets/db6g85xsyg/1>
- **License:** CC BY 4.0.
- **File:** `RAW_FABRID.zip`, 4,133,046,386 bytes. SHA-256 `f6b4f82cce2f0220bbab3968a74b5337ebf2bbfd5e33d0548d34e1ebf80f85b9`.
- **Acquisition:** Basler line-scan camera, **4 pixels per millimetre**, dual LED illumination.
- **Images:** 8-bit single-channel PNG, 1792x1024 (448 mm x 256 mm of fabric per image).

Verified layout (zip central directory, 21,457 files):

```
RAW_FABRID/
  images/                    709 high-res PNGs (defect-free and defective)
  masks/                     204 binary masks (0 background, 255 defect), same filenames
  MVTec/
    train/good/              14,196 patches, 256x256
    test/good/                4,969 patches
    test/defect/                687 patches
    ground_truth/defect/        687 masks
  RAW_FABRID_HighRes_Metadata.csv   filename, roll_id, has_defect, total_area_px, num_defects
  RAW_FABRID_Patches_Metadata.csv   patch_name, parent_raw, type, defect_area
  RAW_FABRID_HighRes_COCO.json
  RAW_FABRID_Patches_COCO.json
  Readme.txt
```

Facts derived from the two metadata CSVs (read remotely, 2.2 MB):

- 709 high-res images: 507 defect-free, 202 defective. Each gives exactly 28 patches (7 x 4 grid of 256x256, no overlap).
- **The official split is by whole image.** All patches of a defect-free image are `train_good`. All patches of a defective image are `test_good` or `test_defect`. No parent image appears on both sides, so there is no tile leakage.
- Test-good patches all come from defective images (the clean regions next to a defect). Note this when reading false-alarm numbers.
- Filenames carry roll, frame index and timestamp, for example `Rollo1A_0000001_D_07-05-2024_10-30-55.085.png`. Frames are about 1.2 s apart, so each roll is an ordered stream.
- Defect area per high-res image: min 227 px, median 3,000 px, max 120,787 px. Per defective patch: min 1 px, 10th percentile 336 px, median 2,801 px.

Per-roll counts:

| Roll | Defect-free images | Defective images | train_good patches | test_good patches | test_defect patches |
|---|---|---|---|---|---|
| Rollo1A | 66 | 21 | 1,848 | 535 | 53 |
| Rollo2A | 55 | 27 | 1,540 | 631 | 125 |
| Rollo3A | 13 | 7 | 364 | 153 | 43 |
| Rollo4A | 92 | 78 | 2,576 | 1,915 | 269 |
| Rollo5A | 18 | 15 | 504 | 364 | 56 |
| Rollo6A | 96 | 28 | 2,688 | 732 | 52 |
| Rollo7A | 167 | 26 | 4,676 | 639 | 89 |
| **Total** | **507** | **202** | **14,196** | **4,969** | **687** |

Member C's handoff records the visual verdict **SAME**. Report RAW-FABRID leave-one-roll-out as roll generalization and use held-out ZJU patterns for unseen-fabric claims. C records two unpaired high-resolution masks outside the metadata inventory. The high-resolution manifest reproduced exactly from the mounted handoff. It has 11,872 training tiles, 2,324 validation tiles, 4,944 good test tiles and 712 defective test tiles. The 25 cross-representation label differences and 532 area differences are reported in C's supplemental audit. Keep shipped patch labels and masks together, and high-resolution crops and masks together. The 15 approved RAW-FABRID jobs call the high-resolution adapter. The current distributed study loader has no shipped-patch benchmark mode. Any implementation of that separate mode needs review.

## 2. ZJU-Leaper (core cross-fabric benchmark)

- **Source:** project page <https://nico-zck.github.io/zju-leaper>. Download link `http://bit.ly/zjuleaper` resolves to Google Drive file `1zzoU3Aj21V62GWS8LBm217TsE5uFeMVB`, `ZJU-Leaper.7z`, **5.8 GB** (alive on 2026-11-12).
- **Code:** <https://github.com/nico-zck/ZJU-Leaper-Dataset> (loader in `dataset_api/zl_fabric.py`, metrics in `dataset_api/zl_eval.py`).
- **License:** The official project page provides a download and citation with no explicit license. Record the applicable terms or obtain permission for the intended execution route before cloud upload. Cite Zhang et al., IEEE Transactions on Artificial Intelligence. The mounted handoff reproduced 170,724 train, 18,968 validation and 126,596 test tiles.
- **Content:** 19 fabric patterns in 5 groups. Papers quote 98,777 images in total. Images are 512x512 (the official loader hard-codes 512x512 for empty masks).

| Group | Patterns |
|---|---|
| 1 | White Plain, Thick Stripe, Thin Stripe |
| 2 | Dot Pattern, Houndstooth, Gingham, Knot Pattern |
| 3 | Twill Plaid, Blue Plaid, Brown Plaid, Gray Plaid, Red Plaid |
| 4 | Floral Print 1, 2, 3 |
| 5 | Pattern 1, 2, 3, 4 |

Layout expected by the official loader:

```
ZJU-Leaper/
  Images/{id}.jpg                  (or .png)
  Annotations/xmls/{id}.xml        pattern_id, defective flag, bboxes
  Annotations/masks/{id}.png       only for defective images
  ImageSets/Patterns/pattern{1..19}.json
  ImageSets/Groups/group{1..5}.json
  ImageSets/total.json
```

Each JSON holds image IDs as `{"normal": {"train": [...], "test": [...]}, "defect": {"train": [...], "test": [...]}}`. LoomGuard uses **Setting 1** (normal samples only for training, test on normal plus defective with masks). The pattern, group and total JSON files map directly onto our three training regimes.

Unverified, check on download: exact image counts per pattern and split, image format, and that every defective test image has a mask.

Risk: Google Drive rate-limits popular large files. Use `gdown` and retry later if the quota error appears. The alternative host listed by the authors is `http://www.qaas.zju.edu.cn/zju-leaper`.

Adapter (`build_zju`, Phase 5, not yet run on real files): reads `ImageSets/Patterns/pattern<n>.json` for the 19 patterns, finds files through an index of `Images/` and `Annotations/masks/` (ids with or without extension, any image extension), uses `normal/train` (90 percent train, 10 percent val, per pattern), `normal/test` and `defect/test`; `defect/train` is unused. Group membership comes from the table above (pattern numbers 1-3, 4-7, 8-12, 13-15, 16-19), which assumes the JSON numbering follows that order. A listed id without image or mask, a mask of the wrong size, or an image listed twice stops the run with a message that counts the problems. The group and total JSON files are not read.

## 3. MVTec AD textures (sanity check)

- **Source:** <https://www.mvtec.com/company/research/datasets/mvtec-ad>. The page now asks for a short form before download.
- **License:** CC BY-NC-SA 4.0. Non-commercial only. Do not redistribute and do not ship MVTec images in the demo.
- **Use:** the five texture categories only: carpet, grid, leather, tile, wood. Standard MVTec folder layout (`train/good`, `test/<defect>`, `ground_truth/<defect>`).
- Purpose: our PatchCore and Reverse Distillation must land near the published numbers (see `EVALUATION.md`) before we trust them on fabric.

Unofficial mirrors exist on Hugging Face. Prefer the official download.

## 4. AITEX (small-data stress test)

- **Source:** <https://www.aitex.es/afid/>. Three 7z archives, all alive on 2026-11-12:
  - `Defect_images.7z` 77.7 MB
  - `NODefect_images.7z` 98.3 MB
  - `Mask_images.7z` 88 KB
- **License:** none stated. The page asks users to cite Silvestre-Blanes et al., AUTEX Research Journal, 2019.
- **Content:** 245 images at 4096x256, 7 fabric structures. 140 defect-free (20 per fabric), 105 defective, 12 defect types, one white-on-black mask per defective image.
- Each image gives 16 tiles of 256x256 at native resolution.
- With about 16 training images per fabric it is a low-data test: PatchCore should cope, a trained decoder may not.

Unverified, check at ingest: mask-to-image pairing (some images are reported to have more than one mask file), and tiles that are mostly black border.

Adapter (`build_aitex`, Phase 5, not yet run on real files): finds `Defect_images`, `NODefect_images` and `Mask_images` anywhere under the root; the fabric is the trailing `_<NN>` of the file stem (an assumption, an error names the first file that does not fit); masks are paired by `<stem>_mask*` and several masks of one image are united; per fabric the 20 normal images split 12 train and 8 val; tiles with more than half of their pixels below gray level 10 are dropped from every split (pixel metrics still run over whole images).

## 5. TSFabrics (optional)

- **Source:** Figshare, DOI `10.6084/m9.figshare.29573381`. File `TSfabrics.zip`, **9.19 GB**. Paper: Scientific Data 13:379 (2026).
- **License:** CC BY 4.0.
- **Content:** 93,196 grayscale JPEGs at 800x640 (centre crop of 1280x1024), 30 fps, from circular knitting machines. 89,900 normal (3,946 of them contain a "cutline"), 3,296 defective. 22 sequence folders over 3 knit textures, with varying machine speed and illumination.
- **Annotations:** `ANNOTATION/<folder>/Cutline` and `ANNOTATION/<folder>/Defect` pixel masks.
- **Scale:** 800x640 px covers 16 cm x 13 cm, about 5 px/mm.
- Interesting twist: cutlines are normal production marks that look like defects. A model trained on frames that include cutlines should not flag them.
- Adjacent frames overlap, so it is a video stream, not a stitched roll.

## 6. Lusitano (dropped)

Dropped on 2026-11-12, decision D-14 in `DECISIONS.md`. The verification facts are kept here in case the dataset is reconsidered at the Phase 7 gate.

- **Source:** <https://kailashhambarde.github.io/Lusitano>. Files hosted at `www.it.ubi.pt`:
  - `nondefects.7z` 5.96 GB (the page labels it "Train Dataset V2, 6000 non-defect images")
  - `test.zip` 13.3 GB (1,646 defect and 1,038 normal images)
- **Images:** 4096x1024, Teledyne Dalsa Linea line-scan camera, MVTec-style folders.
- **License:** none stated for the data. The paper is CC BY.
- **Masks:** the paper (Carrilho, Hambarde, Proença, Applied Sciences 14:5298, 2024) reports only image-level AUC and mAP and documents no ground-truth mask folder. Assume there are none.
- **Published baseline:** Reverse Distillation at 256x256 on a 2000x1000 centre crop reaches image AUC 0.886 with the full training set. Squeezing a 2000x1000 image into 256x256 destroys thin defects, so a tiled pipeline has room to beat it.
- **Access problem:** the host's TLS certificate was expired on 2026-11-12. The contents of `test.zip` were not inspected because that would need certificate checks switched off.
- The page and the paper disagree on counts (32k vs 6,000 training images; 1,300/1,100 vs 1,646/1,038 test images).

## Skipped

Tianchi (restricted access), Roboflow fabric sets (mostly boxes, heavily augmented), TILDA (old, low resolution, no official masks).

## Preprocessing rules common to all datasets

1. Grayscale images are repeated to three channels, then normalised with ImageNet mean and standard deviation.
2. Models always see 256x256 tiles at the dataset's **native resolution**. See `ARCHITECTURE.md` for the tiler.
3. No dataset is copied into tiles on disk. A manifest CSV lists crop boxes and tiles are cropped at load time.
4. Splits are assigned per parent image before tiling. See `EVALUATION.md`.

## Download checklist for the training PC (Phase 1)

| Step | Command or action | Expected |
|---|---|---|
| RAW-FABRID | Download from the Mendeley page, check SHA-256 | 4.13 GB, hash above |
| ZJU-Leaper | `gdown 1zzoU3Aj21V62GWS8LBm217TsE5uFeMVB` then extract with 7-Zip | 5.8 GB |
| MVTec AD | Fill in the form on the MVTec page, extract the 5 texture folders | carpet, grid, leather, tile, wood |
| AITEX | Download the three 7z files from the AITEX page | 176 MB total |
| Verify | Run `scripts/verify_datasets.py` (written in Phase 1) | Counts match this document; mismatches recorded in `PROGRESS_LOG.md` |
