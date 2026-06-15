# Lettuce Corruption Robustness Benchmark

This is a clean benchmark toolkit for static-image lettuce instance segmentation.

## What is included?

The benchmark uses image-quality corruptions only:

1. Clean
2. Gaussian noise
3. Gaussian blur
4. Motion blur
5. Brightness shift
6. Contrast shift
7. Gamma correction
8. Local soft shadow
9. JPEG compression
10. Medium mixed stress
11. Hard mixed stress

Removed by design:

- synthetic leaf overlap
- artificial block masking
- random cutout
- mask-aware visibility degradation
- any label/annotation modification

## Why no label modification?

All corruptions modify image quality only. The object position and ground-truth instance masks remain defined by the original test set. Therefore, every stress folder uses the same original COCO annotation file.

## 1. Generate the benchmark

Main protocol:

```bash
python tools/generate_benchmark.py \
  --input test \
  --root-output stress \
  --mode main \
  --seed 42
```

Windows:

```bat
python tools\generate_benchmark.py ^
  --input test ^
  --root-output stress ^
  --mode main ^
  --seed 42
```

Generated folders:

```text
stress/
  clean/
  noise/
  gaussian_blur/
  motion_blur/
  brightness/
  contrast/
  gamma/
  shadow/
  jpeg/
  medium/
  hard/
```

Detailed protocol with severity 1/2/3:

```bash
python tools/generate_benchmark.py \
  --input test \
  --root-output stress_detailed \
  --mode detailed \
  --seed 42
```

## 2. Evaluate with MMDetection

For each condition, use the same COCO test annotation and only change the image folder.

Example:

```bash
python tools/test.py configs/maskrcnn_r50_lettuce.py work_dirs/maskrcnn_r50/best.pth \
  --cfg-options test_dataloader.dataset.data_prefix.img='stress/hard/'
```

Expected evaluation matrix for the main paper:

```text
9 models × 11 conditions = 99 runs
```

## 3. Metrics

Primary static-image instance segmentation metrics:

- Mask AP
- Mask AP50
- Mask AP75
- Box AP
- Box AP50
- Box AP75

Robustness metrics:

- AP per condition
- Robustness Drop: RD_hard = AP_clean - AP_hard
- Stability Index: SI = AP_hard / AP_clean
- Robustness Area Score: RAS = mean(AP_clean, AP_medium, AP_hard)
- Normalized RAS: nRAS = RAS / AP_clean
- Mean Corruption Performance: mCP
- Ranking Shift: |Rank_clean - Rank_hard|

Optional deployment metrics:

- FPS
- inference time per image
- Params
- FLOPs
- GPU memory

Do not use tracking/video metrics such as MOTA, MOTP, IDF1, or ID switches.

## 4. Summarize results

Save MMDetection metric files as:

```text
results/raw_json/model__condition.json
```

Examples:

```text
results/raw_json/maskrcnn_r50__clean.json
results/raw_json/maskrcnn_r50__hard.json
results/raw_json/solov2__motion_blur.json
```

Then run:

```bash
python tools/summarize_results.py \
  --input-dir results/raw_json \
  --out-all results/all_results.csv \
  --out-robust results/robustness_summary.csv
```

## 5. Plot paper figures

```bash
python tools/plot_results.py \
  --all-results results/all_results.csv \
  --robust-results results/robustness_summary.csv \
  --outdir results/figures
```

Generated figures:

- fig1_clean_saturation.png
- fig2_robustness_decay.png
- fig3_corruption_sensitivity_heatmap.png
- fig4_ranking_shift.png
- fig5_stability_index.png

## 6. Make corruption example grid

```bash
python tools/make_corruption_grid.py \
  --root stress \
  --image-name your_image_name.jpg \
  --output results/figures/corruption_grid.jpg
```

## Recommended paper title

Beyond Clean mAP: A Corruption Robustness Benchmark for Lettuce Instance Segmentation
