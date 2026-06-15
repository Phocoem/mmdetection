# Reproducible Lettuce Segmentation Study

## Scientific protocol

1. Select checkpoints only with validation segmentation mAP.
2. Training has a 200-epoch safety cap, but stops after 20 validation epochs
   without an absolute gain of at least 0.001 segmentation mAP.
3. Evaluate the selected checkpoint once on clean test and unchanged-label
   corruption sets.
4. Use the same corruption manifest and checkpoint for every model.
5. Report at least three seeds as mean and standard deviation.

The safety cap is necessary because an unconstrained optimization run has no
guaranteed stopping point. Early stopping estimates convergence; it does not
prove a global maximum.

## Prepare data

The expected clean layout is:

```text
mmdet_dataset/lettuce/
  annotations/train.json
  annotations/val.json
  annotations/test.json
  images/train/
  images/val/
  images/test/
```

Install benchmark dependencies:

```powershell
pip install -r requirements_research.txt
```

Audit split integrity before training:

```powershell
python tools/research/audit_dataset.py --check-dimensions --hash-images
```

Build the recommended noise benchmark:

```powershell
python tools/research/build_corruption_benchmark.py --suite noise
```

For the extended label-preserving ImageNet-C/COCO-C subset:

```powershell
python tools/research/build_corruption_benchmark.py --suite label_preserving_c --overwrite
```

`elastic_transform` and `glass_blur` are deliberately excluded because moving
image content without moving its instance masks invalidates strict pixel-aligned
evaluation.

## Train And Evaluate

Run one seed end to end:

```powershell
python tools/research/run_experiment.py configs/fair_lettuce/mask_rcnn_r50_fpn.py --seed 2026
```

Every run performs a dimension-aware dataset audit before allocating GPU time.

Repeat with at least three predeclared seeds, for example `2026`, `2027`, and
`2028`. Do not choose seeds after seeing test results.

Run all fair configs with the three predeclared seeds:

```powershell
python tools/research/run_study.py --amp
```

Aggregate completed runs:

```powershell
python tools/research/summarize_study.py
```

Each run stores provenance, source diff, dependency versions, console logs,
resolved config, JSON/TensorBoard curves, best/last checkpoints, condition-level
metrics, clean AP, mPC, rPC, and robustness drop.
