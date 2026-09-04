#!/usr/bin/env bash
set -e
cd /home/pc/mmdet_AI/mmdetection
export PYTHONPATH=/home/pc/mmdet_AI/mmdetection:$PYTHONPATH
python tools/research/visualize_featuremaps_dgcf_paper.py \
  --clean-root mmdet_dataset/lettuce \
  --benchmark-root mmdet_dataset/lettuce_c \
  --benchmark-manifest mmdet_dataset/lettuce_c/manifest.json \
  --out-dir paper_outputs_dgcf_simple/featuremaps_3corr \
  --image-names 0003_000050.png 0007_000050.png 0011_000050.png \
  --conditions clean brightness:3 contrast:3 gaussian_noise:3 \
  --target-layer neck_p2 \
  --heatmap-mode mean_abs \
  --alpha 0.45 \
  --device cuda:0
