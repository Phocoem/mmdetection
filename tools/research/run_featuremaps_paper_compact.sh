#!/usr/bin/env bash
set -e

cd /home/pc/mmdet_AI/mmdetection
export PYTHONPATH=/home/pc/mmdet_AI/mmdetection:$PYTHONPATH

# Recommended compact version for paper:
# C3-C5 show deeper backbone semantics.
# P2-P4 show useful FPN multiscale responses.
python tools/research/visualize_featuremaps_c1c5_p1p5.py \
  --clean-root mmdet_dataset/lettuce \
  --benchmark-root mmdet_dataset/lettuce_c \
  --benchmark-manifest mmdet_dataset/lettuce_c/manifest.json \
  --out-dir paper_outputs_dgcf_simple/featuremaps_paper_compact \
  --image-names 0003_000050.png \
  --conditions clean contrast:3 gaussian_noise:3 \
  --layers C3 C4 C5 P2 P3 P4 \
  --heatmap-mode mean_abs \
  --alpha 0.45 \
  --device cuda:0
