#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-configs/fair_lettuce/mask_rcnn_r50_dab_fpn.py}
CKPT=${2:-work_dirs/research/mask_rcnn_r50_dab_fpn/seed_2026/best_coco_segm_mAP_epoch_XX.pth}
OUT=${3:-work_dirs/research/mask_rcnn_r50_dab_fpn/seed_2026/evaluation}
cd /home/pc/mmdet_AI/mmdetection
export PYTHONPATH=/home/pc/mmdet_AI/mmdetection:${PYTHONPATH:-}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
python tools/research/evaluate_benchmark.py \
  "$CONFIG" \
  "$CKPT" \
  --clean-root /home/pc/mmdet_AI/mmdetection/mmdet_dataset/lettuce \
  --benchmark-root /home/pc/mmdet_AI/mmdetection/mmdet_dataset/lettuce_c \
  --output-dir "$OUT" \
  --seed 2026
