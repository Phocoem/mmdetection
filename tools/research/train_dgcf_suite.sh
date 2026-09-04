#!/usr/bin/env bash
set -e

cd /home/pc/mmdet_AI/mmdetection
export PYTHONPATH=/home/pc/mmdet_AI/mmdetection:$PYTHONPATH

CONFIGS=(
  configs/fair_lettuce/mask_rcnn_r50_fpn.py
  configs/fair_lettuce/mask_rcnn_r50_aspp_fpn.py
  configs/fair_lettuce/mask_rcnn_r50_dgcf_no_gate_fpn.py
  configs/fair_lettuce/mask_rcnn_r50_dgcf_no_detail_fpn.py
  configs/fair_lettuce/mask_rcnn_r50_dgcf_no_context_fpn.py
  configs/fair_lettuce/mask_rcnn_r50_dgcf_fpn.py
)

for cfg in "${CONFIGS[@]}"; do
  echo "Training: $cfg"
  python tools/train.py "$cfg"
done
