#!/usr/bin/env bash
set -euo pipefail
cd /home/pc/mmdet_AI/mmdetection
export PYTHONPATH=/home/pc/mmdet_AI/mmdetection:${PYTHONPATH:-}
python tools/train.py configs/fair_lettuce/mask_rcnn_r50_dab_fpn.py
