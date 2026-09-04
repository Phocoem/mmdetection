#!/usr/bin/env bash
set -e

CONFIG=${1:-configs/ruq/ruq_mask_rcnn_r50_fpn_1x_lettuce.py}

python tools/train.py "$CONFIG"
