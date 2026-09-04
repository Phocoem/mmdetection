#!/usr/bin/env bash
set -e

# Usage:
# bash projects/ruq/tools/test_robustness.sh CONFIG CHECKPOINT ROBUST_ROOT
# Example:
# bash projects/ruq/tools/test_robustness.sh \
#   configs/ruq/ruq_mask_rcnn_r50_fpn_1x_lettuce.py \
#   work_dirs/ruq_mask_rcnn_r50_fpn_1x_lettuce/best_coco_segm_mAP_epoch_50.pth \
#   data/lettuce_robust

CONFIG=$1
CHECKPOINT=$2
ROBUST_ROOT=$3

if [ -z "$CONFIG" ] || [ -z "$CHECKPOINT" ] || [ -z "$ROBUST_ROOT" ]; then
  echo "Usage: bash projects/ruq/tools/test_robustness.sh CONFIG CHECKPOINT ROBUST_ROOT"
  exit 1
fi

CORRUPTIONS=(gaussian_noise motion_blur defocus_blur shadow brightness_low brightness_high contrast_low occlusion jpeg_compression)
SEVERITIES=(1 2 3)

mkdir -p work_dirs/ruq_robust_results

echo "== Clean test =="
python tools/test.py "$CONFIG" "$CHECKPOINT" \
  --work-dir work_dirs/ruq_robust_results/clean \
  --cfg-options \
  test_dataloader.dataset.data_root="$ROBUST_ROOT/" \
  test_dataloader.dataset.ann_file="annotations/test_clean.json" \
  test_dataloader.dataset.data_prefix.img="images/clean/" \
  test_evaluator.ann_file="$ROBUST_ROOT/annotations/test_clean.json"

for C in "${CORRUPTIONS[@]}"; do
  for S in "${SEVERITIES[@]}"; do
    echo "== Testing $C severity $S =="
    python tools/test.py "$CONFIG" "$CHECKPOINT" \
      --work-dir "work_dirs/ruq_robust_results/${C}_s${S}" \
      --cfg-options \
      test_dataloader.dataset.data_root="$ROBUST_ROOT/" \
      test_dataloader.dataset.ann_file="annotations/test_${C}_s${S}.json" \
      test_dataloader.dataset.data_prefix.img="images/${C}/s${S}/" \
      test_evaluator.ann_file="$ROBUST_ROOT/annotations/test_${C}_s${S}.json"
  done
done
