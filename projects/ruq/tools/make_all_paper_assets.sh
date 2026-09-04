#!/usr/bin/env bash
set -euo pipefail

# make_all_paper_assets.sh
# Run from MMDetection root.
# Example:
# bash projects/ruq/tools/make_all_paper_assets.sh \
#   configs/ruq/ruq_mask_rcnn_r50_fpn_1x_lettuce.py \
#   work_dirs/ruq_mask_rcnn_r50_fpn_1x_lettuce/best_coco_segm_mAP_epoch_50.pth \
#   data/lettuce_robust \
#   paper_assets

CONFIG=${1:-configs/ruq/ruq_mask_rcnn_r50_fpn_1x_lettuce.py}
CKPT=${2:-work_dirs/ruq_mask_rcnn_r50_fpn_1x_lettuce/latest.pth}
ROBUST_ROOT=${3:-data/lettuce_robust}
OUT_ROOT=${4:-paper_assets}
DEVICE=${DEVICE:-cuda:0}
SEVERITY=${SEVERITY:-2}
MAX_IMAGES=${MAX_IMAGES:-6}
RESULTS_ROOT=${RESULTS_ROOT:-work_dirs/research}
SEED=${SEED:-seed_2026}

mkdir -p "${OUT_ROOT}"

echo "[1/4] Export robustness image grids"
python projects/ruq/tools/visualize_robustness_grid.py \
  --robust-root "${ROBUST_ROOT}" \
  --out-dir "${OUT_ROOT}/robustness_images" \
  --severity "${SEVERITY}" \
  --num-images "${MAX_IMAGES}" \
  --draw-gt || true

echo "[2/4] Export feature-map heatmaps"
python projects/ruq/tools/export_featuremap_heatmaps.py \
  --config "${CONFIG}" \
  --checkpoint "${CKPT}" \
  --input "${ROBUST_ROOT}/images/clean" \
  --out-dir "${OUT_ROOT}/featuremaps_clean" \
  --device "${DEVICE}" \
  --max-images "${MAX_IMAGES}" \
  --layers neck roi_mask_head || true

echo "[3/4] Plot training curves"
python projects/ruq/tools/plot_training_curves.py \
  --results-root "${RESULTS_ROOT}" \
  --seed "${SEED}" \
  --out-dir "${OUT_ROOT}/training_curves" || true

echo "[4/4] Generate paper tables and heatmaps"
python projects/ruq/tools/make_paper_tables_v3_ruq.py \
  --results-root "${RESULTS_ROOT}" \
  --seed "${SEED}" \
  --manifest "${ROBUST_ROOT}/manifest.json" \
  --out-dir "${OUT_ROOT}/paper_tables" || true

echo "=================================================="
echo "Done. Main outputs:"
echo "  ${OUT_ROOT}/robustness_images/"
echo "  ${OUT_ROOT}/featuremaps_clean/"
echo "  ${OUT_ROOT}/training_curves/"
echo "  ${OUT_ROOT}/paper_tables/"
echo "=================================================="
