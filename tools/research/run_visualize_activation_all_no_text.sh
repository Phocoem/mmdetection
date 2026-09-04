#!/usr/bin/env bash
set -e

cd /home/pc/mmdet_AI/mmdetection
export PYTHONPATH=/home/pc/mmdet_AI/mmdetection:$PYTHONPATH

IMAGE_PATH="mmdet_dataset/lettuce/images/test/0011_000050.png"
RUN_NAME="no_random"
OUT_ROOT="work_dirs/research/feature_activation_vis_${RUN_NAME}"

CONFIGS=(
  "configs/fair_lettuce/mask_rcnn_r50_fpn.py"
  "configs/fair_lettuce/mask_rcnn_r50_dgcf_fpn.py"

)

NAMES=(
  "mask_rcnn_r50_fpn"
  "mask_rcnn_r50_dgcf_fpn"

)

for i in "${!CONFIGS[@]}"; do
  CONFIG="${CONFIGS[$i]}"
  NAME="${NAMES[$i]}"
  CKPTS=( "work_dirs/research/${NAME}/${RUN_NAME}"/best_coco_segm_mAP_epoch_*.pth )

  if [ ! -e "${CKPTS[0]}" ]; then
    echo "[WARN] Missing checkpoint for ${NAME}"
    continue
  fi

  python tools/research/visualize_prediction_and_activation_maps.py     --config "${CONFIG}"     --checkpoint "${CKPTS[0]}"     --image "${IMAGE_PATH}"     --out-dir "${OUT_ROOT}/${NAME}"     --score-thr 0.85     --alpha 0.45     --activation-mode mean_abs
done

echo "[OK] All visualizations exported to: ${OUT_ROOT}"
