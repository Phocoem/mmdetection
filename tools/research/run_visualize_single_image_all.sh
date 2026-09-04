#!/usr/bin/env bash
set -e

cd /home/pc/mmdet_AI/mmdetection
export PYTHONPATH=/home/pc/mmdet_AI/mmdetection:$PYTHONPATH

IMAGE_PATH="mmdet_dataset/lettuce/images/test/0003_000050.png"
OUT_ROOT="work_dirs/research/feature_vis_single_no_random"

CONFIGS=(
  "configs/fair_lettuce/mask_rcnn_r50_fpn.py"
  "configs/fair_lettuce/mask_rcnn_r101_fpn.py"
  "configs/fair_lettuce/mask_rcnn_r50_dgcf_fpn.py"
  "configs/fair_lettuce/mask_rcnn_r50_dgcf_no_context_fpn.py"
  "configs/fair_lettuce/mask_rcnn_r50_dgcf_no_detail_fpn.py"
  "configs/fair_lettuce/mask_rcnn_r50_dgcf_no_gate_fpn.py"
)

NAMES=(
  "mask_rcnn_r50_fpn"
  "mask_rcnn_r101_fpn"
  "mask_rcnn_r50_dgcf_fpn"
  "mask_rcnn_r50_dgcf_no_context_fpn"
  "mask_rcnn_r50_dgcf_no_detail_fpn"
  "mask_rcnn_r50_dgcf_no_gate_fpn"
)

for i in "${!CONFIGS[@]}"; do
  CONFIG="${CONFIGS[$i]}"
  NAME="${NAMES[$i]}"
  CKPTS=( "work_dirs/research/${NAME}/no_random"/best_coco_segm_mAP_epoch_*.pth )
  if [ ! -e "${CKPTS[0]}" ]; then
    echo "[WARN] Missing checkpoint for ${NAME}"
    continue
  fi
  python tools/research/visualize_single_image_features.py     --config "${CONFIG}"     --checkpoint "${CKPTS[0]}"     --image "${IMAGE_PATH}"     --out-dir "${OUT_ROOT}/${NAME}"     --score-thr 0.85
done
