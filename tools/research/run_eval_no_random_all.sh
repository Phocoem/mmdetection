#!/usr/bin/env bash
set -e

cd /home/pc/mmdet_AI/mmdetection
export PYTHONPATH=/home/pc/mmdet_AI/mmdetection:$PYTHONPATH

CLEAN_ROOT="/home/pc/mmdet_AI/mmdetection/mmdet_dataset/lettuce"
BENCHMARK_ROOT="/home/pc/mmdet_AI/mmdetection/mmdet_dataset/lettuce_c"
SEED=2026

CONFIGS=(
  "configs/fair_lettuce/mask_rcnn_r50_fpn.py"
  "configs/fair_lettuce/mask_rcnn_r101_fpn.py"
  "configs/fair_lettuce/mask_rcnn_r50_dgcf_fpn.py"
  "configs/fair_lettuce/mask_rcnn_r50_dgcf_no_context_fpn.py"
  "configs/fair_lettuce/mask_rcnn_r50_dgcf_no_detail_fpn.py"
  "configs/fair_lettuce/mask_rcnn_r50_dgcf_no_gate_fpn.py"
  "configs/fair_lettuce/solo_r50.py"
  "configs/fair_lettuce/solov2_r50.py"
  "configs/fair_lettuce/yolact_r50.py"
)

NAMES=(
  "mask_rcnn_r50_fpn"
  "mask_rcnn_r101_fpn"
  "mask_rcnn_r50_dgcf_fpn"
  "mask_rcnn_r50_dgcf_no_context_fpn"
  "mask_rcnn_r50_dgcf_no_detail_fpn"
  "mask_rcnn_r50_dgcf_no_gate_fpn"
  "solo_r50"
  "solov2_r50"
  "yolact_r50"
)

for i in "${!CONFIGS[@]}"; do
  CONFIG="${CONFIGS[$i]}"
  NAME="${NAMES[$i]}"
  RUN_DIR="work_dirs/research/${NAME}/no_random"
  CKPTS=( "${RUN_DIR}"/best_coco_segm_mAP_epoch_*.pth )

  if [ ! -e "${CKPTS[0]}" ]; then
    echo "[WARN] Missing checkpoint: ${RUN_DIR}/best_coco_segm_mAP_epoch_*.pth"
    echo "[SKIP] ${NAME}"
    continue
  fi

  echo "=================================================="
  echo "Evaluate ${NAME}"
  echo "=================================================="

  python tools/research/evaluate_benchmark.py     "${CONFIG}"     "${CKPTS[0]}"     --clean-root "${CLEAN_ROOT}"     --benchmark-root "${BENCHMARK_ROOT}"     --output-dir "${RUN_DIR}/evaluation"     --seed "${SEED}"
done
