#!/usr/bin/env bash
set -e

cd /home/pc/mmdet_AI/mmdetection
export PYTHONPATH=/home/pc/mmdet_AI/mmdetection:$PYTHONPATH

DEVICE="cuda:0"
SCORE_THR=0.98

IMG_CLEAN="mmdet_dataset/lettuce/images/test/0011_000050.png"

OUT_ROOT="work_dirs/research/predict_0011_000050_all_models"
mkdir -p "$OUT_ROOT"

# Nếu ảnh clean là jpg/jpeg thì tự tìm lại
if [ ! -f "$IMG_CLEAN" ]; then
  IMG_CLEAN="mmdet_dataset/lettuce/images/test/0011_000050.jpg"
fi

if [ ! -f "$IMG_CLEAN" ]; then
  IMG_CLEAN="mmdet_dataset/lettuce/images/test/0011_000050.jpeg"
fi

if [ ! -f "$IMG_CLEAN" ]; then
  echo "[ERROR] Không tìm thấy ảnh 0011_000050 trong images/test"
  find mmdet_dataset/lettuce/images/test -name "0011_000050.*" | head -20
  exit 1
fi

NAMES=(
  "mask_rcnn_r50_fpn"
  "mask_rcnn_r101_fpn"
  "solo_r50"
  "solov2_r50"
  "mask_rcnn_r50_dgcf_fpn"
  "mask_rcnn_r50_dgcf_no_context_fpn"
  "mask_rcnn_r50_dgcf_no_detail_fpn"
  "mask_rcnn_r50_dgcf_no_gate_fpn"
)

CONFIGS=(
  "configs/fair_lettuce/mask_rcnn_r50_fpn.py"
  "configs/fair_lettuce/mask_rcnn_r101_fpn.py"
  "configs/fair_lettuce/solo_r50.py"
  "configs/fair_lettuce/solov2_r50.py"
  "configs/fair_lettuce/mask_rcnn_r50_dgcf_fpn.py"
  "configs/fair_lettuce/mask_rcnn_r50_dgcf_no_context_fpn.py"
  "configs/fair_lettuce/mask_rcnn_r50_dgcf_no_detail_fpn.py"
  "configs/fair_lettuce/mask_rcnn_r50_dgcf_no_gate_fpn.py"
)

for i in "${!NAMES[@]}"; do
  NAME="${NAMES[$i]}"
  CFG="${CONFIGS[$i]}"

  echo "=================================================="
  echo "Model: $NAME"
  echo "Config: $CFG"
  echo "Image : $IMG_CLEAN"
  echo "=================================================="

  if [ ! -f "$CFG" ]; then
    echo "[WARN] Missing config: $CFG"
    continue
  fi

  CKPT_GLOB="work_dirs/research/${NAME}/no_random/best_coco_segm_mAP_epoch_*.pth"
  mapfile -t CKPTS < <(ls $CKPT_GLOB 2>/dev/null || true)

  if [ "${#CKPTS[@]}" -eq 0 ]; then
    echo "[WARN] Không tìm thấy checkpoint cho $NAME"
    echo "       Pattern: $CKPT_GLOB"
    continue
  fi

  CKPT="${CKPTS[0]}"
  OUT_DIR="${OUT_ROOT}/${NAME}/clean"
  mkdir -p "$OUT_DIR"

  python demo/image_demo.py \
    "$IMG_CLEAN" \
    "$CFG" \
    --weights "$CKPT" \
    --device "$DEVICE" \
    --out-dir "$OUT_DIR" \
    --pred-score-thr "$SCORE_THR"

  echo "[OK] Saved to: $OUT_DIR"
done

echo "=================================================="
echo "Done. Results saved in:"
echo "$OUT_ROOT"
echo "=================================================="
