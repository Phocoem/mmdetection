#!/usr/bin/env bash
set -e

cd /home/pc/mmdet_AI/mmdetection
export PYTHONPATH=/home/pc/mmdet_AI/mmdetection:$PYTHONPATH

DEVICE="cuda:0"
SCORE_THR=0.97
OUT_ROOT="work_dirs/research/predict_0011_000050_conditions"
mkdir -p "$OUT_ROOT"

# Ảnh đầu vào cho các condition.
# Generator mới imagecorruptions lưu ảnh corrupted dạng PNG.
declare -A IMAGES
IMAGES["clean"]="mmdet_dataset/lettuce/images/test/0011_000050.png"
IMAGES["brightness_s3"]="mmdet_dataset/lettuce_c/images/brightness/3/0011_000050.png"
IMAGES["contrast_s3"]="mmdet_dataset/lettuce_c/images/contrast/3/0011_000050.png"
IMAGES["gaussian_noise_s3"]="mmdet_dataset/lettuce_c/images/gaussian_noise/3/0011_000050.png"

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

# Fallback clean image nếu ảnh gốc là jpg/jpeg.
if [ ! -f "${IMAGES["clean"]}" ]; then
  if [ -f "mmdet_dataset/lettuce/images/test/0011_000050.jpg" ]; then
    IMAGES["clean"]="mmdet_dataset/lettuce/images/test/0011_000050.jpg"
  elif [ -f "mmdet_dataset/lettuce/images/test/0011_000050.jpeg" ]; then
    IMAGES["clean"]="mmdet_dataset/lettuce/images/test/0011_000050.jpeg"
  fi
fi

for CONDITION in clean brightness_s3 contrast_s3 gaussian_noise_s3; do
  IMG="${IMAGES[$CONDITION]}"

  if [ ! -f "$IMG" ]; then
    echo "[WARN] Missing image for condition: $CONDITION"
    echo "       Expected: $IMG"
    echo "       Searching:"
    find mmdet_dataset -name "0011_000050.*" | head -20
    continue
  fi

  for i in "${!NAMES[@]}"; do
    NAME="${NAMES[$i]}"
    CFG="${CONFIGS[$i]}"

    echo "=================================================="
    echo "Condition: $CONDITION"
    echo "Model    : $NAME"
    echo "Image    : $IMG"
    echo "=================================================="

    if [ ! -f "$CFG" ]; then
      echo "[WARN] Missing config: $CFG"
      continue
    fi

    CKPT_GLOB="work_dirs/research/${NAME}/no_random/best_coco_segm_mAP_epoch_*.pth"
    mapfile -t CKPTS < <(ls $CKPT_GLOB 2>/dev/null || true)

    if [ "${#CKPTS[@]}" -eq 0 ]; then
      echo "[WARN] Missing checkpoint for $NAME"
      echo "       Pattern: $CKPT_GLOB"
      continue
    fi

    CKPT="${CKPTS[0]}"
    OUT_DIR="${OUT_ROOT}/${CONDITION}/${NAME}"
    mkdir -p "$OUT_DIR"

    python demo/image_demo.py \
      "$IMG" \
      "$CFG" \
      --weights "$CKPT" \
      --device "$DEVICE" \
      --out-dir "$OUT_DIR" \
      --pred-score-thr "$SCORE_THR"

    echo "[OK] Saved to: $OUT_DIR"
  done
done

echo "=================================================="
echo "Done. Results saved in:"
echo "$OUT_ROOT"
echo "=================================================="
