#!/usr/bin/env bash
set -e

cd /home/pc/mmdet_AI/mmdetection
export PYTHONPATH=/home/pc/mmdet_AI/mmdetection:$PYTHONPATH

DEVICE="cuda:0"
SCORE_THR=0.3
IMAGE_STEM="0011_000050"
ANN_FILE="mmdet_dataset/lettuce/annotations/test.json"

OUT_ROOT="work_dirs/research/predict_0011_000050_conditions_with_gt"
mkdir -p "$OUT_ROOT"

declare -A IMAGES
IMAGES["clean"]="mmdet_dataset/lettuce/images/test/${IMAGE_STEM}.png"
IMAGES["brightness_s3"]="mmdet_dataset/lettuce_c/images/brightness/3/${IMAGE_STEM}.png"
IMAGES["contrast_s3"]="mmdet_dataset/lettuce_c/images/contrast/3/${IMAGE_STEM}.png"
IMAGES["gaussian_noise_s3"]="mmdet_dataset/lettuce_c/images/gaussian_noise/3/${IMAGE_STEM}.png"

if [ ! -f "${IMAGES["clean"]}" ]; then
  if [ -f "mmdet_dataset/lettuce/images/test/${IMAGE_STEM}.jpg" ]; then
    IMAGES["clean"]="mmdet_dataset/lettuce/images/test/${IMAGE_STEM}.jpg"
  elif [ -f "mmdet_dataset/lettuce/images/test/${IMAGE_STEM}.jpeg" ]; then
    IMAGES["clean"]="mmdet_dataset/lettuce/images/test/${IMAGE_STEM}.jpeg"
  fi
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

if [ ! -f "$ANN_FILE" ]; then
  echo "[ERROR] Missing annotation file: $ANN_FILE"
  exit 1
fi

for CONDITION in clean brightness_s3 contrast_s3 gaussian_noise_s3; do
  IMG="${IMAGES[$CONDITION]}"

  if [ ! -f "$IMG" ]; then
    echo "[WARN] Missing image for condition: $CONDITION"
    echo "       Expected: $IMG"
    find mmdet_dataset -name "${IMAGE_STEM}.*" | head -20
    continue
  fi

  GT_DIR="${OUT_ROOT}/${CONDITION}/ground_truth"
  mkdir -p "$GT_DIR"

  python tools/research/visualize_groundtruth_0011_000050.py \
    --ann "$ANN_FILE" \
    --image "$IMG" \
    --image-stem "$IMAGE_STEM" \
    --out-dir "$GT_DIR" \
    --alpha 0.45

  for i in "${!NAMES[@]}"; do
    NAME="${NAMES[$i]}"
    CFG="${CONFIGS[$i]}"

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
  done
done

echo "Done. Results saved in: $OUT_ROOT"
