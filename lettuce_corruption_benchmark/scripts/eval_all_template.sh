#!/usr/bin/env bash
set -e

# Edit paths for your MMDetection project.
# This script assumes MMDetection v3-style test.py and cfg-options.

RESULT_DIR="results/raw_json"
mkdir -p ${RESULT_DIR}

MODELS=(
"yolact|configs/yolact_lettuce.py|work_dirs/yolact/best.pth"
"solo|configs/solo_lettuce.py|work_dirs/solo/best.pth"
"solov2|configs/solov2_lettuce.py|work_dirs/solov2/best.pth"
"condinst|configs/condinst_lettuce.py|work_dirs/condinst/best.pth"
"maskrcnn_r50|configs/maskrcnn_r50_lettuce.py|work_dirs/maskrcnn_r50/best.pth"
"maskrcnn_r101|configs/maskrcnn_r101_lettuce.py|work_dirs/maskrcnn_r101/best.pth"
"maskrcnn_r50_cbam|configs/maskrcnn_r50_cbam_lettuce.py|work_dirs/maskrcnn_r50_cbam/best.pth"
"maskrcnn_r50_aspp|configs/maskrcnn_r50_aspp_lettuce.py|work_dirs/maskrcnn_r50_aspp/best.pth"
"maskrcnn_r50_fpfpn|configs/maskrcnn_r50_fpfpn_lettuce.py|work_dirs/maskrcnn_r50_fpfpn/best.pth"
)

CONDITIONS=(
"clean"
"noise"
"gaussian_blur"
"motion_blur"
"brightness"
"contrast"
"gamma"
"shadow"
"jpeg"
"medium"
"hard"
)

for item in "${MODELS[@]}"; do
  IFS="|" read -r MODEL CONFIG CKPT <<< "$item"

  for CONDITION in "${CONDITIONS[@]}"; do
    IMG_PREFIX="stress/${CONDITION}/"
    echo "Evaluating ${MODEL} on ${CONDITION}"

    python tools/test.py "${CONFIG}" "${CKPT}" \
      --cfg-options test_dataloader.dataset.data_prefix.img="${IMG_PREFIX}" \
      > "${RESULT_DIR}/${MODEL}__${CONDITION}.log" 2>&1

    echo "Done: ${MODEL} ${CONDITION}"
    echo "IMPORTANT: copy or export the final MMDetection metrics to ${RESULT_DIR}/${MODEL}__${CONDITION}.json"
  done
done
