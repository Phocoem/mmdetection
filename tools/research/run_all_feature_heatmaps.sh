#!/bin/bash
# Chay plot_feature_heatmaps.py HANG LOAT cho ca 16 he thong, chia thanh
# 4 anh rieng theo DUNG 4 nhom cua Table 4 trong bai (Baseline /
# Architectural intervention / Augmentation / Feature-constraint) -
# tranh 1 anh qua khong (16 hang x 6 cot anh full-res).
#
# Cach dung:
#   chmod +x run_all_feature_heatmaps.sh
#   ./run_all_feature_heatmaps.sh <duong_dan_anh_input> [seed]
#
# Vi du:
#   ./run_all_feature_heatmaps.sh \
#       mmdet_dataset/lettuce_c/images/contrast_s3/0011_000010.png 2024

set -u  # KHONG dung set -e - 1 nhom loi khong chan cac nhom con lai

IMAGE="${1:?Can truyen duong dan anh input, vd: mmdet_dataset/lettuce_c/images/contrast_s3/0011_000010.png}"
SEED="${2:-2024}"
CONFIG_DIR="configs/fair_lettuce"
WORK_DIR="work_dirs/research"
SCRIPT="tools/research/plot_feature_heatmaps.py"
OUT_DIR="feature_heatmaps_batch"
FAILED_LOG="${OUT_DIR}/batch_failures.log"

mkdir -p "$OUT_DIR"
> "$FAILED_LOG"

ckpt() {  # ckpt <system_dir_name> -> duong dan checkpoint (glob)
    echo "${WORK_DIR}/$1/seed_${SEED}/best_coco_segm_mAP_epoch_*.pth"
}
cfg() {  # cfg <config_file_name> -> duong dan config
    echo "${CONFIG_DIR}/$1"
}

run_group() {
    local group_name="$1"; shift
    local args=("$@")
    local out_file="${OUT_DIR}/heatmap_${group_name}.png"
    echo ""
    echo "======== NHOM: ${group_name} (${#args[@]} he thong) ========"
    if ! python "$SCRIPT" \
        --image "$IMAGE" \
        --system-configs "${args[@]}" \
        --output "$out_file"; then
        echo "THAT BAI: nhom ${group_name}" | tee -a "$FAILED_LOG"
    fi
}

# ---- Nhom 1: Baseline (2 he thong) ----
run_group "baseline" \
    "Mask R-CNN R50 (FPN)|$(cfg mask_rcnn_r50_fpn.py)|$(ckpt mask_rcnn_r50_fpn)" \
    "Mask R-CNN R101|$(cfg mask_rcnn_r101_fpn.py)|$(ckpt mask_rcnn_r101_fpn)"

# ---- Nhom 2: Architectural intervention (2 he thong) ----
run_group "architecture" \
    "CBAM-FPN|$(cfg mask_rcnn_r50_cbam_fpn.py)|$(ckpt mask_rcnn_r50_cbam_fpn)" \
    "BiFPN|$(cfg mask_rcnn_r50_bifpn.py)|$(ckpt mask_rcnn_r50_bifpn)"

# ---- Nhom 3: Augmentation (1 he thong - so voi Baseline R50 de co doi chieu) ----
run_group "augmentation" \
    "Mask R-CNN R50 (FPN)|$(cfg mask_rcnn_r50_fpn.py)|$(ckpt mask_rcnn_r50_fpn)" \
    "CPU photometric aug|$(cfg mask_rcnn_r50_fpn_aug.py)|$(ckpt mask_rcnn_r50_fpn_aug)"

# ---- Nhom 4a: Feature-constraint - lambda sweep (5 he thong) ----
run_group "featureconstraint_lambda" \
    "IAPC control (l=0)|$(cfg mask_rcnn_r50_gpuaug.py)|$(ckpt mask_rcnn_r50_gpuaug)" \
    "IAPC (l=0.1)|$(cfg mask_rcnn_r50_iapc_lam0p10.py)|$(ckpt mask_rcnn_r50_iapc_lam0p10)" \
    "IAPC (l=0.25, full)|$(cfg mask_rcnn_r50_iapc_lam0p25.py)|$(ckpt mask_rcnn_r50_iapc_lam0p25)" \
    "IAPC (l=0.5)|$(cfg mask_rcnn_r50_iapc_lam0p50.py)|$(ckpt mask_rcnn_r50_iapc_lam0p50)" \
    "IAPC (l=1.0)|$(cfg mask_rcnn_r50_iapc.py)|$(ckpt mask_rcnn_r50_iapc)"

# ---- Nhom 4b: Feature-constraint - combination + ablations (6 he thong) ----
run_group "featureconstraint_ablation" \
    "IAPC (l=0.25, full)|$(cfg mask_rcnn_r50_iapc_lam0p25.py)|$(ckpt mask_rcnn_r50_iapc_lam0p25)" \
    "IAPC + FPN+Aug|$(cfg mask_rcnn_r50_iapc_cpuaug.py)|$(ckpt mask_rcnn_r50_iapc_cpuaug)" \
    "w/o instance-awareness|$(cfg mask_rcnn_r50_abl_global.py)|$(ckpt mask_rcnn_r50_abl_global)" \
    "P2 only|$(cfg mask_rcnn_r50_abl_p2only.py)|$(ckpt mask_rcnn_r50_abl_p2only)" \
    "cosine only|$(cfg mask_rcnn_r50_abl_cosonly.py)|$(ckpt mask_rcnn_r50_abl_cosonly)" \
    "l1 only|$(cfg mask_rcnn_r50_abl_l1only.py)|$(ckpt mask_rcnn_r50_abl_l1only)"
    # Luu y: "w/o stop-gradient" chua dua vao vi 6 he thong/nhom da la
    # nhieu - them tay neu muon: "w/o stop-gradient|$(cfg mask_rcnn_r50_abl_nosg.py)|$(ckpt mask_rcnn_r50_abl_nosg)"

echo ""
echo "========================================"
if [ -s "$FAILED_LOG" ]; then
    echo "CO NHOM THAT BAI - xem chi tiet:"
    cat "$FAILED_LOG"
else
    echo "TAT CA 5 ANH DA SINH THANH CONG trong ${OUT_DIR}/"
fi
ls -la "$OUT_DIR"/*.png 2>/dev/null
