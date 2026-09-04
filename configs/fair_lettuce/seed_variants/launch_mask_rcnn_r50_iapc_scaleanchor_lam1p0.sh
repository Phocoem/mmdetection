#!/bin/bash
# TU DONG SINH - chay lan luot tat ca seed vua tao
# KHONG dung set -e - neu 1 seed loi, cac seed con lai van
# tiep tuc chay, loi duoc ghi vao FAILED_LOG.
FAILED_LOG=mask_rcnn_r50_iapc_scaleanchor_lam1p0_train_failures.log
> "$FAILED_LOG"
if ! python tools/train.py configs/fair_lettuce/seed_variants/mask_rcnn_r50_iapc_scaleanchor_lam1p0_seed2024.py; then
  echo "THAT BAI (train): configs/fair_lettuce/seed_variants/mask_rcnn_r50_iapc_scaleanchor_lam1p0_seed2024.py" | tee -a "$FAILED_LOG"
fi
if ! python tools/train.py configs/fair_lettuce/seed_variants/mask_rcnn_r50_iapc_scaleanchor_lam1p0_seed2025.py; then
  echo "THAT BAI (train): configs/fair_lettuce/seed_variants/mask_rcnn_r50_iapc_scaleanchor_lam1p0_seed2025.py" | tee -a "$FAILED_LOG"
fi
if ! python tools/train.py configs/fair_lettuce/seed_variants/mask_rcnn_r50_iapc_scaleanchor_lam1p0_seed2026.py; then
  echo "THAT BAI (train): configs/fair_lettuce/seed_variants/mask_rcnn_r50_iapc_scaleanchor_lam1p0_seed2026.py" | tee -a "$FAILED_LOG"
fi
if ! python tools/train.py configs/fair_lettuce/seed_variants/mask_rcnn_r50_iapc_scaleanchor_lam1p0_seed2027.py; then
  echo "THAT BAI (train): configs/fair_lettuce/seed_variants/mask_rcnn_r50_iapc_scaleanchor_lam1p0_seed2027.py" | tee -a "$FAILED_LOG"
fi
if ! python tools/train.py configs/fair_lettuce/seed_variants/mask_rcnn_r50_iapc_scaleanchor_lam1p0_seed2028.py; then
  echo "THAT BAI (train): configs/fair_lettuce/seed_variants/mask_rcnn_r50_iapc_scaleanchor_lam1p0_seed2028.py" | tee -a "$FAILED_LOG"
fi
echo ""
if [ -s "$FAILED_LOG" ]; then
  echo "CO SEED THAT BAI - xem chi tiet:"; cat "$FAILED_LOG"
else
  echo "TAT CA SEED TRAIN THANH CONG."
fi
