HƯỚNG DẪN CHẠY PREDICT ẢNH 0011_000050
======================================

Bộ file gồm 2 script:

1. predict_0011_000050_all_models.sh
   - Predict ảnh clean 0011_000050 trên tất cả mô hình.
   - Kết quả lưu tại:
     work_dirs/research/predict_0011_000050_all_models/

2. predict_0011_000050_conditions.sh
   - Predict ảnh 0011_000050 trên các condition:
     clean
     brightness_s3
     contrast_s3
     gaussian_noise_s3
   - Chạy trên tất cả mô hình.
   - Kết quả lưu tại:
     work_dirs/research/predict_0011_000050_conditions/

Cách dùng:

  cd /home/pc/mmdet_AI/mmdetection

  cp /đường/dẫn/tải/về/*.sh tools/research/

  chmod +x tools/research/predict_0011_000050_all_models.sh
  chmod +x tools/research/predict_0011_000050_conditions.sh

  bash tools/research/predict_0011_000050_all_models.sh
  bash tools/research/predict_0011_000050_conditions.sh

Ghi chú:
- Script giả định checkpoint nằm trong:
  work_dirs/research/<model>/no_random/best_coco_segm_mAP_epoch_*.pth

- Nếu checkpoint của bạn nằm ở thư mục khác, sửa biến CKPT_GLOB trong script.

- Với hình cho paper, nên chọn ít mô hình:
  Mask R-CNN R50
  Mask R-CNN R101
  DGCF-FPN

- Các mô hình SOLO/SOLOv2 và ablation nên để supplementary nếu hình quá rộng.
