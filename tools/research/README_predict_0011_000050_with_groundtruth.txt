HƯỚNG DẪN: PREDICT 0011_000050 KÈM GROUND TRUTH
================================================

Bộ file này bổ sung Ground Truth visualization vào workflow predict.

File gồm:

1. visualize_groundtruth_0011_000050.py
   - Đọc COCO annotation.
   - Tìm image_id theo stem 0011_000050.
   - Vẽ mask GT + bbox GT lên ảnh clean/corrupted.
   - Vì Lettuce-C là label-preserving corruption benchmark nên GT của ảnh clean
     vẫn dùng hợp lệ cho brightness/contrast/gaussian_noise.

2. predict_0011_000050_conditions_with_gt.sh
   - Tạo Ground Truth cho từng condition.
   - Chạy predict các mô hình.
   - Condition mặc định:
       clean
       brightness_s3
       contrast_s3
       gaussian_noise_s3
   - Kết quả lưu:
       work_dirs/research/predict_0011_000050_conditions_with_gt/

3. make_prediction_grid_0011_000050.py
   - Ghép hình paper-style:
       Input | Ground Truth | Mask R-CNN R50 | Mask R-CNN R101 | DGCF-FPN
   - Các hàng:
       Clean
       Brightness S3
       Contrast S3
       Gaussian noise S3


CÁCH CÀI VÀ CHẠY
----------------

Từ thư mục project:

    cd /home/pc/mmdet_AI/mmdetection

Copy file vào tools/research:

    cp visualize_groundtruth_0011_000050.py tools/research/
    cp predict_0011_000050_conditions_with_gt.sh tools/research/
    cp make_prediction_grid_0011_000050.py tools/research/

Cấp quyền chạy:

    chmod +x tools/research/predict_0011_000050_conditions_with_gt.sh

Chạy predict + GT:

    bash tools/research/predict_0011_000050_conditions_with_gt.sh

Sau khi predict xong, ghép hình so sánh:

    python tools/research/make_prediction_grid_0011_000050.py


OUTPUT
------

Ground truth từng condition:

    work_dirs/research/predict_0011_000050_conditions_with_gt/<condition>/ground_truth/

Prediction từng mô hình:

    work_dirs/research/predict_0011_000050_conditions_with_gt/<condition>/<model>/

Hình grid cuối:

    work_dirs/research/predict_0011_000050_conditions_with_gt/prediction_grid.png


GHI CHÚ QUAN TRỌNG
------------------

- Script dùng annotation gốc:
    mmdet_dataset/lettuce/annotations/test.json

- Nếu ảnh 0011_000050 không nằm trong test.json, script sẽ báo lỗi.
- Nếu demo/image_demo.py xuất tên file khác, make_prediction_grid sẽ tự tìm file bắt đầu bằng 0011_000050.
- Với hình chính trong paper, chỉ nên dùng:
    Input | Ground Truth | Mask R-CNN R50 | Mask R-CNN R101 | DGCF-FPN
  Không nên đưa quá nhiều mô hình vào hình chính vì sẽ rối.
