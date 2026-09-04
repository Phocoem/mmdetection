Cập nhật Ground Truth màu đỏ
===========================

File visualize_groundtruth_0011_000050.py đã được sửa để:
- Mask GT tô màu đỏ giống prediction visualization.
- Contour và bbox GT cũng màu đỏ.
- Mặc định không ghi nhãn GT 1, GT 2... để hình sạch hơn.
- Có thể bật nhãn bằng --draw-index nếu cần.

Cách dùng:

cd /home/pc/mmdet_AI/mmdetection

cp visualize_groundtruth_0011_000050.py tools/research/
cp predict_0011_000050_conditions_with_gt.sh tools/research/

chmod +x tools/research/predict_0011_000050_conditions_with_gt.sh

bash tools/research/predict_0011_000050_conditions_with_gt.sh

Nếu đã predict rồi và chỉ muốn vẽ lại GT màu đỏ, chạy thủ công ví dụ:

python tools/research/visualize_groundtruth_0011_000050.py \
  --ann mmdet_dataset/lettuce/annotations/test.json \
  --image mmdet_dataset/lettuce_c/images/contrast/3/0011_000050.png \
  --image-stem 0011_000050 \
  --out-dir work_dirs/research/predict_0011_000050_conditions_with_gt/contrast_s3/ground_truth \
  --alpha 0.45
