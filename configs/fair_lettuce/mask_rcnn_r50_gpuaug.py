# ĐỐI CHỨNG THEN CHỐT: dùng ĐÚNG phân phối nhiễu GPU như IAPC nhưng
# TẮT consistency (lambda=0). Đây là baseline công bằng nhất — mọi khác
# biệt so với IAPC chỉ đến từ consistency loss, không phải từ việc đổi
# cách sinh augmentation.
_base_ = './mask_rcnn_r50_iapc.py'
model = dict(consistency_weight=0.0)
work_dir = 'work_dirs/research/mask_rcnn_r50_gpuaug/default'
