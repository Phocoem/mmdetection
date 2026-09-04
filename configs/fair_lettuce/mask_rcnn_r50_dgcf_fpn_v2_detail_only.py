# Ablation "detail-only" (Major Comment 10, mục context-only/detail-only):
# giữ detail (thành phần có bằng chứng 3/3 seed ở gaussian_s2/s3), bỏ context
# (thành phần không có bằng chứng nhất quán ở contrast).
_base_ = './mask_rcnn_r50_dgcf_fpn_v2.py'

model = dict(neck=dict(use_context=False))

work_dir = 'work_dirs/research/mask_rcnn_r50_dgcf_v2_detail_only/default'
