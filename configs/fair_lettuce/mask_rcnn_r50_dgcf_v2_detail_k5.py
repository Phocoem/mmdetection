# Ablation: average kernel 5x5 thay vì 3x3.
_base_ = './mask_rcnn_r50_dgcf_fpn_v2.py'
model = dict(neck=dict(detail_avg_kernel=5))
work_dir = 'work_dirs/research/mask_rcnn_r50_dgcf_v2_k5/default'
