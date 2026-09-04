# Ablation "context-only" (Major Comment 10).
_base_ = './mask_rcnn_r50_dgcf_fpn_v2.py'

model = dict(neck=dict(use_detail=False))

work_dir = 'work_dirs/research/mask_rcnn_r50_dgcf_v2_context_only/default'
