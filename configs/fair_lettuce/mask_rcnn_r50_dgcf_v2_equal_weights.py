# Ablation: trọng số bằng nhau cố định (1/3,1/3,1/3), không gate.
_base_ = './mask_rcnn_r50_dgcf_fpn_v2.py'
custom_imports = dict(
    imports=['mmdet.models.necks.dgcf_fpn', 'mmdet.models.necks.dgcf_fpn_v2',
             'mmdet.models.necks.dgcf_fpn_v2_variants'],
    allow_failed_imports=False)
model = dict(neck=dict(type='DGCFPNv2Flex', gate_mode='equal'))
work_dir = 'work_dirs/research/mask_rcnn_r50_dgcf_v2_equal/default'
