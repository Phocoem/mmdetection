# Ablation: gate sigmoid (mỗi nhánh độc lập [0,1]) thay vì softmax.
_base_ = './mask_rcnn_r50_dgcf_fpn_v2.py'
custom_imports = dict(
    imports=['mmdet.models.necks.dgcf_fpn', 'mmdet.models.necks.dgcf_fpn_v2',
             'mmdet.models.necks.dgcf_fpn_v2_variants'],
    allow_failed_imports=False)
model = dict(neck=dict(type='DGCFPNv2Flex', gate_mode='adaptive_sigmoid'))
work_dir = 'work_dirs/research/mask_rcnn_r50_dgcf_v2_sigmoid/default'
