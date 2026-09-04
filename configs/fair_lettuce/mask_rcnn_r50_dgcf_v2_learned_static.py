# Ablation: fixed learned weights — 3 tham số toàn cục học được, KHÔNG phụ
# thuộc input. Nếu biến thể này đạt ngang adaptive gate => gate không thực sự
# cần input-dependence (trả lời trực tiếp Major Comment 3).
_base_ = './mask_rcnn_r50_dgcf_fpn_v2.py'
custom_imports = dict(
    imports=['mmdet.models.necks.dgcf_fpn', 'mmdet.models.necks.dgcf_fpn_v2',
             'mmdet.models.necks.dgcf_fpn_v2_variants'],
    allow_failed_imports=False)
model = dict(neck=dict(type='DGCFPNv2Flex', gate_mode='learned_static'))
work_dir = 'work_dirs/research/mask_rcnn_r50_dgcf_v2_learned_static/default'
