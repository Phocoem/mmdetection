# Baseline kiến trúc: CBAM-FPN (Major Comment 8).
_base_ = './mask_rcnn_r50_fpn.py'

custom_imports = dict(
    imports=['mmdet.models.necks.cbam_fpn'],
    allow_failed_imports=False)

model = dict(
    neck=dict(
        type='CBAMFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        cbam_reduction=16,
        cbam_spatial_kernel=7,
    ))

work_dir = 'work_dirs/research/mask_rcnn_r50_cbam_fpn/seed2027'
