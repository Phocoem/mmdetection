# Baseline kiến trúc: BiFPN (Major Comment 8).
_base_ = './mask_rcnn_r50_fpn.py'

custom_imports = dict(
    imports=['mmdet.models.necks.bifpn'],
    allow_failed_imports=False)

model = dict(
    neck=dict(
        _delete_=True,
        type='BiFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        num_blocks=2,
    ))

work_dir = 'work_dirs/research/mask_rcnn_r50_bifpn/seed2027'
