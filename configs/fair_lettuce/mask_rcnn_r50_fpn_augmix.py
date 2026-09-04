# FPN + AugMix (Hendrycks et al. 2020) - da duoc trich dan o Related Work
# Section 2.2 nhung chua tung duoc benchmark. File nay bo sung thi nghiem
# con thieu #6 (mmuc "thi_nghiem_con_thieu.txt" / checklist).
_base_ = './mask_rcnn_r50_fpn.py'
custom_imports = dict(
    imports=['mmdet.datasets.transforms.augmix_transform'],
    allow_failed_imports=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(800, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='AugMix', severity=3, width=3, depth_range=(1, 3),
         alpha=1.0, prob=1.0),
    dict(type='PackDetInputs'),
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
work_dir = 'work_dirs/research/mask_rcnn_r50_fpn_augmix/seed2026'
