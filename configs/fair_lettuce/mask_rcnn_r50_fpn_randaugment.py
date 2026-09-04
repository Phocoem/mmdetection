# FPN + RandAugment (dai dien "policy-search augmentation" - Cubuk et al.,
# da trich dan Related Work Section 2.2 nhung chua benchmark). Bo sung
# thi nghiem con thieu #7.
_base_ = './mask_rcnn_r50_fpn.py'
custom_imports = dict(
    imports=['mmdet.datasets.transforms.randaugment_photometric'],
    allow_failed_imports=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(800, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandAugmentPhotometric', num_ops=2, magnitude=5, prob=1.0),
    dict(type='PackDetInputs'),
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
work_dir = 'work_dirs/research/mask_rcnn_r50_fpn_randaugment/seed2026'
