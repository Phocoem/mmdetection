# BiFPN + CPU photometric aug - to hop kien truc tot nhat Tier 2 (BiFPN,
# APcorr=0.649 khong augmentation) voi augmentation Tier 1, chua tung
# duoc test trong ban goc. Bo sung thi nghiem con thieu #8/#12.
_base_ = './mask_rcnn_r50_bifpn.py'
custom_imports = dict(
    imports=['mmdet.datasets.transforms.robust_aug'],
    allow_failed_imports=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(800, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion', brightness_delta=32,
         contrast_range=(0.5, 1.5), saturation_range=(0.7, 1.3),
         hue_delta=10),
    dict(type='RandomGaussianNoise', prob=0.5, sigma_range=(0.01, 0.06)),
    dict(type='PackDetInputs'),
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
work_dir = 'work_dirs/research/mask_rcnn_r50_bifpn_aug/seed2026'
