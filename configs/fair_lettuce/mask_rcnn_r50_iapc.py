# IAPC: Instance-Aware Pyramid Consistency (đề xuất chính)
# Kiến trúc GIỮ NGUYÊN Mask R-CNN + FPN chuẩn -> chi phí suy luận = 0.
# Khác biệt duy nhất: thêm consistency loss lúc train.
_base_ = './mask_rcnn_r50_fpn.py'

custom_imports = dict(
    imports=['mmdet.models.detectors.consistency_mask_rcnn',
             'mmdet.datasets.transforms.robust_aug'],
    allow_failed_imports=False)

model = dict(
    type='IAPCMaskRCNN',
    consistency_weight=1.0,
    consistency_levels=(0, 1, 2, 3),   # P2..P5
    level_weights=(1.0, 1.0, 1.0, 1.0),
    instance_aware=True,               # chỉ ép nhất quán ở vùng cây
    dilate_mask_px=8,                  # bao cả vùng ngay ngoài biên
    detach_clean=True,                 # nhánh sạch làm "teacher"
    cos_weight=0.5,                    # hybrid cosine + L1
    warmup_iters=500,
    corruption_cfg=dict(
        brightness_delta=0.35,
        contrast_range=(0.55, 1.45),
        noise_sigma_range=(0.05, 0.30),
        prob_brightness=0.7,
        prob_contrast=0.7,
        prob_noise=0.7))

# Pipeline train: KHÔNG augmentation ở CPU. Nhánh nhiễu được sinh trên GPU
# bên trong detector để có cặp (sạch, nhiễu) hoàn toàn khớp nhau.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(800, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
work_dir = 'work_dirs/research/mask_rcnn_r50_iapc/default'
