# ĐỀ XUẤT CHÍNH: FPN + BoundaryContrastAttack (BCA) + full uniform augmentation.
# BCA lo robustness contrast (structure-aware ở biên); augmentation đồng đều
# (ngang mức fpn_aug) lo robustness noise/brightness. Kết hợp để vượt aug thường.
# Kiến trúc giữ NGUYÊN FPN chuẩn (chi phí inference = 0).
_base_ = './mask_rcnn_r50_fpn.py'

custom_imports = dict(
    imports=['mmdet.datasets.transforms.boundary_aug',
             'mmdet.datasets.transforms.robust_aug'],
    allow_failed_imports=False)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(keep_ratio=True, scale=(800, 800), type='Resize'),
    dict(prob=0.5, type='RandomFlip'),
    # BCA cần gt_masks -> đặt sau LoadAnnotations, trước PackDetInputs.
    dict(type='BoundaryContrastAttack', prob=0.5, strength=0.45,
         band_width=11, min_keep=0.35, blur_boundary_sigma=2.0),
    # Augmentation đồng đều NGANG mức fpn_aug (bù điểm yếu noise/brightness):
    dict(type='PhotoMetricDistortion', brightness_delta=32,
         contrast_range=(0.5, 1.5), saturation_range=(0.7, 1.3), hue_delta=10),
    dict(type='RandomGaussianNoise', prob=0.5, sigma_range=(0.01, 0.06)),
    dict(type='PackDetInputs'),
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
work_dir = 'work_dirs/research/mask_rcnn_r50_fpn_bca/default'
