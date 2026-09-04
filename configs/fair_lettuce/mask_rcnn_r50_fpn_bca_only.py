# Ablation: CHỈ BoundaryContrastAttack, không kèm augmentation đồng đều.
# So với _bca (có kèm) để đo đóng góp riêng của BCA và vai trò chống shortcut.
_base_ = './mask_rcnn_r50_fpn.py'
custom_imports = dict(
    imports=['mmdet.datasets.transforms.boundary_aug'],
    allow_failed_imports=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(keep_ratio=True, scale=(800, 800), type='Resize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='BoundaryContrastAttack', prob=0.5, strength=0.4,
         band_width=9, min_keep=0.35, blur_boundary_sigma=2.0),
    dict(type='PackDetInputs'),
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
work_dir = 'work_dirs/research/mask_rcnn_r50_fpn_bca_only/default'
