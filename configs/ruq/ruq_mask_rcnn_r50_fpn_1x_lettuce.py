# RUQ-Mask R-CNN config for MMDetection 3.x
# Dataset: COCO format, single class: lettuce/crop
# Put this file under: configs/ruq/ruq_mask_rcnn_r50_fpn_1x_lettuce.py

_base_ = '../mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py'

custom_imports = dict(
    imports=[
        'mmdet.models.roi_heads.ruq_standard_roi_head',
        'mmdet.models.roi_heads.mask_heads.ruq_fcn_mask_head',
    ],
    allow_failed_imports=False)

# ===== Dataset =====
data_root = 'data/lettuce_coco/'
metainfo = {
    'classes': ('lettuce', ),
    'palette': [(0, 220, 0)]
}

# Robust but not too destructive training augmentation.
# This is separate from the robustness benchmark used for testing.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomResize', scale=[(640, 640), (800, 800), (1024, 1024)], keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.6, 1.4),
        saturation_range=(0.6, 1.4),
        hue_delta=12),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='annotations/train.json',
        data_prefix=dict(img='train/'),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=True, min_size=16),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='annotations/val.json',
        data_prefix=dict(img='val/'),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline))

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='annotations/test.json',
        data_prefix=dict(img='test/'),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/val.json',
    metric=['bbox', 'segm'],
    format_only=False)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/test.json',
    metric=['bbox', 'segm'],
    format_only=False)

# ===== Model =====
model = dict(
    roi_head=dict(
        type='RUQStandardRoIHead',
        mask_head=dict(
            _delete_=True,
            type='RUQFCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=1,
            loss_mask=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
            # Extra losses for RUQ
            loss_quality_weight=0.5,
            loss_uncertainty_weight=0.3,
            boundary_kernel=3)))

# ===== Training schedule =====
# For a small agricultural dataset, 50 epochs is a reasonable first run.
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=50, by_epoch=True, milestones=[35, 45], gamma=0.1)
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2))

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=5, save_best='coco/segm_mAP', rule='greater', max_keep_ckpts=3),
    logger=dict(type='LoggerHook', interval=50))

# Load COCO pretrained Mask R-CNN weights if available from the base config.
# If your MMDetection version does not auto-load it, pass --cfg-options load_from=PATH_OR_URL.
