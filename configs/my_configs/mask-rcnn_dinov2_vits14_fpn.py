_base_ = '../mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py'

dataset_type = 'CocoDataset'
data_root = 'data/mmdet_dataset_0003/'

metainfo = {
    'classes': ('lettuce',),
    'palette': [(220, 20, 60)]
}

custom_imports = dict(
    imports=['mmpretrain.models'],
    allow_failed_imports=False
)

model = dict(
backbone=dict(
    _delete_=True,
    type='mmpretrain.VisionTransformer',
    arch=dict(
        embed_dims=384,
        num_layers=12,
        num_heads=6,
        feedforward_channels=1536
    ),
    img_size=518,
    patch_size=14,
    out_indices=(2, 5, 8, 11),
    out_type='featmap',
    final_norm=False,
    init_cfg=dict(
        type='Pretrained',
        checkpoint='https://download.openmmlab.com/mmpretrain/v1.0/dinov2/vit-small-p14_dinov2-pre_3rdparty_20230426-5641ca5a.pth',
        prefix='backbone.'
    )
    ),
    neck=dict(
        type='FPN',
        in_channels=[384, 384, 384, 384],
        out_channels=256,
        num_outs=5
    ),
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)
    )
)

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/train/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=0)
    )
)

val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/val.json',
        data_prefix=dict(img='images/val/'),
        test_mode=True
    )
)

test_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/test.json',
        data_prefix=dict(img='images/'),
        test_mode=True
    )
)

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/val.json',
    metric=['bbox', 'segm']
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/test.json',
    metric=['bbox', 'segm']
)

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=50,
    val_interval=1
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        betas=(0.9, 0.999),
        weight_decay=0.05
    )
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=500
    ),
    dict(
        type='MultiStepLR',
        begin=0,
        end=50,
        by_epoch=True,
        milestones=[35, 45],
        gamma=0.1
    )
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='coco/segm_mAP',
        rule='greater',
        max_keep_ckpts=3
    ),
    logger=dict(
        type='LoggerHook',
        interval=50
    )
)

load_from = None

work_dir = './work_dirs/mask-rcnn_dinov2_vits14_fpn'