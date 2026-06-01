_base_ = '../mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py'

dataset_type = 'CocoDataset'
data_root = 'mmdet_dataset/'

metainfo = {
    'classes': ('lettuce',),
    'palette': [(220, 20, 60)]
}

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)
    ),
    neck=dict(
        type='ASPPFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        aspp_dilations=(1, 3, 6, 9)
    )
)



train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/train/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=0)
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/val.json',
        data_prefix=dict(img='images/val/'),
        test_mode=True
    )
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/test.json',
    metric=['bbox', 'segm']
)
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/val.json',
    metric=['bbox', 'segm']
)

test_dataloader = dict(
    batch_size=1,
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

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=40,
    val_interval=2
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=0.0025,
        momentum=0.9,
        weight_decay=0.0001
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
        end=40,
        by_epoch=True,
        milestones=[28, 36],
        gamma=0.1
    )
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='coco/segm_mAP',
        rule='greater',
        max_keep_ckpts=2
    ),
    logger=dict(
        type='LoggerHook',
        interval=50
    )
)

load_from = 'checkpoints/mask_rcnn_r50.pth'

work_dir = './work_dirs/mask-rcnn_r50_aspp-fpn'