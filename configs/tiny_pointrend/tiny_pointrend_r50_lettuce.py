_base_ = '../point_rend/point-rend_r50-caffe_fpn_ms-1x_coco.py'

dataset_type = 'CocoDataset'
data_root = 'data/mmdet_dataset/'

metainfo = {
    'classes': ('lettuce',),
    'palette': [(0, 255, 0)]
}

model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=1
        ),

        mask_head=dict(
            num_classes=1
        ),

        point_head=dict(
            num_classes=1
            # KHÔNG để num_points ở đây
        )
    ),

    train_cfg=dict(
        rcnn=dict(
            mask_point=dict(
                num_points=784,
                oversample_ratio=3,
                importance_sample_ratio=0.95
            )
        )
    )
)

train_dataloader = dict(
    batch_size=2,
    num_workers=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/train/')
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/val.json',
        data_prefix=dict(img='images/val/')
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/test.json',
        data_prefix=dict(img='images/')
    )
)

val_evaluator = dict(
    ann_file=data_root + 'annotations/val.json',
    metric=['bbox', 'segm']
)

test_evaluator = dict(
    ann_file=data_root + 'annotations/test.json',
    metric=['bbox', 'segm']
)

train_cfg = dict(max_epochs=50)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='coco/segm_mAP',
        rule='greater',
        max_keep_ckpts=3
    )
)

load_from = 'checkpoints/mask_rcnn_r50.pth'