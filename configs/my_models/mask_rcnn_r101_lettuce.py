_base_ = '../mask_rcnn/mask-rcnn_r101_fpn_1x_coco.py'


data_root = 'data/mmdet_dataset/'

metainfo = {
    'classes': ('lettuce',),
    'palette': [(220, 20, 60)]
}

num_classes = 1

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
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

test_dataloader = val_dataloader

val_evaluator = dict(
    ann_file=data_root + 'annotations/val.json',
    metric=['bbox', 'segm']
)

test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=40,
    val_interval=2
)

default_hooks = dict(
    checkpoint=dict(interval=1, save_best='coco/segm_mAP', rule='greater', max_keep_ckpts=2),
    logger=dict(interval=50)
)


model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=num_classes),
        mask_head=dict(num_classes=num_classes)
    )
)

load_from = 'checkpoints/mask_rcnn_r101.pth'

work_dir = './work_dirs/mask_rcnn_r101_lettuce'
