_base_ = '../condinst/condinst_r50_fpn_ms-poly-90k_coco_instance.py'


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

model = dict(
    bbox_head=dict(
        num_classes=num_classes
    )
)

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=40,
    val_interval=2
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    checkpoint=dict(interval=1, save_best='coco/segm_mAP', rule='greater', max_keep_ckpts=2),
    logger=dict(interval=50)
)

load_from = 'checkpoints/condinst_r50.pth'
work_dir = './work_dirs/condinst_r50_lettuce'
