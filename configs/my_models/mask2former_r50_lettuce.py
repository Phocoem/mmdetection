_base_ = '../mask2former/mask2former_r50_8xb2-lsj-50e_coco.py'


data_root = 'mmdet_dataset/'

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
    max_epochs=50,
    val_interval=5
)

default_hooks = dict(
    checkpoint=dict(interval=5, max_keep_ckpts=3),
    logger=dict(interval=50)
)


model = dict(
    panoptic_head=dict(
        num_classes=num_classes
    ),
    panoptic_fusion_head=dict(
        num_instances=num_classes
    )
)

train_cfg = dict(
    type='IterBasedTrainLoop', 
    max_iters=10000,         
    val_interval=500      
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', 
        by_epoch=False, 
        interval=1000, 
        max_keep_ckpts=3
    ),
    logger=dict(type='LoggerHook', interval=50)
)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.05),
    clip_grad=dict(max_norm=0.01, norm_type=2)
)

load_from = 'checkpoints/mask2former_r50.pth'
work_dir = './work_dirs/mask2former_r50_lettuce'
