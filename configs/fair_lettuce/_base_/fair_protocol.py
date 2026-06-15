"""Shared experimental protocol for fair lettuce instance segmentation."""

fair_data_root = 'mmdet_dataset/lettuce/'
fair_metainfo = dict(classes=('lettuce',), palette=[(220, 20, 60)])

fair_data_preprocessor = dict(
    type='DetDataPreprocessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_mask=True,
    pad_size_divisor=32)

fair_r50_backbone = dict(
    _delete_=True,
    type='ResNet',
    depth=50,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=1,
    norm_cfg=dict(type='BN', requires_grad=True),
    norm_eval=True,
    style='pytorch',
    init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'))

fair_r101_backbone = dict(
    _delete_=True,
    type='ResNet',
    depth=101,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    frozen_stages=1,
    norm_cfg=dict(type='BN', requires_grad=True),
    norm_eval=True,
    style='pytorch',
    init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101'))

fair_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(800, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]

fair_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(800, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor')),
]

fair_train_dataloader = dict(
    _delete_=True,
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='CocoDataset',
        data_root=fair_data_root,
        metainfo=fair_metainfo,
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/train/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=0),
        pipeline=fair_train_pipeline))

fair_val_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=fair_data_root,
        metainfo=fair_metainfo,
        ann_file='annotations/val.json',
        data_prefix=dict(img='images/val/'),
        test_mode=True,
        pipeline=fair_test_pipeline))

fair_test_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=fair_data_root,
        metainfo=fair_metainfo,
        ann_file='annotations/test.json',
        data_prefix=dict(img='images/test/'),
        test_mode=True,
        pipeline=fair_test_pipeline))

fair_val_evaluator = dict(
    _delete_=True,
    type='CocoMetric',
    ann_file=fair_data_root + 'annotations/val.json',
    metric='segm',
    format_only=False)

fair_test_evaluator = dict(
    _delete_=True,
    type='CocoMetric',
    ann_file=fair_data_root + 'annotations/test.json',
    metric='segm',
    format_only=False)

fair_train_cfg = dict(
    _delete_=True, type='EpochBasedTrainLoop', max_epochs=200, val_interval=1)
fair_val_cfg = dict(_delete_=True, type='ValLoop')
fair_test_cfg = dict(_delete_=True, type='TestLoop')

fair_optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05),
    clip_grad=dict(max_norm=1.0, norm_type=2))

fair_param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='ReduceOnPlateauLR',
        monitor='coco/segm_mAP',
        rule='greater',
        factor=0.5,
        patience=5,
        threshold=0.001,
        threshold_rule='abs',
        cooldown=1,
        min_value=1e-6,
        begin=0,
        end=200,
        by_epoch=True,
        verbose=True),
]

fair_default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='coco/segm_mAP',
        rule='greater',
        max_keep_ckpts=3,
        save_last=True),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

fair_randomness = dict(seed=2026, deterministic=True)
fair_auto_scale_lr = dict(enable=False, base_batch_size=2)
fair_custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='coco/segm_mAP',
        rule='greater',
        min_delta=0.001,
        patience=20,
        strict=True,
        check_finite=True)
]

fair_vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
fair_visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=fair_vis_backends,
    name='visualizer')
fair_log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
fair_log_level = 'INFO'
fair_env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
