_base_ = [
    '../mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py',
    './_base_/fair_protocol.py',
]

train_dataloader = _base_.fair_train_dataloader
val_dataloader = _base_.fair_val_dataloader
test_dataloader = _base_.fair_test_dataloader
val_evaluator = _base_.fair_val_evaluator
test_evaluator = _base_.fair_test_evaluator
train_cfg = _base_.fair_train_cfg
val_cfg = _base_.fair_val_cfg
test_cfg = _base_.fair_test_cfg
optim_wrapper = _base_.fair_optim_wrapper
param_scheduler = _base_.fair_param_scheduler
default_hooks = _base_.fair_default_hooks
randomness = _base_.fair_randomness
auto_scale_lr = _base_.fair_auto_scale_lr
custom_hooks = _base_.fair_custom_hooks
visualizer = _base_.fair_visualizer
log_processor = _base_.fair_log_processor
log_level = _base_.fair_log_level
env_cfg = _base_.fair_env_cfg
load_from = None
resume = False

model = dict(
    data_preprocessor=_base_.fair_data_preprocessor,
    backbone=_base_.fair_r50_backbone,
    neck=dict(
        type='FPFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)))

work_dir = './work_dirs/fair_lettuce/mask_rcnn_r50_fpfpn'
