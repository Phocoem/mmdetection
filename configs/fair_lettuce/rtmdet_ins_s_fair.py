# Modern instance segmentation baseline (Major Comment 8): RTMDet-Ins-S
# áp fair protocol (cùng data/optimizer/schedule/early-stopping với các model
# khác). LƯU Ý ghi trong bài: pipeline Mosaic/MixUp gốc của RTMDet bị thay
# bằng fair pipeline để so sánh công bằng theo protocol — RTMDet có thể đạt
# cao hơn với recipe gốc của nó; đây là protocol-fair comparison, không phải
# best-recipe comparison.
_base_ = [
    '../rtmdet/rtmdet-ins_s_8xb32-300e_coco.py',
    './_base_/fair_protocol.py',
]
train_dataloader = _base_.fair_train_dataloader
val_dataloader = _base_.fair_val_dataloader
test_dataloader = _base_.fair_test_dataloader
val_evaluator = _base_.fair_val_evaluator
test_evaluator = _base_.fair_test_evaluator
train_cfg = dict(_delete_=True, max_epochs=200,
                 type='EpochBasedTrainLoop', val_interval=1)
val_cfg = _base_.fair_val_cfg
test_cfg = _base_.fair_test_cfg
optim_wrapper = _base_.fair_optim_wrapper
param_scheduler = _base_.fair_param_scheduler
default_hooks = _base_.fair_default_hooks
randomness = _base_.fair_randomness
auto_scale_lr = _base_.fair_auto_scale_lr
custom_hooks = _base_.fair_custom_hooks  # early stopping; bỏ PipelineSwitchHook gốc
visualizer = _base_.fair_visualizer
log_processor = _base_.fair_log_processor
log_level = _base_.fair_log_level
env_cfg = _base_.fair_env_cfg
load_from = None
resume = False
model = dict(bbox_head=dict(num_classes=1))
work_dir = 'work_dirs/research/rtmdet_ins_s_fair/default'
