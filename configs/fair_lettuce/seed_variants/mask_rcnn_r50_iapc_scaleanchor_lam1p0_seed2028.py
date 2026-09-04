# TU DONG SINH boi generate_seed_configs.py - KHONG sua tay.
# fair_protocol.py hardcode randomness.seed=2025 dung chung cho moi
# he thong qua _base_ - file nay OVERRIDE rieng de dam bao seed=2028
# thuc su duoc ap dung doc lap, khong bi ke thua nham gia tri 2025.
_base_ = '/home/pc/mmdet_AI/mmdetection/configs/fair_lettuce/mask_rcnn_r50_iapc_scaleanchor_lam1p0.py'

randomness = dict(seed=2028, deterministic=True)
work_dir = 'work_dirs/research/mask_rcnn_r50_iapc_scaleanchor_lam1p0/seed_2028'
