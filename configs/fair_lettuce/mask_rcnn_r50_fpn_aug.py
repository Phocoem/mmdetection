# Baseline robustness bằng augmentation (Major Comment 8):
# PhotoMetricDistortion (brightness/contrast/saturation/hue jitter)
# + RandomGaussianNoise (sigma nhẹ hơn severity test để tránh train-on-test).
# Kiến trúc giữ nguyên FPN chuẩn — trả lời câu hỏi phản biện:
# "DGCF-FPN có tốt hơn việc đơn giản dùng color jitter/noise aug không?"
_base_ = './mask_rcnn_r50_fpn.py'

custom_imports = dict(
    imports=['mmdet.datasets.transforms.robust_aug'],
    allow_failed_imports=False)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(keep_ratio=True, scale=(800, 800), type='Resize'),
    dict(prob=0.5, type='RandomFlip'),
    # brightness_delta 32/255 ~ 0.125; contrast_range bao phủ vùng nhẹ-vừa
    dict(type='PhotoMetricDistortion',
         brightness_delta=32,
         contrast_range=(0.5, 1.5),
         saturation_range=(0.7, 1.3),
         hue_delta=10),
    dict(type='RandomGaussianNoise', prob=0.5, sigma_range=(0.01, 0.06)),
    dict(type='PackDetInputs'),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

work_dir = 'work_dirs/research/mask_rcnn_r50_fpn_aug/default'
