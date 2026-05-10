_base_ = '../cascade_rcnn/cascade-mask-rcnn_r50_fpn_1x_coco.py'

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS
from mmdet.models.detectors import CascadeRCNN


# =========================
# Frequency Attention Module
# =========================

class FA(nn.Module):

    def __init__(self, c=256):
        super().__init__()

        self.fuse = nn.Sequential(
            nn.Conv2d(c * 2, c, 1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        # low-frequency
        low = F.avg_pool2d(x, 3, 1, 1)

        # high-frequency
        high = x - low

        out = torch.cat([low, high], dim=1)

        out = self.fuse(out)

        return x + out


# =========================
# FA Cascade Mask R-CNN
# =========================

@MODELS.register_module()
class FACascadeMaskRCNN(CascadeRCNN):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.fa = nn.ModuleList([
            FA(256) for _ in range(5)
        ])

    def extract_feat(self, batch_inputs):

        feats = super().extract_feat(batch_inputs)

        feats = tuple(
            m(f)
            for m, f in zip(self.fa, feats)
        )

        return feats


# =========================
# DATASET
# =========================

dataset_type = 'CocoDataset'

data_root = 'data/mmdet_dataset/'

metainfo = {
    'classes': ('lettuce',),
    'palette': [(0, 255, 0)]
}


# =========================
# MODEL
# =========================

model = dict(

    type='FACascadeMaskRCNN',

    roi_head=dict(

        bbox_head=[

            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(
                    type='SmoothL1Loss',
                    beta=1.0,
                    loss_weight=1.0)
            ),

            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(
                    type='SmoothL1Loss',
                    beta=1.0,
                    loss_weight=1.0)
            ),

            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(
                    type='SmoothL1Loss',
                    beta=1.0,
                    loss_weight=1.0)
            )
        ],

        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=1,
            loss_mask=dict(
                type='CrossEntropyLoss',
                use_mask=True,
                loss_weight=1.0)
        )
    )
)


# =========================
# DATALOADER
# =========================

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


# =========================
# EVALUATOR
# =========================

val_evaluator = dict(
    ann_file=data_root + 'annotations/val.json',
    metric=['bbox', 'segm']
)

test_evaluator = dict(
    ann_file=data_root + 'annotations/test.json',
    metric=['bbox', 'segm']
)


# =========================
# TRAIN
# =========================

train_cfg = dict(max_epochs=50)

load_from = 'checkpoints/cascade_mask_rcnn_r50.pth'