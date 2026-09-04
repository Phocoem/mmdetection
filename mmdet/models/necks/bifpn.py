# -*- coding: utf-8 -*-
"""BiFPN đơn giản hóa cho MMDetection 3.x (baseline theo Major Comment 8).

Tham khảo: Tan et al., "EfficientDet: Scalable and Efficient Object Detection",
CVPR 2020. Bản này dùng conv thường (không depthwise) để công bằng về capacity
với FPN chuẩn, fast normalized fusion với trọng số học được >= 0.
Nhận C2-C5, xuất P2-P6 (5 mức) tương thích Mask R-CNN strides [4,8,16,32,64].
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmdet.registry import MODELS
from mmengine.model import BaseModule


class WeightedFusion(nn.Module):
    """Fast normalized fusion: sum(w_i * x_i) / (sum(w_i) + eps), w_i >= 0."""

    def __init__(self, num_inputs: int):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_inputs))

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        w = F.relu(self.weights)
        w = w / (w.sum() + 1e-4)
        out = 0
        for i, x in enumerate(inputs):
            out = out + w[i] * x
        return out


class BiFPNBlock(nn.Module):
    """Một block BiFPN: top-down rồi bottom-up trên 5 mức."""

    def __init__(self, channels: int, num_levels: int = 5,
                 conv_cfg=None, norm_cfg=None, act_cfg=dict(type='ReLU')):
        super().__init__()
        self.num_levels = num_levels
        # top-down: các node trung gian cho level 0..num_levels-2
        self.td_fuse = nn.ModuleList(
            [WeightedFusion(2) for _ in range(num_levels - 1)])
        self.td_conv = nn.ModuleList([
            ConvModule(channels, channels, 3, padding=1, conv_cfg=conv_cfg,
                       norm_cfg=norm_cfg, act_cfg=act_cfg)
            for _ in range(num_levels - 1)
        ])
        # bottom-up: level 1..num_levels-1; level trung gian nhận 3 input
        self.bu_fuse = nn.ModuleList()
        for i in range(1, num_levels):
            self.bu_fuse.append(WeightedFusion(3 if i < num_levels - 1 else 2))
        self.bu_conv = nn.ModuleList([
            ConvModule(channels, channels, 3, padding=1, conv_cfg=conv_cfg,
                       norm_cfg=norm_cfg, act_cfg=act_cfg)
            for _ in range(num_levels - 1)
        ])

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        n = self.num_levels
        # Top-down pathway
        td = [None] * n
        td[n - 1] = feats[n - 1]
        for i in range(n - 2, -1, -1):
            up = F.interpolate(td[i + 1], size=feats[i].shape[-2:],
                               mode='nearest')
            td[i] = self.td_conv[i](self.td_fuse[i]([feats[i], up]))
        # Bottom-up pathway
        out = [None] * n
        out[0] = td[0]
        for i in range(1, n):
            down = F.max_pool2d(out[i - 1], kernel_size=2, ceil_mode=True)
            if down.shape[-2:] != td[i].shape[-2:]:
                down = F.interpolate(down, size=td[i].shape[-2:],
                                     mode='nearest')
            if i < n - 1:
                fused = self.bu_fuse[i - 1]([feats[i], td[i], down])
            else:
                fused = self.bu_fuse[i - 1]([td[i], down])
            out[i] = self.bu_conv[i - 1](fused)
        return out


@MODELS.register_module()
class BiFPN(BaseModule):
    """BiFPN neck: in_channels (C2-C5) -> num_outs=5 (P2-P6)."""

    def __init__(self,
                 in_channels: List[int],
                 out_channels: int = 256,
                 num_outs: int = 5,
                 num_blocks: int = 2,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 init_cfg=dict(type='Xavier', layer='Conv2d',
                               distribution='uniform')):
        super().__init__(init_cfg=init_cfg)
        assert num_outs == len(in_channels) + 1, \
            'Bản này hỗ trợ num_outs = số backbone levels + 1 (thêm P6).'
        self.lateral_convs = nn.ModuleList([
            ConvModule(c, out_channels, 1, conv_cfg=conv_cfg,
                       norm_cfg=norm_cfg, act_cfg=None)
            for c in in_channels
        ])
        self.blocks = nn.ModuleList([
            BiFPNBlock(out_channels, num_levels=num_outs, conv_cfg=conv_cfg,
                       norm_cfg=norm_cfg, act_cfg=act_cfg)
            for _ in range(num_blocks)
        ])

    def forward(self, inputs):
        feats = [l(x) for l, x in zip(self.lateral_convs, inputs)]
        # Sinh P6 từ mức cao nhất bằng maxpool stride 2 (giống FPN chuẩn)
        feats.append(F.max_pool2d(feats[-1], kernel_size=1, stride=2))
        for block in self.blocks:
            feats = block(feats)
        return tuple(feats)
