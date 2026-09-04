# -*- coding: utf-8 -*-
"""CBAM-FPN: FPN + Convolutional Block Attention Module trên mỗi mức pyramid.

Baseline kiến trúc robustness theo yêu cầu phản biện (Major Comment 8).
Tham khảo: Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018.
Compatible with MMDetection 3.x.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS
from .fpn import FPN


class ChannelAttention(nn.Module):

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False))

    def forward(self, x):
        # adaptive_avg_pool2d(x,1) == mean toàn cục; adaptive_max_pool2d(x,1)
        # == max toàn cục nhưng KHÔNG có backward deterministic. Thay bằng
        # amax (có deterministic) để chạy được với use_deterministic_algorithms.
        avg = self.mlp(x.mean(dim=(2, 3), keepdim=True))
        mx = self.mlp(x.amax(dim=(2, 3), keepdim=True))
        return torch.sigmoid(avg + mx)


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size,
                              padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAMBlock(nn.Module):

    def __init__(self, channels: int, reduction: int = 16,
                 spatial_kernel: int = 7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


@MODELS.register_module()
class CBAMFPN(FPN):
    """FPN với CBAM áp lên từng output pyramid level (residual)."""

    def __init__(self, *args, cbam_reduction: int = 16,
                 cbam_spatial_kernel: int = 7, **kwargs):
        super().__init__(*args, **kwargs)
        self.cbam_blocks = nn.ModuleList([
            CBAMBlock(self.out_channels, cbam_reduction, cbam_spatial_kernel)
            for _ in range(self.num_outs)
        ])

    def forward(self, inputs):
        outs = list(super().forward(inputs))
        for i in range(len(outs)):
            outs[i] = outs[i] + self.cbam_blocks[i](outs[i])
        return tuple(outs)
