# -*- coding: utf-8 -*-
"""DGCFPNv2Flex: mở rộng DGCFPNv2 cho ablation sâu (Major Comment 10).

Thêm 3 trục ablation mà v2 chưa hỗ trợ:
1. gate_mode:
   - 'adaptive_softmax' (mặc định, = v2)
   - 'adaptive_sigmoid'  (softmax vs sigmoid — mỗi nhánh gate độc lập [0,1])
   - 'learned_static'    (fixed learned weights: nn.Parameter toàn cục,
                          KHÔNG phụ thuộc input — kiểm tra gate có thực sự
                          cần input-dependence hay chỉ cần 3 số học được)
   - 'equal'             (trọng số bằng nhau cố định 1/K)
2. detail_filter: 'avg' | 'laplacian' | 'sobel'
   (average filter so với Laplacian/Sobel high-pass — Major Comment 10 mục 12)
3. detail_avg_kernel: 3 | 5 | 7 (kernel size ablation)

Compatible with MMDetection 3.x.
"""

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmdet.registry import MODELS
from .dgcf_fpn_v2 import DGCFPNv2, _zero_init_convmodule

_LAPLACIAN = torch.tensor(
    [[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])
_SOBEL_X = torch.tensor(
    [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
_SOBEL_Y = torch.tensor(
    [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])


class DetailFilterBlock(nn.Module):
    """Detail branch với bộ lọc high-pass thay thế được."""

    def __init__(self, channels: int, filter_type: str = 'avg',
                 avg_kernel: int = 3, conv_cfg=None, norm_cfg=None,
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        assert filter_type in ('avg', 'laplacian', 'sobel')
        self.filter_type = filter_type
        self.avg_kernel = avg_kernel
        self.channels = channels
        if filter_type == 'laplacian':
            k = _LAPLACIAN.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
            self.register_buffer('lap_kernel', k)
        elif filter_type == 'sobel':
            kx = _SOBEL_X.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
            ky = _SOBEL_Y.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
            self.register_buffer('sobel_x', kx)
            self.register_buffer('sobel_y', ky)
        self.align = ConvModule(channels, channels, kernel_size=1,
                                conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                                act_cfg=act_cfg)

    def _extract(self, x: torch.Tensor) -> torch.Tensor:
        if self.filter_type == 'avg':
            pad = self.avg_kernel // 2
            return x - F.avg_pool2d(x, self.avg_kernel, stride=1, padding=pad)
        if self.filter_type == 'laplacian':
            return F.conv2d(x, self.lap_kernel, padding=1,
                            groups=self.channels)
        gx = F.conv2d(x, self.sobel_x, padding=1, groups=self.channels)
        gy = F.conv2d(x, self.sobel_y, padding=1, groups=self.channels)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.align(self._extract(x))


class LearnedStaticWeights(nn.Module):
    """Trọng số học được toàn cục, không phụ thuộc input (softmax(param))."""

    def __init__(self, num_branches: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_branches))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.logits, dim=0)
        return w.view(1, -1, 1, 1).expand(x.size(0), -1, 1, 1)


class SigmoidGateWrapper(nn.Module):
    """Bọc DegradationGate, thay softmax cuối bằng sigmoid độc lập từng nhánh."""

    def __init__(self, base_gate: nn.Module):
        super().__init__()
        self.base = base_gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = F.adaptive_avg_pool2d(x, 1)
        var = F.adaptive_avg_pool2d((x - mean).pow(2), 1)
        std = torch.sqrt(var + 1e-6)
        stats = torch.cat([mean, std], dim=1)
        logits = self.base.fc2(self.base.act(self.base.fc1(stats)))
        return torch.sigmoid(logits)


@MODELS.register_module()
class DGCFPNv2Flex(DGCFPNv2):

    def __init__(self, *args,
                 gate_mode: str = 'adaptive_softmax',
                 detail_filter: str = 'avg',
                 conv_cfg=None, norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 **kwargs):
        assert gate_mode in ('adaptive_softmax', 'adaptive_sigmoid',
                             'learned_static', 'equal')
        # 'equal' và 'learned_static' xây trên use_adaptive_gate=False/custom
        if gate_mode == 'equal':
            kwargs['use_adaptive_gate'] = False
            k = 1 + int(kwargs.get('use_context', True)) \
                + int(kwargs.get('use_detail', True))
            kwargs['static_weights'] = tuple([1.0 / k] * 3)
        super().__init__(*args, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                         act_cfg=act_cfg, **kwargs)
        self.gate_mode = gate_mode

        # Thay detail block nếu filter khác 'avg'
        if self.use_detail and detail_filter != 'avg':
            zero_init = kwargs.get('zero_init_branches', True)
            new_blocks = nn.ModuleList()
            for _ in range(len(self.detail_blocks)):
                blk = DetailFilterBlock(
                    channels=self.out_channels, filter_type=detail_filter,
                    avg_kernel=kwargs.get('detail_avg_kernel', 3),
                    conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
                if zero_init:
                    _zero_init_convmodule(blk.align)
                new_blocks.append(blk)
            self.detail_blocks = new_blocks

        # Thay gate theo mode
        num_branches = 1 + int(self.use_context) + int(self.use_detail)
        if gate_mode == 'learned_static':
            self.use_adaptive_gate = True  # để _fuse_level gọi gates[i](x)
            self.gates = nn.ModuleList([
                LearnedStaticWeights(num_branches)
                for _ in range(len(self.gates))])
        elif gate_mode == 'adaptive_sigmoid':
            self.gates = nn.ModuleList([
                SigmoidGateWrapper(g) for g in self.gates])
