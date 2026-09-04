# -*- coding: utf-8 -*-
"""DGCFPN: Degradation-Guided Context-Detail Fusion FPN.

This neck is designed for robustness-oriented instance segmentation.

Core idea:
    Standard FPN features are stable but may lose robustness under image
    degradation. ASPP-like context is strong under several corruptions, but it is
    fixed and may over-smooth details. DGCFPN keeps three paths:
        1) original FPN feature
        2) multi-scale context feature
        3) high-frequency/detail feature
    and learns a degradation-guided adaptive gate to fuse them per FPN level.

This is not plain ASPP. The ASPP-like branch is used as a context candidate.
The novelty is the degradation-guided context-detail fusion mechanism.

Compatible with MMDetection 3.x.
"""

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmdet.registry import MODELS
from .fpn import FPN


class ASPPContextBlock(nn.Module):
    """ASPP-like context block for one FPN level."""

    def __init__(self,
                 channels: int = 256,
                 branch_channels: int = 64,
                 dilations: Sequence[int] = (1, 3, 6, 9),
                 with_image_pool: bool = True,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        self.with_image_pool = with_image_pool
        self.branches = nn.ModuleList()

        for dilation in dilations:
            if dilation == 1:
                self.branches.append(
                    ConvModule(
                        channels,
                        branch_channels,
                        kernel_size=1,
                        padding=0,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))
            else:
                self.branches.append(
                    ConvModule(
                        channels,
                        branch_channels,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))

        if self.with_image_pool:
            self.image_pool = ConvModule(
                channels,
                branch_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            num_branches = len(dilations) + 1
        else:
            self.image_pool = None
            num_branches = len(dilations)

        self.project = ConvModule(
            branch_channels * num_branches,
            channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [branch(x) for branch in self.branches]
        if self.with_image_pool:
            pooled = F.adaptive_avg_pool2d(x, output_size=1)
            pooled = self.image_pool(pooled)
            pooled = F.interpolate(
                pooled, size=x.shape[-2:], mode='bilinear', align_corners=False)
            outs.append(pooled)
        out = torch.cat(outs, dim=1)
        return self.project(out)


class DetailEnhanceBlock(nn.Module):
    """High-frequency/detail enhancement block.

    It estimates local detail by subtracting local average response and then
    fuses the original feature with the detail residual. This branch is useful
    for blur/motion degradation but is adaptively gated to avoid amplifying
    noise when not needed.
    """

    def __init__(self,
                 channels: int = 256,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        self.fuse = ConvModule(
            channels * 2,
            channels,
            kernel_size=3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.refine = ConvModule(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_mean = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        detail = x - local_mean
        out = self.fuse(torch.cat([x, detail], dim=1))
        return self.refine(out)


class DegradationGate(nn.Module):
    """Global degradation-aware branch gate.

    The gate uses feature mean/std statistics to estimate which path should be
    trusted more at the current level:
        original FPN / context branch / detail branch.
    """

    def __init__(self,
                 channels: int = 256,
                 num_branches: int = 3,
                 reduction: int = 4,
                 init_bias: Sequence[float] = (1.0, 0.0, 0.0)):
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.fc1 = nn.Conv2d(channels * 2, hidden, kernel_size=1)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, num_branches, kernel_size=1)
        self.num_branches = num_branches

        nn.init.normal_(self.fc2.weight, std=0.001)
        nn.init.constant_(self.fc2.bias, 0.0)
        if init_bias is not None:
            bias = torch.tensor(list(init_bias), dtype=torch.float32)
            if bias.numel() == num_branches:
                with torch.no_grad():
                    self.fc2.bias.copy_(bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = F.adaptive_avg_pool2d(x, output_size=1)
        var = F.adaptive_avg_pool2d((x - mean).pow(2), output_size=1)
        std = torch.sqrt(var + 1e-6)
        stats = torch.cat([mean, std], dim=1)
        logits = self.fc2(self.act(self.fc1(stats)))
        return torch.softmax(logits, dim=1)


@MODELS.register_module()
class DGCFPN(FPN):
    """Degradation-Guided Context-Detail Fusion FPN.

    Args:
        use_context: use ASPP-like multi-scale context branch.
        use_detail: use high-frequency/detail branch.
        use_adaptive_gate: if False, static_weights are used.
        static_weights: fixed fusion weights for ablation without adaptive gate.
            Order is [original, context, detail]. If a branch is disabled, the
            remaining weights are re-normalized.
        residual_alpha: output = x + alpha * (fused - x). A value in [0,1]
            keeps training stable. 1.0 means direct fused output.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 context_dilations: Sequence[int] = (1, 3, 6, 9),
                 context_branch_channels: int = 64,
                 context_with_image_pool: bool = True,
                 use_context: bool = True,
                 use_detail: bool = True,
                 use_adaptive_gate: bool = True,
                 static_weights: Sequence[float] = (0.34, 0.33, 0.33),
                 gate_reduction: int = 4,
                 gate_init_bias: Sequence[float] = (1.0, 0.0, 0.0),
                 residual_alpha: float = 1.0,
                 apply_to_levels: Sequence[int] = (0, 1, 2, 3, 4),
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 **kwargs):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_outs=num_outs,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **kwargs)

        self.use_context = use_context
        self.use_detail = use_detail
        self.use_adaptive_gate = use_adaptive_gate
        self.residual_alpha = float(residual_alpha)
        self.apply_to_levels = tuple(apply_to_levels)
        self.static_weights = tuple(float(v) for v in static_weights)

        self.context_blocks = nn.ModuleList()
        self.detail_blocks = nn.ModuleList()
        self.gates = nn.ModuleList()

        for _ in range(num_outs):
            if use_context:
                self.context_blocks.append(
                    ASPPContextBlock(
                        channels=out_channels,
                        branch_channels=context_branch_channels,
                        dilations=context_dilations,
                        with_image_pool=context_with_image_pool,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))
            else:
                self.context_blocks.append(nn.Identity())

            if use_detail:
                self.detail_blocks.append(
                    DetailEnhanceBlock(
                        channels=out_channels,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))
            else:
                self.detail_blocks.append(nn.Identity())

            num_branches = 1 + int(use_context) + int(use_detail)
            if use_adaptive_gate:
                bias = [gate_init_bias[0]]
                if use_context:
                    bias.append(gate_init_bias[1])
                if use_detail:
                    bias.append(gate_init_bias[2])
                self.gates.append(
                    DegradationGate(
                        channels=out_channels,
                        num_branches=num_branches,
                        reduction=gate_reduction,
                        init_bias=bias))
            else:
                self.gates.append(nn.Identity())

    def _static_branch_weights(self, x: torch.Tensor, branches):
        weights = [self.static_weights[0]]
        if self.use_context:
            weights.append(self.static_weights[1])
        if self.use_detail:
            weights.append(self.static_weights[2])
        w = x.new_tensor(weights, dtype=x.dtype)
        w = torch.clamp(w, min=0)
        w = w / (w.sum() + 1e-6)
        return w.view(1, len(branches), 1, 1, 1)

    def _fuse_level(self, x: torch.Tensor, level_idx: int) -> torch.Tensor:
        branches = [x]
        if self.use_context:
            branches.append(self.context_blocks[level_idx](x))
        if self.use_detail:
            branches.append(self.detail_blocks[level_idx](x))

        stack = torch.stack(branches, dim=1)  # B,K,C,H,W

        if self.use_adaptive_gate:
            weights = self.gates[level_idx](x)  # B,K,1,1
            weights = weights.unsqueeze(2)      # B,K,1,1,1
        else:
            weights = self._static_branch_weights(x, branches)

        fused = (stack * weights).sum(dim=1)
        alpha = max(0.0, min(1.0, self.residual_alpha))
        return x + alpha * (fused - x)

    def forward(self, inputs):
        outs = list(super().forward(inputs))
        for i in self.apply_to_levels:
            if i < len(outs):
                outs[i] = self._fuse_level(outs[i], i)
        return tuple(outs)
