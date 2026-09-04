# -*- coding: utf-8 -*-
"""
CORA-FPN: Clean-Only Reliability-Guided ASPP-FPN for MMDetection 3.x.

Train only on clean images. During training, this neck stores clean feature
statistics as EMA buffers. During inference, each FPN level is compared with
this clean reference. If a feature level deviates from the clean distribution,
the neck increases the contribution of an ASPP context branch.

Expected path:
  mmdet/models/necks/clean_reliability_aspp_fpn.py
"""
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmdet.registry import MODELS
from mmdet.models.necks.fpn import FPN


class LiteASPP(nn.Module):
    """Lightweight ASPP block for 256-channel FPN features."""

    def __init__(self, channels: int, rates: Sequence[int] = (1, 3, 6, 9),
                 branch_channels: int = 64, with_image_pool: bool = True) -> None:
        super().__init__()
        self.with_image_pool = with_image_pool
        self.branches = nn.ModuleList()

        for rate in rates:
            if rate == 1:
                self.branches.append(
                    ConvModule(channels, branch_channels, kernel_size=1, padding=0,
                               norm_cfg=None, act_cfg=dict(type='ReLU')))
            else:
                self.branches.append(
                    ConvModule(channels, branch_channels, kernel_size=3,
                               padding=rate, dilation=rate,
                               norm_cfg=None, act_cfg=dict(type='ReLU')))

        concat_channels = branch_channels * len(self.branches)
        if with_image_pool:
            self.image_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                ConvModule(channels, branch_channels, kernel_size=1,
                           norm_cfg=None, act_cfg=dict(type='ReLU')))
            concat_channels += branch_channels
        else:
            self.image_pool = None

        self.project = ConvModule(concat_channels, channels, kernel_size=1,
                                  norm_cfg=None, act_cfg=dict(type='ReLU'))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [branch(x) for branch in self.branches]
        if self.image_pool is not None:
            pooled = self.image_pool(x)
            pooled = F.interpolate(pooled, size=x.shape[-2:], mode='bilinear',
                                   align_corners=False)
            outs.append(pooled)
        return self.project(torch.cat(outs, dim=1))


@MODELS.register_module()
class CleanReliabilityASPPFPN(FPN):
    """FPN with clean-reference reliability-guided ASPP fusion.

    No corrupted images are required during training.

    Fusion:
      P'_l = (1 - alpha_l) * P_l + alpha_l * ASPP(P_l)

    alpha_l increases when feature level l deviates from clean training statistics.
    """

    def __init__(self,
                 aspp_rates: Sequence[int] = (1, 3, 6, 9),
                 aspp_branch_channels: int = 64,
                 aspp_with_image_pool: bool = True,
                 stat_momentum: float = 0.01,
                 reliability_tau: float = 1.0,
                 gate_temperature: float = 3.0,
                 context_min: float = 0.05,
                 context_max: float = 0.65,
                 update_clean_stats: bool = True,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.stat_momentum = stat_momentum
        self.reliability_tau = reliability_tau
        self.gate_temperature = gate_temperature
        self.context_min = context_min
        self.context_max = context_max
        self.update_clean_stats = update_clean_stats

        self.aspp_blocks = nn.ModuleList([
            LiteASPP(self.out_channels, rates=aspp_rates,
                     branch_channels=aspp_branch_channels,
                     with_image_pool=aspp_with_image_pool)
            for _ in range(self.num_outs)
        ])

        self.register_buffer('clean_mean', torch.zeros(self.num_outs))
        self.register_buffer('clean_var', torch.ones(self.num_outs))
        self.register_buffer('stats_initialized', torch.tensor(0, dtype=torch.uint8))

        self.latest_alpha_context = None
        self.latest_deviation = None

    @staticmethod
    def _feature_score(outs):
        scores = []
        for feat in outs:
            s = feat.detach().float().pow(2).mean(dim=(1, 2, 3)).sqrt()
            scores.append(s)
        return torch.stack(scores, dim=1)

    @torch.no_grad()
    def _update_stats(self, scores: torch.Tensor) -> None:
        if not self.update_clean_stats:
            return
        batch_mean = scores.mean(dim=0)
        batch_var = scores.var(dim=0, unbiased=False).clamp_min(1e-6)
        L = scores.shape[1]
        if self.stats_initialized.item() == 0:
            self.clean_mean[:L].copy_(batch_mean)
            self.clean_var[:L].copy_(batch_var)
            self.stats_initialized.fill_(1)
        else:
            m = self.stat_momentum
            self.clean_mean[:L].mul_(1.0 - m).add_(batch_mean * m)
            self.clean_var[:L].mul_(1.0 - m).add_(batch_var * m)

    def _compute_alpha(self, scores: torch.Tensor) -> torch.Tensor:
        L = scores.shape[1]
        if self.stats_initialized.item() == 0:
            return scores.new_full((scores.shape[0], L), self.context_min)

        mean = self.clean_mean[:L].view(1, L)
        std = self.clean_var[:L].clamp_min(1e-6).sqrt().view(1, L)
        deviation = (scores - mean).abs() / std
        alpha_raw = torch.sigmoid(self.gate_temperature * (deviation - self.reliability_tau))
        alpha_context = self.context_min + (self.context_max - self.context_min) * alpha_raw
        self.latest_deviation = deviation.detach()
        self.latest_alpha_context = alpha_context.detach()
        return alpha_context

    def forward(self, inputs):
        outs = list(super().forward(inputs))
        scores = self._feature_score(outs)
        if self.training:
            self._update_stats(scores)
        alpha = self._compute_alpha(scores)

        fused_outs = []
        for i, feat in enumerate(outs):
            context_feat = self.aspp_blocks[i](feat)
            a = alpha[:, i].view(-1, 1, 1, 1).to(dtype=feat.dtype, device=feat.device)
            fused_outs.append((1.0 - a) * feat + a * context_feat)
        return tuple(fused_outs)
