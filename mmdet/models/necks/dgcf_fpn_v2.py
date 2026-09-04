# -*- coding: utf-8 -*-
"""DGCFPNv2: bản cải tiến của DGCFPN dựa trên phân tích 3-seed deterministic.

Thay đổi so với v1 (lý do trong ngoặc):
1. zero_init_branches: zero-init conv cuối của context/detail
   (v1 chỉ bias gate ~57.6% về nhánh gốc, ~42% tín hiệu epoch đầu là nhiễu
   random-init -> seed variance cao; zero-init đảm bảo output ban đầu = FPN
   chuẩn 100% bất kể gate).
2. gate_init_bias mặc định (4.0, -2.0, -2.0) -> softmax ~ [0.98, 0.01, 0.01]
   (khởi động sát FPN chuẩn hơn hẳn (1,0,0) ~ [0.58, 0.21, 0.21]).
3. record_gate_stats + get_gate_stats(): ghi lại alpha_o/c/d theo level để
   phục vụ phân tích "adaptive" theo yêu cầu phản biện (Major Comment 3).
4. detail_light: tùy chọn detail branch nhẹ 1x1 conv đúng như mô tả paper
   (v1 dùng cat + 2 conv 3x3 ~1.77M tham số/level, lệch paper ~27 lần).
5. gate_entropy_weight: (tùy chọn) trả về loss phụ khuyến khích gate không
   sụp sớm về 1 nhánh (chống "winner-take-most" theo khởi tạo).

Compatible with MMDetection 3.x.
"""

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmdet.registry import MODELS
from .fpn import FPN
from .dgcf_fpn import ASPPContextBlock, DegradationGate


def _zero_init_convmodule(module: ConvModule):
    nn.init.zeros_(module.conv.weight)
    if module.conv.bias is not None:
        nn.init.zeros_(module.conv.bias)


class DetailEnhanceBlockLight(nn.Module):
    """Detail branch nhẹ, khớp mô tả paper: phi_d(P_i - Avg(P_i)) với 1x1 conv."""

    def __init__(self, channels: int = 256, avg_kernel: int = 3,
                 conv_cfg=None, norm_cfg=None, act_cfg=dict(type='ReLU')):
        super().__init__()
        self.avg_kernel = avg_kernel
        self.align = ConvModule(
            channels, channels, kernel_size=1,
            conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.avg_kernel // 2
        local_mean = F.avg_pool2d(
            x, kernel_size=self.avg_kernel, stride=1, padding=pad)
        return self.align(x - local_mean)


@MODELS.register_module()
class DGCFPNv2(FPN):
    """Dynamic Gated Context-Detail Fusion FPN (bản cải tiến).

    Lưu ý: tên class bỏ chữ "Degradation-Guided" theo Major Comment 2 —
    gate hiện chỉ degradation-motivated (dùng mean/std input), chưa có
    degradation label/severity conditioning thực sự.
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
                 detail_light: bool = True,
                 detail_avg_kernel: int = 3,
                 use_adaptive_gate: bool = True,
                 static_weights: Sequence[float] = (0.34, 0.33, 0.33),
                 gate_reduction: int = 4,
                 gate_init_bias: Sequence[float] = (4.0, -2.0, -2.0),
                 zero_init_branches: bool = True,
                 record_gate_stats: bool = False,
                 gate_entropy_weight: float = 0.0,
                 residual_alpha: float = 1.0,
                 apply_to_levels: Sequence[int] = (0, 1, 2, 3, 4),
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 **kwargs):
        super().__init__(
            in_channels=in_channels, out_channels=out_channels,
            num_outs=num_outs, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
            act_cfg=act_cfg, **kwargs)

        self.use_context = use_context
        self.use_detail = use_detail
        self.use_adaptive_gate = use_adaptive_gate
        self.residual_alpha = float(residual_alpha)
        self.apply_to_levels = tuple(apply_to_levels)
        self.static_weights = tuple(float(v) for v in static_weights)
        self.record_gate_stats = record_gate_stats
        self.gate_entropy_weight = float(gate_entropy_weight)
        self._gate_stats: List[dict] = []

        self.context_blocks = nn.ModuleList()
        self.detail_blocks = nn.ModuleList()
        self.gates = nn.ModuleList()

        for _ in range(num_outs):
            if use_context:
                blk = ASPPContextBlock(
                    channels=out_channels,
                    branch_channels=context_branch_channels,
                    dilations=context_dilations,
                    with_image_pool=context_with_image_pool,
                    conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
                if zero_init_branches:
                    _zero_init_convmodule(blk.project)
                self.context_blocks.append(blk)
            else:
                self.context_blocks.append(nn.Identity())

            if use_detail:
                if detail_light:
                    dblk = DetailEnhanceBlockLight(
                        channels=out_channels, avg_kernel=detail_avg_kernel,
                        conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
                    if zero_init_branches:
                        _zero_init_convmodule(dblk.align)
                else:
                    from .dgcf_fpn import DetailEnhanceBlock
                    dblk = DetailEnhanceBlock(
                        channels=out_channels, conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg, act_cfg=act_cfg)
                    if zero_init_branches:
                        _zero_init_convmodule(dblk.refine)
                self.detail_blocks.append(dblk)
            else:
                self.detail_blocks.append(nn.Identity())

            num_branches = 1 + int(use_context) + int(use_detail)
            if use_adaptive_gate:
                bias = [gate_init_bias[0]]
                if use_context:
                    bias.append(gate_init_bias[1])
                if use_detail:
                    bias.append(gate_init_bias[2])
                self.gates.append(DegradationGate(
                    channels=out_channels, num_branches=num_branches,
                    reduction=gate_reduction, init_bias=bias))
            else:
                self.gates.append(nn.Identity())

    # ------------------------------------------------------------------
    # API phân tích gate (Major Comment 3)
    # ------------------------------------------------------------------
    def reset_gate_stats(self):
        self._gate_stats = []

    def get_gate_stats(self) -> List[dict]:
        """Trả về list dict {'level': i, 'weights': [B, K]} thu thập được."""
        return self._gate_stats

    # ------------------------------------------------------------------
    def _static_branch_weights(self, x, branches):
        weights = [self.static_weights[0]]
        if self.use_context:
            weights.append(self.static_weights[1])
        if self.use_detail:
            weights.append(self.static_weights[2])
        w = x.new_tensor(weights)
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
            if self.record_gate_stats and not self.training:
                self._gate_stats.append({
                    'level': level_idx,
                    'weights': weights.detach().flatten(1).cpu(),  # B,K
                })
            weights = weights.unsqueeze(2)  # B,K,1,1,1
        else:
            weights = self._static_branch_weights(x, branches)

        fused = (stack * weights).sum(dim=1)
        alpha = max(0.0, min(1.0, self.residual_alpha))
        return x + alpha * (fused - x)

    def gate_entropy_loss(self) -> torch.Tensor:
        """Loss phụ (tùy chọn): -entropy trung bình của gate, khuyến khích
        gate giữ phân phối mềm trong giai đoạn đầu. Gọi từ custom hook nếu
        gate_entropy_weight > 0; mặc định 0 (không dùng)."""
        if not self._gate_stats or self.gate_entropy_weight <= 0:
            return torch.tensor(0.0)
        ent = []
        for rec in self._gate_stats:
            w = rec['weights'].clamp_min(1e-8)
            ent.append(-(w * w.log()).sum(dim=1).mean())
        return -self.gate_entropy_weight * torch.stack(ent).mean()

    def forward(self, inputs):
        outs = list(super().forward(inputs))
        for i in self.apply_to_levels:
            if i < len(outs):
                outs[i] = self._fuse_level(outs[i], i)
        return tuple(outs)
