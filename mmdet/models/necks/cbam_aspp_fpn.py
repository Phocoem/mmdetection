import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS
from mmcv.cnn import ConvModule
from mmdet.models.necks.fpn import FPN

from visualize_feature import save_feature_map


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False)
        )

    def forward(self, x):
        avg_out = self.mlp(F.adaptive_avg_pool2d(x, 1))
        max_out = self.mlp(F.adaptive_max_pool2d(x, 1))
        return torch.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return torch.sigmoid(self.conv(x_cat))


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=(1, 3, 6, 9)):
        super().__init__()

        self.convs = nn.ModuleList()
        for d in dilations:
            self.convs.append(
                ConvModule(
                    in_channels,
                    out_channels,
                    kernel_size=3 if d > 1 else 1,
                    padding=d if d > 1 else 0,
                    dilation=d,
                    norm_cfg=dict(type='BN'),
                    act_cfg=dict(type='ReLU')
                )
            )

        self.project = ConvModule(
            out_channels * len(dilations),
            out_channels,
            kernel_size=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU')
        )

    def forward(self, x):
        outs = [conv(x) for conv in self.convs]
        outs = torch.cat(outs, dim=1)
        return self.project(outs)


class ECA(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)  # N,1,C
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * y


@MODELS.register_module()
class CBAMASPPFPN(FPN):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 cbam_reduction=16,
                 aspp_dilations=(1, 3, 6, 9),
                 **kwargs):

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_outs=num_outs,
            **kwargs
        )

        self.cbams = nn.ModuleList([
            CBAM(out_channels, cbam_reduction)
            for _ in range(num_outs)
        ])

        self.aspps = nn.ModuleList([
            ASPP(out_channels, out_channels, aspp_dilations)
            for _ in range(num_outs)
        ])

    # =====================================================
    # FORWARD
    # =====================================================

    def forward(self, inputs):

        import os
        log_file = "model_shapes_log.txt"
        
        # Đảm bảo file tồn tại
        if not os.path.exists(log_file):
            with open(log_file, "w", encoding='utf-8') as f:
                f.write("=== LOG KÍCH THƯỚC NECK (CBAM-ASPP-FPN) ===\n")

        # Ghi log cho từng tầng đặc trưng đầu vào từ Backbone
        with open(log_file, "a", encoding='utf-8') as f:
            f.write(f"\n--- Forward Pass tại: {self.__class__.__name__} ---\n")
            for i, feat in enumerate(inputs):
                # Thay vì dùng 'x', chúng ta dùng 'feat' trong vòng lặp
                f.write(f"  - Input C{i+2} shape: {list(feat.shape)}\n")
            f.write("-" * 30 + "\n")
            
        # FPN output
        outs = super().forward(inputs)

        new_outs = []

        for i, out in enumerate(outs):

            # =====================================
            # INPUT FEATURE
            # =====================================

            input_feat = out

            # =====================================
            # CBAM
            # =====================================

            cbam_out = self.cbams[i](input_feat)

            # =====================================
            # ASPP
            # =====================================

            aspp_out = self.aspps[i](input_feat)

            # =====================================
            # CBAM + ASPP
            # =====================================

            cbam_aspp_out = self.aspps[i](cbam_out)

            # =====================================
            # SAVE ONLY ONCE
            # =====================================

            if not hasattr(self, 'saved'):

                self.saved = True

                save_feature_map(
                    input_feat,
                    f'./work_dirs/feature_vis/P{i+2}_input.png'
                )

                save_feature_map(
                    cbam_out,
                    f'./work_dirs/feature_vis/P{i+2}_cbam.png'
                )

                save_feature_map(
                    aspp_out,
                    f'./work_dirs/feature_vis/P{i+2}_aspp.png'
                )

                save_feature_map(
                    cbam_aspp_out,
                    f'./work_dirs/feature_vis/P{i+2}_cbam_aspp.png'
                )

            new_outs.append(cbam_aspp_out)

        return tuple(new_outs)


@MODELS.register_module()
class ECAFPN(FPN):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_outs,
        eca_kernel_size=3,
        **kwargs
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_outs=num_outs,
            **kwargs
        )

        self.ecas = nn.ModuleList([
            ECA(out_channels, k_size=eca_kernel_size)
            for _ in range(num_outs)
        ])

    def forward(self, inputs):
        outs = super().forward(inputs)
        outs = [self.ecas[i](outs[i]) for i in range(len(outs))]
        return tuple(outs)


@MODELS.register_module()
class CBAMASPPBiFPN(FPN):

    def __init__(
        self,
        in_channels,
        out_channels,
        num_outs,
        cbam_reduction=16,
        aspp_dilations=(1, 3, 6, 9),
        **kwargs
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_outs=num_outs,
            **kwargs
        )

        self.cbams = nn.ModuleList([
            CBAM(out_channels, cbam_reduction)
            for _ in range(num_outs)
        ])

        self.aspps = nn.ModuleList([
            ASPP(out_channels, out_channels, aspp_dilations)
            for _ in range(num_outs)
        ])

        self.res_convs = nn.ModuleList([
            ConvModule(
                out_channels * 4,
                out_channels,
                kernel_size=1,
                norm_cfg=dict(type='BN'),
                act_cfg=None
            )
            for _ in range(num_outs)
        ])

        self.downsample_convs = nn.ModuleList([
            ConvModule(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU')
            )
            for _ in range(num_outs - 1)
        ])

        self.bifpn_fuse = nn.ModuleList([
            ConvModule(
                out_channels * 2,
                out_channels,
                kernel_size=3,
                padding=1,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU')
            )
            for _ in range(num_outs - 1)
        ])

    def forward(self, inputs):
        outs = super().forward(inputs)

        enhanced_outs = []
        for i, out in enumerate(outs):
            cbam_out = self.cbams[i](out)
            aspp_out = self.aspps[i](out)
            cbam_aspp_out = self.aspps[i](cbam_out)

            fused = torch.cat([out, cbam_out, aspp_out, cbam_aspp_out], dim=1)
            fused = self.res_convs[i](fused)
            enhanced_outs.append(out + fused)

        for i in range(len(enhanced_outs) - 1):
            down = self.downsample_convs[i](enhanced_outs[i])
            fused = self.bifpn_fuse[i](torch.cat([enhanced_outs[i + 1], down], dim=1))
            enhanced_outs[i + 1] = enhanced_outs[i + 1] + fused

        return tuple(enhanced_outs)
