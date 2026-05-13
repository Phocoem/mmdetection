import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS
from mmcv.cnn import ConvModule
from mmdet.models.necks.fpn import FPN


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

    def forward(self, inputs):
        outs = super().forward(inputs)

        new_outs = []
        for i, out in enumerate(outs):
            out = self.cbams[i](out)
            out = self.aspps[i](out)
            new_outs.append(out)

        return tuple(new_outs)