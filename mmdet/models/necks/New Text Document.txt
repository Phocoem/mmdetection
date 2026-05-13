import torch
import torch.nn as nn

from mmcv.cnn import ConvModule
from mmdet.registry import MODELS
from mmdet.models.necks.fpn import FPN


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
class ASPPFPN(FPN):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 aspp_dilations=(1, 3, 6, 9),
                 **kwargs):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            num_outs=num_outs,
            **kwargs
        )

        self.aspps = nn.ModuleList([
            ASPP(out_channels, out_channels, aspp_dilations)
            for _ in range(num_outs)
        ])

    def forward(self, inputs):
        outs = super().forward(inputs)
        outs = [self.aspps[i](out) for i, out in enumerate(outs)]
        return tuple(outs)