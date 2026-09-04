import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.registry import MODELS
from mmdet.models.backbones.resnet import ResNet


class DGCFModule(nn.Module):
    """C4-level Degradation-Guided Context-Detail Fusion.

    C4' = C4 + gamma * (G * Context(C4) + (1-G) * Detail(C4))
    """

    def __init__(self,
                 in_channels,
                 reduction=4,
                 dilations=(1, 3, 5),
                 use_context=True,
                 use_detail=True,
                 use_gate=True,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 init_gamma=0.0):
        super().__init__()
        mid_channels = max(in_channels // reduction, 64)
        self.use_context = use_context
        self.use_detail = use_detail
        self.use_gate = use_gate

        if use_context:
            self.context_reduce = ConvModule(
                in_channels, mid_channels, 1,
                norm_cfg=norm_cfg, act_cfg=act_cfg)
            self.context_convs = nn.ModuleList([
                ConvModule(
                    mid_channels, mid_channels, 3,
                    padding=d, dilation=d,
                    norm_cfg=norm_cfg, act_cfg=act_cfg)
                for d in dilations
            ])
            self.context_fuse = ConvModule(
                mid_channels * len(dilations), in_channels, 1,
                norm_cfg=norm_cfg, act_cfg=act_cfg)

        if use_detail:
            self.detail = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1,
                          groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, 1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
            )

        if use_gate:
            self.gate = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 1, bias=True),
                nn.Sigmoid()
            )

        self.gamma = nn.Parameter(torch.tensor(float(init_gamma)))

    def forward(self, x):
        if self.use_context:
            xc = self.context_reduce(x)
            f_context = self.context_fuse(
                torch.cat([conv(xc) for conv in self.context_convs], dim=1)
            )
        else:
            f_context = torch.zeros_like(x)

        if self.use_detail:
            f_detail = self.detail(x)
        else:
            f_detail = torch.zeros_like(x)

        if self.use_gate:
            g = self.gate(x)
        else:
            g = torch.full_like(x, 0.5)

        return x + self.gamma * (g * f_context + (1.0 - g) * f_detail)


@MODELS.register_module()
class C4DGCFResNet(ResNet):
    """ResNet that enhances C4 before FPN.

    Standard Mask R-CNN FPN uses:
        C2, C3, C4, C5 -> FPN -> P2-P6 -> RPN/RoI heads

    This backbone returns:
        C2, C3, C4', C5
    """

    def __init__(self,
                 dgcf=dict(
                     reduction=4,
                     dilations=(1, 3, 5),
                     use_context=True,
                     use_detail=True,
                     use_gate=True,
                     init_gamma=0.0),
                 **kwargs):
        super().__init__(**kwargs)

        if self.depth in [50, 101, 152]:
            c4_channels = 1024
        elif self.depth in [18, 34]:
            c4_channels = 256
        else:
            c4_channels = 1024

        self.dgcf = DGCFModule(c4_channels, **dgcf)

    def forward(self, x):
        outs = list(super().forward(x))
        if len(outs) >= 3:
            outs[2] = self.dgcf(outs[2])
        return tuple(outs)
