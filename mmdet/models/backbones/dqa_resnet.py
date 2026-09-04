import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.registry import MODELS
from mmdet.models.backbones.resnet import ResNet


class QualityEncoder(nn.Module):
    """Estimate degradation-quality vector from C2."""

    def __init__(self,
                 in_channels,
                 quality_dim=128,
                 hidden_channels=256,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super().__init__()
        self.conv = nn.Sequential(
            ConvModule(in_channels, hidden_channels, 3, padding=1,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
            ConvModule(hidden_channels, hidden_channels, 3, padding=1,
                       norm_cfg=norm_cfg, act_cfg=act_cfg),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_channels, quality_dim),
            nn.ReLU(inplace=True),
            nn.Linear(quality_dim, quality_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class DQAModule(nn.Module):
    """Degradation-Quality-Aware C4 modulation.

    mode='modulate':
        q = QualityEncoder(C2)
        F = Conv(C4)
        scale,bias = MLP(q)
        C4' = C4 + gamma * (scale * F + bias)

    mode='context_detail':
        q = QualityEncoder(C2)
        F_context = ASPP-like(C4)
        F_detail = DetailConv(C4)
        alpha,beta = MLP(q)
        C4' = C4 + gamma * (alpha * F_context + beta * F_detail)
    """

    def __init__(self,
                 c4_channels=1024,
                 quality_dim=128,
                 mode='modulate',
                 reduction=4,
                 dilations=(1, 3, 5),
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 init_gamma=0.1):
        super().__init__()
        assert mode in ['modulate', 'context_detail']
        self.mode = mode

        if mode == 'modulate':
            self.feature_proj = ConvModule(
                c4_channels, c4_channels, 3, padding=1,
                norm_cfg=norm_cfg, act_cfg=act_cfg)
            self.affine = nn.Sequential(
                nn.Linear(quality_dim, quality_dim),
                nn.ReLU(inplace=True),
                nn.Linear(quality_dim, c4_channels * 2)
            )
        else:
            mid_channels = max(c4_channels // reduction, 64)
            self.context_reduce = ConvModule(
                c4_channels, mid_channels, 1,
                norm_cfg=norm_cfg, act_cfg=act_cfg)
            self.context_convs = nn.ModuleList([
                ConvModule(mid_channels, mid_channels, 3,
                           padding=d, dilation=d,
                           norm_cfg=norm_cfg, act_cfg=act_cfg)
                for d in dilations
            ])
            self.context_fuse = ConvModule(
                mid_channels * len(dilations), c4_channels, 1,
                norm_cfg=norm_cfg, act_cfg=act_cfg)

            self.detail = nn.Sequential(
                nn.Conv2d(c4_channels, c4_channels, 3, padding=1,
                          groups=c4_channels, bias=False),
                nn.BatchNorm2d(c4_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(c4_channels, c4_channels, 1, bias=False),
                nn.BatchNorm2d(c4_channels),
                nn.ReLU(inplace=True),
            )

            self.weights = nn.Sequential(
                nn.Linear(quality_dim, quality_dim),
                nn.ReLU(inplace=True),
                nn.Linear(quality_dim, c4_channels * 2),
                nn.Sigmoid()
            )

        self.gamma = nn.Parameter(torch.tensor(float(init_gamma)))

    def forward(self, c4, q):
        b, c, h, w = c4.shape

        if self.mode == 'modulate':
            f = self.feature_proj(c4)
            affine = self.affine(q).view(b, 2, c, 1, 1)
            scale = torch.sigmoid(affine[:, 0])
            bias = torch.tanh(affine[:, 1])
            return c4 + self.gamma * (scale * f + bias)

        xc = self.context_reduce(c4)
        f_context = self.context_fuse(
            torch.cat([conv(xc) for conv in self.context_convs], dim=1)
        )
        f_detail = self.detail(c4)
        weights = self.weights(q).view(b, 2, c, 1, 1)
        alpha = weights[:, 0]
        beta = weights[:, 1]
        return c4 + self.gamma * (alpha * f_context + beta * f_detail)


@MODELS.register_module()
class DQAResNet(ResNet):
    """ResNet with DQA C4 modulation.

    Returns C2, C3, C4', C5.
    """

    def __init__(self,
                 dqa=dict(
                     quality_dim=128,
                     mode='modulate',
                     reduction=4,
                     dilations=(1, 3, 5),
                     init_gamma=0.1),
                 **kwargs):
        super().__init__(**kwargs)

        if self.depth in [50, 101, 152]:
            c2_channels = 256
            c4_channels = 1024
        elif self.depth in [18, 34]:
            c2_channels = 64
            c4_channels = 256
        else:
            c2_channels = 256
            c4_channels = 1024

        quality_dim = dqa.get('quality_dim', 128)

        self.quality_encoder = QualityEncoder(
            in_channels=c2_channels,
            quality_dim=quality_dim)

        self.dqa = DQAModule(
            c4_channels=c4_channels,
            quality_dim=quality_dim,
            mode=dqa.get('mode', 'modulate'),
            reduction=dqa.get('reduction', 4),
            dilations=dqa.get('dilations', (1, 3, 5)),
            init_gamma=dqa.get('init_gamma', 0.1))

    def forward(self, x):
        outs = list(super().forward(x))
        if len(outs) >= 3:
            q = self.quality_encoder(outs[0])  # C2
            outs[2] = self.dqa(outs[2], q)     # C4 -> C4'
        return tuple(outs)
