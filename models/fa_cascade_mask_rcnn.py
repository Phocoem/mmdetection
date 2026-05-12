import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS
from mmdet.models.detectors.cascade_rcnn import CascadeRCNN


class FA(nn.Module):
    def __init__(self, c=256):
        super().__init__()

        self.fuse = nn.Sequential(
            nn.Conv2d(c * 2, c, kernel_size=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        low = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        high = x - low

        out = torch.cat([low, high], dim=1)
        out = self.fuse(out)

        return x + out


@MODELS.register_module()
class FACascadeMaskRCNN(CascadeRCNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.fa = nn.ModuleList([
            FA(256),
            FA(256),
            FA(256),
            FA(256),
            FA(256)
        ])

    def extract_feat(self, batch_inputs):
        feats = super().extract_feat(batch_inputs)

        feats = tuple(
            self.fa[i](feats[i]) for i in range(len(feats))
        )

        return feats