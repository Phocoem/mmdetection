import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS
from mmdet.models.detectors import CascadeRCNN


class FA(nn.Module):
    def __init__(self, c=256):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(c * 2, c, 1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        low = F.avg_pool2d(x, 3, 1, 1)
        high = x - low
        return x + self.fuse(torch.cat([low, high], dim=1))


@MODELS.register_module()
class FACascadeMaskRCNN(CascadeRCNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fa = nn.ModuleList([FA(256) for _ in range(5)])

    def extract_feat(self, batch_inputs):
        feats = super().extract_feat(batch_inputs)
        return tuple(m(f) for m, f in zip(self.fa, feats))