import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class MFGNet(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.reduce = nn.Conv2d(2048, 256, kernel_size=1)

        self.decoder = nn.Sequential(
            ConvBlock(256, 256),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(256, 128),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(128, 64),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(64, 64),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
        )

        self.mask_head = nn.Conv2d(64, 1, kernel_size=1)
        self.center_head = nn.Conv2d(64, 1, kernel_size=1)
        self.radial_head = nn.Conv2d(64, 2, kernel_size=1)
        self.boundary_head = nn.Conv2d(64, 1, kernel_size=1)
        self.distance_head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[-2:]

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.reduce(x)
        feat = self.decoder(x)

        feat = F.interpolate(feat, size=input_size, mode="bilinear", align_corners=False)

        mask = torch.sigmoid(self.mask_head(feat))
        center = torch.sigmoid(self.center_head(feat))
        radial = F.normalize(self.radial_head(feat), p=2, dim=1)
        boundary = torch.sigmoid(self.boundary_head(feat))
        distance = torch.sigmoid(self.distance_head(feat))

        return {
            "mask": mask,
            "center": center,
            "radial": radial,
            "boundary": boundary,
            "distance": distance
        }