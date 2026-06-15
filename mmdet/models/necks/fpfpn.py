from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmdet.registry import MODELS
from .fpn import FPN


class FrequencySplit(nn.Module):

    def __init__(self):
        super().__init__()

        self.blur = nn.AvgPool2d(
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, x):

        low = self.blur(x)

        high = x - low

        return high, low


@MODELS.register_module()
class FPFPN(FPN):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.2))

        self.freq_split = FrequencySplit()

    def forward(self, inputs: Tuple[Tensor]):

        assert len(inputs) == len(self.in_channels)

        # --------------------------
        # Build laterals
        # --------------------------

        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # --------------------------
        # Top-down path
        # --------------------------

        used_backbone_levels = len(laterals)

        for i in range(used_backbone_levels - 1, 0, -1):

            if 'scale_factor' in self.upsample_cfg:

                up = F.interpolate(
                    laterals[i],
                    **self.upsample_cfg
                )

            else:

                prev_shape = laterals[i - 1].shape[2:]

                up = F.interpolate(
                    laterals[i],
                    size=prev_shape,
                    **self.upsample_cfg
                )

            # ==========================
            # Frequency Preserving
            # ==========================

            high, low = self.freq_split(up)

            laterals[i - 1] = (
                laterals[i - 1]
                + self.alpha * low
                + self.beta * high
            )

        # --------------------------
        # Outputs
        # --------------------------

        outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels)
        ]

        # --------------------------
        # Extra levels
        # --------------------------

        if self.num_outs > len(outs):

            if not self.add_extra_convs:

                for i in range(
                        self.num_outs - used_backbone_levels):

                    outs.append(
                        F.max_pool2d(
                            outs[-1],
                            1,
                            stride=2
                        )
                    )

            else:

                if self.add_extra_convs == 'on_input':

                    extra_source = \
                        inputs[self.backbone_end_level - 1]

                elif self.add_extra_convs == 'on_lateral':

                    extra_source = laterals[-1]

                elif self.add_extra_convs == 'on_output':

                    extra_source = outs[-1]

                else:

                    raise NotImplementedError

                outs.append(
                    self.fpn_convs[used_backbone_levels](
                        extra_source
                    )
                )

                for i in range(
                        used_backbone_levels + 1,
                        self.num_outs):

                    if self.relu_before_extra_convs:

                        outs.append(
                            self.fpn_convs[i](
                                F.relu(outs[-1])
                            )
                        )

                    else:

                        outs.append(
                            self.fpn_convs[i](
                                outs[-1]
                            )
                        )
        if not self.training:
            print(
            f'alpha={self.alpha.item():.3f}, '
            f'beta={self.beta.item():.3f}'
    )

        return tuple(outs)