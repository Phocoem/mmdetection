from typing import List, Optional, Tuple

import torch
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import DetDataSample, SampleList
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import ConfigType, InstanceList
from ..task_modules.samplers import SamplingResult
from ..utils import empty_instances, unpack_gt_instances
from .standard_roi_head import StandardRoIHead


@MODELS.register_module()
class StandardRoIHeadPCN(StandardRoIHead):

    def __init__(
        self,
        proposal_compression_head=None,
        *args,
        **kwargs
    ):

        super().__init__(*args, **kwargs)

        self.proposal_compression_head = None

        if proposal_compression_head is not None:
            self.proposal_compression_head = MODELS.build(
                proposal_compression_head
            )

    def compress_proposals(
        self,
        x,
        rpn_results_list
    ):

        if self.proposal_compression_head is None:
            return rpn_results_list

        compressed_results = []

        for proposals in rpn_results_list:

            proposal_boxes = proposals.bboxes

            if proposal_boxes.numel() == 0:
                compressed_results.append(proposals)
                continue

            proposal_feats = proposal_boxes

            compressed_boxes, keep_idx = \
                self.proposal_compression_head(
                    proposal_feats,
                    proposal_boxes
                )

            proposals.bboxes = compressed_boxes

            if hasattr(proposals, "scores"):
                proposals.scores = \
                    proposals.scores[keep_idx]

            if hasattr(proposals, "labels"):
                proposals.labels = \
                    proposals.labels[keep_idx]

            compressed_results.append(
                proposals
            )

        return compressed_results

    def forward(
        self,
        x: Tuple[Tensor],
        rpn_results_list: InstanceList,
        batch_data_samples: SampleList = None
    ):

        rpn_results_list = self.compress_proposals(
            x,
            rpn_results_list
        )

        return super().forward(
            x,
            rpn_results_list,
            batch_data_samples
        )

    def loss(
        self,
        x: Tuple[Tensor],
        rpn_results_list: InstanceList,
        batch_data_samples: List[DetDataSample]
    ):

        rpn_results_list = self.compress_proposals(
            x,
            rpn_results_list
        )

        return super().loss(
            x,
            rpn_results_list,
            batch_data_samples
        )

    def predict_bbox(
        self,
        x: Tuple[Tensor],
        batch_img_metas: List[dict],
        rpn_results_list: InstanceList,
        rcnn_test_cfg: ConfigType,
        rescale: bool = False
    ):

        rpn_results_list = self.compress_proposals(
            x,
            rpn_results_list
        )

        return super().predict_bbox(
            x,
            batch_img_metas,
            rpn_results_list,
            rcnn_test_cfg,
            rescale
        )