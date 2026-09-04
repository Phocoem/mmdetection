# Copyright (c) OpenMMLab. All rights reserved.
# RUQ Standard RoI Head for MMDetection 3.x

from typing import Tuple

from mmdet.registry import MODELS
from mmdet.models.roi_heads.standard_roi_head import StandardRoIHead
from mmdet.structures.bbox import bbox2roi


@MODELS.register_module()
class RUQStandardRoIHead(StandardRoIHead):
    """Standard RoI head that supports RUQFCNMaskHead dict outputs.

    The standard MMDetection ROI head expects mask_head(mask_feats) to return a
    tensor. RUQFCNMaskHead returns a dict containing mask, quality, and
    uncertainty predictions, so this class adapts _mask_forward and
    _mask_forward_train.
    """

    def _mask_forward(self, x: Tuple, rois):
        mask_feats = self.mask_roi_extractor(
            x[:self.mask_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            mask_feats = self.shared_head(mask_feats)

        out = self.mask_head(mask_feats)
        if isinstance(out, dict):
            mask_results = out
            mask_results['mask_feats'] = mask_feats
        else:
            mask_results = dict(mask_preds=out, mask_feats=mask_feats)
        return mask_results

    def _mask_forward_train(self, x: Tuple, sampling_results,
                            bbox_feats, batch_gt_instances):
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_priors for res in sampling_results])
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    res.pos_priors.new_ones(
                        res.pos_priors.size(0), dtype=bool, device=device))
                pos_inds.append(
                    res.neg_priors.new_zeros(
                        res.neg_priors.size(0), dtype=bool, device=device))
            pos_inds = self._concat_bool(pos_inds)
            out = self.mask_head(bbox_feats[pos_inds])
            if isinstance(out, dict):
                mask_results = out
                mask_results['mask_feats'] = bbox_feats[pos_inds]
            else:
                mask_results = dict(mask_preds=out, mask_feats=bbox_feats[pos_inds])

        # Pass full dict into RUQFCNMaskHead.loss_and_target().
        mask_loss_and_target = self.mask_head.loss_and_target(
            mask_preds=mask_results,
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=self.train_cfg)
        mask_results.update(mask_loss_and_target)
        return mask_results

    @staticmethod
    def _concat_bool(items):
        import torch
        return torch.cat(items)
