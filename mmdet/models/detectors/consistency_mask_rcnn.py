# -*- coding: utf-8 -*-
"""Instance-Aware Pyramid Consistency (IAPC) for corruption-robust
instance segmentation.

ĐỘNG CƠ (từ thực nghiệm):
    Data augmentation cải thiện robustness đáng kể (AP_corr 0.655 -> 0.714)
    nhưng Robustness Drop vẫn ~0.150. Nguyên nhân: augmentation chỉ dạy
    invariance GIÁN TIẾP qua dữ liệu — model thấy ảnh biến dạng và tự suy
    ra nên bất biến, nhưng KHÔNG có tín hiệu trực tiếp nào ép biểu diễn
    của ảnh sạch và ảnh nhiễu phải trùng nhau.

Ý TƯỞNG:
    Thêm ràng buộc TRỰC TIẾP: với cùng một ảnh, đặc trưng pyramid của
    phiên bản sạch và phiên bản bị nhiễu phải gần nhau — nhưng CHỈ ở vùng
    có đối tượng (instance-aware), và có trọng số riêng cho từng mức
    pyramid (level-adaptive).

BA THÀNH PHẦN MỚI so với consistency regularization đã có:
    1. INSTANCE-AWARE: chỉ ép nhất quán tại vùng foreground (dùng gt_masks
       downsample về từng level). Nền đất thay đổi dưới corruption là
       chuyện bình thường và không nên bị phạt; chỉ vùng cây mới cần bất
       biến. Đây là điểm khác biệt với consistency loss toàn ảnh.
    2. MULTI-LEVEL PYRAMID: áp ở P2..P6 với trọng số riêng, vì các mức
       pyramid nhạy cảm khác nhau với corruption (mức thấp = texture,
       nhạy nhiễu; mức cao = ngữ nghĩa, ít nhạy hơn).
    3. NORMALIZED COSINE + L1 HYBRID: dùng khoảng cách kết hợp để ép cả
       hướng (cosine, bất biến với scale) lẫn độ lớn (L1), tránh trường
       hợp mạng "gian lận" bằng cách co nhỏ toàn bộ feature.

CHI PHÍ:
    Train: thêm 1 lần forward backbone+neck cho nhánh sạch (KHÔNG chạy
    head) -> khoảng +35-40% thời gian train.
    Inference: BẰNG 0. Model triển khai y hệt Mask R-CNN + FPN chuẩn.
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from mmdet.models.detectors.mask_rcnn import MaskRCNN
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig


@MODELS.register_module()
class IAPCMaskRCNN(MaskRCNN):
    """Mask R-CNN với Instance-Aware Pyramid Consistency.

    Args:
        consistency_weight: hệ số lambda của L_cons trong tổng loss.
        consistency_levels: các mức pyramid áp consistency (index trong
            tuple neck output, 0 = P2). Mặc định (0,1,2,3) = P2..P5;
            P6 thường quá thô, ít ý nghĩa.
        level_weights: trọng số riêng cho từng level trong
            consistency_levels. None -> đều nhau.
        instance_aware: True -> chỉ áp consistency ở vùng foreground.
        dilate_mask_px: nới rộng mask foreground (theo pixel ở ảnh gốc)
            trước khi downsample, để bao gồm cả vùng biên bên ngoài.
        detach_clean: True -> nhánh sạch dùng stop-gradient (giống
            teacher). False -> gradient chảy cả hai nhánh (symmetric).
        cos_weight: trọng số của thành phần cosine trong khoảng cách
            hybrid; phần còn lại (1 - cos_weight) cho L1 chuẩn hoá.
        warmup_iters: số iteration đầu KHÔNG áp consistency (để mạng ổn
            định trước). 0 -> áp ngay.
        corruption_cfg: cấu hình sinh nhánh nhiễu trên GPU. Xem
            :meth:`_make_corrupted_view`.
    """

    def __init__(self,
                 backbone: ConfigType,
                 rpn_head: ConfigType,
                 roi_head: ConfigType,
                 train_cfg: ConfigType,
                 test_cfg: ConfigType,
                 neck: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 # --- IAPC ---
                 consistency_weight: float = 1.0,
                 consistency_levels: Tuple[int, ...] = (0, 1, 2, 3),
                 level_weights: Optional[Tuple[float, ...]] = None,
                 instance_aware: bool = True,
                 dilate_mask_px: int = 8,
                 detach_clean: bool = True,
                 cos_weight: float = 0.5,
                 warmup_iters: int = 500,
                 corruption_cfg: Optional[dict] = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)

        self.consistency_weight = float(consistency_weight)
        self.consistency_levels = tuple(consistency_levels)
        if level_weights is None:
            level_weights = tuple(1.0 for _ in self.consistency_levels)
        assert len(level_weights) == len(self.consistency_levels), \
            'level_weights phải cùng độ dài với consistency_levels'
        self.level_weights = tuple(float(w) for w in level_weights)
        self.instance_aware = bool(instance_aware)
        self.dilate_mask_px = int(dilate_mask_px)
        self.detach_clean = bool(detach_clean)
        self.cos_weight = float(cos_weight)
        assert 0.0 <= self.cos_weight <= 1.0
        self.warmup_iters = int(warmup_iters)

        default_corruption = dict(
            brightness_delta=0.35,     # theo đơn vị std ảnh đã chuẩn hoá
            contrast_range=(0.55, 1.45),
            noise_sigma_range=(0.05, 0.30),
            prob_brightness=0.7,
            prob_contrast=0.7,
            prob_noise=0.7)
        if corruption_cfg is not None:
            default_corruption.update(corruption_cfg)
        self.corruption_cfg = default_corruption

        self.register_buffer('_iter', torch.zeros(1, dtype=torch.long),
                             persistent=False)

    # ------------------------------------------------------------------
    # Sinh nhánh nhiễu trên GPU
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _make_corrupted_view(self, x: Tensor) -> Tensor:
        """Tạo phiên bản nhiễu của batch ảnh ĐÃ chuẩn hoá.

        Thao tác trực tiếp trên không gian đã normalize (mean/std
        ImageNet), nên các hằng số ở đây tính theo đơn vị std chứ không
        phải [0,255]:
            - brightness: cộng một hằng số
            - contrast: co giãn quanh giá trị trung bình từng ảnh
            - noise: cộng nhiễu Gauss

        Mỗi ảnh trong batch được sinh tham số ĐỘC LẬP, nên nhánh nhiễu đa
        dạng hơn so với dùng chung một tham số cho cả batch.
        """
        cfg = self.corruption_cfg
        b = x.size(0)
        dev, dtype = x.device, x.dtype
        out = x.clone()

        def rand(shape, lo, hi):
            return torch.empty(shape, device=dev, dtype=dtype).uniform_(lo, hi)

        # --- brightness: cộng offset ---
        m = (torch.rand(b, device=dev) < cfg['prob_brightness']).to(dtype)
        delta = rand((b, 1, 1, 1), -cfg['brightness_delta'],
                     cfg['brightness_delta']) * m.view(b, 1, 1, 1)
        out = out + delta

        # --- contrast: co giãn quanh mean từng ảnh ---
        m = (torch.rand(b, device=dev) < cfg['prob_contrast']).to(dtype)
        lo, hi = cfg['contrast_range']
        factor = rand((b, 1, 1, 1), lo, hi)
        factor = factor * m.view(b, 1, 1, 1) + (1.0 - m.view(b, 1, 1, 1))
        mean = out.mean(dim=(1, 2, 3), keepdim=True)
        out = (out - mean) * factor + mean

        # --- gaussian noise ---
        m = (torch.rand(b, device=dev) < cfg['prob_noise']).to(dtype)
        lo, hi = cfg['noise_sigma_range']
        sigma = rand((b, 1, 1, 1), lo, hi) * m.view(b, 1, 1, 1)
        out = out + torch.randn_like(out) * sigma

        return out

    # ------------------------------------------------------------------
    # Mask foreground theo từng mức pyramid
    # ------------------------------------------------------------------
    def _foreground_masks(self, batch_data_samples: SampleList,
                          feats: List[Tensor],
                          input_hw: Tuple[int, int]) -> List[Tensor]:
        """Trả về list mask (B,1,H_l,W_l) cho mỗi level trong
        consistency_levels; giá trị trong [0,1].

        Nếu instance_aware=False hoặc ảnh không có instance nào, trả về
        mask toàn 1 (tương đương consistency toàn ảnh).
        """
        B = len(batch_data_samples)
        H, W = input_hw
        dev = feats[0].device
        dtype = feats[0].dtype

        # Gom mask foreground ở độ phân giải ảnh đầu vào
        full = torch.zeros((B, 1, H, W), device=dev, dtype=dtype)
        any_instance = False
        for i, ds in enumerate(batch_data_samples):
            gt = getattr(ds, 'gt_instances', None)
            masks = getattr(gt, 'masks', None) if gt is not None else None
            if masks is None or len(masks) == 0:
                continue
            # BitmapMasks / PolygonMasks -> numpy (N, h, w)
            if hasattr(masks, 'to_ndarray'):
                arr = masks.to_ndarray()
            elif hasattr(masks, 'masks'):
                arr = masks.masks
            else:
                continue
            if arr is None or len(arr) == 0:
                continue
            any_instance = True
            m = torch.as_tensor(arr, device=dev).any(dim=0).to(dtype)
            mh, mw = m.shape[-2:]
            full[i, 0, :min(mh, H), :min(mw, W)] = \
                m[:min(mh, H), :min(mw, W)]

        if not (self.instance_aware and any_instance):
            return [torch.ones((B, 1, f.shape[-2], f.shape[-1]),
                               device=dev, dtype=dtype)
                    for f in feats]

        # Nới rộng mask để bao cả vùng ngay ngoài biên (nơi lỗi hay xảy ra)
        if self.dilate_mask_px > 0:
            k = 2 * self.dilate_mask_px + 1
            full = F.max_pool2d(full, kernel_size=k, stride=1,
                                padding=self.dilate_mask_px)

        out = []
        for f in feats:
            m = F.interpolate(full, size=f.shape[-2:], mode='bilinear',
                              align_corners=False)
            out.append(m.clamp_(0.0, 1.0))
        return out

    # ------------------------------------------------------------------
    # Consistency loss
    # ------------------------------------------------------------------
    def _consistency_loss(self, feats_clean: List[Tensor],
                          feats_corrupt: List[Tensor],
                          masks: List[Tensor]) -> Tensor:
        """Khoảng cách hybrid cosine + L1 chuẩn hoá, có trọng số mask.

        Cosine phạt lệch HƯỚNG của vector đặc trưng (bất biến với scale),
        L1 phạt lệch ĐỘ LỚN. Chỉ dùng cosine sẽ để mạng tự do co giãn
        biên độ; chỉ dùng L1 thì dễ bị chi phối bởi các kênh biên độ lớn.
        """
        total = feats_clean[0].new_zeros(())
        wsum = 0.0

        for w, fc, fk, m in zip(self.level_weights, feats_clean,
                                feats_corrupt, masks):
            denom = m.sum().clamp_min(1.0)

            # --- cosine distance theo từng vị trí không gian ---
            cos = F.cosine_similarity(fc, fk, dim=1, eps=1e-6)  # (B,H,W)
            cos_d = (1.0 - cos).unsqueeze(1)                    # (B,1,H,W)
            cos_term = (cos_d * m).sum() / denom

            # --- L1 chuẩn hoá theo độ lớn của nhánh sạch ---
            l1 = (fc - fk).abs().mean(dim=1, keepdim=True)      # (B,1,H,W)
            scale = fc.abs().mean(dim=1, keepdim=True).clamp_min(1e-3)
            l1_term = ((l1 / scale) * m).sum() / denom

            total = total + w * (self.cos_weight * cos_term +
                                 (1.0 - self.cos_weight) * l1_term)
            wsum += w

        return total / max(wsum, 1e-6)

    # ------------------------------------------------------------------
    # loss()
    # ------------------------------------------------------------------
    def _detection_losses(self, x: Tuple[Tensor, ...],
                          batch_data_samples: SampleList) -> dict:
        """Bản sao logic TwoStageDetector.loss() nhưng nhận sẵn feature.

        Tách ra để nhánh nhiễu dùng lại feature đã trích, tránh forward
        backbone hai lần cho cùng một ảnh.
        """
        import copy

        losses = dict()
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # RPN chỉ phân biệt foreground/background -> gán nhãn về 0
            for ds in rpn_data_samples:
                ds.gt_instances.labels = \
                    torch.zeros_like(ds.gt_instances.labels)
            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg)
            for key in list(rpn_losses.keys()):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None, \
                'Không có RPN thì data samples phải chứa proposals'
            rpn_results_list = [ds.proposals for ds in batch_data_samples]

        roi_losses = self.roi_head.loss(x, rpn_results_list,
                                        batch_data_samples)
        losses.update(roi_losses)
        return losses

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Detection loss tính trên nhánh NHIỄU; thêm consistency giữa
        pyramid của nhánh sạch và nhánh nhiễu."""
        self._iter += 1
        in_warmup = int(self._iter.item()) <= self.warmup_iters

        corrupted = self._make_corrupted_view(batch_inputs)

        # --- nhánh nhiễu: forward đầy đủ (backbone + neck + heads) ---
        x_corrupt = self.extract_feat(corrupted)
        losses = self._detection_losses(x_corrupt, batch_data_samples)

        # --- consistency (chỉ backbone + neck cho nhánh sạch) ---
        if self.consistency_weight > 0 and not in_warmup:
            if self.detach_clean:
                with torch.no_grad():
                    x_clean_all = self.extract_feat(batch_inputs)
                sel_clean = [x_clean_all[i].detach()
                             for i in self.consistency_levels]
            else:
                x_clean_all = self.extract_feat(batch_inputs)
                sel_clean = [x_clean_all[i]
                             for i in self.consistency_levels]

            sel_corrupt = [x_corrupt[i] for i in self.consistency_levels]
            masks = self._foreground_masks(
                batch_data_samples, sel_corrupt,
                input_hw=batch_inputs.shape[-2:])
            loss_cons = self._consistency_loss(sel_clean, sel_corrupt, masks)
            losses['loss_consistency'] = self.consistency_weight * loss_cons
        else:
            # giữ key ổn định giữa các iteration để log không nhảy cột
            losses['loss_consistency'] = batch_inputs.new_zeros(())

        return losses
