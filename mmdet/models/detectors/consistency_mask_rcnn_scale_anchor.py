# -*- coding: utf-8 -*-
"""IAPC voi TRAN TREN (upper bound) cho scale chuan hoa s_l(x) trong Eq. (7)
cua bai - va Section 3.5.4.

VAN DE DA PHAT HIEN (do bang plot_feature_heatmaps.py --per-row-normalize):
    scale = fc.abs().mean(dim=1, keepdim=True).clamp_min(1e-3)
Dong nay CHI CO SAN DUOI (1e-3), KHONG CO TRAN TREN. Do thuc te:
    Baseline (khong consistency): P2=0.36 P3=0.36 P4=0.36 P5=0.31
    IAPC lambda=0.25:             P2=0.72 P3=0.93 P4=2.48 P5=52.36 (167x!)
    IAPC lambda=1.0:              P2=3.36 P3=4.00 P4=3.50 P5=31.02 (99x!)
Scale phinh khong kiem soat lam SO HANG L1 CHUAN HOA yeu di theo thoi
gian (cung 1 chenh lech tuyet doi / mau so cang lon = ty so cang nho),
lam giam dan hieu luc cua rang buoc consistency ma khong can 2 nhanh
that su giong nhau hon - mot bien the khac cua "collapse" ma Section
3.5.4 da canh bao (nhung KHONG PHAI bien the da duoc chan boi cos term
bat bien scale + chuan hoa L1 - day la lo hong RIENG qua MAU SO).

CACH VA: them tran tren (scale_anchor_cap) cho scale, dua tren magnitude
"binh thuong" cua baseline (~0.3-0.36) - mac dinh cap=1.0 (khoang 3x
baseline, du khong gian hoc nhung chan dut kha nang phinh 100x+).

Dang ky them class nay, KHONG sua truc tiep consistency_mask_rcnn.py -
giu nguyen ban goc de con so sanh "truoc/sau vá" (Section B trong ke
hoach so sanh da thong nhat).
"""
from typing import List, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from mmdet.registry import MODELS
from .consistency_mask_rcnn import IAPCMaskRCNN


@MODELS.register_module()
class IAPCMaskRCNNScaleAnchor(IAPCMaskRCNN):
    """IAPC + tran tren cho scale chuan hoa (vá lỗ hổng magnitude).

    Args (them ngoai cac tham so cua IAPCMaskRCNN):
        scale_anchor_cap: float hoac list[float] - tran tren cho scale o
            MOI tang pyramid. Neu la 1 float, ap dung CHUNG cho tat ca
            tang (mac dinh 1.0). Neu la list, phai cung do dai voi
            level_weights (1 gia tri rieng cho tung tang P2..P5).
    """

    def __init__(self, *args,
                 scale_anchor_cap: Union[float, List[float]] = 1.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.scale_anchor_cap = scale_anchor_cap

    def _consistency_loss(self, feats_clean: List[Tensor],
                           feats_corrupt: List[Tensor],
                           masks: List[Tensor]) -> Tensor:
        """Giong het ham goc trong consistency_mask_rcnn.py, CHI KHAC
        DUNG 1 DONG: them .clamp(max=cap) vao scale (dong danh dau SUA
        ben duoi). Toan bo phan con lai (cosine term, tong hop cuoi)
        giu nguyen 100% khong doi."""
        total = feats_clean[0].new_zeros(())
        wsum = 0.0

        caps = self.scale_anchor_cap
        if not isinstance(caps, (list, tuple)):
            caps = [caps] * len(self.level_weights)
        if len(caps) != len(self.level_weights):
            raise ValueError(
                f'scale_anchor_cap phai la 1 so hoac list cung do dai '
                f'voi level_weights ({len(self.level_weights)} tang), '
                f'nhung nhan duoc {len(caps)} gia tri.')

        for w, fc, fk, m, cap in zip(self.level_weights, feats_clean,
                                      feats_corrupt, masks, caps):
            denom = m.sum().clamp_min(1.0)
            # --- cosine distance theo tung vi tri khong gian (KHONG DOI) ---
            cos = F.cosine_similarity(fc, fk, dim=1, eps=1e-6)
            cos_d = (1.0 - cos).unsqueeze(1)
            cos_term = (cos_d * m).sum() / denom
            # --- L1 chuan hoa theo do lon cua nhanh sach ---
            l1 = (fc - fk).abs().mean(dim=1, keepdim=True)
            # *** SUA: them max=cap - day la DONG DUY NHAT thay doi so voi
            # ban goc (vốn chi co clamp_min(1e-3), khong co tran tren). ***
            scale = fc.abs().mean(dim=1, keepdim=True).clamp(min=1e-3, max=cap)
            l1_term = ((l1 / scale) * m).sum() / denom
            total = total + w * (self.cos_weight * cos_term +
                                  (1.0 - self.cos_weight) * l1_term)
            wsum += w
        return total / max(wsum, 1e-6)
