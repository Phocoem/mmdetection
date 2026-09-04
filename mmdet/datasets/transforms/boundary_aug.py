# -*- coding: utf-8 -*-
"""BoundaryContrastAttack: mask-guided boundary augmentation (đề xuất chính).

Động lực: trong instance segmentation cây trồng, lỗi tập trung ở BIÊN mask
(mask AP nhạy với chất lượng biên). Dưới suy giảm tương phản, ranh giới lá-đất
mờ đi và model mất biên. Thay vì áp corruption đồng đều (như AugMix/
PhotoMetricDistortion), transform này GIẢM TƯƠNG PHẢN CÂY-NỀN CÓ ĐỊNH HƯỚNG
tại vùng biên, buộc model học biểu diễn biên vững chắc — tấn công trực tiếp
vào điểm yếu nhất của bài toán.

Thiết kế chống 2 cạm bẫy:
1. LABEL-PRESERVING: mức giảm tương phản bị chặn (min_keep > 0), chỉ áp ở dải
   biên hẹp, giữ nguyên trong lòng cây và nền xa. Biên KHÓ HƠN nhưng VẪN CÒN,
   annotation gốc vẫn đúng.
2. CHỐNG SHORTCUT: (a) áp ngẫu nhiên prob<1; (b) dải biên làm mềm bằng Gaussian
   (không có ranh giới cứng "vùng bị can thiệp"); (c) nên dùng kèm augmentation
   đồng đều thông thường trong pipeline để model không học "vùng can thiệp=biên".

Yêu cầu: chạy trong pipeline SAU LoadAnnotations (cần gt_masks), TRƯỚC Resize
hoặc sau đều được (transform tự khớp kích thước mask với ảnh).

Cường độ khuyến nghị (nhẹ, an toàn label-preserving): strength=0.4,
band_width=7, prob=0.5. Tăng dần nếu cần mạnh hơn nhưng kiểm tra biên còn
phân biệt được bằng mắt.
"""

from typing import Tuple

import cv2
import numpy as np
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class BoundaryContrastAttack(BaseTransform):
    """Giảm tương phản cây-nền có định hướng tại vùng biên mask.

    Args:
        prob: xác suất áp dụng (chống shortcut, giữ đa dạng).
        strength: mức giảm tương phản tối đa tại tâm biên, [0,1].
            0 = không đổi; 1 = kéo hoàn toàn về mức nền cục bộ (KHÔNG khuyến
            nghị vì phá label-preserving). Khuyến nghị 0.3-0.5.
        band_width: độ rộng dải biên (pixel) — vùng chịu tác động quanh biên.
        min_keep: hệ số tương phản tối thiểu được giữ lại (chặn dưới để đảm
            bảo biên không biến mất hoàn toàn). Ví dụ 0.35 = luôn giữ >=35%
            tương phản gốc ngay tại tâm biên.
        blur_boundary_sigma: làm mềm mặt nạ biên (chống ranh giới cứng).
    """

    def __init__(self,
                 prob: float = 0.5,
                 strength: float = 0.4,
                 band_width: int = 7,
                 min_keep: float = 0.35,
                 blur_boundary_sigma: float = 2.0):
        assert 0.0 <= strength <= 1.0
        assert 0.0 < min_keep <= 1.0
        self.prob = prob
        self.strength = strength
        self.band_width = band_width
        self.min_keep = min_keep
        self.blur_boundary_sigma = blur_boundary_sigma

    def _get_binary_mask(self, results: dict, h: int, w: int):
        """Gộp mọi instance mask thành 1 mask nhị phân foreground (cây)."""
        masks = results.get('gt_masks', None)
        if masks is None or len(masks) == 0:
            return None
        # mmdet BitmapMasks -> .masks (N,H,W); PolygonMasks -> cần to_bitmap
        if hasattr(masks, 'to_bitmap'):
            try:
                masks = masks.to_bitmap()
            except Exception:  # noqa: BLE001
                pass
        arr = getattr(masks, 'masks', None)
        if arr is None:
            return None
        fg = (np.asarray(arr).sum(axis=0) > 0).astype(np.uint8)
        if fg.shape != (h, w):
            fg = cv2.resize(fg, (w, h), interpolation=cv2.INTER_NEAREST)
        return fg

    def _boundary_weight(self, fg: np.ndarray) -> np.ndarray:
        """Trọng số [0,1] cao ở biên, giảm dần vào trong/ra ngoài, làm mềm."""
        # Morphological gradient = dilation - erosion => dải biên
        k = np.ones((3, 3), np.uint8)
        grad = cv2.morphologyEx(fg, cv2.MORPH_GRADIENT, k)
        # Nới dải biên tới band_width bằng distance transform 2 phía
        # Khoảng cách tới biên (cả trong lẫn ngoài cây)
        dist_out = cv2.distanceTransform(1 - grad, cv2.DIST_L2, 3)
        band = np.clip(1.0 - dist_out / max(self.band_width, 1), 0.0, 1.0)
        # Làm mềm để không có ranh giới cứng (chống shortcut)
        if self.blur_boundary_sigma > 0:
            band = cv2.GaussianBlur(band, (0, 0), self.blur_boundary_sigma)
            band = np.clip(band, 0.0, 1.0)
        return band.astype(np.float32)

    def transform(self, results: dict) -> dict:
        if np.random.rand() > self.prob:
            return results
        img = results['img']
        h, w = img.shape[:2]

        fg = self._get_binary_mask(results, h, w)
        if fg is None:
            return results  # không có mask -> bỏ qua an toàn

        band = self._boundary_weight(fg)  # H,W in [0,1]

        # Mục tiêu nội suy để GIẢM tương phản = mức xám cục bộ (làm mờ ranh
        # giới cây-nền). Dùng mean cục bộ trên cửa sổ tỉ lệ theo band_width để
        # hiệu ứng độc lập kích thước ảnh (ổn định giữa 100px test và 800px thật).
        ksize = max(3, int(self.band_width * 2) | 1)  # lẻ
        local = cv2.blur(img.astype(np.float32), (ksize, ksize))

        # Hệ số giữ tương phản theo từng pixel:
        #   keep = 1 ở vùng không phải biên; giảm tới (1-strength) tại tâm biên
        #   nhưng không dưới min_keep (chặn label-preserving).
        rnd = np.random.uniform(0.5, 1.0)  # dao động cường độ mỗi ảnh
        reduce = self.strength * rnd * band  # [0, strength]
        keep = np.clip(1.0 - reduce, self.min_keep, 1.0)[..., None]

        # Nội suy về nền cục bộ theo keep (keep=1 giữ nguyên, nhỏ -> gần nền)
        out = keep * img.astype(np.float32) + (1.0 - keep) * local
        results['img'] = np.clip(out, 0, 255).astype(img.dtype)
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(prob={self.prob}, '
                f'strength={self.strength}, band_width={self.band_width}, '
                f'min_keep={self.min_keep})')
