# -*- coding: utf-8 -*-
"""PhotoAugMix: AugMix chỉ dùng phép biến đổi quang học (photometric-only).

Tham khảo: Hendrycks et al., "AugMix: A Simple Data Processing Method to
Improve Robustness and Uncertainty", ICLR 2020.

Khác bản gốc: LOẠI BỎ toàn bộ phép hình học (rotate/shear/translate) vì chúng
làm sai lệch bbox/mask annotation trong detection & instance segmentation.
Chỉ giữ: autocontrast, equalize, posterize, solarize, color, contrast,
brightness, sharpness — annotation giữ nguyên hợp lệ.

Ghi rõ điều này trong bài để phản biện không hỏi vì sao AugMix "thiếu" ops.
"""

import numpy as np
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS
from PIL import Image, ImageEnhance, ImageOps


def _autocontrast(img, _):
    return ImageOps.autocontrast(img)


def _equalize(img, _):
    return ImageOps.equalize(img)


def _posterize(img, level):
    bits = 4 - int(level * 3)  # 4 -> 1 bits
    return ImageOps.posterize(img, max(1, bits))


def _solarize(img, level):
    thresh = int(256 - level * 128)
    return ImageOps.solarize(img, thresh)


def _color(img, level):
    return ImageEnhance.Color(img).enhance(1.0 + (level - 0.5) * 1.2)


def _contrast(img, level):
    return ImageEnhance.Contrast(img).enhance(1.0 + (level - 0.5) * 1.2)


def _brightness(img, level):
    return ImageEnhance.Brightness(img).enhance(1.0 + (level - 0.5) * 1.2)


def _sharpness(img, level):
    return ImageEnhance.Sharpness(img).enhance(1.0 + (level - 0.5) * 1.2)


_OPS = [_autocontrast, _equalize, _posterize, _solarize,
        _color, _contrast, _brightness, _sharpness]


@TRANSFORMS.register_module()
class PhotoAugMix(BaseTransform):
    """AugMix photometric-only.

    Args:
        severity: cường độ mỗi op, thang [0,1].
        width: số chuỗi augmentation trộn song song (mặc định 3).
        depth: độ dài mỗi chuỗi; -1 = ngẫu nhiên 1-3.
        alpha: tham số Dirichlet/Beta cho trọng số trộn.
        prob: xác suất áp dụng.
    """

    def __init__(self, severity: float = 0.3, width: int = 3,
                 depth: int = -1, alpha: float = 1.0, prob: float = 0.5):
        self.severity = severity
        self.width = width
        self.depth = depth
        self.alpha = alpha
        self.prob = prob

    def transform(self, results: dict) -> dict:
        if np.random.rand() > self.prob:
            return results
        img = results['img']  # BGR uint8 (mmcv)
        pil = Image.fromarray(img[..., ::-1])  # -> RGB

        ws = np.random.dirichlet([self.alpha] * self.width)
        m = np.random.beta(self.alpha, self.alpha)

        mix = np.zeros_like(np.asarray(pil), dtype=np.float32)
        for i in range(self.width):
            aug = pil.copy()
            depth = self.depth if self.depth > 0 else np.random.randint(1, 4)
            for _ in range(depth):
                op = _OPS[np.random.randint(len(_OPS))]
                level = np.random.uniform(0, self.severity)
                aug = op(aug, level)
            mix += ws[i] * np.asarray(aug, dtype=np.float32)

        out = (1 - m) * np.asarray(pil, dtype=np.float32) + m * mix
        out = np.clip(out, 0, 255).astype(np.uint8)
        results['img'] = out[..., ::-1].copy()  # RGB -> BGR
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(severity={self.severity}, '
                f'width={self.width}, depth={self.depth}, prob={self.prob})')
