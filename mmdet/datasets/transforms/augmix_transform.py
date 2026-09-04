# -*- coding: utf-8 -*-
"""AugMix (Hendrycks et al. 2020) cho MMDetection, ban rut gon cho
instance segmentation.

GIOI HAN PHAM VI - CAN NEU RO NEU DUNG TRONG BAI:
    AugMix goc dung ca toan tu HINH HOC (ShearX/Y, TranslateX/Y, Rotate).
    Voi instance segmentation, toan tu hinh hoc se lam lech mask/bbox neu
    khong duoc ap dung dong bo len annotation. De giu "label-preserving"
    (dung tinh than Section 3.2 - loai glass blur/elastic transform vi
    lam lech pixel), ban nay CHI dung toan tu KHONG hinh hoc: AutoContrast,
    Equalize, Posterize, Solarize.

    AugMix KHONG dung Brightness/Color/Contrast/Sharpness - dung theo
    thiet ke goc de tranh trung voi corruption test-time (ImageNet-C
    style), giu dung nguyen tac "khong train-on-test" da co trong
    robust_aug.py.
"""
import random

import numpy as np
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS
from PIL import Image
from torchvision.transforms import functional as TF


def _autocontrast(img, _severity):
    return TF.autocontrast(img)


def _equalize(img, _severity):
    return TF.equalize(img)


def _posterize(img, severity):
    bits = max(1, 8 - int(severity))
    return TF.posterize(img, bits)


def _solarize(img, severity):
    threshold = max(0, 256 - int(severity) * 32)
    return TF.solarize(img, threshold)


AUGMIX_OPS = [_autocontrast, _equalize, _posterize, _solarize]


@TRANSFORMS.register_module()
class AugMix(BaseTransform):
    """AugMix (ban rut gon, khong toan tu hinh hoc) cho anh uint8 HxWxC (BGR).

    Args:
        severity: cuong do moi toan tu (1-10). Mac dinh 3 - nhe, dung tinh
            than "train nhe hon test" cua robust_aug.py.
        width: so chain augmentation duoc tron (AugMix goc dung 3).
        depth_range: do sau moi chain (so toan tu noi tiep), lay ngau
            nhien trong khoang nay cho MOI chain.
        alpha: tham so Dirichlet (trong so cac chain) va Beta (tron voi
            anh goc).
        prob: xac suat ap dung AugMix cho moi anh.
    """

    def __init__(self, severity: int = 3, width: int = 3,
                 depth_range=(1, 3), alpha: float = 1.0, prob: float = 1.0):
        self.severity = severity
        self.width = width
        self.depth_range = depth_range
        self.alpha = alpha
        self.prob = prob

    def _apply_chain(self, pil_img: Image.Image) -> Image.Image:
        depth = random.randint(*self.depth_range)
        out = pil_img
        for _ in range(depth):
            op = random.choice(AUGMIX_OPS)
            out = op(out, self.severity)
        return out

    def transform(self, results: dict) -> dict:
        if np.random.rand() > self.prob:
            return results

        img = results['img']  # HxWxC, uint8, BGR (quy uoc mmdet)
        pil_img = Image.fromarray(img[:, :, ::-1])  # BGR -> RGB cho PIL/TF
        base = np.asarray(pil_img).astype(np.float32)

        ws = np.random.dirichlet([self.alpha] * self.width).astype(np.float32)
        m = np.float32(np.random.beta(self.alpha, self.alpha))

        mix = np.zeros_like(base)
        for i in range(self.width):
            chain_img = np.asarray(self._apply_chain(pil_img)).astype(np.float32)
            mix += ws[i] * chain_img

        mixed = (1.0 - m) * base + m * mix
        mixed = np.clip(mixed, 0, 255).astype(np.uint8)

        results['img'] = np.ascontiguousarray(mixed[:, :, ::-1])  # -> BGR
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(severity={self.severity}, '
                f'width={self.width}, depth_range={self.depth_range}, '
                f'alpha={self.alpha}, prob={self.prob})')
