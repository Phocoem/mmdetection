# -*- coding: utf-8 -*-
"""RandAugment (Cubuk et al.), ban rut gon photometric-only cho detection.

Day la dai dien thuc te cho "policy-search augmentation" da trich dan o
Related Work Section 2.2 (Cubuk et al. 2019 - AutoAugment) nhung chua tung
benchmark. Dung RandAugment thay vi AutoAugment vi khong can tim policy
rieng bang search (thuc te hon), gioi han CHI toan tu KHONG hinh hoc
(cung ly do nhu augmix_transform.py) de mask/bbox khong bi lech:
Brightness, Color, Contrast, Sharpness, Posterize, Solarize, AutoContrast,
Equalize.
"""
import random

import numpy as np
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS
from PIL import Image
from torchvision.transforms import functional as TF

_MAX_MAGNITUDE = 10  # thang chuan RandAugment 0-10


def _brightness(img, m):
    factor = max(0.1, 1.0 + (m / _MAX_MAGNITUDE) * 0.9 * random.choice([-1, 1]))
    return TF.adjust_brightness(img, factor)


def _color(img, m):
    factor = max(0.1, 1.0 + (m / _MAX_MAGNITUDE) * 0.9 * random.choice([-1, 1]))
    return TF.adjust_saturation(img, factor)


def _contrast(img, m):
    factor = max(0.1, 1.0 + (m / _MAX_MAGNITUDE) * 0.9 * random.choice([-1, 1]))
    return TF.adjust_contrast(img, factor)


def _sharpness(img, m):
    factor = max(0.0, 1.0 + (m / _MAX_MAGNITUDE) * 0.9 * random.choice([-1, 1]))
    return TF.adjust_sharpness(img, factor)


def _posterize(img, m):
    bits = max(1, 8 - int(round((m / _MAX_MAGNITUDE) * 4)))
    return TF.posterize(img, bits)


def _solarize(img, m):
    threshold = max(0, int(256 - (m / _MAX_MAGNITUDE) * 256))
    return TF.solarize(img, threshold)


def _autocontrast(img, _m):
    return TF.autocontrast(img)


def _equalize(img, _m):
    return TF.equalize(img)


RANDAUG_OPS = [_brightness, _color, _contrast, _sharpness, _posterize,
               _solarize, _autocontrast, _equalize]


@TRANSFORMS.register_module()
class RandAugmentPhotometric(BaseTransform):
    """RandAugment ban photometric-only (khong toan tu hinh hoc).

    Args:
        num_ops: so toan tu ap dung lien tiep moi anh (N trong RandAugment).
        magnitude: cuong do moi toan tu (0-10, M trong RandAugment).
        prob: xac suat ap dung cho moi anh.
    """

    def __init__(self, num_ops: int = 2, magnitude: int = 5,
                 prob: float = 1.0):
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.prob = prob

    def transform(self, results: dict) -> dict:
        if np.random.rand() > self.prob:
            return results

        img = results['img']
        pil_img = Image.fromarray(img[:, :, ::-1])  # BGR -> RGB

        ops = random.sample(RANDAUG_OPS,
                             k=min(self.num_ops, len(RANDAUG_OPS)))
        for op in ops:
            pil_img = op(pil_img, self.magnitude)

        out = np.ascontiguousarray(np.asarray(pil_img)[:, :, ::-1])
        results['img'] = out
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(num_ops={self.num_ops}, '
                f'magnitude={self.magnitude}, prob={self.prob})')
