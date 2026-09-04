# -*- coding: utf-8 -*-
"""Augmentation transforms cho robustness baseline (Major Comment 8).

Đăng ký transform RandomGaussianNoise vì MMDetection chưa có sẵn.
Lưu ý: cường độ augmentation cố tình đặt NHẸ HƠN severity test (sigma tối đa
0.06 so với test sigma 0.08-0.18) để tránh "train on test corruption" —
augmentation chỉ được thấy phân phối nhiễu tương tự, không thấy đúng cường độ
test, giữ tính công bằng của đánh giá robustness.
"""

import numpy as np
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class RandomGaussianNoise(BaseTransform):
    """Thêm Gaussian noise ngẫu nhiên vào ảnh (uint8, 0-255).

    Args:
        prob: xác suất áp dụng.
        sigma_range: khoảng sigma tính trên thang [0,1] (nhân 255 nội bộ).
    """

    def __init__(self, prob: float = 0.5,
                 sigma_range=(0.01, 0.06)):
        self.prob = prob
        self.sigma_range = sigma_range

    def transform(self, results: dict) -> dict:
        if np.random.rand() > self.prob:
            return results
        img = results['img'].astype(np.float32)
        sigma = np.random.uniform(*self.sigma_range) * 255.0
        noise = np.random.normal(0.0, sigma, size=img.shape).astype(np.float32)
        img = np.clip(img + noise, 0, 255)
        results['img'] = img.astype(np.uint8)
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(prob={self.prob}, '
                f'sigma_range={self.sigma_range})')
