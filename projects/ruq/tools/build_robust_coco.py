#!/usr/bin/env python3
"""Build a corruption-based robustness benchmark for COCO instance segmentation.

The script keeps the original COCO annotation unchanged and creates corrupted
copies of the test images. This is valid for corruptions that do not change image
geometry, e.g. blur, noise, brightness, shadow, JPEG, occlusion.

Example:
python projects/ruq/tools/build_robust_coco.py \
  --img-dir data/lettuce_coco/test \
  --ann data/lettuce_coco/annotations/test.json \
  --out data/lettuce_robust \
  --severities 1 2 3
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def imread_rgb(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f'Cannot read image: {path}')
    return img


def save_img(path: Path, img: np.ndarray, jpeg_quality=None):
    ensure_parent(path)
    if jpeg_quality is not None:
        cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    else:
        cv2.imwrite(str(path), img)


def gaussian_noise(img, severity):
    sigmas = {1: 8, 2: 16, 3: 28}
    sigma = sigmas[severity]
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def motion_blur(img, severity):
    kernels = {1: 5, 2: 9, 3: 15}
    k = kernels[severity]
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0 / k
    return cv2.filter2D(img, -1, kernel)


def defocus_blur(img, severity):
    kernels = {1: 5, 2: 9, 3: 13}
    k = kernels[severity]
    return cv2.GaussianBlur(img, (k, k), 0)


def brightness_low(img, severity):
    factors = {1: 0.75, 2: 0.55, 3: 0.38}
    return np.clip(img.astype(np.float32) * factors[severity], 0, 255).astype(np.uint8)


def brightness_high(img, severity):
    factors = {1: 1.20, 2: 1.45, 3: 1.75}
    out = img.astype(np.float32) * factors[severity]
    return np.clip(out, 0, 255).astype(np.uint8)


def contrast_low(img, severity):
    factors = {1: 0.75, 2: 0.55, 3: 0.35}
    mean = img.mean(axis=(0, 1), keepdims=True)
    out = mean + factors[severity] * (img.astype(np.float32) - mean)
    return np.clip(out, 0, 255).astype(np.uint8)


def shadow(img, severity):
    """Add random soft shadow without changing geometry."""
    h, w = img.shape[:2]
    alpha = {1: 0.75, 2: 0.58, 3: 0.42}[severity]
    mask = np.zeros((h, w), dtype=np.uint8)

    # Random quadrilateral crossing the image.
    x1 = random.randint(0, max(1, w // 2))
    x2 = random.randint(max(1, w // 2), w)
    pts = np.array([
        [x1, 0],
        [min(w - 1, x2), 0],
        [random.randint(max(0, w // 3), w - 1), h - 1],
        [random.randint(0, max(1, w // 3)), h - 1],
    ], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    mask = cv2.GaussianBlur(mask, (51, 51), 0).astype(np.float32) / 255.0
    factor = 1.0 - mask[..., None] * (1.0 - alpha)
    out = img.astype(np.float32) * factor
    return np.clip(out, 0, 255).astype(np.uint8)


def occlusion(img, severity):
    """Add one rectangular occluder; GT masks are kept for robustness testing."""
    h, w = img.shape[:2]
    frac = {1: 0.06, 2: 0.12, 3: 0.20}[severity]
    occ_area = int(h * w * frac)
    rw = random.randint(max(10, int(w * 0.12)), max(12, int(w * 0.30)))
    rh = max(10, occ_area // max(1, rw))
    rw = min(rw, w - 1)
    rh = min(rh, h - 1)
    x = random.randint(0, max(0, w - rw - 1))
    y = random.randint(0, max(0, h - rh - 1))
    out = img.copy()
    color = [int(c) for c in img.reshape(-1, 3).mean(axis=0)]
    cv2.rectangle(out, (x, y), (x + rw, y + rh), color, -1)
    return out


def jpeg_compression(img, severity):
    qualities = {1: 55, 2: 30, 3: 15}
    q = qualities[severity]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), q]
    ok, enc = cv2.imencode('.jpg', img, encode_param)
    if not ok:
        return img
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


CORRUPTIONS = {
    'gaussian_noise': gaussian_noise,
    'motion_blur': motion_blur,
    'defocus_blur': defocus_blur,
    'shadow': shadow,
    'brightness_low': brightness_low,
    'brightness_high': brightness_high,
    'contrast_low': contrast_low,
    'occlusion': occlusion,
    'jpeg_compression': jpeg_compression,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-dir', required=True, help='Clean test image directory')
    parser.add_argument('--ann', required=True, help='Clean COCO annotation json')
    parser.add_argument('--out', required=True, help='Output robustness dataset root')
    parser.add_argument('--severities', nargs='+', type=int, default=[1, 2, 3])
    parser.add_argument('--corruptions', nargs='+', default=list(CORRUPTIONS.keys()))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    img_dir = Path(args.img_dir)
    ann_path = Path(args.ann)
    out_root = Path(args.out)
    (out_root / 'annotations').mkdir(parents=True, exist_ok=True)
    (out_root / 'images').mkdir(parents=True, exist_ok=True)

    with open(ann_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    # Copy clean images and annotation for convenience.
    clean_out = out_root / 'images' / 'clean'
    clean_out.mkdir(parents=True, exist_ok=True)
    for im in coco['images']:
        rel_name = im['file_name']
        src = img_dir / rel_name
        dst = clean_out / rel_name
        ensure_parent(dst)
        shutil.copy2(src, dst)

    # Copy clean annotation for every corruption. The image file_name remains the same.
    for corr in args.corruptions:
        if corr not in CORRUPTIONS:
            raise ValueError(f'Unknown corruption: {corr}')
        fn = CORRUPTIONS[corr]
        for sev in args.severities:
            print(f'Building {corr}, severity={sev}')
            out_img_dir = out_root / 'images' / corr / f's{sev}'
            out_img_dir.mkdir(parents=True, exist_ok=True)

            for im in coco['images']:
                rel_name = im['file_name']
                src = img_dir / rel_name
                dst = out_img_dir / rel_name
                img = imread_rgb(src)
                out = fn(img, sev)
                save_img(dst, out)

            out_ann = out_root / 'annotations' / f'test_{corr}_s{sev}.json'
            shutil.copy2(ann_path, out_ann)

    # Also copy clean annotation for convenience.
    shutil.copy2(ann_path, out_root / 'annotations' / 'test_clean.json')
    print(f'Done. Robust dataset saved to: {out_root}')


if __name__ == '__main__':
    main()
