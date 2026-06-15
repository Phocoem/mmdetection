"""
Lettuce Corruption Benchmark Engine
-----------------------------------

This script generates corrupted test image folders for static-image lettuce instance segmentation.

Design:
- Image-only corruption.
- COCO annotations are NOT changed.
- No occlusion, no fake leaves, no block masking.
- Corruptions are selected to match realistic image-quality degradations:
  Gaussian noise, Gaussian blur, motion blur, brightness, contrast, gamma,
  local soft shadow, JPEG compression, medium mixed stress, hard mixed stress.

Example:
python tools/stress_engine.py --input test --output stress/noise_s2 --condition noise --severity 2 --seed 42
"""

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


SEVERITY_PRESETS: Dict[str, Dict[int, dict]] = {
    "noise": {
        1: {"sigma": 5},
        2: {"sigma": 10},
        3: {"sigma": 20},
    },
    "gaussian_blur": {
        1: {"kernel": 3},
        2: {"kernel": 5},
        3: {"kernel": 7},
    },
    "motion_blur": {
        1: {"kernel": 5},
        2: {"kernel": 9},
        3: {"kernel": 13},
    },
    "brightness": {
        1: {"factor_range": (0.85, 1.15)},
        2: {"factor_range": (0.70, 1.30)},
        3: {"factor_range": (0.55, 1.45)},
    },
    "contrast": {
        1: {"factor_range": (0.85, 1.15)},
        2: {"factor_range": (0.70, 1.30)},
        3: {"factor_range": (0.55, 1.45)},
    },
    "gamma": {
        1: {"gamma_range": (0.85, 1.15)},
        2: {"gamma_range": (0.70, 1.35)},
        3: {"gamma_range": (0.55, 1.60)},
    },
    "shadow": {
        1: {"opacity": 0.12},
        2: {"opacity": 0.22},
        3: {"opacity": 0.35},
    },
    "jpeg": {
        1: {"quality": 90},
        2: {"quality": 70},
        3: {"quality": 50},
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def safe_imwrite(path: Path, img: np.ndarray, quality: int = 95) -> None:
    """
    Save output as JPG to avoid libpng write errors on Windows.
    The output file keeps the original relative path but uses .jpg extension.
    """
    path = Path(path).with_suffix(".jpg")
    path.parent.mkdir(parents=True, exist_ok=True)

    ok = cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise IOError(f"Failed to write image: {path}")


def gaussian_noise(img: np.ndarray, sigma: float) -> np.ndarray:
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def gaussian_blur(img: np.ndarray, kernel: int) -> np.ndarray:
    kernel = int(kernel)
    if kernel % 2 == 0:
        kernel += 1
    return cv2.GaussianBlur(img, (kernel, kernel), 0)


def motion_blur(img: np.ndarray, kernel: int) -> np.ndarray:
    kernel = int(kernel)
    if kernel < 3:
        kernel = 3

    k = np.zeros((kernel, kernel), dtype=np.float32)
    k[kernel // 2, :] = 1.0
    k /= kernel

    angle = random.choice([0, 30, 60, 90, 120, 150])
    center = (kernel / 2 - 0.5, kernel / 2 - 0.5)
    rot = cv2.getRotationMatrix2D(center, angle, 1.0)
    k = cv2.warpAffine(k, rot, (kernel, kernel))
    k /= max(k.sum(), 1e-6)

    return cv2.filter2D(img, -1, k)


def brightness_shift(img: np.ndarray, factor: float) -> np.ndarray:
    out = img.astype(np.float32) * float(factor)
    return np.clip(out, 0, 255).astype(np.uint8)


def contrast_shift(img: np.ndarray, factor: float) -> np.ndarray:
    img_f = img.astype(np.float32)
    mean = np.mean(img_f, axis=(0, 1), keepdims=True)
    out = (img_f - mean) * float(factor) + mean
    return np.clip(out, 0, 255).astype(np.uint8)


def gamma_correction(img: np.ndarray, gamma: float) -> np.ndarray:
    gamma = max(float(gamma), 1e-6)
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, table)


def local_soft_shadow(img: np.ndarray, opacity: float) -> np.ndarray:
    """
    Local, smooth, natural-looking illumination reduction.
    Avoids large triangular full-image shadows.
    """
    h, w = img.shape[:2]
    shadow = np.ones((h, w), dtype=np.float32)

    cx = random.randint(int(0.15 * w), int(0.85 * w))
    cy = random.randint(int(0.15 * h), int(0.85 * h))
    ax = random.randint(max(8, int(0.15 * w)), max(12, int(0.35 * w)))
    ay = random.randint(max(8, int(0.08 * h)), max(12, int(0.25 * h)))
    angle = random.randint(0, 180)

    mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(mask, (cx, cy), (ax, ay), angle, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(w, h) * 0.035)

    shadow = shadow - float(opacity) * mask
    out = img.astype(np.float32) * shadow[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def jpeg_compression(img: np.ndarray, quality: int) -> np.ndarray:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, enc = cv2.imencode(".jpg", img, encode_param)
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return dec


def apply_condition(img: np.ndarray, condition: str, severity: int) -> Tuple[np.ndarray, dict]:
    severity = int(severity)

    if condition == "clean":
        return img, {"condition": "clean", "severity": 0}

    if condition == "noise":
        p = SEVERITY_PRESETS["noise"][severity]
        return gaussian_noise(img, p["sigma"]), {"condition": condition, "severity": severity, **p}

    if condition == "gaussian_blur":
        p = SEVERITY_PRESETS["gaussian_blur"][severity]
        return gaussian_blur(img, p["kernel"]), {"condition": condition, "severity": severity, **p}

    if condition == "motion_blur":
        p = SEVERITY_PRESETS["motion_blur"][severity]
        return motion_blur(img, p["kernel"]), {"condition": condition, "severity": severity, **p}

    if condition == "brightness":
        p = SEVERITY_PRESETS["brightness"][severity]
        factor = random.uniform(*p["factor_range"])
        return brightness_shift(img, factor), {
            "condition": condition, "severity": severity, "factor": round(factor, 4)
        }

    if condition == "contrast":
        p = SEVERITY_PRESETS["contrast"][severity]
        factor = random.uniform(*p["factor_range"])
        return contrast_shift(img, factor), {
            "condition": condition, "severity": severity, "factor": round(factor, 4)
        }

    if condition == "gamma":
        p = SEVERITY_PRESETS["gamma"][severity]
        gamma = random.uniform(*p["gamma_range"])
        return gamma_correction(img, gamma), {
            "condition": condition, "severity": severity, "gamma": round(gamma, 4)
        }

    if condition == "shadow":
        p = SEVERITY_PRESETS["shadow"][severity]
        return local_soft_shadow(img, p["opacity"]), {"condition": condition, "severity": severity, **p}

    if condition == "jpeg":
        p = SEVERITY_PRESETS["jpeg"][severity]
        return jpeg_compression(img, p["quality"]), {"condition": condition, "severity": severity, **p}

    if condition == "medium":
        # Mild realistic image-quality degradation.
        out = gaussian_noise(img, sigma=5)
        out = gaussian_blur(out, kernel=3)
        out = brightness_shift(out, random.uniform(0.85, 1.15))
        out = contrast_shift(out, random.uniform(0.90, 1.10))
        return out, {"condition": condition, "severity": 1}

    if condition == "hard":
        # Strong but still realistic mixed image-quality degradation.
        out = gaussian_noise(img, sigma=18)
        out = motion_blur(out, kernel=9)
        out = brightness_shift(out, random.uniform(0.65, 1.35))
        out = contrast_shift(out, random.uniform(0.70, 1.30))
        out = gamma_correction(out, random.uniform(0.70, 1.40))
        out = local_soft_shadow(out, opacity=0.22)
        out = jpeg_compression(out, quality=65)
        return out, {"condition": condition, "severity": 3}

    raise ValueError(f"Unknown condition: {condition}")


def list_images(input_dir: Path):
    return sorted([p for p in input_dir.rglob("*") if p.suffix.lower() in IMG_EXTS])


def generate_dataset(input_dir: str, output_dir: str, condition: str, severity: int, seed: int, metadata_csv: str = None):
    set_seed(seed)

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(input_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {input_dir}")

    rows = []
    n_written = 0

    for img_path in image_paths:
        rel = img_path.relative_to(input_dir)
        out_path = output_dir / rel

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Cannot read image: {img_path}")
            continue

        out, meta = apply_condition(img, condition, severity)
        safe_imwrite(out_path, out, quality=95)

        rows.append({
            "source": str(img_path),
            "output": str(out_path.with_suffix(".jpg")),
            "relative_path": str(rel.with_suffix(".jpg")),
            **meta
        })
        n_written += 1

    if metadata_csv:
        metadata_csv = Path(metadata_csv)
        metadata_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted(set(k for row in rows for k in row.keys()))
        with open(metadata_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"Generated {n_written} images: {condition}, severity={severity} -> {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input clean test image folder")
    parser.add_argument("--output", required=True, help="Output corrupted image folder")
    parser.add_argument("--condition", required=True,
                        choices=[
                            "clean",
                            "noise",
                            "gaussian_blur",
                            "motion_blur",
                            "brightness",
                            "contrast",
                            "gamma",
                            "shadow",
                            "jpeg",
                            "medium",
                            "hard",
                        ])
    parser.add_argument("--severity", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metadata-csv", default=None)
    args = parser.parse_args()

    generate_dataset(
        input_dir=args.input,
        output_dir=args.output,
        condition=args.condition,
        severity=args.severity,
        seed=args.seed,
        metadata_csv=args.metadata_csv,
    )


if __name__ == "__main__":
    main()
