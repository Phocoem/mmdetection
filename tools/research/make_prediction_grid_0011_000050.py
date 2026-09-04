#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create a paper-style prediction comparison grid.

Default rows:
- Clean
- Brightness S3
- Contrast S3
- Gaussian noise S3

Default columns:
- Input
- Ground Truth
- Mask R-CNN R50
- Mask R-CNN R101
- DGCF-FPN

Run after predict_0011_000050_conditions_with_gt.sh.

Example:
python tools/research/make_prediction_grid_0011_000050.py \
  --root work_dirs/research/predict_0011_000050_conditions_with_gt \
  --out work_dirs/research/predict_0011_000050_conditions_with_gt/prediction_grid.png
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


CONDITION_LABELS = [
    ("clean", "Clean"),
    ("brightness_s3", "Brightness S3"),
    ("contrast_s3", "Contrast S3"),
    ("gaussian_noise_s3", "Gaussian noise S3"),
]

MODEL_COLUMNS = [
    ("mask_rcnn_r50_fpn", "Mask R-CNN R50"),
    ("mask_rcnn_r101_fpn", "Mask R-CNN R101"),
    ("mask_rcnn_r50_dgcf_fpn", "DGCF-FPN"),
]


def find_image(path_or_dir: Path, stem: str):
    if path_or_dir.is_file():
        return path_or_dir
    if not path_or_dir.exists():
        return None

    # Common demo output names.
    candidates = []
    for ext in [".png", ".jpg", ".jpeg"]:
        candidates.extend([
            path_or_dir / f"{stem}{ext}",
            path_or_dir / f"{stem}_groundtruth{ext}",
        ])

    for p in candidates:
        if p.exists():
            return p

    matches = sorted(path_or_dir.rglob(f"{stem}*.*"))
    image_matches = [p for p in matches if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    return image_matches[0] if image_matches else None


def read_resize(path: Path, height: int):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read image: {path}")

    h, w = img.shape[:2]
    scale = height / h
    new_w = int(round(w * scale))
    return cv2.resize(img, (new_w, height), interpolation=cv2.INTER_AREA)


def add_top_label(img, label):
    h, w = img.shape[:2]
    pad = 34
    canvas = np.full((h + pad, w, 3), 255, dtype=np.uint8)
    canvas[pad:, :] = img
    cv2.putText(canvas, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2, cv2.LINE_AA)
    return canvas


def add_left_label(row_img, label):
    h, w = row_img.shape[:2]
    pad = 150
    canvas = np.full((h, w + pad, 3), 255, dtype=np.uint8)
    canvas[:, pad:] = row_img

    # Put horizontal row label.
    cv2.putText(canvas, label, (8, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2, cv2.LINE_AA)
    return canvas


def pad_to_width(img, width):
    h, w = img.shape[:2]
    if w == width:
        return img
    canvas = np.full((h, width, 3), 255, dtype=np.uint8)
    canvas[:, :w] = img
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="work_dirs/research/predict_0011_000050_conditions_with_gt")
    parser.add_argument("--image-stem", default="0011_000050")
    parser.add_argument("--clean-image", default="mmdet_dataset/lettuce/images/test/0011_000050.png")
    parser.add_argument("--out", default="work_dirs/research/predict_0011_000050_conditions_with_gt/prediction_grid.png")
    parser.add_argument("--height", type=int, default=300)
    args = parser.parse_args()

    root = Path(args.root)
    stem = args.image_stem

    # Inputs for each condition.
    inputs = {
        "clean": Path(args.clean_image),
        "brightness_s3": Path(f"mmdet_dataset/lettuce_c/images/brightness/3/{stem}.png"),
        "contrast_s3": Path(f"mmdet_dataset/lettuce_c/images/contrast/3/{stem}.png"),
        "gaussian_noise_s3": Path(f"mmdet_dataset/lettuce_c/images/gaussian_noise/3/{stem}.png"),
    }

    if not inputs["clean"].exists():
        for ext in [".jpg", ".jpeg"]:
            p = Path(f"mmdet_dataset/lettuce/images/test/{stem}{ext}")
            if p.exists():
                inputs["clean"] = p
                break

    rows = []
    header_labels = ["Input", "Ground Truth"] + [label for _, label in MODEL_COLUMNS]

    for cond, cond_label in CONDITION_LABELS:
        cells = []

        # Input.
        input_path = inputs[cond]
        if not input_path.exists():
            print(f"[WARN] Missing input: {input_path}")
            continue
        cells.append(add_top_label(read_resize(input_path, args.height), header_labels[0]))

        # Ground truth.
        gt_dir = root / cond / "ground_truth"
        gt_path = find_image(gt_dir, stem)
        if gt_path is None:
            print(f"[WARN] Missing GT for {cond}: {gt_dir}")
            continue
        cells.append(add_top_label(read_resize(gt_path, args.height), header_labels[1]))

        # Model predictions.
        for model_name, model_label in MODEL_COLUMNS:
            pred_dir = root / cond / model_name
            pred_path = find_image(pred_dir, stem)
            if pred_path is None:
                print(f"[WARN] Missing prediction for {cond}/{model_name}: {pred_dir}")
                # Blank placeholder with same size as input.
                blank = np.full_like(read_resize(input_path, args.height), 255)
                cells.append(add_top_label(blank, model_label))
            else:
                cells.append(add_top_label(read_resize(pred_path, args.height), model_label))

        max_h = max(c.shape[0] for c in cells)
        cells = [np.pad(c, ((0, max_h - c.shape[0]), (0, 0), (0, 0)), constant_values=255) for c in cells]

        gap = np.full((max_h, 18, 3), 255, dtype=np.uint8)
        row = cells[0]
        for cell in cells[1:]:
            row = np.hstack([row, gap, cell])

        row = add_left_label(row, cond_label)
        rows.append(row)

    if not rows:
        raise RuntimeError("No rows created. Check input/prediction paths.")

    max_w = max(r.shape[1] for r in rows)
    rows = [pad_to_width(r, max_w) for r in rows]
    gap_y = np.full((22, max_w, 3), 255, dtype=np.uint8)

    grid = rows[0]
    for row in rows[1:]:
        grid = np.vstack([grid, gap_y, row])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)
    print(f"[OK] Saved grid: {out_path}")


if __name__ == "__main__":
    main()
