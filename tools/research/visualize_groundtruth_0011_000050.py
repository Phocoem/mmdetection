#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualize COCO ground-truth instance masks for one lettuce image.

GT masks are drawn in red to match MMDetection-style prediction visualization.

Example:
python tools/research/visualize_groundtruth_0011_000050.py \
  --ann mmdet_dataset/lettuce/annotations/test.json \
  --image mmdet_dataset/lettuce_c/images/contrast/3/0011_000050.png \
  --image-stem 0011_000050 \
  --out-dir work_dirs/research/predict_0011_000050_conditions_with_gt/contrast_s3/ground_truth
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def load_json(path: Path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def find_image_record(coco: dict, image_stem: str):
    matches = []
    for img in coco.get("images", []):
        if Path(img["file_name"]).stem == image_stem:
            matches.append(img)
    if not matches:
        raise FileNotFoundError(f"No image record with stem '{image_stem}' in annotation file.")
    if len(matches) > 1:
        print(f"[WARN] Found {len(matches)} image records for stem {image_stem}; using first.")
    return matches[0]


def segmentation_to_mask(segmentation, height: int, width: int):
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:
        raise ImportError("pycocotools is required to decode COCO masks.") from exc

    if segmentation is None:
        return np.zeros((height, width), dtype=np.uint8)

    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles)
        mask = mask_utils.decode(rle)
    elif isinstance(segmentation, dict):
        if isinstance(segmentation.get("counts"), list):
            rle = mask_utils.frPyObjects(segmentation, height, width)
        else:
            rle = segmentation
        mask = mask_utils.decode(rle)
    else:
        raise TypeError(f"Unsupported segmentation type: {type(segmentation)}")

    if mask.ndim == 3:
        mask = np.any(mask, axis=2)

    return mask.astype(np.uint8)


def draw_ground_truth(image_bgr, anns, alpha=0.45, draw_box=True, draw_index=False):
    h, w = image_bgr.shape[:2]
    out = image_bgr.copy()

    # Red in BGR, close to common prediction mask visualization.
    mask_color = np.array([0, 0, 255], dtype=np.uint8)
    outline_color = (0, 0, 255)

    anns = sorted(anns, key=lambda a: float(a.get("area", 0)), reverse=True)

    for idx, ann in enumerate(anns, start=1):
        mask = segmentation_to_mask(ann.get("segmentation"), h, w).astype(bool)
        if not np.any(mask):
            continue

        color_layer = np.zeros_like(out)
        color_layer[mask] = mask_color
        out = np.where(
            mask[:, :, None],
            (out.astype(np.float32) * (1.0 - alpha) + color_layer.astype(np.float32) * alpha).astype(np.uint8),
            out,
        )

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, outline_color, 2)

        if draw_box and "bbox" in ann:
            x, y, bw, bh = ann["bbox"]
            x1, y1 = int(round(x)), int(round(y))
            x2, y2 = int(round(x + bw)), int(round(y + bh))
            cv2.rectangle(out, (x1, y1), (x2, y2), outline_color, 1)

            if draw_index:
                label = f"GT {idx}"
                tx, ty = max(0, x1), max(14, y1 - 4)
                cv2.putText(out, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, outline_color, 1, cv2.LINE_AA)

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann", default="mmdet_dataset/lettuce/annotations/test.json")
    parser.add_argument("--image", required=True)
    parser.add_argument("--image-stem", default="0011_000050")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--no-box", action="store_true")
    parser.add_argument("--draw-index", action="store_true")
    args = parser.parse_args()

    ann_path = Path(args.ann)
    image_path = Path(args.image)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ann_path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    coco = load_json(ann_path)
    img_record = find_image_record(coco, args.image_stem)
    image_id = img_record["id"]

    anns = [a for a in coco.get("annotations", []) if a.get("image_id") == image_id]

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    gt = draw_ground_truth(
        image_bgr,
        anns,
        alpha=args.alpha,
        draw_box=not args.no_box,
        draw_index=args.draw_index,
    )

    out_path = out_dir / f"{args.image_stem}_groundtruth.png"
    cv2.imwrite(str(out_path), gt)
    print(f"[OK] GT instances: {len(anns)}")
    print(f"[OK] Saved: {out_path}")


if __name__ == "__main__":
    main()
