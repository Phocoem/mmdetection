"""
Create a visual grid of corruption examples for the paper.

Example:
python tools/make_corruption_grid.py --root stress --image-name IMG_001.jpg --output results/figures/corruption_grid.jpg
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


CONDITIONS = ["clean", "noise", "gaussian_blur", "motion_blur", "brightness", "contrast", "gamma", "shadow", "jpeg", "medium", "hard"]


def find_image(folder: Path, image_name: str):
    p = folder / image_name
    if p.exists():
        return p
    stem = Path(image_name).stem
    for ext in [".jpg", ".jpeg", ".png"]:
        q = folder / f"{stem}{ext}"
        if q.exists():
            return q
    matches = list(folder.rglob(f"{stem}.*"))
    return matches[0] if matches else None


def add_label(img, label):
    out = img.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w, 32), (255, 255, 255), -1)
    cv2.putText(out, label, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
    return out


def resize_keep(img, width=320):
    h, w = img.shape[:2]
    scale = width / w
    return cv2.resize(img, (width, int(h * scale)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Stress benchmark root folder")
    parser.add_argument("--image-name", required=True, help="Example image filename")
    parser.add_argument("--output", default="results/figures/corruption_grid.jpg")
    args = parser.parse_args()

    root = Path(args.root)
    imgs = []

    for cond in CONDITIONS:
        p = find_image(root / cond, args.image_name)
        if p is None:
            print(f"[WARN] Missing {cond}/{args.image_name}")
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = resize_keep(img, width=320)
        img = add_label(img, cond)
        imgs.append(img)

    if not imgs:
        raise RuntimeError("No images found")

    # normalize heights
    min_h = min(i.shape[0] for i in imgs)
    imgs = [i[:min_h] for i in imgs]

    rows = []
    for i in range(0, len(imgs), 4):
        row = imgs[i:i+4]
        if len(row) < 4:
            pad = np.ones_like(row[0]) * 255
            row += [pad] * (4 - len(row))
        rows.append(np.hstack(row))

    grid = np.vstack(rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
