#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_robustness_grid.py

Create paper-ready grids showing real clean images and their corrupted
robustness variants. Optionally overlays COCO ground-truth masks/boxes.

Expected robustness dataset layout from build_robust_coco.py:

robust_root/
  images/
    clean/
    gaussian_noise/s1/
    gaussian_noise/s2/
    ...
  annotations/
    test_clean.json
    test_gaussian_noise_s1.json
    ...

Example:
python projects/ruq/tools/visualize_robustness_grid.py \
  --robust-root data/lettuce_robust \
  --out-dir paper_assets/robustness_images \
  --corruptions gaussian_noise motion_blur defocus_blur shadow brightness_low brightness_high contrast_low jpeg_compression \
  --severity 2 \
  --num-images 6 \
  --draw-gt
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

# Use Agg so this script works on servers without display.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from pycocotools.coco import COCO
except Exception:  # pragma: no cover
    COCO = None

DISPLAY_NAMES = {
    "clean": "Clean",
    "gaussian_noise": "Gaussian Noise",
    "shot_noise": "Shot Noise",
    "impulse_noise": "Impulse Noise",
    "motion_blur": "Motion Blur",
    "defocus_blur": "Defocus Blur",
    "zoom_blur": "Zoom Blur",
    "shadow": "Shadow",
    "brightness_low": "Low Brightness",
    "brightness_high": "High Brightness",
    "brightness": "Brightness",
    "contrast_low": "Low Contrast",
    "contrast": "Contrast",
    "jpeg_compression": "JPEG",
    "pixelate": "Pixelate",
    "fog": "Fog",
    "snow": "Snow",
    "frost": "Frost",
    "occlusion": "Occlusion",
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def dname(x: str) -> str:
    return DISPLAY_NAMES.get(x, x.replace("_", " ").title())


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def find_clean_annotation(robust_root: Path) -> Optional[Path]:
    candidates = [
        robust_root / "annotations" / "test_clean.json",
        robust_root / "annotations" / "clean.json",
        robust_root / "test_clean.json",
    ]
    for p in candidates:
        if p.is_file():
            return p
    anns = sorted((robust_root / "annotations").glob("*clean*.json"))
    return anns[0] if anns else None


def list_clean_images(robust_root: Path, annotation: Optional[Path]) -> List[str]:
    if annotation and annotation.is_file():
        data = read_json(annotation)
        names = [Path(img["file_name"]).name for img in data.get("images", [])]
        # Keep order in COCO file and remove duplicates.
        out, seen = [], set()
        for n in names:
            if n not in seen:
                out.append(n)
                seen.add(n)
        if out:
            return out
    clean_dir = robust_root / "images" / "clean"
    return sorted([p.name for p in clean_dir.iterdir() if p.suffix.lower() in IMG_EXTS])


def image_path_for_condition(robust_root: Path, image_name: str, corruption: str, severity: Optional[int]) -> Optional[Path]:
    if corruption == "clean":
        candidates = [
            robust_root / "images" / "clean" / image_name,
            robust_root / "clean" / image_name,
        ]
    else:
        s = f"s{severity}" if severity is not None else ""
        candidates = [
            robust_root / "images" / corruption / s / image_name,
            robust_root / "images" / corruption / str(severity) / image_name if severity is not None else robust_root / "images" / corruption / image_name,
            robust_root / corruption / s / image_name,
            robust_root / corruption / str(severity) / image_name if severity is not None else robust_root / corruption / image_name,
            robust_root / "images" / f"{corruption}_s{severity}" / image_name if severity is not None else robust_root / "images" / corruption / image_name,
        ]
    for p in candidates:
        if p and p.is_file():
            return p
    # fallback: recursive search, robust to custom folder names
    roots = [robust_root / "images", robust_root]
    suffixes = []
    if corruption == "clean":
        suffixes.append(f"clean/{image_name}")
    elif severity is not None:
        suffixes.extend([
            f"{corruption}/s{severity}/{image_name}",
            f"{corruption}/{severity}/{image_name}",
            f"{corruption}_s{severity}/{image_name}",
            f"{corruption}_{severity}/{image_name}",
        ])
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob(image_name):
            sp = str(p).replace("\\", "/")
            if any(sp.endswith(suf) for suf in suffixes):
                return p
    return None


def coco_index(annotation: Optional[Path]):
    if annotation is None or not annotation.is_file() or COCO is None:
        return None, {}
    coco = COCO(str(annotation))
    name_to_id = {Path(v["file_name"]).name: k for k, v in coco.imgs.items()}
    return coco, name_to_id


def overlay_gt(image: Image.Image, coco, image_id: int, alpha: float = 0.35) -> Image.Image:
    """Overlay GT masks and boxes from COCO annotation."""
    if coco is None or image_id is None:
        return image
    img = image.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    ann_ids = coco.getAnnIds(imgIds=[image_id])
    anns = coco.loadAnns(ann_ids)
    # green translucent mask + red boundary/box. Chosen for paper visibility.
    for ann in anns:
        try:
            mask = coco.annToMask(ann).astype(np.uint8)
        except Exception:
            mask = None
        if mask is not None:
            mask_img = Image.fromarray(mask * int(255 * alpha), mode="L").resize(img.size)
            color = Image.new("RGBA", img.size, (0, 220, 0, int(255 * alpha)))
            overlay = Image.composite(color, overlay, mask_img)
        if "bbox" in ann:
            x, y, w, h = ann["bbox"]
            sx = img.size[0] / max(1, coco.imgs[image_id].get("width", img.size[0]))
            sy = img.size[1] / max(1, coco.imgs[image_id].get("height", img.size[1]))
            draw.rectangle([x * sx, y * sy, (x + w) * sx, (y + h) * sy], outline=(255, 0, 0, 230), width=2)
    return Image.alpha_composite(img, overlay).convert("RGB")


def load_rgb(path: Path, resize_long: int = 512) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if resize_long > 0:
        w, h = img.size
        scale = resize_long / max(w, h)
        if scale < 1:
            img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    return img


def make_grid_for_image(
    robust_root: Path,
    image_name: str,
    corruptions: List[str],
    severity: int,
    out_dir: Path,
    draw_gt: bool,
    coco,
    name_to_id: Dict[str, int],
    resize_long: int,
) -> Optional[Path]:
    conditions = ["clean"] + corruptions
    imgs: List[Tuple[str, Image.Image]] = []
    for cond in conditions:
        p = image_path_for_condition(robust_root, image_name, cond, None if cond == "clean" else severity)
        if p is None:
            print(f"[WARN] Missing image for {cond} severity={severity}: {image_name}")
            continue
        im = load_rgb(p, resize_long=resize_long)
        if draw_gt:
            im = overlay_gt(im, coco, name_to_id.get(image_name))
        label = dname(cond) if cond == "clean" else f"{dname(cond)} S{severity}"
        imgs.append((label, im))
    if not imgs:
        return None
    n = len(imgs)
    fig_w = max(3.2 * n, 8)
    fig_h = 3.8
    fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h))
    if n == 1:
        axes = [axes]
    for ax, (label, im) in zip(axes, imgs):
        ax.imshow(im)
        ax.set_title(label, fontsize=12)
        ax.axis("off")
    fig.suptitle(Path(image_name).stem, fontsize=14, y=1.02)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"robust_grid_{Path(image_name).stem}_s{severity}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_contact_sheet(paths: List[Path], out_path: Path, title: str = "Robustness samples") -> None:
    if not paths:
        return
    images = [Image.open(p).convert("RGB") for p in paths]
    widths, heights = zip(*(im.size for im in images))
    max_w = max(widths)
    total_h = sum(heights)
    sheet = Image.new("RGB", (max_w, total_h), (255, 255, 255))
    y = 0
    for im in images:
        sheet.paste(im, ((max_w - im.width) // 2, y))
        y += im.height
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robust-root", required=True, help="Robustness dataset root")
    parser.add_argument("--out-dir", default="paper_assets/robustness_images")
    parser.add_argument("--corruptions", nargs="+", default=[
        "gaussian_noise", "motion_blur", "defocus_blur", "shadow",
        "brightness_low", "brightness_high", "contrast_low", "jpeg_compression"
    ])
    parser.add_argument("--severity", type=int, default=2)
    parser.add_argument("--num-images", type=int, default=6)
    parser.add_argument("--image-names", nargs="*", default=None, help="Specific image file names to visualize")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--draw-gt", action="store_true", help="Overlay COCO ground-truth masks/boxes if pycocotools is available")
    parser.add_argument("--resize-long", type=int, default=512)
    args = parser.parse_args()

    robust_root = Path(args.robust_root)
    out_dir = Path(args.out_dir)
    ann = find_clean_annotation(robust_root)
    coco, name_to_id = coco_index(ann)
    if args.draw_gt and coco is None:
        print("[WARN] --draw-gt requested but pycocotools/annotation is unavailable. Continue without overlay.")

    names = args.image_names or list_clean_images(robust_root, ann)
    if not args.image_names:
        random.Random(args.seed).shuffle(names)
        names = names[: args.num_images]

    exported: List[Path] = []
    for name in names:
        p = make_grid_for_image(
            robust_root=robust_root,
            image_name=name,
            corruptions=args.corruptions,
            severity=args.severity,
            out_dir=out_dir,
            draw_gt=args.draw_gt,
            coco=coco,
            name_to_id=name_to_id,
            resize_long=args.resize_long,
        )
        if p:
            exported.append(p)
            print(f"[OK] {p}")
    make_contact_sheet(exported, out_dir / f"robustness_contact_sheet_s{args.severity}.png")
    print(f"[DONE] Exported {len(exported)} robustness grid figures to {out_dir}")


if __name__ == "__main__":
    main()
