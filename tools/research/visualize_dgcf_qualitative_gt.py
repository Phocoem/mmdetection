#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qualitative visualization for DGCF paper.

Supports BOTH:
  --image-indices 0 5 10
and:
  --image-names 0003_000000.png 0007_000000.png 0011_000000.png

Columns:
  Input | Ground truth | Mask R-CNN R50 | Enhanced Mask R-CNN (DGCF)

This script:
- uses existing checkpoints
- uses high-contrast overlay colors
- supports clean and corrupted images
- creates individual figures and optional grid figures
- auto-fixes missing mmdet.models.necks.pafpn.py if needed
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt


HIGH_CONTRAST_COLORS = [
    (0, 255, 255),     # cyan
    (255, 0, 255),     # magenta
    (255, 255, 0),     # yellow
    (255, 60, 60),     # red
    (0, 120, 255),     # blue
    (80, 255, 80),     # lime
    (255, 165, 0),     # orange
    (255, 255, 255),   # white
    (180, 0, 255),     # violet
    (0, 255, 160),     # aqua green
    (255, 100, 180),   # hot pink
    (120, 220, 255),   # light cyan-blue
]


def get_vis_color(i):
    return HIGH_CONTRAST_COLORS[i % len(HIGH_CONTRAST_COLORS)]


def ensure_pafpn_stub():
    """Create a small PAFPN compatibility stub if local MMDetection is missing it."""
    target = Path.cwd() / "mmdet" / "models" / "necks" / "pafpn.py"
    if target.exists() or not target.parent.exists():
        return

    target.write_text(
        "from mmdet.registry import MODELS\n"
        "from .fpn import FPN\n\n"
        "@MODELS.register_module()\n"
        "class PAFPN(FPN):\n"
        "    pass\n",
        encoding="utf-8",
    )
    print(f"[AUTO-FIX] created missing {target}")


def patch_torch_load():
    """PyTorch 2.6+ compatibility for old MMDetection checkpoints."""
    if getattr(torch.load, "_lettuce_patch_weights_only_false", False):
        return

    old = torch.load

    def patched(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return old(*args, **kwargs)

    patched._lettuce_patch_weights_only_false = True
    torch.load = patched


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def normalize_image_name(name: str) -> str:
    """Accept file name, stem, or path; return stem for robust matching."""
    return Path(str(name)).stem


def find_checkpoint(work_dir):
    wd = Path(work_dir)
    for pat in ["best_coco_segm_mAP*.pth", "best_*.pth", "epoch_*.pth", "*.pth"]:
        hits = list(wd.glob(pat))
        if hits:
            return sorted(hits, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    return None


def rle_decode(rle):
    try:
        from pycocotools import mask as mask_utils
        return mask_utils.decode(rle).astype(bool)
    except Exception:
        pass

    counts = rle["counts"]
    h, w = rle["size"]
    flat = np.zeros(h * w, dtype=np.uint8)
    idx = 0
    val = 0
    for c in counts:
        if val == 1:
            flat[idx:idx + c] = 1
        idx += c
        val = 1 - val
    return flat.reshape((w, h)).T.astype(bool)


def polygon_to_mask(segmentation, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    if not isinstance(segmentation, list):
        return mask.astype(bool)

    for poly in segmentation:
        if len(poly) < 6:
            continue
        pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
        pts = np.round(pts).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)

    return mask.astype(bool)


def load_coco(root, ann_file):
    coco = read_json(Path(root) / ann_file)
    anns_by_img = {}
    for ann in coco.get("annotations", []):
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    imgs = sorted(coco["images"], key=lambda x: x.get("id", 0))
    return coco, imgs, anns_by_img


def select_image_indices_from_names(imgs, image_names):
    """Map image file names/stems to indices in COCO image list."""
    if not image_names:
        return None

    wanted = [normalize_image_name(x) for x in image_names]
    mapping = {}
    for idx, img in enumerate(imgs):
        file_name = img.get("file_name", "")
        mapping[normalize_image_name(file_name)] = idx

    selected = []
    missing = []
    for name in wanted:
        if name in mapping:
            selected.append(mapping[name])
        else:
            missing.append(name)

    if missing:
        print("[WARNING] These image names were not found in clean test annotation:")
        for m in missing:
            print(f"  - {m}")
        print("[INFO] Available examples:")
        for img in imgs[:10]:
            print(f"  - {img.get('file_name', '')}")

    return selected


def get_image_and_gt_by_index(clean_root, ann_file, prefix, index):
    coco, imgs, anns_by_img = load_coco(clean_root, ann_file)
    index = max(0, min(index, len(imgs) - 1))
    img_info = imgs[index]
    img_path = Path(clean_root) / prefix / img_info["file_name"]
    anns = anns_by_img.get(img_info["id"], [])
    return img_path, img_info, anns, index


def get_condition_image(clean_root, benchmark_root, benchmark_manifest, condition, index):
    """Return condition image path plus clean GT info/annotations."""
    gt_img_path, img_info, anns, index = get_image_and_gt_by_index(
        clean_root, "annotations/test.json", "images/test", index
    )

    if condition == "clean":
        return gt_img_path, img_info, anns, index

    corruption, severity = condition.split(":")
    severity = int(severity)
    man = read_json(benchmark_manifest)
    out_ann = man.get("output_annotation", "annotations/test_png.json")

    for item in man.get("conditions", []):
        if item.get("corruption") == corruption and int(item.get("severity")) == severity:
            bench_coco, bench_imgs, _ = load_coco(benchmark_root, out_ann)
            index = max(0, min(index, len(bench_imgs) - 1))
            bench_info = bench_imgs[index]
            img_path = Path(benchmark_root) / item["image_prefix"].rstrip("/") / bench_info["file_name"]
            return img_path, img_info, anns, index

    raise RuntimeError(f"condition not found in benchmark manifest: {condition}")


def overlay_gt(image_bgr, img_info, anns, alpha=0.38):
    out = image_bgr.copy()
    h, w = image_bgr.shape[:2]

    for i, ann in enumerate(anns):
        seg = ann.get("segmentation", None)
        if seg is None:
            continue

        if isinstance(seg, dict):
            mask = rle_decode(seg)
        else:
            mask = polygon_to_mask(seg, h, w)

        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

        color_tuple = get_vis_color(i)
        color = np.zeros_like(out, dtype=np.uint8)
        color[:] = np.array(color_tuple, dtype=np.uint8)
        out[mask] = ((1 - alpha) * out[mask] + alpha * color[mask]).astype(np.uint8)

        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, tuple(int(x) for x in color_tuple), 2)

    return out


def load_detector(config, checkpoint, device):
    ensure_pafpn_stub()
    patch_torch_load()

    from mmdet.utils import register_all_modules
    from mmdet.apis import init_detector

    register_all_modules(init_default_scope=True)
    return init_detector(str(config), str(checkpoint), device=device)


def overlay_prediction(image_bgr, result, score_thr, alpha=0.38):
    out = image_bgr.copy()
    inst = getattr(result, "pred_instances", None)
    if inst is None or len(inst) == 0:
        return out

    scores = getattr(inst, "scores", None)
    masks = getattr(inst, "masks", None)
    bboxes = getattr(inst, "bboxes", None)

    if scores is not None:
        keep = scores.detach().cpu().numpy() >= score_thr
    else:
        keep = np.ones(len(inst), dtype=bool)

    if masks is not None:
        masks_np = masks.detach().cpu().numpy().astype(bool)
        for i, mask in enumerate(masks_np):
            if not keep[i]:
                continue

            color_tuple = get_vis_color(i)
            color = np.zeros_like(out, dtype=np.uint8)
            color[:] = np.array(color_tuple, dtype=np.uint8)
            out[mask] = ((1 - alpha) * out[mask] + alpha * color[mask]).astype(np.uint8)

            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, tuple(int(x) for x in color_tuple), 2)

    if bboxes is not None:
        boxes = bboxes.detach().cpu().numpy()
        for i, box in enumerate(boxes):
            if not keep[i]:
                continue

            color_tuple = get_vis_color(i)
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(out, (x1, y1), (x2, y2), tuple(int(x) for x in color_tuple), 2)

    return out


def build_models(args):
    models = [
        {
            "name": "Mask R-CNN R50",
            "config": "configs/fair_lettuce/mask_rcnn_r50_fpn.py",
            "work_dir": "work_dirs/research/mask_rcnn_r50_fpn/seed_2026",
        },
        {
            "name": "Enhanced Mask R-CNN (DGCF)",
            "config": "configs/fair_lettuce/mask_rcnn_r50_dgcf_fpn.py",
            "work_dir": "work_dirs/research/mask_rcnn_r50_dgcf_fpn/seed_2026",
        },
    ]

    loaded = []
    for m in models:
        cfg = Path(m["config"])
        ckpt = find_checkpoint(m["work_dir"])

        if not cfg.is_file():
            print(f"[SKIP] missing config: {cfg}")
            continue
        if ckpt is None:
            print(f"[SKIP] missing checkpoint: {m['work_dir']}")
            continue

        print(f"[LOAD] {m['name']} | {ckpt}")
        loaded.append((m["name"], load_detector(cfg, ckpt, args.device)))

    if not loaded:
        raise RuntimeError("No models loaded.")

    return loaded


def save_single_condition_figure(args, idx, condition, models, display_name=None):
    from mmdet.apis import inference_detector

    img_path, img_info, anns, real_idx = get_condition_image(
        args.clean_root, args.benchmark_root, args.benchmark_manifest, condition, idx
    )
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[WARN] cannot read {img_path}")
        return

    panels = [
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(overlay_gt(img, img_info, anns, args.alpha), cv2.COLOR_BGR2RGB),
    ]
    titles = ["Input", "Ground truth"]

    for name, model in models:
        with torch.no_grad():
            result = inference_detector(model, str(img_path))

        vis = overlay_prediction(img, result, args.score_thr, args.alpha)
        panels.append(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        titles.append(name)

    img_label = display_name or Path(img_info.get("file_name", f"idx{real_idx}")).name

    plt.figure(figsize=(4 * len(panels), 4.2))
    for i, (panel, title) in enumerate(zip(panels, titles), 1):
        ax = plt.subplot(1, len(panels), i)
        ax.imshow(panel)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.suptitle(f"{condition} | {img_label}", fontsize=12)
    plt.tight_layout()

    safe_label = normalize_image_name(img_label)
    out = Path(args.out_dir) / f"qual_{safe_label}_{condition.replace(':','_')}.png"
    plt.savefig(out, dpi=300)
    plt.savefig(out.with_suffix(".pdf"))
    plt.close()
    print(f"[OK] {out}")


def save_grid_figure(args, idx, models, display_name=None):
    from mmdet.apis import inference_detector

    rows = []
    img_label = display_name

    for condition in args.conditions:
        img_path, img_info, anns, real_idx = get_condition_image(
            args.clean_root, args.benchmark_root, args.benchmark_manifest, condition, idx
        )
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        if img_label is None:
            img_label = Path(img_info.get("file_name", f"idx{real_idx}")).name

        row = [
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(overlay_gt(img, img_info, anns, args.alpha), cv2.COLOR_BGR2RGB),
        ]

        for name, model in models:
            with torch.no_grad():
                result = inference_detector(model, str(img_path))
            row.append(cv2.cvtColor(overlay_prediction(img, result, args.score_thr, args.alpha), cv2.COLOR_BGR2RGB))

        rows.append((condition, row))

    if not rows:
        return

    col_titles = ["Input", "Ground truth"] + [m[0] for m in models]
    nrows = len(rows)
    ncols = len(col_titles)

    plt.figure(figsize=(4 * ncols, 3.5 * nrows))

    for r, (condition, row) in enumerate(rows):
        for c, panel in enumerate(row):
            ax = plt.subplot(nrows, ncols, r * ncols + c + 1)
            ax.imshow(panel)
            ax.axis("off")

            if r == 0:
                ax.set_title(col_titles[c], fontsize=10)

            if c == 0:
                ax.text(
                    -0.02, 0.5, condition,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=10,
                )

    plt.suptitle(str(img_label), fontsize=12)
    plt.tight_layout()

    safe_label = normalize_image_name(img_label or f"idx{idx}")
    out = Path(args.out_dir) / f"qual_grid_{safe_label}.png"
    plt.savefig(out, dpi=300)
    plt.savefig(out.with_suffix(".pdf"))
    plt.close()
    print(f"[OK] {out}")


def resolve_selected_indices(args):
    """Return list of (index, display_name). image-names takes priority."""
    _, imgs, _ = load_coco(args.clean_root, "annotations/test.json")

    if args.image_names:
        selected = select_image_indices_from_names(imgs, args.image_names)
        pairs = []
        for idx in selected:
            pairs.append((idx, Path(imgs[idx].get("file_name", f"idx{idx}")).name))
        return pairs

    return [(idx, None) for idx in args.image_indices]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clean-root", default="mmdet_dataset/lettuce")
    p.add_argument("--benchmark-root", default="mmdet_dataset/lettuce_c")
    p.add_argument("--benchmark-manifest", default="mmdet_dataset/lettuce_c/manifest.json")
    p.add_argument("--out-dir", default="paper_outputs_dgcf_simple/qualitative")
    p.add_argument("--image-indices", nargs="+", type=int, default=[0, 5, 10])
    p.add_argument("--image-names", nargs="+", default=None,
                   help="Specific image file names or stems, e.g. 0003_000000.png 0007_000000.png")
    p.add_argument("--conditions", nargs="+", default=[
        "clean", "brightness:1", "brightness:2", "brightness:3",
        "contrast:1", "contrast:2", "contrast:3",
        "gaussian_noise:1", "gaussian_noise:2", "gaussian_noise:3",
    ])
    p.add_argument("--score-thr", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=0.38,
                   help="Overlay alpha. Higher means stronger color.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--grid", action="store_true")
    args = p.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    if args.image_names:
        print("[INFO] --image-names provided. It will override --image-indices.")

    selected = resolve_selected_indices(args)
    if not selected:
        raise RuntimeError("No images selected.")

    print("[SELECTED IMAGES]")
    for idx, name in selected:
        print(f"  - index={idx} | name={name}")

    models = build_models(args)

    for idx, display_name in selected:
        for cond in args.conditions:
            save_single_condition_figure(args, idx, cond, models, display_name)
        if args.grid:
            save_grid_figure(args, idx, models, display_name)


if __name__ == "__main__":
    main()
