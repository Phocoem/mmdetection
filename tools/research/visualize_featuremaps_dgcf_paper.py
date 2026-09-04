#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature-map visualization for lettuce robustness paper.

Default comparison:
- Mask R-CNN R50-FPN
- Enhanced Mask R-CNN (DGCF)

Default conditions:
- clean
- brightness:3
- contrast:3
- gaussian_noise:3

Default images:
- 0003_000050.png
- 0007_000050.png
- 0011_000050.png

Output per image:
- PNG and PDF grid figure
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt


def patch_torch_load():
    """Fix PyTorch >= 2.6 old-checkpoint loading issue."""
    if getattr(torch.load, "_patched_weights_only_false", False):
        return
    old_load = torch.load
    def new_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return old_load(*args, **kwargs)
    new_load._patched_weights_only_false = True
    torch.load = new_load


def ensure_pafpn_stub():
    """Create small PAFPN compatibility file if the local MMDet tree misses it."""
    pafpn = Path.cwd() / "mmdet" / "models" / "necks" / "pafpn.py"
    if pafpn.exists() or not pafpn.parent.exists():
        return
    pafpn.write_text(
        "from mmdet.registry import MODELS\n"
        "from .fpn import FPN\n\n"
        "@MODELS.register_module()\n"
        "class PAFPN(FPN):\n"
        "    pass\n",
        encoding="utf-8",
    )
    print(f"[AUTO-FIX] created missing {pafpn}")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def normalize_name(name):
    return Path(str(name)).stem


def find_checkpoint(work_dir):
    wd = Path(work_dir)
    if not wd.exists():
        return None
    for pat in ["best_coco_segm_mAP*.pth", "best_*.pth", "epoch_*.pth", "*.pth"]:
        hits = list(wd.glob(pat))
        if hits:
            return sorted(hits, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    return None


def normalize_heatmap(x):
    x = x.astype(np.float32)
    x = x - np.min(x)
    mx = np.max(x)
    if mx > 1e-8:
        x = x / mx
    return x


def feature_to_heatmap(feat, mode="mean_abs"):
    if isinstance(feat, (list, tuple)):
        feat = feat[0]
    if feat.ndim == 4:
        feat = feat[0]
    feat = feat.detach().float().cpu()
    if mode == "mean":
        heat = feat.mean(dim=0).numpy()
    elif mode == "max":
        heat = feat.max(dim=0).values.numpy()
    else:
        heat = feat.abs().mean(dim=0).numpy()
    return normalize_heatmap(heat)


def overlay_heatmap(image_bgr, heat, alpha=0.45):
    h, w = image_bgr.shape[:2]
    heat = cv2.resize(heat, (w, h), interpolation=cv2.INTER_LINEAR)
    heat_u8 = np.uint8(255 * heat)
    color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    return cv2.addWeighted(image_bgr, 1 - alpha, color, alpha, 0)


def load_clean_images(clean_root):
    coco = read_json(Path(clean_root) / "annotations" / "test.json")
    return sorted(coco["images"], key=lambda x: x.get("file_name", ""))


def find_image_index(clean_root, image_name):
    imgs = load_clean_images(clean_root)
    target = normalize_name(image_name)
    for i, img in enumerate(imgs):
        if normalize_name(img.get("file_name", "")) == target:
            return i, img.get("file_name", "")
    print(f"[WARNING] image not found: {image_name}")
    print("[INFO] first 20 images:")
    for img in imgs[:20]:
        print("  -", img.get("file_name", ""))
    raise RuntimeError(f"Image not found: {image_name}")


def get_condition_image_path(clean_root, benchmark_root, benchmark_manifest, image_name, condition):
    idx, clean_file_name = find_image_index(clean_root, image_name)
    if condition == "clean":
        return Path(clean_root) / "images" / "test" / clean_file_name

    corruption, sev = condition.split(":")
    sev = int(sev)
    man = read_json(benchmark_manifest)
    out_ann = man.get("output_annotation", "annotations/test_png.json")
    bench_coco = read_json(Path(benchmark_root) / out_ann)
    bench_imgs = sorted(bench_coco["images"], key=lambda x: x.get("file_name", ""))
    if idx >= len(bench_imgs):
        raise RuntimeError(f"Index {idx} out of range for benchmark images.")
    bench_file = bench_imgs[idx].get("file_name", clean_file_name)

    for item in man.get("conditions", []):
        if item.get("corruption") == corruption and int(item.get("severity")) == sev:
            return Path(benchmark_root) / item["image_prefix"].rstrip("/") / bench_file
    raise RuntimeError(f"Condition not found in manifest: {condition}")


def load_detector_model(config, checkpoint, device):
    patch_torch_load()
    ensure_pafpn_stub()
    from mmdet.utils import register_all_modules
    from mmdet.apis import init_detector
    register_all_modules(init_default_scope=True)
    model = init_detector(str(config), str(checkpoint), device=device)
    model.eval()
    return model


class FeatureCatcher:
    def __init__(self):
        self.value = None
        self.handle = None
    def hook(self, module, inp, out):
        self.value = out
    def register(self, module):
        self.handle = module.register_forward_hook(self.hook)
    def remove(self):
        if self.handle is not None:
            self.handle.remove()


def get_target_module(model, target="neck_p2"):
    if target == "backbone_layer4":
        if hasattr(model.backbone, "layer4"):
            return model.backbone.layer4, None
        raise RuntimeError("model.backbone.layer4 not found")
    if target.startswith("neck_p"):
        level = int(target.replace("neck_p", ""))
        if hasattr(model, "neck"):
            return model.neck, level
        raise RuntimeError("model.neck not found")
    raise RuntimeError(f"Unknown target layer: {target}")


def extract_heatmap(model, image_path, target="neck_p2", heatmap_mode="mean_abs"):
    from mmdet.apis import inference_detector
    module, level = get_target_module(model, target)
    catcher = FeatureCatcher()
    catcher.register(module)
    with torch.no_grad():
        _ = inference_detector(model, str(image_path))
    catcher.remove()
    feat = catcher.value
    if feat is None:
        raise RuntimeError("No feature captured.")
    if isinstance(feat, (list, tuple)):
        if level is None:
            level = 0
        if level >= len(feat):
            raise RuntimeError(f"Feature level {level} out of range. Got {len(feat)} levels.")
        feat = feat[level]
    return feature_to_heatmap(feat, mode=heatmap_mode)


def default_models():
    return [
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


def load_models(args):
    loaded = []
    for info in default_models():
        cfg = Path(info["config"])
        ckpt = find_checkpoint(info["work_dir"])
        if not cfg.is_file():
            print(f"[SKIP] missing config: {cfg}")
            continue
        if ckpt is None:
            print(f"[SKIP] missing checkpoint: {info['work_dir']}")
            continue
        print(f"[LOAD] {info['name']} | {ckpt}")
        model = load_detector_model(cfg, ckpt, args.device)
        loaded.append((info["name"], model))
    return loaded


def make_feature_grid_for_image(args, image_name, models):
    rows = []
    for cond in args.conditions:
        img_path = get_condition_image_path(args.clean_root, args.benchmark_root, args.benchmark_manifest, image_name, cond)
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"[WARN] cannot read image: {img_path}")
            continue
        row = {"condition": cond, "input": cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), "items": []}
        for model_name, model in models:
            heat = extract_heatmap(model, img_path, target=args.target_layer, heatmap_mode=args.heatmap_mode)
            overlay = overlay_heatmap(image_bgr, heat, alpha=args.alpha)
            row["items"].append({"name": model_name, "heat": heat, "overlay": cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)})
        rows.append(row)

    if not rows:
        return
    ncols = 1 + len(models) * 2
    nrows = len(rows)
    plt.figure(figsize=(4.0 * ncols, 3.2 * nrows))
    for r, row in enumerate(rows):
        ax = plt.subplot(nrows, ncols, r * ncols + 1)
        ax.imshow(row["input"])
        ax.axis("off")
        if r == 0:
            ax.set_title("Input", fontsize=10)
        ax.text(-0.04, 0.5, row["condition"], transform=ax.transAxes, rotation=90, va="center", ha="right", fontsize=10)
        col = 2
        for item in row["items"]:
            ax = plt.subplot(nrows, ncols, r * ncols + col)
            ax.imshow(item["heat"], cmap="jet")
            ax.axis("off")
            if r == 0:
                ax.set_title(f"{item['name']}\nFeature", fontsize=10)
            col += 1
            ax = plt.subplot(nrows, ncols, r * ncols + col)
            ax.imshow(item["overlay"])
            ax.axis("off")
            if r == 0:
                ax.set_title(f"{item['name']}\nOverlay", fontsize=10)
            col += 1
    plt.suptitle(f"Feature-map visualization | {image_name} | {args.target_layer}", fontsize=12)
    plt.tight_layout()
    ensure_dir(args.out_dir)
    safe = normalize_name(image_name)
    out_png = Path(args.out_dir) / f"featuremap_grid_{safe}_{args.target_layer}.png"
    out_pdf = Path(args.out_dir) / f"featuremap_grid_{safe}_{args.target_layer}.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out_png}")
    print(f"[OK] {out_pdf}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clean-root", default="mmdet_dataset/lettuce")
    p.add_argument("--benchmark-root", default="mmdet_dataset/lettuce_c")
    p.add_argument("--benchmark-manifest", default="mmdet_dataset/lettuce_c/manifest.json")
    p.add_argument("--out-dir", default="paper_outputs_dgcf_simple/featuremaps_3corr")
    p.add_argument("--image-names", nargs="+", default=["0003_000050.png", "0007_000050.png", "0011_000050.png"])
    p.add_argument("--conditions", nargs="+", default=["clean", "brightness:3", "contrast:3", "gaussian_noise:3"])
    p.add_argument("--target-layer", default="neck_p2", choices=["backbone_layer4", "neck_p0", "neck_p1", "neck_p2", "neck_p3"])
    p.add_argument("--heatmap-mode", default="mean_abs", choices=["mean_abs", "mean", "max"])
    p.add_argument("--alpha", type=float, default=0.45)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["PYTHONPATH"] = str(Path.cwd()) + os.pathsep + os.environ.get("PYTHONPATH", "")
    ensure_dir(args.out_dir)
    models = load_models(args)
    if not models:
        raise RuntimeError("No models loaded.")
    print("[INFO] target layer:", args.target_layer)
    print("[INFO] conditions:", args.conditions)
    print("[INFO] images:", args.image_names)
    for image_name in args.image_names:
        make_feature_grid_for_image(args, image_name, models)
    print("[DONE]")
    print(f"Saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
