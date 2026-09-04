#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize backbone feature maps C1-C5 and FPN feature maps P1-P5.

Designed for the lettuce robustness paper:
- Baseline: Mask R-CNN R50-FPN
- Proposed: Enhanced Mask R-CNN (DGCF)

Backbone mapping for ResNet in MMDetection:
- C1: output after stem conv/relu/maxpool
- C2: layer1
- C3: layer2
- C4: layer3
- C5: layer4

FPN mapping:
- P1/P2/P3/P4/P5 are mapped to FPN output list indices.
- In standard MMDetection FPN with ResNet, outputs usually correspond to strides [4, 8, 16, 32, 64].
- Therefore P1 here means the first FPN output, not necessarily the original FPN paper notation.
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt


# -------------------------
# Runtime compatibility
# -------------------------
def patch_torch_load():
    if getattr(torch.load, "_patched_weights_only_false", False):
        return

    old_load = torch.load

    def new_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return old_load(*args, **kwargs)

    new_load._patched_weights_only_false = True
    torch.load = new_load


def ensure_pafpn_stub():
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


# -------------------------
# Utilities
# -------------------------
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
    if isinstance(feat, (tuple, list)):
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


# -------------------------
# Dataset resolver
# -------------------------
def load_clean_coco(clean_root):
    coco_path = Path(clean_root) / "annotations" / "test.json"
    coco = read_json(coco_path)
    imgs = sorted(coco["images"], key=lambda x: x.get("file_name", ""))
    return coco, imgs


def find_image_index(clean_root, image_name):
    _, imgs = load_clean_coco(clean_root)
    target = normalize_name(image_name)

    for i, img in enumerate(imgs):
        if normalize_name(img.get("file_name", "")) == target:
            return i, img.get("file_name", "")

    print(f"[WARNING] image not found: {image_name}")
    print("[INFO] first 20 available files:")
    for img in imgs[:20]:
        print("  -", img.get("file_name", ""))
    raise RuntimeError(f"Image not found: {image_name}")


def get_condition_image_path(clean_root, benchmark_root, benchmark_manifest, image_name, condition):
    idx, clean_file = find_image_index(clean_root, image_name)

    if condition == "clean":
        return Path(clean_root) / "images" / "test" / clean_file

    corruption, sev = condition.split(":")
    sev = int(sev)

    man = read_json(benchmark_manifest)
    out_ann = man.get("output_annotation", "annotations/test_png.json")
    bench_coco = read_json(Path(benchmark_root) / out_ann)
    bench_imgs = sorted(bench_coco["images"], key=lambda x: x.get("file_name", ""))

    if idx >= len(bench_imgs):
        raise RuntimeError(f"Index {idx} is out of benchmark range.")

    bench_file = bench_imgs[idx].get("file_name", clean_file)

    for item in man.get("conditions", []):
        if item.get("corruption") == corruption and int(item.get("severity")) == sev:
            return Path(benchmark_root) / item["image_prefix"].rstrip("/") / bench_file

    raise RuntimeError(f"Condition not found: {condition}")


# -------------------------
# Model loading
# -------------------------
def load_detector_model(config, checkpoint, device):
    patch_torch_load()
    ensure_pafpn_stub()

    from mmdet.utils import register_all_modules
    from mmdet.apis import init_detector

    register_all_modules(init_default_scope=True)
    model = init_detector(str(config), str(checkpoint), device=device)
    model.eval()
    return model


def build_default_models():
    return [
        {
            "name": "Mask R-CNN R50",
            "safe_name": "mask_rcnn_r50",
            "config": "configs/fair_lettuce/mask_rcnn_r50_fpn.py",
            "work_dir": "work_dirs/research/mask_rcnn_r50_fpn/seed_2026",
        },
        {
            "name": "Enhanced Mask R-CNN (DGCF)",
            "safe_name": "dgcf",
            "config": "configs/fair_lettuce/mask_rcnn_r50_dgcf_fpn.py",
            "work_dir": "work_dirs/research/mask_rcnn_r50_dgcf_fpn/seed_2026",
        },
    ]


def load_models(args):
    loaded = []
    for info in build_default_models():
        cfg = Path(info["config"])
        ckpt = find_checkpoint(info["work_dir"])

        if not cfg.is_file():
            print(f"[SKIP] missing config: {cfg}")
            continue
        if ckpt is None:
            print(f"[SKIP] missing checkpoint: {info['work_dir']}")
            continue

        print(f"[LOAD] {info['name']}")
        print(f"       cfg : {cfg}")
        print(f"       ckpt: {ckpt}")
        loaded.append((info, load_detector_model(cfg, ckpt, args.device)))

    if not loaded:
        raise RuntimeError("No model loaded.")

    return loaded


# -------------------------
# Hooks and extraction
# -------------------------
class MultiFeatureCatcher:
    def __init__(self):
        self.features = {}
        self.handles = []

    def add(self, name, module):
        def _hook(module, inp, out):
            self.features[name] = out
        self.handles.append(module.register_forward_hook(_hook))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def register_c_features(model, catcher):
    """Register C1-C5 from ResNet backbone."""
    bb = model.backbone

    # C1 after maxpool/stem. Hook maxpool because it is normally after conv1/bn/relu.
    if hasattr(bb, "maxpool"):
        catcher.add("C1", bb.maxpool)

    if hasattr(bb, "layer1"):
        catcher.add("C2", bb.layer1)
    if hasattr(bb, "layer2"):
        catcher.add("C3", bb.layer2)
    if hasattr(bb, "layer3"):
        catcher.add("C4", bb.layer3)
    if hasattr(bb, "layer4"):
        catcher.add("C5", bb.layer4)


def register_p_features(model, catcher):
    """Register all P outputs from FPN neck."""
    if not hasattr(model, "neck"):
        raise RuntimeError("model.neck not found")
    catcher.add("FPN_OUT", model.neck)


def extract_all_features(model, image_path, layers, heatmap_mode="mean_abs"):
    """Return dict layer name -> 2D heatmap."""
    from mmdet.apis import inference_detector

    catcher = MultiFeatureCatcher()

    need_c = any(x.startswith("C") for x in layers)
    need_p = any(x.startswith("P") for x in layers)

    if need_c:
        register_c_features(model, catcher)
    if need_p:
        register_p_features(model, catcher)

    with torch.no_grad():
        _ = inference_detector(model, str(image_path))

    catcher.remove()

    out = {}

    # C maps.
    for c in ["C1", "C2", "C3", "C4", "C5"]:
        if c in layers and c in catcher.features:
            out[c] = feature_to_heatmap(catcher.features[c], mode=heatmap_mode)

    # P maps.
    if need_p and "FPN_OUT" in catcher.features:
        fpn_out = catcher.features["FPN_OUT"]
        if not isinstance(fpn_out, (list, tuple)):
            fpn_out = [fpn_out]

        # User-facing P1..P5 mapped to indices 0..4.
        for pidx in range(1, 6):
            lname = f"P{pidx}"
            if lname in layers:
                feat_idx = pidx - 1
                if feat_idx < len(fpn_out):
                    out[lname] = feature_to_heatmap(fpn_out[feat_idx], mode=heatmap_mode)
                else:
                    print(f"[WARN] {lname} requested but FPN has only {len(fpn_out)} outputs.")

    return out


# -------------------------
# Figure creation
# -------------------------
def make_layer_grid(args, image_name, condition, models):
    img_path = get_condition_image_path(
        args.clean_root,
        args.benchmark_root,
        args.benchmark_manifest,
        image_name,
        condition,
    )
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Cannot read image: {img_path}")

    # Extract feature maps.
    all_model_features = []
    for info, model in models:
        feats = extract_all_features(
            model,
            img_path,
            layers=args.layers,
            heatmap_mode=args.heatmap_mode,
        )
        all_model_features.append((info, feats))

    # Layout:
    # rows = layers
    # cols = Input + for each model [heatmap, overlay]
    nrows = len(args.layers)
    ncols = 1 + len(models) * 2

    plt.figure(figsize=(4.0 * ncols, 3.2 * nrows))

    for r, layer in enumerate(args.layers):
        # Input only on first model-block row, but repeated per layer for readability.
        ax = plt.subplot(nrows, ncols, r * ncols + 1)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis("off")
        if r == 0:
            ax.set_title("Input", fontsize=10)
        ax.text(
            -0.04, 0.5, layer,
            transform=ax.transAxes,
            rotation=90,
            va="center",
            ha="right",
            fontsize=11,
            fontweight="bold",
        )

        col = 2
        for info, feats in all_model_features:
            if layer not in feats:
                blank = np.zeros(img.shape[:2], dtype=np.float32)
                heat = blank
            else:
                heat = feats[layer]

            overlay = overlay_heatmap(img, heat, alpha=args.alpha)

            ax = plt.subplot(nrows, ncols, r * ncols + col)
            ax.imshow(heat, cmap="jet")
            ax.axis("off")
            if r == 0:
                ax.set_title(f"{info['name']}\n{layer} heatmap", fontsize=10)
            col += 1

            ax = plt.subplot(nrows, ncols, r * ncols + col)
            ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            ax.axis("off")
            if r == 0:
                ax.set_title(f"{info['name']}\n{layer} overlay", fontsize=10)
            col += 1

    title = f"C/P feature maps | {image_name} | {condition}"
    plt.suptitle(title, fontsize=13)
    plt.tight_layout()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    safe_img = normalize_name(image_name)
    safe_cond = condition.replace(":", "_")
    layer_tag = "_".join(args.layers)
    out_png = out_dir / f"featuremap_{safe_img}_{safe_cond}_{layer_tag}.png"
    out_pdf = out_dir / f"featuremap_{safe_img}_{safe_cond}_{layer_tag}.pdf"

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
    p.add_argument("--out-dir", default="paper_outputs_dgcf_simple/featuremaps_c1c5_p1p5")
    p.add_argument("--image-names", nargs="+", default=["0003_000050.png"])
    p.add_argument("--conditions", nargs="+", default=["clean", "brightness:3", "contrast:3", "gaussian_noise:3"])
    p.add_argument("--layers", nargs="+", default=["C1", "C2", "C3", "C4", "C5", "P1", "P2", "P3", "P4", "P5"],
                   help="Choose from C1 C2 C3 C4 C5 P1 P2 P3 P4 P5")
    p.add_argument("--heatmap-mode", default="mean_abs", choices=["mean_abs", "mean", "max"])
    p.add_argument("--alpha", type=float, default=0.45)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def main():
    args = parse_args()

    os.environ["PYTHONPATH"] = str(Path.cwd()) + os.pathsep + os.environ.get("PYTHONPATH", "")
    ensure_dir(args.out_dir)

    # Normalize layer names.
    args.layers = [x.upper() for x in args.layers]

    valid = {"C1", "C2", "C3", "C4", "C5", "P1", "P2", "P3", "P4", "P5"}
    for x in args.layers:
        if x not in valid:
            raise RuntimeError(f"Invalid layer {x}. Valid choices: {sorted(valid)}")

    print("[INFO] layers:", args.layers)
    print("[INFO] conditions:", args.conditions)
    print("[INFO] images:", args.image_names)

    models = load_models(args)

    for image_name in args.image_names:
        for condition in args.conditions:
            make_layer_grid(args, image_name, condition, models)

    print("[DONE]")
    print(f"Saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
