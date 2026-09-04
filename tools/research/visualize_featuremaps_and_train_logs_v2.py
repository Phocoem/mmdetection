#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_featuremaps_and_train_logs_v2.py

FIXED for PyTorch >= 2.6:
- Avoids init_detector checkpoint loading problem when torch.load defaults to weights_only=True.
- Loads MMDetection checkpoint manually with weights_only=False.
- Adds safe globals fallback.
- Generates feature maps for clean + robustness/corruption sets.
- Each condition takes one image.
- Parses train_console.log / txt logs and plots training curves.

Run from MMDetection root:
    cd /home/pc/mmdet_AI/mmdetection
    conda activate mmdet
    export PYTHONPATH=/home/pc/mmdet_AI/mmdetection:$PYTHONPATH

Example:
    python tools/research/visualize_featuremaps_and_train_logs_v2.py \
      --config configs/fair_lettuce/mask_rcnn_r50_aspp_boundary_fpn.py \
      --checkpoint work_dirs/research/mask_rcnn_r50_aspp_boundary_fpn/seed_2026/best_coco_segm_mAP_epoch_65.pth \
      --clean-root mmdet_dataset/lettuce \
      --benchmark-root mmdet_dataset/lettuce_c \
      --manifest mmdet_dataset/lettuce_c/manifest.json \
      --out-dir work_dirs/research/featuremap_report/aspp_boundary \
      --layer neck \
      --levels 0 1 2 3 \
      --only brightness contrast gaussian_noise defocus_blur motion_blur \
      --log-file work_dirs/research/mask_rcnn_r50_aspp_boundary_fpn/seed_2026/train_console.log
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# PyTorch 2.6+ checkpoint compatibility
# ----------------------------------------------------------------------

def patch_torch_load_for_mmengine() -> None:
    """Patch torch.load so MMEngine checkpoints can load under PyTorch >= 2.6.

    PyTorch 2.6 changed torch.load default weights_only from False to True.
    MMEngine checkpoints may contain HistoryBuffer and metadata objects.
    This patch forces weights_only=False only when caller did not explicitly set it.
    Use only for checkpoints that you trust, e.g., checkpoints you trained yourself.
    """
    if getattr(torch.load, "_mmdet_weights_only_patched", False):
        return

    old_torch_load = torch.load

    def patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return old_torch_load(*args, **kwargs)

    patched_load._mmdet_weights_only_patched = True
    torch.load = patched_load

    try:
        from mmengine.logging.history_buffer import HistoryBuffer
        torch.serialization.add_safe_globals([HistoryBuffer])
    except Exception:
        pass

    try:
        from mmengine.config.config import ConfigDict
        torch.serialization.add_safe_globals([ConfigDict])
    except Exception:
        pass


# ----------------------------------------------------------------------
# Basic utils
# ----------------------------------------------------------------------

def mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))


def read_image_rgb(path: Path) -> np.ndarray:
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def save_rgb(path: Path, img_rgb: np.ndarray) -> None:
    mkdir(path.parent)
    cv2.imwrite(str(path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))


def normalize_map(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - x.min()
    mx = x.max()
    if mx < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return x / mx


def save_heatmap(path: Path, fmap: np.ndarray, title: Optional[str] = None) -> None:
    mkdir(path.parent)
    fmap = normalize_map(fmap)
    plt.figure(figsize=(5, 5))
    plt.imshow(fmap)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_overlay(path: Path, img_rgb: np.ndarray, fmap: np.ndarray, alpha: float = 0.45) -> None:
    mkdir(path.parent)
    h, w = img_rgb.shape[:2]
    fmap = normalize_map(fmap)
    fmap = cv2.resize(fmap, (w, h), interpolation=cv2.INTER_LINEAR)
    heat = np.uint8(255 * fmap)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    overlay = np.uint8((1 - alpha) * img_rgb + alpha * heat)
    save_rgb(path, overlay)


# ----------------------------------------------------------------------
# Conditions
# ----------------------------------------------------------------------

def get_image_from_coco(data_root: Path, ann_file: str, image_prefix: str, image_index: int) -> Path:
    ann_path = data_root / ann_file
    if not ann_path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")

    ann = read_json(ann_path)
    images = sorted(ann.get("images", []), key=lambda x: x.get("id", 0))
    if not images:
        raise RuntimeError(f"No images found in {ann_path}")

    idx = max(0, min(image_index, len(images) - 1))
    file_name = images[idx]["file_name"]
    image_path = data_root / image_prefix / file_name

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    return image_path


def build_conditions(args) -> List[Dict[str, Any]]:
    clean_root = Path(args.clean_root).resolve()
    benchmark_root = Path(args.benchmark_root).resolve()
    conditions: List[Dict[str, Any]] = []

    clean_image = get_image_from_coco(
        clean_root,
        "annotations/test.json",
        "images/test",
        args.image_index,
    )
    conditions.append({
        "name": "clean",
        "corruption": "clean",
        "severity": "",
        "image_path": clean_image,
    })

    manifest_path = Path(args.manifest).resolve() if args.manifest else None
    if manifest_path and manifest_path.is_file():
        manifest = read_json(manifest_path)
        output_ann = manifest.get("output_annotation", "annotations/test_png.json")

        for item in manifest.get("conditions", []):
            corruption = item["corruption"]
            severity = int(item["severity"])

            if args.only and corruption not in args.only:
                continue

            if args.severities and severity not in args.severities:
                continue

            image_prefix = item["image_prefix"].rstrip("/")
            image_path = get_image_from_coco(
                benchmark_root,
                output_ann,
                image_prefix,
                args.image_index,
            )

            conditions.append({
                "name": f"{corruption}_s{severity}",
                "corruption": corruption,
                "severity": severity,
                "image_path": image_path,
            })

    return conditions


# ----------------------------------------------------------------------
# Model loading and feature hook
# ----------------------------------------------------------------------

def load_model(config_path: str, checkpoint_path: str, device: str):
    patch_torch_load_for_mmengine()

    from mmengine.config import Config
    from mmdet.registry import MODELS
    from mmengine.runner import load_checkpoint
    from mmdet.utils import register_all_modules
    register_all_modules(init_default_scope=True)

    cfg = Config.fromfile(config_path)

    # Do not load pretrained backbone when loading own checkpoint.
    if "model" in cfg and "backbone" in cfg.model and "init_cfg" in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None

    model = MODELS.build(cfg.model)
    checkpoint = load_checkpoint(
        model,
        checkpoint_path,
        map_location="cpu",
        strict=False,
        revise_keys=[(r"^module\.", "")],
    )

    # Set dataset_meta if present.
    if isinstance(checkpoint, dict):
        meta = checkpoint.get("meta", {})
        if "dataset_meta" in meta:
            model.dataset_meta = meta["dataset_meta"]
        elif "CLASSES" in meta:
            model.dataset_meta = {"classes": meta["CLASSES"]}

    model.cfg = cfg
    model.to(device)
    model.eval()
    return model


class FeatureHook:
    def __init__(self, module):
        self.features = None
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.features = output

    def close(self):
        self.handle.remove()


def get_hook_module(model, layer: str):
    if layer == "backbone":
        return model.backbone
    if layer == "neck":
        return model.neck
    if layer == "rpn_head":
        return model.rpn_head
    if layer == "roi_head":
        return model.roi_head
    raise ValueError(f"Unknown layer: {layer}")


def run_inference(model, image_path: Path) -> None:
    from mmdet.apis import inference_detector
    with torch.no_grad():
        _ = inference_detector(model, str(image_path))


def output_to_feature_maps(features: Any, levels: List[int]) -> List[Tuple[int, np.ndarray]]:
    if isinstance(features, torch.Tensor):
        feat_list = [features]
    elif isinstance(features, (list, tuple)):
        feat_list = list(features)
    elif isinstance(features, dict):
        feat_list = list(features.values())
    else:
        raise TypeError(f"Unsupported feature output type: {type(features)}")

    maps: List[Tuple[int, np.ndarray]] = []

    for i, feat in enumerate(feat_list):
        if i not in levels:
            continue
        if not isinstance(feat, torch.Tensor):
            continue
        if feat.ndim != 4:
            continue

        # B,C,H,W -> H,W
        fmap = feat.detach().float().abs().mean(dim=1)[0].cpu().numpy()
        maps.append((i, fmap))

    return maps


def visualize_featuremaps(args) -> None:
    out_dir = Path(args.out_dir).resolve()
    mkdir(out_dir)

    conditions = build_conditions(args)
    print(f"[INFO] Number of conditions: {len(conditions)}")
    for c in conditions:
        print(f"  - {c['name']}: {c['image_path']}")

    print(f"[INFO] Loading model: {args.checkpoint}")
    model = load_model(args.config, args.checkpoint, args.device)

    module = get_hook_module(model, args.layer)
    hook = FeatureHook(module)

    rows = []

    for cond in conditions:
        cond_name = safe_name(cond["name"])
        img_path = Path(cond["image_path"])
        img_rgb = read_image_rgb(img_path)

        print(f"[FeatureMap] {cond['name']}")
        hook.features = None
        run_inference(model, img_path)

        if hook.features is None:
            print(f"[WARN] No feature captured: {cond['name']}")
            continue

        maps = output_to_feature_maps(hook.features, args.levels)

        if not maps:
            print(f"[WARN] No valid 4D feature map at selected levels for {cond['name']}")
            continue

        cond_dir = out_dir / "featuremaps" / cond_name
        mkdir(cond_dir)
        save_rgb(cond_dir / "input.png", img_rgb)

        for level, fmap in maps:
            heat_path = cond_dir / f"{args.layer}_level{level}_heatmap.png"
            overlay_path = cond_dir / f"{args.layer}_level{level}_overlay.png"

            save_heatmap(heat_path, fmap)
            save_overlay(overlay_path, img_rgb, fmap, alpha=args.overlay_alpha)

            rows.append({
                "condition": cond["name"],
                "corruption": cond["corruption"],
                "severity": cond["severity"],
                "image_path": str(img_path),
                "layer": args.layer,
                "level": level,
                "input": str(cond_dir / "input.png"),
                "heatmap": str(heat_path),
                "overlay": str(overlay_path),
            })

    hook.close()

    csv_path = out_dir / "featuremap_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "condition", "corruption", "severity", "image_path",
                "layer", "level", "input", "heatmap", "overlay"
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Featuremap CSV: {csv_path}")


# ----------------------------------------------------------------------
# Log parser
# ----------------------------------------------------------------------

def parse_float(s: str) -> float:
    return float(s.replace(",", ""))


def parse_log(log_file: Path) -> Dict[str, List[Tuple[float, float]]]:
    text = log_file.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    data = {
        "loss": [],
        "lr": [],
        "segm_mAP": [],
        "segm_mAP50": [],
        "segm_mAP75": [],
        "bbox_mAP": [],
        "bbox_mAP50": [],
        "bbox_mAP75": [],
    }

    last_x = 0.0

    for line_no, line in enumerate(lines, 1):
        # Train line: Epoch(train) [12][50/100]
        m_train = re.search(r"Epoch\(train\)\s+\[(\d+)\]\[(\d+)/(\d+)\]", line)
        if m_train:
            epoch = int(m_train.group(1))
            it = int(m_train.group(2))
            total = int(m_train.group(3))
            x = epoch + it / max(1, total)
            last_x = x
        else:
            x = last_x if last_x > 0 else float(line_no)

        if "Epoch(train)" in line or " loss:" in line:
            m_loss = re.search(r"(?:^|\s)loss:\s*([0-9.eE+-]+)", line)
            if m_loss:
                data["loss"].append((x, parse_float(m_loss.group(1))))

            m_lr = re.search(r"\blr:\s*([0-9.eE+-]+)", line)
            if m_lr:
                data["lr"].append((x, parse_float(m_lr.group(1))))

        # Val epoch can be extracted.
        m_val = re.search(r"Epoch\(val\)\s+\[(\d+)\]", line)
        val_x = float(m_val.group(1)) if m_val else x

        patterns = {
            "segm_mAP": r"coco/segm_mAP:\s*([0-9.eE+-]+)",
            "segm_mAP50": r"coco/segm_mAP_50:\s*([0-9.eE+-]+)",
            "segm_mAP75": r"coco/segm_mAP_75:\s*([0-9.eE+-]+)",
            "bbox_mAP": r"coco/bbox_mAP:\s*([0-9.eE+-]+)",
            "bbox_mAP50": r"coco/bbox_mAP_50:\s*([0-9.eE+-]+)",
            "bbox_mAP75": r"coco/bbox_mAP_75:\s*([0-9.eE+-]+)",
        }

        for key, pat in patterns.items():
            mm = re.search(pat, line)
            if mm:
                data[key].append((val_x, parse_float(mm.group(1))))

    return data


def plot_series(path: Path, points: List[Tuple[float, float]], title: str, ylabel: str) -> None:
    if not points:
        return
    mkdir(path.parent)
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    plt.figure(figsize=(8, 4.5))
    plt.plot(x, y, linewidth=1.6)
    plt.scatter(x, y, s=8)
    plt.title(title)
    plt.xlabel("Epoch / iteration progress")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_multi(path: Path, series: Dict[str, List[Tuple[float, float]]], title: str, ylabel: str) -> None:
    if not any(series.values()):
        return
    mkdir(path.parent)
    plt.figure(figsize=(8, 4.5))
    for label, points in series.items():
        if not points:
            continue
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        plt.plot(x, y, linewidth=1.6, marker="o", markersize=3, label=label)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def visualize_log(args) -> None:
    if not args.log_file:
        return

    log_path = Path(args.log_file).resolve()
    if not log_path.is_file():
        print(f"[WARN] Log not found: {log_path}")
        return

    print(f"[INFO] Parsing log: {log_path}")
    data = parse_log(log_path)

    out = Path(args.out_dir).resolve() / "training_curves"
    mkdir(out)

    plot_series(out / "train_loss.png", data["loss"], "Training Loss", "Loss")
    plot_series(out / "learning_rate.png", data["lr"], "Learning Rate", "LR")

    plot_multi(
        out / "validation_mask_ap.png",
        {
            "segm_mAP": data["segm_mAP"],
            "segm_mAP50": data["segm_mAP50"],
            "segm_mAP75": data["segm_mAP75"],
        },
        "Validation Mask AP",
        "AP",
    )

    plot_multi(
        out / "validation_bbox_ap.png",
        {
            "bbox_mAP": data["bbox_mAP"],
            "bbox_mAP50": data["bbox_mAP50"],
            "bbox_mAP75": data["bbox_mAP75"],
        },
        "Validation BBox AP",
        "AP",
    )

    json_path = out / "parsed_training_log.json"
    serial = {k: [{"x": x, "y": y} for x, y in v] for k, v in data.items()}
    json_path.write_text(json.dumps(serial, indent=2), encoding="utf-8")
    print(f"[OK] Training curves: {out}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--clean-root", default="mmdet_dataset/lettuce")
    parser.add_argument("--benchmark-root", default="mmdet_dataset/lettuce_c")
    parser.add_argument("--manifest", default="mmdet_dataset/lettuce_c/manifest.json")
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--layer", default="neck", choices=["backbone", "neck", "rpn_head", "roi_head"])
    parser.add_argument("--levels", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-index", type=int, default=0)
    parser.add_argument("--overlay-alpha", type=float, default=0.45)

    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--severities", nargs="+", type=int, default=None)

    parser.add_argument("--log-file", default=None)
    parser.add_argument("--skip-featuremaps", action="store_true")
    parser.add_argument("--skip-log", action="store_true")

    args = parser.parse_args()

    mkdir(Path(args.out_dir).resolve())

    if not args.skip_featuremaps:
        visualize_featuremaps(args)

    if not args.skip_log:
        visualize_log(args)

    print(f"[DONE] Output saved to: {Path(args.out_dir).resolve()}")


if __name__ == "__main__":
    main()
