#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_featuremap_heatmaps.py

Export paper-ready feature-map heatmaps and prediction visualizations from an
MMDetection model. Works for Mask R-CNN, PointRend, and RUQ-Mask R-CNN.

Outputs per image:
- prediction overlay: boxes/masks/scores
- FPN/neck feature heatmaps P2-P5/P6 if available
- backbone stage heatmaps if requested
- RUQ raw ROI uncertainty/quality summaries if the mask head returns them

Run inside the MMDetection environment:
python projects/ruq/tools/export_featuremap_heatmaps.py \
  --config configs/ruq/ruq_mask_rcnn_r50_fpn_1x_lettuce.py \
  --checkpoint work_dirs/ruq_mask_rcnn_r50_fpn_1x_lettuce/best_coco_segm_mAP_epoch_50.pth \
  --input data/lettuce_robust/images/clean \
  --out-dir paper_assets/featuremaps \
  --score-thr 0.3 \
  --max-images 8 \
  --layers neck backbone roi_mask_head
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize01(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = np.nanmin(x), np.nanmax(x)
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn + eps)


def tensor_to_heatmap(t: torch.Tensor) -> np.ndarray:
    """Convert a feature tensor [N,C,H,W] or [C,H,W] to 2D normalized heatmap."""
    if isinstance(t, (list, tuple)):
        t = t[0]
    if not torch.is_tensor(t):
        raise TypeError(f"Expected torch tensor, got {type(t)}")
    with torch.no_grad():
        t = t.detach()
        if t.ndim == 4:
            t = t[0]
        if t.ndim == 3:
            # Mean absolute response over channels gives stable visualization.
            h = t.abs().mean(dim=0)
        elif t.ndim == 2:
            h = t.abs()
        else:
            h = t.reshape(-1).abs()
        h = h.float().cpu().numpy()
    return normalize01(h)


def colorize_heatmap(h: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize heatmap to image size and apply OpenCV color map."""
    w, h_img = size
    hm = cv2.resize(h, (w, h_img), interpolation=cv2.INTER_LINEAR)
    hm_u8 = np.uint8(np.clip(hm * 255, 0, 255))
    colored = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return colored


def overlay_heatmap(image_rgb: np.ndarray, h: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    colored = colorize_heatmap(h, (image_rgb.shape[1], image_rgb.shape[0]))
    out = (image_rgb.astype(np.float32) * (1 - alpha) + colored.astype(np.float32) * alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def list_images(input_path: Path, max_images: int = 0) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() == ".txt":
            paths = [Path(x.strip()) for x in input_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        else:
            paths = [input_path]
    else:
        paths = sorted([p for p in input_path.rglob("*") if p.suffix.lower() in IMG_EXTS])
    if max_images and max_images > 0:
        paths = paths[:max_images]
    return paths


def get_instances(result: Any):
    if hasattr(result, "pred_instances"):
        return result.pred_instances
    return None


def safe_tensor_to_numpy(x: Any) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def draw_predictions(image_path: Path, result: Any, out_path: Path, score_thr: float = 0.3, max_instances: int = 50) -> None:
    """Simple prediction visualization independent of MMDet Visualizer."""
    img = Image.open(image_path).convert("RGB")
    rgba = img.convert("RGBA")
    overlay = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    inst = get_instances(result)
    if inst is None or len(inst) == 0:
        img.save(out_path)
        return
    scores = safe_tensor_to_numpy(inst.scores) if hasattr(inst, "scores") else np.ones(len(inst))
    bboxes = safe_tensor_to_numpy(inst.bboxes) if hasattr(inst, "bboxes") else None
    masks = None
    if hasattr(inst, "masks"):
        masks = safe_tensor_to_numpy(inst.masks)
    order = np.argsort(-scores)[:max_instances]
    for idx in order:
        score = float(scores[idx])
        if score < score_thr:
            continue
        if masks is not None and idx < len(masks):
            m = masks[idx]
            if m.ndim == 3:
                m = m[0]
            m = (m > 0.5).astype(np.uint8)
            mask_img = Image.fromarray(m * 120, mode="L").resize(rgba.size)
            color = Image.new("RGBA", rgba.size, (0, 220, 0, 110))
            overlay = Image.composite(color, overlay, mask_img)
        if bboxes is not None:
            x1, y1, x2, y2 = [float(v) for v in bboxes[idx]]
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0, 230), width=3)
            draw.text((x1, max(0, y1 - 14)), f"{score:.2f}", fill=(255, 0, 0, 255))
    out = Image.alpha_composite(rgba, overlay).convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)


class ActivationCollector:
    def __init__(self, model, layer_names: List[str]):
        self.model = model
        self.layer_names = layer_names
        self.handles = []
        self.outputs: Dict[str, Any] = {}

    def _hook(self, name: str):
        def fn(module, inputs, output):
            self.outputs[name] = output
        return fn

    def _get_module(self, name: str):
        # Friendly aliases for common MMDet components.
        aliases = {
            "backbone": getattr(self.model, "backbone", None),
            "neck": getattr(self.model, "neck", None),
            "roi_head": getattr(self.model, "roi_head", None),
            "roi_mask_head": getattr(getattr(self.model, "roi_head", None), "mask_head", None),
            "mask_head": getattr(getattr(self.model, "roi_head", None), "mask_head", None),
        }
        if name in aliases:
            return aliases[name]
        # Dot-path fallback, e.g. backbone.layer4 or neck.lateral_convs.0
        obj = self.model
        for part in name.split("."):
            if part.isdigit():
                obj = obj[int(part)]
            else:
                obj = getattr(obj, part)
        return obj

    def register(self):
        for name in self.layer_names:
            module = self._get_module(name)
            if module is None:
                print(f"[WARN] Layer alias not found: {name}")
                continue
            self.handles.append(module.register_forward_hook(self._hook(name)))
            print(f"[INFO] Hooked layer: {name}")

    def clear(self):
        self.outputs.clear()

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def export_feature_outputs(image_rgb: np.ndarray, outputs: Dict[str, Any], out_base: Path, alpha: float = 0.45, max_levels: int = 5) -> Dict[str, Any]:
    """Export heatmaps for hooked layer outputs."""
    summary: Dict[str, Any] = {}
    for name, out in outputs.items():
        # RUQ mask head may return dict: mask_preds, quality_preds, uncertainty_preds.
        if isinstance(out, dict):
            if "uncertainty_preds" in out and torch.is_tensor(out["uncertainty_preds"]):
                u = out["uncertainty_preds"]
                # Save first several ROI uncertainty maps as a grid image.
                n = min(16, u.shape[0]) if u.ndim >= 4 else 1
                maps = []
                for i in range(n):
                    ui = u[i, 0].detach().sigmoid().cpu().numpy() if u.ndim == 4 else u.detach().sigmoid().cpu().numpy()
                    maps.append(normalize01(ui))
                if maps:
                    grid_cols = min(4, len(maps))
                    grid_rows = int(np.ceil(len(maps) / grid_cols))
                    fig_w, fig_h = grid_cols * 2.2, grid_rows * 2.2
                    import matplotlib.pyplot as plt
                    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(fig_w, fig_h))
                    axes = np.array(axes).reshape(-1)
                    for ax, m in zip(axes, maps):
                        ax.imshow(m, cmap="magma")
                        ax.axis("off")
                    for ax in axes[len(maps):]:
                        ax.axis("off")
                    fig.suptitle("RUQ ROI uncertainty maps", fontsize=12)
                    fig.tight_layout()
                    p = out_base.parent / f"{out_base.name}_{name}_uncertainty_roi_grid.png"
                    fig.savefig(p, dpi=300, bbox_inches="tight")
                    plt.close(fig)
                    summary[f"{name}_uncertainty_grid"] = str(p)
            if "quality_preds" in out and torch.is_tensor(out["quality_preds"]):
                q = out["quality_preds"].detach().cpu().float().numpy().reshape(-1)
                summary[f"{name}_quality_mean"] = float(q.mean()) if q.size else None
                summary[f"{name}_quality_min"] = float(q.min()) if q.size else None
                summary[f"{name}_quality_max"] = float(q.max()) if q.size else None
            # Continue and try heatmap from mask_preds if available.
            out = out.get("mask_preds", None)
            if out is None:
                continue
        if isinstance(out, (tuple, list)):
            for level, tensor in enumerate(out[:max_levels]):
                if not torch.is_tensor(tensor):
                    continue
                h = tensor_to_heatmap(tensor)
                over = overlay_heatmap(image_rgb, h, alpha=alpha)
                p = out_base.parent / f"{out_base.name}_{name}_L{level}.png"
                Image.fromarray(over).save(p)
                summary[f"{name}_L{level}"] = str(p)
        elif torch.is_tensor(out):
            h = tensor_to_heatmap(out)
            over = overlay_heatmap(image_rgb, h, alpha=alpha)
            p = out_base.parent / f"{out_base.name}_{name}.png"
            Image.fromarray(over).save(p)
            summary[name] = str(p)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Image file, image folder, or txt list")
    parser.add_argument("--out-dir", default="paper_assets/featuremaps")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-thr", type=float, default=0.3)
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument("--layers", nargs="+", default=["neck", "roi_mask_head"], help="Layer aliases or dot paths")
    parser.add_argument("--alpha", type=float, default=0.45)
    args = parser.parse_args()

    # Import inside main so the file can be inspected without MMDetection installed.
    from mmdet.apis import init_detector, inference_detector

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = init_detector(args.config, args.checkpoint, device=args.device)
    model.eval()

    collector = ActivationCollector(model, args.layers)
    collector.register()

    image_paths = list_images(Path(args.input), max_images=args.max_images)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.input}")

    all_summary = []
    for img_path in image_paths:
        print(f"[INFO] Inference: {img_path}")
        collector.clear()
        result = inference_detector(model, str(img_path))
        image_rgb = np.asarray(Image.open(img_path).convert("RGB"))
        stem = img_path.stem
        img_out_dir = out_dir / stem
        img_out_dir.mkdir(parents=True, exist_ok=True)
        pred_path = img_out_dir / f"{stem}_prediction.png"
        draw_predictions(img_path, result, pred_path, score_thr=args.score_thr)
        summary = {
            "image": str(img_path),
            "prediction_overlay": str(pred_path),
        }
        summary.update(export_feature_outputs(image_rgb, collector.outputs, img_out_dir / stem, alpha=args.alpha))
        all_summary.append(summary)

    collector.close()
    (out_dir / "featuremap_export_summary.json").write_text(json.dumps(all_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] Feature-map heatmaps exported to: {out_dir}")


if __name__ == "__main__":
    main()
