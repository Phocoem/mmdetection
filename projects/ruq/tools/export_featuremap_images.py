#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_featuremap_images.py

Export image-level prediction visualizations and CNN/FPN feature-map heatmaps
for MMDetection/MMEngine models.

Designed for Mask R-CNN / PointRend / RUQ-Mask R-CNN experiments.
It saves, for each input image:
  - original image
  - prediction overlay
  - backbone feature-map heatmaps and overlays
  - neck/FPN feature-map heatmaps and overlays
  - optional contact sheet

Example:
python projects/ruq/tools/export_featuremap_images.py \
  --config configs/ruq/ruq_mask_rcnn_r50_fpn_1x_lettuce.py \
  --checkpoint work_dirs/ruq_mask_rcnn_r50_fpn_1x_lettuce/best_coco_segm_mAP_epoch_50.pth \
  --input data/lettuce_robust/images/clean \
  --out-dir paper_assets/featuremaps/ruq_clean \
  --layers neck \
  --max-images 8 \
  --score-thr 0.3
"""

import argparse
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

try:
    from mmdet.apis import init_detector, inference_detector
except Exception as exc:  # pragma: no cover
    print("[ERROR] Cannot import MMDetection. Run this inside your mmdetection environment.", file=sys.stderr)
    print(str(exc), file=sys.stderr)
    raise

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_stem(path: Path) -> str:
    stem = path.stem
    stem = re.sub(r"[^a-zA-Z0-9_.-]+", "_", stem)
    return stem[:120]


def collect_images(input_path: Path, max_images: int = 0, shuffle: bool = False, seed: int = 0) -> List[Path]:
    if input_path.is_file():
        paths = [input_path]
    else:
        paths = sorted([p for p in input_path.rglob("*") if p.suffix.lower() in IMG_EXTS])
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(paths)
    if max_images and max_images > 0:
        paths = paths[:max_images]
    return paths


def read_image_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def normalize_01(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = arr.astype(np.float32)
    arr = arr - float(np.nanmin(arr))
    denom = float(np.nanmax(arr)) + eps
    return arr / denom


def heatmap_to_color(hm01: np.ndarray) -> np.ndarray:
    hm_uint8 = np.clip(hm01 * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)


def overlay_heatmap(img_bgr: np.ndarray, hm01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    hm_color = heatmap_to_color(hm01)
    if hm_color.shape[:2] != img_bgr.shape[:2]:
        hm_color = cv2.resize(hm_color, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    out = cv2.addWeighted(img_bgr, 1 - alpha, hm_color, alpha, 0)
    return out


def tensor_to_heatmap(feat: torch.Tensor, img_hw: Tuple[int, int], channel_reduce: str = "absmean") -> np.ndarray:
    """Convert feature tensor [B,C,H,W] or [C,H,W] to image-sized heatmap [H,W] in [0,1]."""
    with torch.no_grad():
        if feat.dim() == 4:
            feat = feat[0]
        if feat.dim() != 3:
            raise ValueError(f"Expected [C,H,W] feature tensor, got shape={tuple(feat.shape)}")
        x = feat.detach().float().cpu()
        if channel_reduce == "mean":
            x = x.mean(dim=0)
        elif channel_reduce == "max":
            x = x.max(dim=0).values
        elif channel_reduce == "l2":
            x = torch.sqrt((x * x).mean(dim=0) + 1e-8)
        else:
            x = x.abs().mean(dim=0)
        hm = x.numpy()
    hm = normalize_01(hm)
    img_h, img_w = img_hw
    hm = cv2.resize(hm, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
    return normalize_01(hm)


def unpack_features(obj: Any) -> List[torch.Tensor]:
    """Return a list of feature tensors from a hook output."""
    if obj is None:
        return []
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (list, tuple)):
        out: List[torch.Tensor] = []
        for item in obj:
            out.extend(unpack_features(item))
        return out
    if isinstance(obj, dict):
        out = []
        for _, item in obj.items():
            out.extend(unpack_features(item))
        return out
    return []


class FeatureCatcher:
    def __init__(self, model: Any, layer_names: Sequence[str]) -> None:
        self.model = model
        self.layer_names = list(layer_names)
        self.storage: Dict[str, Any] = {}
        self.handles = []

    def _hook(self, name: str):
        def fn(module, inputs, output):
            self.storage[name] = output
        return fn

    def register(self) -> None:
        for name in self.layer_names:
            module = getattr(self.model, name, None)
            if module is None:
                print(f"[WARN] Model has no layer '{name}', skip hook.")
                continue
            self.handles.append(module.register_forward_hook(self._hook(name)))
            print(f"[INFO] Hook registered: model.{name}")

    def clear(self) -> None:
        self.storage.clear()

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()


def masks_to_numpy(masks: Any) -> Optional[np.ndarray]:
    if masks is None:
        return None
    if isinstance(masks, torch.Tensor):
        return masks.detach().cpu().numpy()
    # BitmapMasks/PolygonMasks-like objects
    if hasattr(masks, "to_ndarray"):
        try:
            return masks.to_ndarray()
        except Exception:
            pass
    try:
        return np.asarray(masks)
    except Exception:
        return None


def get_pred_instances(result: Any) -> Any:
    if hasattr(result, "pred_instances"):
        return result.pred_instances
    if isinstance(result, dict) and "pred_instances" in result:
        return result["pred_instances"]
    return None


def draw_predictions(img_bgr: np.ndarray, result: Any, score_thr: float = 0.3) -> np.ndarray:
    """Lightweight prediction renderer independent of DetLocalVisualizer."""
    out = img_bgr.copy()
    pred = get_pred_instances(result)
    if pred is None:
        return out
    try:
        scores = pred.scores.detach().cpu().numpy() if hasattr(pred, "scores") else np.array([])
        bboxes = pred.bboxes.detach().cpu().numpy() if hasattr(pred, "bboxes") else np.empty((0, 4))
        labels = pred.labels.detach().cpu().numpy() if hasattr(pred, "labels") else np.zeros(len(scores), dtype=np.int64)
        masks = masks_to_numpy(pred.masks if hasattr(pred, "masks") else None)
    except Exception:
        return out

    if scores is None or len(scores) == 0:
        return out
    keep = np.where(scores >= score_thr)[0]
    # Stable palette, enough for paper visualization. Not model-critical.
    palette = [
        (0, 255, 0), (0, 200, 255), (255, 0, 255), (255, 180, 0),
        (0, 128, 255), (180, 0, 255), (255, 255, 0), (128, 255, 128),
    ]
    overlay = out.copy()
    for rank, idx in enumerate(keep):
        color = palette[int(labels[idx]) % len(palette)]
        if masks is not None and idx < len(masks):
            mask = masks[idx]
            if mask.dtype != np.bool_:
                mask = mask > 0.5
            if mask.shape[:2] != out.shape[:2]:
                mask = cv2.resize(mask.astype(np.uint8), (out.shape[1], out.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            overlay[mask] = color
            # contour for readable boundaries
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, color, 2)
        if idx < len(bboxes):
            x1, y1, x2, y2 = bboxes[idx].astype(int).tolist()
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            text = f"{scores[idx]:.2f}"
            cv2.putText(out, text, (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    out = cv2.addWeighted(out, 0.72, overlay, 0.28, 0)
    return out


def resize_to_height(img: np.ndarray, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == height:
        return img
    new_w = max(1, int(round(w * height / h)))
    return cv2.resize(img, (new_w, height), interpolation=cv2.INTER_AREA)


def make_contact_sheet(images: List[Tuple[str, np.ndarray]], out_path: Path, cell_h: int = 260) -> None:
    if not images:
        return
    prepared = []
    for title, img in images:
        im = resize_to_height(img, cell_h)
        top = np.full((38, im.shape[1], 3), 255, dtype=np.uint8)
        cv2.putText(top, title[:32], (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
        prepared.append(np.vstack([top, im]))
    max_h = max(im.shape[0] for im in prepared)
    padded = []
    for im in prepared:
        if im.shape[0] < max_h:
            pad = np.full((max_h - im.shape[0], im.shape[1], 3), 255, dtype=np.uint8)
            im = np.vstack([im, pad])
        padded.append(im)
    sheet = np.hstack(padded)
    cv2.imwrite(str(out_path), sheet)


def export_one_image(
    model: Any,
    catcher: FeatureCatcher,
    img_path: Path,
    out_dir: Path,
    layer_names: Sequence[str],
    levels: Sequence[int],
    score_thr: float,
    alpha: float,
    channel_reduce: str,
    make_sheet: bool,
) -> None:
    img_bgr = read_image_bgr(img_path)
    img_h, img_w = img_bgr.shape[:2]
    stem = safe_stem(img_path)
    img_out = out_dir / stem
    ensure_dir(img_out)

    cv2.imwrite(str(img_out / "00_original.png"), img_bgr)
    catcher.clear()
    result = inference_detector(model, str(img_path))
    pred_img = draw_predictions(img_bgr, result, score_thr=score_thr)
    cv2.imwrite(str(img_out / "01_prediction.png"), pred_img)

    sheet_items: List[Tuple[str, np.ndarray]] = [("Original", img_bgr), ("Prediction", pred_img)]

    for layer in layer_names:
        feats = unpack_features(catcher.storage.get(layer))
        if not feats:
            continue
        for li, feat in enumerate(feats):
            if levels and li not in levels:
                continue
            try:
                hm = tensor_to_heatmap(feat, (img_h, img_w), channel_reduce=channel_reduce)
            except Exception as exc:
                print(f"[WARN] Skip {stem} {layer} L{li}: {exc}")
                continue
            color = heatmap_to_color(hm)
            overlay = overlay_heatmap(img_bgr, hm, alpha=alpha)
            cv2.imwrite(str(img_out / f"{layer}_L{li}_heatmap.png"), color)
            cv2.imwrite(str(img_out / f"{layer}_L{li}_overlay.png"), overlay)
            if len(sheet_items) < 8:
                sheet_items.append((f"{layer} L{li}", overlay))

    if make_sheet:
        make_contact_sheet(sheet_items, img_out / "contact_sheet.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export feature-map heatmaps from MMDetection models.")
    parser.add_argument("--config", required=True, help="MMDetection config file")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    parser.add_argument("--input", required=True, help="Image file or image directory")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--device", default="cuda:0", help="cuda:0 or cpu")
    parser.add_argument("--layers", nargs="+", default=["neck"], choices=["backbone", "neck"], help="Layers to hook")
    parser.add_argument("--levels", nargs="*", type=int, default=[], help="Feature levels to save. Empty = all levels")
    parser.add_argument("--score-thr", type=float, default=0.3)
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.45, help="Heatmap overlay alpha")
    parser.add_argument("--channel-reduce", default="absmean", choices=["absmean", "mean", "max", "l2"])
    parser.add_argument("--no-contact-sheet", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    images = collect_images(input_path, max_images=args.max_images, shuffle=args.shuffle, seed=args.seed)
    if not images:
        raise FileNotFoundError(f"No images found in {input_path}")

    print(f"[INFO] Init model: {args.config}")
    model = init_detector(args.config, args.checkpoint, device=args.device)
    model.eval()

    catcher = FeatureCatcher(model, args.layers)
    catcher.register()
    try:
        for idx, img_path in enumerate(images, 1):
            print(f"[{idx}/{len(images)}] {img_path}")
            export_one_image(
                model=model,
                catcher=catcher,
                img_path=img_path,
                out_dir=out_dir,
                layer_names=args.layers,
                levels=args.levels,
                score_thr=args.score_thr,
                alpha=args.alpha,
                channel_reduce=args.channel_reduce,
                make_sheet=not args.no_contact_sheet,
            )
    finally:
        catcher.remove()

    print(f"[OK] Saved feature maps to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
