#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_prediction_and_activation_maps.py

Use this script to export:
- input image
- prediction image
- C2 C3 C4 C5 activation heatmaps/overlays
- P2-P6 activation heatmaps/overlays before DGCF
- P2'-P6' activation heatmaps/overlays after DGCF
- |P' - P| difference heatmaps/overlays

No text is inserted inside the images.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from mmengine.config import Config
from mmengine.dataset import Compose, pseudo_collate
from mmdet.apis import init_detector, inference_detector


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def normalize_01(x):
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    return np.clip((x - lo) / (hi - lo + 1e-6), 0.0, 1.0)


def activation_2d(feat, mode="mean_abs"):
    if feat.dim() == 4:
        feat = feat[0]
    feat = feat.detach().float()

    if mode == "mean_abs":
        amap = feat.abs().mean(dim=0)
    elif mode == "mean":
        amap = feat.mean(dim=0)
    elif mode == "max_abs":
        amap = feat.abs().max(dim=0)[0]
    elif mode == "l2":
        amap = torch.sqrt((feat ** 2).mean(dim=0) + 1e-6)
    else:
        raise ValueError(f"Unsupported activation mode: {mode}")

    return amap.cpu().numpy()


def make_heatmap(amap, image_hw):
    amap = normalize_01(amap)
    amap = cv2.resize(amap, (image_hw[1], image_hw[0]), interpolation=cv2.INTER_LINEAR)
    gray = (amap * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return heatmap, amap


def overlay_heatmap(img_bgr, heatmap_bgr, alpha):
    return cv2.addWeighted(img_bgr, 1.0 - alpha, heatmap_bgr, alpha, 0)


def save_activation(out_dir, name, feat, img_bgr, alpha=0.45, mode="mean_abs"):
    amap = activation_2d(feat, mode=mode)
    heatmap, amap_resized = make_heatmap(amap, img_bgr.shape[:2])
    overlay = overlay_heatmap(img_bgr, heatmap, alpha=alpha)

    np.save(str(out_dir / f"{name}_activation.npy"), amap_resized)
    cv2.imwrite(str(out_dir / f"{name}_activation_heatmap.png"), heatmap)
    cv2.imwrite(str(out_dir / f"{name}_activation_overlay.png"), overlay)


def resize_like(feat, ref):
    if feat.dim() == 3:
        feat = feat.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    if feat.shape[-2:] != ref.shape[-2:]:
        feat = torch.nn.functional.interpolate(
            feat, size=ref.shape[-2:], mode="bilinear", align_corners=False
        )

    return feat[0] if squeeze else feat


def save_difference(out_dir, name, feat_before, feat_after, img_bgr, alpha=0.45):
    if feat_before.dim() == 4:
        feat_before = feat_before[0]
    if feat_after.dim() == 4:
        feat_after = feat_after[0]

    feat_after = resize_like(feat_after, feat_before)
    diff = (feat_after.detach().float() - feat_before.detach().float()).abs()
    amap = diff.mean(dim=0).cpu().numpy()

    heatmap, amap_resized = make_heatmap(amap, img_bgr.shape[:2])
    overlay = overlay_heatmap(img_bgr, heatmap, alpha=alpha)

    np.save(str(out_dir / f"{name}_diff_activation.npy"), amap_resized)
    cv2.imwrite(str(out_dir / f"{name}_diff_heatmap.png"), heatmap)
    cv2.imwrite(str(out_dir / f"{name}_diff_overlay.png"), overlay)


def build_data_batch(cfg, image_path, model):
    pipeline = Compose(cfg.test_dataloader.dataset.pipeline)
    data = dict(img_path=str(image_path), img_id=0)
    data = pipeline(data)
    data_batch = pseudo_collate([data])
    data_batch = model.data_preprocessor(data_batch, False)
    return data_batch


def prediction_no_text(img_bgr, result, score_thr=0.3):
    out = img_bgr.copy()

    if not hasattr(result, "pred_instances"):
        return out

    pred = result.pred_instances
    if not hasattr(pred, "scores"):
        return out

    scores = pred.scores.detach().cpu().numpy()
    keep = scores >= score_thr

    if hasattr(pred, "masks"):
        masks = pred.masks.detach().cpu().numpy()
        for mask in masks[keep]:
            mask = mask.astype(bool)
            color = np.array([0, 255, 0], dtype=np.uint8)
            out[mask] = (0.55 * out[mask] + 0.45 * color).astype(np.uint8)

    if hasattr(pred, "bboxes"):
        bboxes = pred.bboxes.detach().cpu().numpy()
        for box in bboxes[keep]:
            x1, y1, x2, y2 = box.astype(int).tolist()
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return out


def make_grid(paths, out_path, cols=5, tile_w=360):
    imgs = []
    for path in paths:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        new_h = int(h * tile_w / max(w, 1))
        img = cv2.resize(img, (tile_w, new_h), interpolation=cv2.INTER_AREA)
        imgs.append(img)

    if not imgs:
        return

    max_h = max(img.shape[0] for img in imgs)
    padded = []
    for img in imgs:
        if img.shape[0] < max_h:
            pad = np.zeros((max_h - img.shape[0], img.shape[1], 3), dtype=np.uint8)
            img = np.vstack([img, pad])
        padded.append(img)

    rows = []
    for start in range(0, len(padded), cols):
        row = padded[start:start + cols]
        while len(row) < cols:
            row.append(np.zeros_like(padded[0]))
        rows.append(np.hstack(row))

    cv2.imwrite(str(out_path), np.vstack(rows))


def get_p_before_after(model, c_feats):
    p_after = model.neck(c_feats)

    try:
        from mmdet.models.necks.fpn import FPN
        p_before = FPN.forward(model.neck, c_feats)
    except Exception:
        p_before = p_after

    return p_before, p_after


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-thr", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--activation-mode", default="mean_abs",
                        choices=["mean_abs", "mean", "max_abs", "l2"])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    mkdir(out_dir)

    cfg = Config.fromfile(args.config)
    model = init_detector(cfg, args.checkpoint, device=args.device)
    model.eval()

    img_bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(args.image)

    cv2.imwrite(str(out_dir / "input.png"), img_bgr)

    result = inference_detector(model, str(args.image))
    pred_img = prediction_no_text(img_bgr, result, score_thr=args.score_thr)
    cv2.imwrite(str(out_dir / "prediction.png"), pred_img)

    data_batch = build_data_batch(cfg, Path(args.image), model)

    with torch.no_grad():
        inputs = data_batch["inputs"]
        if isinstance(inputs, list):
            inputs = torch.stack(inputs, dim=0)
        inputs = inputs.to(args.device)

        c_feats = model.backbone(inputs)
        p_before, p_after = get_p_before_after(model, c_feats)

    for name, feat in zip(["C2", "C3", "C4", "C5"], c_feats):
        save_activation(out_dir, name, feat, img_bgr, args.alpha, args.activation_mode)

    for name, feat in zip(["P2", "P3", "P4", "P5", "P6"], p_before):
        save_activation(out_dir, name, feat, img_bgr, args.alpha, args.activation_mode)

    for name, feat in zip(["P2_prime", "P3_prime", "P4_prime", "P5_prime", "P6_prime"], p_after):
        save_activation(out_dir, name, feat, img_bgr, args.alpha, args.activation_mode)

    for name, fb, fa in zip(["P2", "P3", "P4", "P5", "P6"], p_before, p_after):
        save_difference(out_dir, name, fb, fa, img_bgr, args.alpha)

    make_grid(
        [
            out_dir / "input.png",
            out_dir / "prediction.png",
            out_dir / "P3_activation_overlay.png",
            out_dir / "P3_prime_activation_overlay.png",
            out_dir / "P3_diff_overlay.png",
        ],
        out_dir / "grid_main_P3_comparison.png",
        cols=5,
    )

    make_grid(
        [
            out_dir / "C2_activation_overlay.png",
            out_dir / "C3_activation_overlay.png",
            out_dir / "C4_activation_overlay.png",
            out_dir / "C5_activation_overlay.png",
        ],
        out_dir / "grid_C2_C5_activation_overlay.png",
        cols=4,
    )

    make_grid(
        [
            out_dir / "P2_activation_overlay.png",
            out_dir / "P3_activation_overlay.png",
            out_dir / "P4_activation_overlay.png",
            out_dir / "P5_activation_overlay.png",
            out_dir / "P6_activation_overlay.png",
        ],
        out_dir / "grid_P_before_activation_overlay.png",
        cols=5,
    )

    make_grid(
        [
            out_dir / "P2_prime_activation_overlay.png",
            out_dir / "P3_prime_activation_overlay.png",
            out_dir / "P4_prime_activation_overlay.png",
            out_dir / "P5_prime_activation_overlay.png",
            out_dir / "P6_prime_activation_overlay.png",
        ],
        out_dir / "grid_P_after_activation_overlay.png",
        cols=5,
    )

    metadata = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "image": args.image,
        "score_thr": args.score_thr,
        "alpha": args.alpha,
        "activation_mode": args.activation_mode,
        "main_grid_order": [
            "input",
            "prediction",
            "P3 before DGCF overlay",
            "P3 after DGCF overlay",
            "|P3_after - P3_before| overlay",
        ],
        "terminology": {
            "feature_map": "internal tensor used by the model",
            "activation_heatmap": "2D visualization converted from a feature map",
            "overlay": "activation heatmap blended with the input image",
            "difference_map": "visualization of absolute difference between P' and P",
        },
    }
    (out_dir / "visualization_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"[OK] Saved to: {out_dir}")
    print(f"[OK] Main grid: {out_dir / 'grid_main_P3_comparison.png'}")


if __name__ == "__main__":
    main()
