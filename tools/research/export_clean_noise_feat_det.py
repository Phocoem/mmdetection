#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Export feature-map overlays and detection visualizations for clean/noise images."""

import argparse
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
import torch

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def safe_name(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', s).strip('_')


def parse_condition(cond: str):
    if cond == 'clean':
        return 'clean', None
    if ':' in cond:
        c, s = cond.split(':', 1)
        return c, s
    return cond, None


def severity_match(path_lower: str, severity):
    if severity is None:
        return True
    s = str(severity).lower()
    tokens = [
        f'severity_{s}', f'severity-{s}', f'severity{s}',
        f'sev_{s}', f'sev-{s}', f'sev{s}',
        f'/s{s}/', f'_s{s}_', f'-s{s}-',
        f'/{s}/', f'_{s}/', f'/{s}_',
    ]
    return any(t in path_lower for t in tokens)


def list_images(root: Path):
    return sorted([p for p in root.rglob('*') if p.suffix.lower() in IMG_EXTS])


def find_clean_images(clean_root: Path, image_names, num_images):
    imgs = list_images(clean_root)
    if image_names:
        idx = {p.name: p for p in imgs}
        out = []
        for n in image_names:
            if n in idx:
                out.append(idx[n])
            else:
                hits = [p for p in imgs if p.name == n or str(p).endswith(n)]
                if hits:
                    out.append(hits[0])
                else:
                    print(f'[WARN] clean image not found: {n}')
        return out
    # Prefer test folders if available
    test_imgs = [p for p in imgs if 'test' in str(p).lower()]
    return (test_imgs or imgs)[:num_images]


def find_condition_image(benchmark_root: Path, condition: str, clean_img: Path):
    cond, sev = parse_condition(condition)
    if cond == 'clean':
        return clean_img
    name = clean_img.name
    candidates = []
    for p in benchmark_root.rglob(name):
        if p.suffix.lower() in IMG_EXTS:
            pl = str(p).replace('\\', '/').lower()
            if cond.lower() in pl and severity_match(pl, sev):
                candidates.append(p)
    if not candidates:
        # fallback: condition only
        for p in benchmark_root.rglob(name):
            if p.suffix.lower() in IMG_EXTS and cond.lower() in str(p).lower():
                candidates.append(p)
    return candidates[0] if candidates else None


def load_models(args):
    if args.models_json:
        data = json.loads(Path(args.models_json).read_text(encoding='utf-8'))
        return data['models']
    return [{
        'name': args.name or Path(args.config).stem,
        'config': args.config,
        'checkpoint': args.checkpoint,
        'work_dir': args.work_dir,
    }]


def find_checkpoint(model):
    if model.get('checkpoint'):
        return model['checkpoint']
    wd = Path(model.get('work_dir', ''))
    for pat in ['best_coco_segm_mAP*.pth', 'best*.pth', 'epoch_*.pth', '*.pth']:
        hits = sorted(wd.glob(pat), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
        if hits:
            return str(hits[0])
    raise FileNotFoundError(f'No checkpoint found for {model.get("name")} in {wd}')


def layer_to_index(layer: str):
    layer = layer.upper()
    if layer.startswith('P'):
        return int(layer[1:]) - 2
    return int(layer)


def normalize_map(x):
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    if hi <= lo:
        lo, hi = float(x.min()), float(x.max())
    x = (x - lo) / (hi - lo + 1e-6)
    return np.clip(x, 0, 1)


def save_feature_overlay(feat, image_bgr, out_path):
    if isinstance(feat, torch.Tensor):
        feat = feat.detach().float().cpu()
    if feat.ndim == 4:
        feat = feat[0]
    heat = feat.abs().mean(dim=0).numpy()
    heat = normalize_map(heat)
    heat = cv2.resize(heat, (image_bgr.shape[1], image_bgr.shape[0]))
    heat_u8 = np.uint8(255 * heat)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_bgr, 0.55, heat_color, 0.45, 0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), overlay)


def draw_detection(result, image_bgr, score_thr=0.5, max_det=80):
    img = image_bgr.copy()
    if not hasattr(result, 'pred_instances'):
        return img
    pred = result.pred_instances
    if len(pred) == 0:
        return img

    scores = pred.scores.detach().cpu().numpy() if hasattr(pred, 'scores') else np.ones(len(pred))
    order = np.argsort(scores)[::-1]
    keep = [i for i in order if scores[i] >= score_thr][:max_det]

    bboxes = pred.bboxes.detach().cpu().numpy() if hasattr(pred, 'bboxes') else None
    masks = None
    if hasattr(pred, 'masks'):
        masks = pred.masks
        if isinstance(masks, torch.Tensor):
            masks = masks.detach().cpu().numpy()
        else:
            masks = masks.cpu().numpy() if hasattr(masks, 'cpu') else np.asarray(masks)

    rng = np.random.default_rng(2026)
    colors = rng.integers(50, 255, size=(max(1, len(keep)), 3), dtype=np.uint8)

    for j, i in enumerate(keep):
        color = tuple(int(x) for x in colors[j].tolist())
        if masks is not None:
            m = masks[i].astype(bool)
            if m.shape[:2] != img.shape[:2]:
                m = cv2.resize(m.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            overlay = img.copy()
            overlay[m] = color
            img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
        if bboxes is not None:
            x1, y1, x2, y2 = bboxes[i].astype(int).tolist()
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f'{scores[i]:.2f}', (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models-json', default=None)
    ap.add_argument('--config', default=None)
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--work-dir', default=None)
    ap.add_argument('--name', default=None)
    ap.add_argument('--clean-root', default='mmdet_dataset/lettuce')
    ap.add_argument('--benchmark-root', default='mmdet_dataset/lettuce_c')
    ap.add_argument('--conditions', nargs='+', default=['clean', 'gaussian_noise:3'])
    ap.add_argument('--image-names', nargs='*', default=None)
    ap.add_argument('--num-images', type=int, default=3)
    ap.add_argument('--layers', nargs='+', default=['P2', 'P3', 'P4'])
    ap.add_argument('--out-dir', default='paper_outputs_clean_noise_visuals')
    ap.add_argument('--score-thr', type=float, default=0.5)
    ap.add_argument('--max-det', type=int, default=80)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    from mmdet.apis import init_detector, inference_detector
    try:
        from mmdet.utils import register_all_modules
        register_all_modules(init_default_scope=True)
    except Exception:
        pass

    clean_root = Path(args.clean_root)
    benchmark_root = Path(args.benchmark_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_imgs = find_clean_images(clean_root, args.image_names, args.num_images)
    if not clean_imgs:
        raise FileNotFoundError(f'No clean images found under {clean_root}')

    models = load_models(args)
    layer_indices = [(l, layer_to_index(l)) for l in args.layers]

    for model_info in models:
        model_name = model_info.get('name') or Path(model_info['config']).stem
        model_safe = safe_name(model_name)
        ckpt = find_checkpoint(model_info)
        print(f'\n=== Load model: {model_name} ===')
        print('Config:', model_info['config'])
        print('Checkpoint:', ckpt)
        model = init_detector(model_info['config'], ckpt, device=args.device)

        captured = {}
        def hook(module, inputs, outputs):
            captured['neck'] = outputs
        handle = model.neck.register_forward_hook(hook)

        for cond in args.conditions:
            cond_safe = safe_name(cond)
            for clean_img in clean_imgs:
                img_path = find_condition_image(benchmark_root, cond, clean_img)
                if img_path is None:
                    print(f'[WARN] condition image not found: {cond} / {clean_img.name}')
                    continue

                captured.clear()
                image_bgr = cv2.imread(str(img_path))
                if image_bgr is None:
                    print(f'[WARN] failed to read image: {img_path}')
                    continue
                result = inference_detector(model, str(img_path))

                base = Path(clean_img).stem
                det = draw_detection(result, image_bgr, score_thr=args.score_thr, max_det=args.max_det)
                det_path = out_dir / model_safe / cond_safe / 'detections' / f'{base}_{cond_safe}_det.png'
                det_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(det_path), det)

                neck_outs = captured.get('neck')
                if neck_outs is None:
                    print(f'[WARN] no neck feature captured for {img_path}')
                    continue
                for layer_name, idx in layer_indices:
                    if idx < 0 or idx >= len(neck_outs):
                        print(f'[WARN] layer {layer_name} idx={idx} out of range, num_outs={len(neck_outs)}')
                        continue
                    feat_path = out_dir / model_safe / cond_safe / 'featuremaps' / layer_name / f'{base}_{cond_safe}_{layer_name}.png'
                    save_feature_overlay(neck_outs[idx], image_bgr, feat_path)

        handle.remove()

    print(f'\nSaved visualizations to: {out_dir}')


if __name__ == '__main__':
    main()
