#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_featuremaps_and_train_logs.py

Sinh feature map cho clean + robustness/corruption, mỗi condition lấy 1 ảnh.
Đọc train_console.log để vẽ loss, lr, val AP.

Chạy từ MMDetection root:
  cd /home/pc/mmdet_AI/mmdetection
  conda activate mmdet
  export PYTHONPATH=/home/pc/mmdet_AI/mmdetection:$PYTHONPATH

Ví dụ:
  python tools/research/visualize_featuremaps_and_train_logs.py \
    --config configs/fair_lettuce/mask_rcnn_r50_aspp_boundary_fpn.py \
    --checkpoint work_dirs/research/mask_rcnn_r50_aspp_boundary_fpn/seed_2026/best_coco_segm_mAP_epoch_35.pth \
    --clean-root mmdet_dataset/lettuce \
    --benchmark-root mmdet_dataset/lettuce_c \
    --manifest mmdet_dataset/lettuce_c/manifest.json \
    --out-dir work_dirs/research/featuremap_report/aspp_boundary \
    --layer neck \
    --levels 0 1 2 3 \
    --log-file work_dirs/research/mask_rcnn_r50_aspp_boundary_fpn/seed_2026/train_console.log
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt


def mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8-sig'))


def safe_name(text: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(text))


def normalize_map(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.nanmin(x)
    return x / (np.nanmax(x) + 1e-8)


def read_image_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f'Cannot read image: {path}')
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_rgb(path: Path, img_rgb: np.ndarray):
    mkdir(path.parent)
    cv2.imwrite(str(path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))


def save_heatmap(path: Path, fmap: np.ndarray):
    mkdir(path.parent)
    fmap = normalize_map(fmap)
    plt.figure(figsize=(5, 5))
    plt.imshow(fmap)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()


def save_overlay(path: Path, img_rgb: np.ndarray, fmap: np.ndarray, alpha: float = 0.45):
    mkdir(path.parent)
    h, w = img_rgb.shape[:2]
    fmap = normalize_map(fmap)
    fmap = cv2.resize(fmap, (w, h), interpolation=cv2.INTER_LINEAR)
    heat = cv2.applyColorMap(np.uint8(255 * fmap), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    overlay = np.uint8((1.0 - alpha) * img_rgb + alpha * heat)
    save_rgb(path, overlay)


def first_image_from_coco(data_root: Path, ann_file: str, image_prefix: str, image_index: int) -> Path:
    coco = read_json(data_root / ann_file)
    images = sorted(coco['images'], key=lambda x: x.get('id', 0))
    if not images:
        raise RuntimeError(f'No images in {data_root / ann_file}')
    image_index = max(0, min(image_index, len(images) - 1))
    return data_root / image_prefix / images[image_index]['file_name']


def build_conditions(clean_root: Path, benchmark_root: Path, manifest: Optional[Path], image_index: int, only: Optional[List[str]]):
    conditions = []
    conditions.append({
        'name': 'clean',
        'corruption': 'clean',
        'severity': '',
        'image_path': first_image_from_coco(clean_root, 'annotations/test.json', 'images/test', image_index),
    })

    if manifest is None or not manifest.is_file():
        return conditions

    m = read_json(manifest)
    output_ann = m.get('output_annotation', 'annotations/test_png.json')
    for item in m.get('conditions', []):
        corruption = item['corruption']
        severity = int(item['severity'])
        if only and corruption not in only:
            continue
        prefix = item['image_prefix'].rstrip('/')
        conditions.append({
            'name': f'{corruption}_s{severity}',
            'corruption': corruption,
            'severity': severity,
            'image_path': first_image_from_coco(benchmark_root, output_ann, prefix, image_index),
        })
    return conditions


def load_model(config: str, checkpoint: str, device: str):
    from mmengine.config import Config
    from mmdet.apis import init_detector
    cfg = Config.fromfile(config)
    if 'model' in cfg and 'backbone' in cfg.model and 'init_cfg' in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None
    model = init_detector(cfg, checkpoint, device=device)
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


def hook_module(model, layer: str):
    if layer == 'backbone':
        return model.backbone
    if layer == 'neck':
        return model.neck
    if layer == 'rpn_head':
        return model.rpn_head
    if layer == 'roi_head':
        return model.roi_head
    raise ValueError(layer)


def output_to_maps(features: Any, levels: List[int]) -> List[Tuple[int, np.ndarray]]:
    if isinstance(features, torch.Tensor):
        feats = [features]
    elif isinstance(features, (list, tuple)):
        feats = list(features)
    else:
        raise TypeError(f'Unsupported feature type: {type(features)}')

    maps = []
    for i, feat in enumerate(feats):
        if i not in levels:
            continue
        if not isinstance(feat, torch.Tensor) or feat.ndim != 4:
            continue
        fmap = feat.detach().float().abs().mean(dim=1)[0].cpu().numpy()
        maps.append((i, fmap))
    return maps


def visualize_featuremaps(args):
    from mmdet.apis import inference_detector
    model = load_model(args.config, args.checkpoint, args.device)
    hook = FeatureHook(hook_module(model, args.layer))

    conditions = build_conditions(
        Path(args.clean_root).resolve(),
        Path(args.benchmark_root).resolve(),
        Path(args.manifest).resolve() if args.manifest else None,
        args.image_index,
        args.only,
    )

    rows = []
    for c in conditions:
        image_path = Path(c['image_path'])
        img_rgb = read_image_rgb(image_path)
        print(f"[FeatureMap] {c['name']} -> {image_path}")
        hook.features = None
        with torch.no_grad():
            _ = inference_detector(model, str(image_path))
        if hook.features is None:
            print(f"[WARN] no features captured: {c['name']}")
            continue

        cdir = Path(args.out_dir).resolve() / 'featuremaps' / safe_name(c['name'])
        mkdir(cdir)
        save_rgb(cdir / 'input.png', img_rgb)

        for level, fmap in output_to_maps(hook.features, args.levels):
            heat_path = cdir / f'{args.layer}_level{level}_heatmap.png'
            overlay_path = cdir / f'{args.layer}_level{level}_overlay.png'
            save_heatmap(heat_path, fmap)
            save_overlay(overlay_path, img_rgb, fmap, args.overlay_alpha)
            rows.append([c['name'], c['corruption'], c['severity'], str(image_path), args.layer, level, str(heat_path), str(overlay_path)])

    hook.close()
    summary = Path(args.out_dir).resolve() / 'featuremap_summary.csv'
    mkdir(summary.parent)
    with summary.open('w', encoding='utf-8') as f:
        f.write('condition,corruption,severity,image_path,layer,level,heatmap,overlay\n')
        for r in rows:
            f.write(','.join(map(str, r)) + '\n')
    print(f'[OK] Featuremap summary: {summary}')


def parse_log(log_path: Path):
    text = log_path.read_text(encoding='utf-8', errors='ignore')
    data = {k: [] for k in ['train_loss', 'lr', 'val_mask_ap', 'val_box_ap', 'val_mask_ap50', 'val_mask_ap75']}
    last_x = 0.0
    for line_idx, line in enumerate(text.splitlines(), 1):
        m = re.search(r'Epoch\(train\)\s+\[(\d+)\]\[(\d+)/(\d+)\]', line)
        if m:
            epoch, it, total = map(int, m.groups())
            last_x = epoch + it / max(1, total)
            lm = re.search(r'\blr:\s*([0-9.eE+-]+)', line)
            if lm:
                data['lr'].append((last_x, float(lm.group(1))))
            lossm = re.search(r'(?:^|\s)loss:\s*([0-9.eE+-]+)', line)
            if lossm:
                data['train_loss'].append((last_x, float(lossm.group(1))))
        else:
            x = last_x if last_x > 0 else float(line_idx)
            lm = re.search(r'\blr:\s*([0-9.eE+-]+)', line)
            if lm:
                data['lr'].append((x, float(lm.group(1))))
            lossm = re.search(r'(?:^|\s)loss:\s*([0-9.eE+-]+)', line)
            if lossm:
                data['train_loss'].append((x, float(lossm.group(1))))

        vm = re.search(r'Epoch\(val\)\s+\[(\d+)\]', line)
        vx = float(vm.group(1)) if vm else last_x
        pats = {
            'val_mask_ap': r'coco/segm_mAP:\s*([0-9.eE+-]+)',
            'val_mask_ap50': r'coco/segm_mAP_50:\s*([0-9.eE+-]+)',
            'val_mask_ap75': r'coco/segm_mAP_75:\s*([0-9.eE+-]+)',
            'val_box_ap': r'coco/bbox_mAP:\s*([0-9.eE+-]+)',
        }
        for key, pat in pats.items():
            mm = re.search(pat, line)
            if mm:
                data[key].append((vx, float(mm.group(1))))
    return data


def plot_series(path: Path, series, title: str, xlabel: str, ylabel: str):
    if not series:
        return
    mkdir(path.parent)
    x = [a for a, b in series]
    y = [b for a, b in series]
    plt.figure(figsize=(8, 4.5))
    plt.plot(x, y, marker='o', markersize=2, linewidth=1.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_multi(path: Path, data_dict, title: str, xlabel: str, ylabel: str):
    if not any(data_dict.values()):
        return
    mkdir(path.parent)
    plt.figure(figsize=(8, 4.5))
    for label, series in data_dict.items():
        if not series:
            continue
        x = [a for a, b in series]
        y = [b for a, b in series]
        plt.plot(x, y, marker='o', markersize=2, linewidth=1.5, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def process_log(args):
    if not args.log_file:
        return
    log_path = Path(args.log_file).resolve()
    if not log_path.is_file():
        print(f'[WARN] log not found: {log_path}')
        return
    data = parse_log(log_path)
    out = Path(args.out_dir).resolve() / 'training_curves'
    mkdir(out)
    plot_series(out / 'train_loss.png', data['train_loss'], 'Training Loss', 'Epoch', 'Loss')
    plot_series(out / 'learning_rate.png', data['lr'], 'Learning Rate', 'Epoch', 'LR')
    plot_multi(out / 'validation_ap.png', {'segm_mAP': data['val_mask_ap'], 'bbox_mAP': data['val_box_ap']}, 'Validation AP', 'Epoch', 'AP')
    plot_multi(out / 'validation_mask_ap50_ap75.png', {'segm_mAP50': data['val_mask_ap50'], 'segm_mAP75': data['val_mask_ap75']}, 'Validation Mask AP50/AP75', 'Epoch', 'AP')
    (out / 'parsed_training_log.json').write_text(json.dumps({k: [{'x': x, 'y': y} for x, y in v] for k, v in data.items()}, indent=2), encoding='utf-8')
    print(f'[OK] Training curves: {out}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--clean-root', default='mmdet_dataset/lettuce')
    p.add_argument('--benchmark-root', default='mmdet_dataset/lettuce_c')
    p.add_argument('--manifest', default='mmdet_dataset/lettuce_c/manifest.json')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--layer', default='neck', choices=['backbone', 'neck', 'rpn_head', 'roi_head'])
    p.add_argument('--levels', nargs='+', type=int, default=[0, 1, 2, 3])
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--image-index', type=int, default=0)
    p.add_argument('--overlay-alpha', type=float, default=0.45)
    p.add_argument('--only', nargs='+', default=None)
    p.add_argument('--log-file', default=None)
    p.add_argument('--skip-featuremaps', action='store_true')
    p.add_argument('--skip-log', action='store_true')
    args = p.parse_args()
    mkdir(Path(args.out_dir).resolve())
    if not args.skip_featuremaps:
        visualize_featuremaps(args)
    if not args.skip_log:
        process_log(args)
    print(f'[DONE] Output: {Path(args.out_dir).resolve()}')


if __name__ == '__main__':
    main()
