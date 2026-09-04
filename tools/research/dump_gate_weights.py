# -*- coding: utf-8 -*-
"""Dump gate weights alpha_o/alpha_c/alpha_d theo (condition, pyramid level)
để chứng minh/bác bỏ tính "adaptive" của gate (Major Comment 3).

Chạy inference trên từng thư mục ảnh condition, thu thập softmax gate qua
DGCFPNv2.record_gate_stats, xuất CSV + biểu đồ:
- gate weight theo corruption type & severity, cho từng level P2-P6.

Cách dùng:
    python tools/research/dump_gate_weights.py \
        configs/fair_lettuce/mask_rcnn_r50_dgcf_fpn_v2.py \
        work_dirs/research/mask_rcnn_r50_dgcf_fpn_v2/seed_2026/best_coco_segm_mAP_epoch_*.pth \
        --conditions-root ${BENCHMARK_ROOT} \
        --clean-root ${CLEAN_ROOT} \
        --out-dir gate_analysis/seed_2026 \
        --max-images 100
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
from mmengine.config import Config
from mmdet.apis import init_detector, inference_detector

BRANCH_NAMES = ['original', 'context', 'detail']


def collect_for_dir(model, img_dir, max_images):
    neck = model.neck
    neck.record_gate_stats = True
    neck.reset_gate_stats()
    imgs = sorted(
        glob.glob(os.path.join(img_dir, '*.jpg')) +
        glob.glob(os.path.join(img_dir, '*.png')))[:max_images]
    if not imgs:
        print(f'[Cảnh báo] Không có ảnh trong {img_dir}')
        return None
    for p in imgs:
        _ = inference_detector(model, p)
    # Gom theo level: mean/std của từng branch weight
    stats = {}
    for rec in neck.get_gate_stats():
        lvl = rec['level']
        stats.setdefault(lvl, []).append(rec['weights'])  # list of [B,K]
    out = {}
    for lvl, ws in stats.items():
        w = torch.cat(ws, dim=0).numpy()  # N,K
        out[f'P{lvl + 2}'] = {
            'mean': w.mean(axis=0).tolist(),
            'std': w.std(axis=0).tolist(),
            'n': int(w.shape[0]),
        }
    neck.record_gate_stats = False
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--conditions-root', required=True,
                        help='Thư mục chứa các thư mục con condition '
                             '(brightness_s1, ..., gaussian_noise_s3), '
                             'mỗi thư mục chứa ảnh test đã corrupt')
    parser.add_argument('--clean-root', required=True,
                        help='Thư mục ảnh test sạch')
    parser.add_argument('--out-dir', default='gate_analysis')
    parser.add_argument('--max-images', type=int, default=100)
    args = parser.parse_args()

    ckpts = glob.glob(args.checkpoint)
    assert ckpts, f'Không tìm thấy checkpoint: {args.checkpoint}'
    model = init_detector(args.config, ckpts[0], device='cuda:0')
    assert hasattr(model.neck, 'record_gate_stats'), \
        'Neck không phải DGCFPNv2 (thiếu record_gate_stats).'

    os.makedirs(args.out_dir, exist_ok=True)
    results = {}

    print('== clean ==')
    results['clean'] = collect_for_dir(model, args.clean_root,
                                       args.max_images)

    for cond_dir in sorted(os.listdir(args.conditions_root)):
        full = os.path.join(args.conditions_root, cond_dir)
        if not os.path.isdir(full):
            continue
        print(f'== {cond_dir} ==')
        results[cond_dir] = collect_for_dir(model, full, args.max_images)

    json_path = os.path.join(args.out_dir, 'gate_weights.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Đã lưu: {json_path}')

    # CSV phẳng: condition, level, branch, mean, std
    import csv
    csv_path = os.path.join(args.out_dir, 'gate_weights.csv')
    with open(csv_path, 'w', newline='') as f:
        wtr = csv.writer(f)
        wtr.writerow(['condition', 'level', 'branch', 'mean', 'std', 'n'])
        for cond, levels in results.items():
            if levels is None:
                continue
            for lvl, s in levels.items():
                for bi, bname in enumerate(BRANCH_NAMES[:len(s['mean'])]):
                    wtr.writerow([cond, lvl, bname,
                                  f"{s['mean'][bi]:.4f}",
                                  f"{s['std'][bi]:.4f}", s['n']])
    print(f'Đã lưu: {csv_path}')

    # Biểu đồ: mỗi level 1 panel, x = condition, các đường = branch
    try:
        import matplotlib.pyplot as plt
        conds = [c for c in results if results[c] is not None]
        levels = sorted({l for c in conds for l in results[c].keys()})
        fig, axes = plt.subplots(len(levels), 1,
                                 figsize=(max(8, len(conds)), 2.6 * len(levels)),
                                 sharex=True)
        if len(levels) == 1:
            axes = [axes]
        for ax, lvl in zip(axes, levels):
            k = len(results[conds[0]][lvl]['mean'])
            for bi in range(k):
                means = [results[c][lvl]['mean'][bi] for c in conds]
                stds = [results[c][lvl]['std'][bi] for c in conds]
                ax.errorbar(range(len(conds)), means, yerr=stds,
                            marker='o', capsize=3, label=BRANCH_NAMES[bi])
            ax.set_title(lvl, loc='left', fontweight='bold')
            ax.set_ylabel('gate weight')
            ax.legend(fontsize=8)
        axes[-1].set_xticks(range(len(conds)))
        axes[-1].set_xticklabels(conds, rotation=40, ha='right')
        fig.suptitle('Gate weights theo condition và pyramid level')
        fig.tight_layout()
        png_path = os.path.join(args.out_dir, 'gate_weights.png')
        fig.savefig(png_path, dpi=150, bbox_inches='tight')
        print(f'Đã lưu: {png_path}')
    except Exception as e:  # noqa: BLE001
        print(f'[Cảnh báo] Không vẽ được biểu đồ: {e}')


if __name__ == '__main__':
    main()
