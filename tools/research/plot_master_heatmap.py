# -*- coding: utf-8 -*-
"""Vẽ heatmap tổng hợp AP mean±std cho mọi model trên MỌI điều kiện.

Gộp kết quả từ nhiều thư mục evaluation (uniform + spatial), tính mean±std
qua các seed, hiển thị 1 hoặc 2 heatmap song song:
  - Heatmap chính: mean AP
  - (Tùy chọn) heatmap std bên cạnh — cho thấy độ ổn định

Cách dùng:
    python tools/research/plot_master_heatmap.py \
        --root work_dirs/research \
        --seeds 2024 2025 2026 \
        --eval-subdirs evaluation evaluation_spatial \
        --models mask_rcnn_r50_fpn mask_rcnn_r50_fpn_aug mask_rcnn_r50_fpn_bca \
                 mask_rcnn_r50_cbam_fpn mask_rcnn_r50_bifpn \
                 mask_rcnn_r50_dgcf_fpn_v2 mask_rcnn_r101_fpn \
        --out heatmap_master.png \
        --show-std
"""

import argparse
import glob
import json
import os
from collections import defaultdict
from statistics import mean, stdev

import numpy as np

METRIC = 'coco/segm_mAP'

# Thứ tự cột chuẩn (nếu điều kiện có)
COND_ORDER = [
    'clean',
    'brightness_s1', 'brightness_s2', 'brightness_s3',
    'contrast_s1', 'contrast_s2', 'contrast_s3',
    'gaussian_noise_s1', 'gaussian_noise_s2', 'gaussian_noise_s3',
    'uneven_contrast_s1', 'uneven_contrast_s2', 'uneven_contrast_s3',
    'dappled_light_s1', 'dappled_light_s2', 'dappled_light_s3',
]


def load_all(root, seeds, eval_subdirs, models):
    """rec[model][cond] = list AP qua các seed."""
    rec = defaultdict(lambda: defaultdict(list))
    for model in models:
        for seed in seeds:
            for subdir in eval_subdirs:
                cond_root = os.path.join(root, model, f'seed_{seed}',
                                         subdir, 'conditions')
                if not os.path.isdir(cond_root):
                    continue
                for cond_dir in glob.glob(os.path.join(cond_root, '*')):
                    mfile = os.path.join(cond_dir, 'metrics.json')
                    if not os.path.isfile(mfile):
                        continue
                    try:
                        d = json.load(open(mfile))
                        if METRIC in d:
                            cond = os.path.basename(cond_dir)
                            rec[model][cond].append(float(d[METRIC]))
                    except Exception:  # noqa: BLE001
                        pass
    return rec


def build_matrix(rec, models, conditions):
    """Ma trận mean, std kích thước (n_models, n_conditions)."""
    M = np.full((len(models), len(conditions)), np.nan)
    S = np.full((len(models), len(conditions)), np.nan)
    N = np.zeros((len(models), len(conditions)), dtype=int)
    for i, m in enumerate(models):
        for j, c in enumerate(conditions):
            vals = rec[m].get(c, [])
            if vals:
                M[i, j] = mean(vals)
                S[i, j] = stdev(vals) if len(vals) > 1 else 0.0
                N[i, j] = len(vals)
    return M, S, N


def sorted_conditions(rec, models):
    seen = set()
    for m in models:
        seen.update(rec[m].keys())
    ordered = [c for c in COND_ORDER if c in seen]
    ordered += sorted(c for c in seen if c not in COND_ORDER)
    return ordered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='work_dirs/research')
    ap.add_argument('--seeds', nargs='+', default=['2024', '2025', '2026'])
    ap.add_argument('--eval-subdirs', nargs='+',
                    default=['evaluation', 'evaluation_spatial'])
    ap.add_argument('--models', nargs='+', required=True)
    ap.add_argument('--out', default='heatmap_master.png')
    ap.add_argument('--show-std', action='store_true',
                    help='Vẽ thêm 1 heatmap std cạnh heatmap mean')
    ap.add_argument('--sort-by-corr', action='store_true',
                    help='Xếp model theo AP_corr giảm dần')
    ap.add_argument('--cmap', default='RdYlGn',
                    help='matplotlib colormap (RdYlGn, viridis, RdYlBu...)')
    args = ap.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rec = load_all(args.root, args.seeds, args.eval_subdirs, args.models)
    conditions = sorted_conditions(rec, args.models)
    if not conditions:
        raise SystemExit('Không đọc được điều kiện nào. Kiểm tra --root, '
                         '--seeds, --eval-subdirs, --models.')

    models = list(args.models)
    if args.sort_by_corr:
        def corr(m):
            corrs = [rec[m][c] for c in conditions if c != 'clean']
            flat = [v for l in corrs for v in l]
            return -mean(flat) if flat else 0
        models.sort(key=corr)

    M, S, N = build_matrix(rec, models, conditions)

    # Vẽ
    n_panels = 2 if args.show_std else 1
    fig_w = max(9, 0.9 * len(conditions) * n_panels)
    fig_h = max(4, 0.55 * len(models) + 2)
    fig, axes = plt.subplots(1, n_panels,
                             figsize=(fig_w, fig_h), squeeze=False)

    def draw(ax, mat, title, vmin, vmax, annot_fmt):
        im = ax.imshow(mat, aspect='auto', cmap=args.cmap,
                       vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(conditions, rotation=40, ha='right', fontsize=9)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight='bold')
        # Vẽ chữ ô
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isnan(v):
                    ax.text(j, i, '—', ha='center', va='center',
                            fontsize=8, color='gray')
                else:
                    color = 'white' if v < (vmin + vmax) / 2 else 'black'
                    if args.cmap in ('RdYlGn', 'RdYlBu', 'viridis'):
                        color = 'black'  # colormap đủ nét, để đen hết
                    ax.text(j, i, f'{v:{annot_fmt}}', ha='center',
                            va='center', fontsize=8, color=color)
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    draw(axes[0, 0], M, 'Mean AP (3 seeds)',
         float(np.nanmin(M)), float(np.nanmax(M)), '.3f')
    if args.show_std:
        smax = float(np.nanmax(S)) if np.any(~np.isnan(S)) else 0.05
        draw(axes[0, 1], S, 'Std across seeds', 0.0, smax, '.3f')

    fig.suptitle(
        f'AP theo (model × điều kiện) — {len(models)} model × '
        f'{len(conditions)} điều kiện × {len(args.seeds)} seed',
        fontsize=12, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f'Đã lưu: {args.out}')

    # Cảnh báo ô thiếu (n<len(seeds))
    missing = []
    for i, m in enumerate(models):
        for j, c in enumerate(conditions):
            if N[i, j] < len(args.seeds):
                missing.append((m, c, N[i, j]))
    if missing:
        print(f'\nCẢNH BÁO: {len(missing)} ô có ít hơn {len(args.seeds)} seed:')
        for m, c, n in missing[:20]:
            print(f'  {m:<40} {c:<25} n={n}')
        if len(missing) > 20:
            print(f'  ... và {len(missing) - 20} ô khác')


if __name__ == '__main__':
    main()
