# -*- coding: utf-8 -*-
"""Vẽ TỪNG panel (a)-(h) của BCA riêng biệt.

Mỗi panel là một hàm draw_a, draw_b, ..., draw_h nhận:
  - ax: matplotlib Axes để vẽ vào
  - ctx: dict chứa ảnh + kết quả BCA (img_rgb, M, w, L, k, out_rgb, ...)
  - hp: dict tham số (band, k_min, s_used, ...)

Bạn có thể:
  1. Xuất TẤT CẢ 8 panel riêng: --mode separate → 8 file PNG riêng.
  2. Ghép các panel chọn lọc vào 1 hình: --mode compose --panels a c f h.

Cách dùng:
  # Xuất 8 file riêng
  python draw_bca_panels.py --ann test.json --img-root images \
      --mode separate --out-dir figures_bca/

  # Ghép 4 panel tối thiểu vào 1 hình
  python draw_bca_panels.py --ann test.json --img-root images \
      --mode compose --panels a c f h --out fig_bca_minimal.pdf

  # Ghép 5 panel (thêm profile định lượng)
  python draw_bca_panels.py --ann test.json --img-root images \
      --mode compose --panels a c f g h --out fig_bca_extended.pdf
"""

import argparse
import json
import os

import cv2
import numpy as np
from pycocotools import mask as coco_mask

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# =============================================================================
# BCA transform (khớp boundary_aug.py)
# =============================================================================
def bca_transform(img, M, strength, band_width, k_min, sigma_b):
    """Chạy BCA đầy đủ, trả về mọi bước trung gian."""
    k3 = np.ones((3, 3), np.uint8)
    boundary = cv2.morphologyEx(M, cv2.MORPH_GRADIENT, k3)
    d = cv2.distanceTransform(1 - boundary, cv2.DIST_L2, 3)
    w_raw = np.clip(1.0 - d / max(band_width, 1), 0.0, 1.0)
    w = cv2.GaussianBlur(w_raw, (0, 0), sigmaX=sigma_b)
    w = np.clip(w, 0.0, 1.0).astype(np.float32)
    Kl = 2 * band_width + 1
    L = cv2.boxFilter(img, -1, (Kl, Kl))
    keep = np.maximum(1.0 - strength * w, k_min).astype(np.float32)
    out = keep[..., None] * img.astype(np.float32) + \
        (1.0 - keep[..., None]) * L.astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return {'boundary': boundary, 'w': w, 'L': L, 'k': keep,
            'out': out, 's_used': strength}


def build_foreground_union(coco_json, image_info, img_shape):
    """Gộp mọi instance mask thành 1 foreground map."""
    H, W = img_shape[:2]
    img_id = image_info['id']
    anns = [a for a in coco_json['annotations'] if a['image_id'] == img_id]
    M = np.zeros((H, W), np.uint8)
    for a in anns:
        seg = a['segmentation']
        if isinstance(seg, list):
            rles = coco_mask.frPyObjects(seg, H, W)
            m = coco_mask.decode(rles)
            if m.ndim == 3:
                m = m.any(axis=2)
        elif isinstance(seg, dict):
            m = coco_mask.decode(seg)
        else:
            continue
        M |= m.astype(np.uint8)
    return M, len(anns)


def pick_image(coco_json, img_root, preferred_id=None):
    if preferred_id is not None:
        for im in coco_json['images']:
            if im['id'] == preferred_id:
                return im
        raise ValueError(f'image_id {preferred_id} không tồn tại')
    ann_count = {}
    for a in coco_json['annotations']:
        ann_count[a['image_id']] = ann_count.get(a['image_id'], 0) + 1
    scored = []
    for im in coco_json['images']:
        n = ann_count.get(im['id'], 0)
        if os.path.isfile(os.path.join(img_root, im['file_name'])):
            scored.append((-abs(n - 10), im))
    scored.sort(key=lambda x: -x[0])
    if not scored:
        raise SystemExit('Không tìm thấy ảnh')
    return scored[0][1]


# =============================================================================
# 8 HÀM VẼ TỪNG PANEL — mỗi hàm nhận (ax, ctx, hp) và tự lo phần vẽ
# =============================================================================
def draw_a(ax, ctx, hp):
    """(a) Input image I — ảnh gốc, cho reader bối cảnh."""
    ax.imshow(ctx['img_rgb'])
    ax.set_title('(a) Input image $I$', fontsize=11, fontweight='bold',
                 pad=6)
    ax.set_xticks([]); ax.set_yticks([])


def draw_b(ax, ctx, hp):
    """(b) Foreground union M — mask nhị phân gộp mọi instance."""
    ax.imshow(ctx['M'], cmap='gray', vmin=0, vmax=1)
    ax.set_title('(b) Foreground union $M$', fontsize=11, fontweight='bold',
                 pad=6)
    ax.set_xticks([]); ax.set_yticks([])


def draw_c(ax, ctx, hp):
    """(c) Boundary band weight w — điểm nhấn của BCA.
    Heatmap thể hiện dải biên: sáng ở viền, mờ dần ra hai bên.
    """
    im = ax.imshow(ctx['w'], cmap='inferno', vmin=0, vmax=1)
    ax.set_title('(c) Boundary band weight $w$', fontsize=11,
                 fontweight='bold', pad=6)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02, shrink=0.85)


def draw_d(ax, ctx, hp):
    """(d) Local surround L — ảnh đã box-mean, ở vùng biên là màu trung gian."""
    L_rgb = cv2.cvtColor(ctx['L'], cv2.COLOR_BGR2RGB)
    ax.imshow(L_rgb)
    ax.set_title('(d) Local surround $L$', fontsize=11, fontweight='bold',
                 pad=6)
    ax.set_xticks([]); ax.set_yticks([])


def draw_e(ax, ctx, hp):
    """(e) Keep coefficient k — verify k >= k_min (label-preserving)."""
    im = ax.imshow(ctx['k'], cmap='viridis', vmin=hp['k_min'], vmax=1)
    ax.set_title(
        f"(e) Keep coefficient $k$\n"
        f"($s{{=}}{hp['s_used']:.2f}$, $k_{{\\min}}{{=}}{hp['k_min']}$)",
        fontsize=11, fontweight='bold', pad=6)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02, shrink=0.85)


def draw_f(ax, ctx, hp):
    """(f) Output I' — kết quả cuối, before/after so với (a)."""
    ax.imshow(ctx['out_rgb'])
    ax.set_title("(f) Augmented output $I' = kI + (1-k)L$",
                 fontsize=11, fontweight='bold', pad=6)
    ax.set_xticks([]); ax.set_yticks([])


def draw_g(ax, ctx, hp):
    """(g) Intensity profile: bằng chứng định lượng label-preserving.
    Chọn dòng y có nhiều biên nhất. Tô cam vùng band.
    """
    boundary = ctx['boundary']
    y_row = int(np.argmax(boundary.sum(axis=1)))
    W = ctx['img_bgr'].shape[1]
    row_orig = ctx['img_bgr'][y_row, :, 1].astype(float)  # green
    row_out = ctx['out_bgr'][y_row, :, 1].astype(float)
    ax.plot(row_orig, color='#1976d2', lw=1.3, label='original $I$',
            alpha=0.85)
    ax.plot(row_out, color='#d32f2f', lw=1.3, label="BCA output $I'$",
            alpha=0.85)
    # Tô các đoạn liên tục có w > 0.1 (vùng bị BCA tác động)
    w_row = ctx['w'][y_row, :]
    band_mask = w_row > 0.1
    in_band = False
    seg_start = 0
    for x in range(W):
        if band_mask[x] and not in_band:
            seg_start = x; in_band = True
        elif not band_mask[x] and in_band:
            ax.axvspan(seg_start, x, alpha=0.15, color='orange')
            in_band = False
    if in_band:
        ax.axvspan(seg_start, W, alpha=0.15, color='orange')
    ax.set_title(f'(g) Intensity profile at $y{{=}}{y_row}$\n'
                 'orange = BCA-affected regions',
                 fontsize=11, fontweight='bold', pad=6)
    ax.set_xlabel('x (pixel)', fontsize=9)
    ax.set_ylabel('green channel', fontsize=9)
    ax.legend(fontsize=9, loc='best')
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.2)


def draw_h(ax, ctx, hp):
    """(h) |I' - I| heatmap: bằng chứng chỉ vùng biên bị tác động."""
    diff = np.abs(ctx['out_bgr'].astype(int)
                  - ctx['img_bgr'].astype(int)).mean(axis=2)
    vmax = max(np.percentile(diff, 99), 15)
    im = ax.imshow(diff, cmap='hot', vmin=0, vmax=vmax)
    ax.set_title("(h) $|I' - I|$: change concentrated\nin boundary band",
                 fontsize=11, fontweight='bold', pad=6)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02, shrink=0.85)


PANEL_FUNCS = {'a': draw_a, 'b': draw_b, 'c': draw_c, 'd': draw_d,
               'e': draw_e, 'f': draw_f, 'g': draw_g, 'h': draw_h}


# =============================================================================
# 2 chế độ output
# =============================================================================
def make_ctx(img_bgr, M, results):
    return {
        'img_bgr': img_bgr,
        'img_rgb': cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
        'M': M,
        'w': results['w'],
        'L': results['L'],
        'k': results['k'],
        'boundary': results['boundary'],
        'out_bgr': results['out'],
        'out_rgb': cv2.cvtColor(results['out'], cv2.COLOR_BGR2RGB),
    }


def save_separate(ctx, hp, out_dir):
    """Xuất 8 file PNG + PDF riêng cho mỗi panel."""
    os.makedirs(out_dir, exist_ok=True)
    for key, fn in PANEL_FUNCS.items():
        # Kích thước tỉ lệ với ảnh gốc cho panel ảnh, cỡ chuẩn cho graph
        if key == 'g':
            fig, ax = plt.subplots(figsize=(7, 4))
        else:
            H, W = ctx['img_bgr'].shape[:2]
            aspect = W / H
            fig, ax = plt.subplots(figsize=(4.5 * aspect, 4.5))
        fn(ax, ctx, hp)
        for ext in ('pdf', 'png'):
            p = os.path.join(out_dir, f'panel_{key}.{ext}')
            plt.savefig(p, bbox_inches='tight', dpi=200)
        plt.close()
        print(f'  saved {out_dir}/panel_{key}.{{pdf,png}}')


def save_compose(ctx, hp, panels, out_path):
    """Ghép các panel được chọn vào 1 hình. Tự chọn grid layout hợp lý."""
    n = len(panels)
    if n <= 2: rows, cols = 1, n
    elif n <= 4: rows, cols = 1, n
    elif n <= 6: rows, cols = 2, 3
    elif n <= 8: rows, cols = 2, 4
    else: raise ValueError('quá nhiều panel')

    # Ước lượng figsize
    fig_w = 4.5 * cols
    fig_h = 4.5 * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axes = np.atleast_1d(axes).flatten()

    for i, key in enumerate(panels):
        PANEL_FUNCS[key](axes[i], ctx, hp)
    # Ẩn axes thừa
    for j in range(len(panels), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.savefig(out_path.replace('.pdf', '.png'),
                bbox_inches='tight', dpi=200)
    plt.close()
    print(f'Saved: {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ann', required=True)
    ap.add_argument('--img-root', required=True)
    ap.add_argument('--image-id', type=int, default=None)
    ap.add_argument('--mode', choices=['separate', 'compose'],
                    default='separate')
    ap.add_argument('--panels', nargs='+', default=['a', 'c', 'f', 'h'],
                    help='các panel cần ghép (dùng với --mode compose)')
    ap.add_argument('--out-dir', default='figures_bca',
                    help='thư mục xuất (dùng với --mode separate)')
    ap.add_argument('--out', default='fig_bca_compose.pdf',
                    help='file xuất (dùng với --mode compose)')
    ap.add_argument('--strength', type=float, default=0.55)
    ap.add_argument('--band', type=int, default=15)
    ap.add_argument('--k-min', type=float, default=0.35)
    ap.add_argument('--sigma-b', type=float, default=3.0)
    args = ap.parse_args()

    coco = json.load(open(args.ann))
    im_info = pick_image(coco, args.img_root, args.image_id)
    img_path = os.path.join(args.img_root, im_info['file_name'])
    img_bgr = cv2.imread(img_path)
    print(f'Ảnh: {im_info["file_name"]}, shape={img_bgr.shape}, '
          f'id={im_info["id"]}')

    M, n_ann = build_foreground_union(coco, im_info, img_bgr.shape)
    print(f'  {n_ann} instances, coverage {100*M.mean():.1f}%')

    r = bca_transform(img_bgr, M,
                      strength=args.strength,
                      band_width=args.band,
                      k_min=args.k_min,
                      sigma_b=args.sigma_b)
    ctx = make_ctx(img_bgr, M, r)
    hp = {'band': args.band, 'k_min': args.k_min,
          'sigma_b': args.sigma_b, 's_used': r['s_used']}

    if args.mode == 'separate':
        save_separate(ctx, hp, args.out_dir)
    else:
        # Validate panels
        for p in args.panels:
            if p not in PANEL_FUNCS:
                raise ValueError(f'panel không hợp lệ: {p}')
        save_compose(ctx, hp, args.panels, args.out)


if __name__ == '__main__':
    main()
