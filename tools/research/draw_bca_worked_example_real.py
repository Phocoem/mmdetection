# -*- coding: utf-8 -*-
"""Vẽ Fig BCA worked example bằng ảnh + annotation THẬT (COCO).

Chọn 1 ảnh test đẹp (nhiều cây, tương phản rõ), chạy BCA lên nó, xuất
8-panel hình publication-quality.

Cách dùng:
    python draw_bca_worked_example_real.py \
        --ann mmdet_dataset/lettuce/annotations/test.json \
        --img-root mmdet_dataset/lettuce/images/test \
        --out fig_bca_worked_example_real.pdf \
        --image-id 0                       # tùy chọn: chọn ảnh cụ thể
        --strength 0.55 --band 15 --k-min 0.35 --sigma-b 3.0

Nếu không truyền --image-id, script chọn tự động ảnh có nhiều instance
kích thước vừa phải (dễ nhìn nhất khi minh họa biên).
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
# BCA ops (khớp với transform trong boundary_aug.py)
# =============================================================================
def build_foreground_union(coco_json, image_info, img_shape):
    """Gộp mọi instance mask của 1 ảnh thành binary foreground map."""
    H, W = img_shape[:2]
    img_id = image_info['id']
    anns = [a for a in coco_json['annotations'] if a['image_id'] == img_id]
    M = np.zeros((H, W), dtype=np.uint8)
    for a in anns:
        seg = a['segmentation']
        if isinstance(seg, list):  # polygon
            rles = coco_mask.frPyObjects(seg, H, W)
            m = coco_mask.decode(rles)
            if m.ndim == 3:
                m = m.any(axis=2)
        elif isinstance(seg, dict):  # RLE
            m = coco_mask.decode(seg)
        else:
            continue
        M |= m.astype(np.uint8)
    return M, len(anns)


def bca_transform(img, M, strength, band_width, k_min, sigma_b, rng):
    """Chạy BCA đầy đủ, trả về mọi bước trung gian để minh họa."""
    H, W = img.shape[:2]

    # (1) Boundary set = morphological gradient
    k3 = np.ones((3, 3), np.uint8)
    boundary = cv2.morphologyEx(M, cv2.MORPH_GRADIENT, k3)

    # (2) Distance transform
    d = cv2.distanceTransform(1 - boundary, cv2.DIST_L2, 3)

    # (3) Raw band weight (linear decay)
    w_raw = np.clip(1.0 - d / max(band_width, 1), 0.0, 1.0)

    # (4) Gaussian smoothed band weight
    w = cv2.GaussianBlur(w_raw, (0, 0), sigmaX=sigma_b)
    w = np.clip(w, 0.0, 1.0).astype(np.float32)

    # (5) Local surround (box-mean, K_l = 2B+1)
    Kl = 2 * band_width + 1
    L = cv2.boxFilter(img, -1, (Kl, Kl))

    # (6) Sample strength (nếu chưa đưa cụ thể)
    s = float(strength) if strength is not None else rng.uniform(0.225, 0.45)

    # (7) Keep coefficient
    keep = np.maximum(1.0 - s * w, k_min).astype(np.float32)

    # (8) Output
    out = keep[..., None] * img.astype(np.float32) + \
        (1.0 - keep[..., None]) * L.astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)

    return {'boundary': boundary, 'w_raw': w_raw, 'w': w,
            'L': L, 'k': keep, 'out': out, 's_used': s}


# =============================================================================
# Chọn ảnh minh họa
# =============================================================================
def pick_image(coco_json, img_root, preferred_id=None):
    """Chọn ảnh test có 5-20 instance, kích thước trung bình để dễ minh họa."""
    imgs = coco_json['images']
    if preferred_id is not None:
        for im in imgs:
            if im['id'] == preferred_id:
                return im
        raise ValueError(f'image_id {preferred_id} không tồn tại')

    # Tự chọn: đếm ann mỗi ảnh, ưu tiên 5-15 instance
    ann_count = {}
    for a in coco_json['annotations']:
        ann_count[a['image_id']] = ann_count.get(a['image_id'], 0) + 1
    scored = []
    for im in imgs:
        n = ann_count.get(im['id'], 0)
        # score: gần 10 nhất thì tốt nhất
        score = -abs(n - 10)
        # bonus nếu file tồn tại
        p = os.path.join(img_root, im['file_name'])
        if os.path.isfile(p):
            scored.append((score, im))
    scored.sort(key=lambda x: -x[0])
    if not scored:
        raise SystemExit('Không tìm thấy ảnh nào tồn tại trong img_root')
    return scored[0][1]


# =============================================================================
# Vẽ hình
# =============================================================================
def draw_figure(img_bgr, M, results, out_path, image_name, hp):
    """Vẽ 8-panel như hình synthetic nhưng bằng ảnh thật."""
    H, W = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    L_rgb = cv2.cvtColor(results['L'], cv2.COLOR_BGR2RGB)
    out_rgb = cv2.cvtColor(results['out'], cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.0))

    def show(ax, im, title, cmap=None, cbar=False, vmin=None, vmax=None):
        h = ax.imshow(im, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10.5, fontweight='bold', pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        if cbar:
            plt.colorbar(h, ax=ax, fraction=0.045, pad=0.02, shrink=0.9)

    # Hàng 1
    show(axes[0, 0], img_rgb, '(a) Input image $I$')
    show(axes[0, 1], M, '(b) Foreground union $M$', cmap='gray',
         vmin=0, vmax=1)
    show(axes[0, 2], results['w'], '(c) Boundary band weight $w$',
         cmap='inferno', cbar=True, vmin=0, vmax=1)
    show(axes[0, 3], L_rgb, '(d) Local surround $L$')

    # Hàng 2
    show(axes[1, 0], results['k'],
         f"(e) Keep coefficient $k$\n$(s={results['s_used']:.2f},"
         f"\\ k_{{\\min}}={hp['k_min']})$",
         cmap='viridis', cbar=True, vmin=hp['k_min'], vmax=1)
    show(axes[1, 1], out_rgb, "(f) Output $I' = kI + (1-k)L$")

    # (g) Profile — chọn dòng đi qua nhiều biên nhất
    boundary = results['boundary']
    boundary_per_row = boundary.sum(axis=1)
    y_row = int(np.argmax(boundary_per_row))
    row_orig = img_bgr[y_row, :, 1].astype(float)   # green channel
    row_out = results['out'][y_row, :, 1].astype(float)
    axes[1, 2].plot(row_orig, color='#1976d2', lw=1.3, label='original',
                    alpha=0.9)
    axes[1, 2].plot(row_out, color='#d32f2f', lw=1.3, label='BCA output',
                    alpha=0.9)
    # Shade nơi w > 0.1 (dải biên)
    w_row = results['w'][y_row, :]
    band_mask = w_row > 0.1
    if band_mask.any():
        # tô sáng các đoạn liên tục
        in_band = False
        seg_start = 0
        for x in range(W):
            if band_mask[x] and not in_band:
                seg_start = x; in_band = True
            elif not band_mask[x] and in_band:
                axes[1, 2].axvspan(seg_start, x, alpha=0.15, color='orange')
                in_band = False
        if in_band:
            axes[1, 2].axvspan(seg_start, W, alpha=0.15, color='orange')
    axes[1, 2].set_title(
        f'(g) Intensity profile at $y{{=}}{y_row}$\n'
        'orange bands mark BCA-affected regions',
        fontsize=10.5, fontweight='bold', pad=6)
    axes[1, 2].set_xlabel('x (pixel)', fontsize=8.5)
    axes[1, 2].set_ylabel('green channel', fontsize=8.5)
    axes[1, 2].legend(fontsize=8.5, loc='best')
    axes[1, 2].tick_params(labelsize=8)
    axes[1, 2].grid(alpha=0.2)

    # (h) diff heatmap — chứng minh chỉ vùng biên bị tác động
    diff = np.abs(results['out'].astype(int) - img_bgr.astype(int)).mean(
        axis=2)
    vmax = max(np.percentile(diff, 99), 15)
    im8 = axes[1, 3].imshow(diff, cmap='hot', vmin=0, vmax=vmax)
    axes[1, 3].set_title(
        "(h) $|I' - I|$ (per-pixel mean)\n"
        'concentrated in boundary band',
        fontsize=10.5, fontweight='bold', pad=6)
    axes[1, 3].set_xticks([]); axes[1, 3].set_yticks([])
    plt.colorbar(im8, ax=axes[1, 3], fraction=0.045, pad=0.02, shrink=0.9)

    # Suptitle nhỏ với thông tin
    fig.suptitle(
        f'BCA on real LettuceMOTS test image: {image_name}   '
        f'({H}$\\times${W} px, $B={hp["band"]}$, '
        f'$k_{{\\min}}={hp["k_min"]}$, $\\sigma_b={hp["sigma_b"]}$)',
        fontsize=11, y=1.01)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.savefig(out_path.replace('.pdf', '.png'),
                bbox_inches='tight', dpi=200)
    plt.close()
    print(f'Đã lưu: {out_path}')
    print(f'         {out_path.replace(".pdf", ".png")}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ann', required=True,
                    help='đường dẫn test.json (COCO annotation)')
    ap.add_argument('--img-root', required=True,
                    help='thư mục chứa ảnh test (COCO images)')
    ap.add_argument('--image-id', type=int, default=None,
                    help='chọn ảnh cụ thể; nếu bỏ, script tự chọn')
    ap.add_argument('--out', default='fig_bca_worked_example_real.pdf')
    ap.add_argument('--strength', type=float, default=0.55,
                    help='strength s cho minh họa (0.5-0.7 hình đẹp nhất)')
    ap.add_argument('--band', type=int, default=15,
                    help='band radius B')
    ap.add_argument('--k-min', type=float, default=0.35)
    ap.add_argument('--sigma-b', type=float, default=3.0)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    coco = json.load(open(args.ann))
    print(f'Loaded {len(coco["images"])} images, '
          f'{len(coco["annotations"])} annotations')

    im_info = pick_image(coco, args.img_root, args.image_id)
    img_path = os.path.join(args.img_root, im_info['file_name'])
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f'không tìm thấy: {img_path}')
    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError(f'không đọc được: {img_path}')
    print(f'Ảnh: {im_info["file_name"]} shape={img.shape}, id={im_info["id"]}')

    M, n_ann = build_foreground_union(coco, im_info, img.shape)
    print(f'  {n_ann} instance masks, foreground coverage '
          f'{100.0 * M.mean():.1f}%')
    if n_ann == 0:
        raise SystemExit('Ảnh này không có annotation — chọn ảnh khác')

    rng = np.random.default_rng(args.seed)
    results = bca_transform(img, M,
                            strength=args.strength,
                            band_width=args.band,
                            k_min=args.k_min,
                            sigma_b=args.sigma_b,
                            rng=rng)

    hp = {'band': args.band, 'k_min': args.k_min, 'sigma_b': args.sigma_b}
    draw_figure(img, M, results, args.out, im_info['file_name'], hp)


if __name__ == '__main__':
    main()
