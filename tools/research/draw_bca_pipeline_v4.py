# -*- coding: utf-8 -*-
"""Pipeline BCA — layout đơn tuyến trục dọc, có mini-preview mỗi bước.

Ý tưởng: mọi thứ chảy TỪ TRÊN XUỐNG theo 1 trục dọc trung tâm. Bên phải
mỗi block là 1 mini-preview (thumbnail thật) minh họa "block này biến gì
thành gì". Không có mũi tên chéo, không có nhánh song song lẫn lộn.

Cách dùng:
    python draw_bca_pipeline_v4.py \
        --ann test.json --img-root images \
        --image-id 11 \
        --out fig_bca_pipeline_v4.pdf
"""
import argparse, json, os
import cv2
import numpy as np
from pycocotools import mask as coco_mask

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


# ---- BCA util (giống các script trước) ----
def bca_transform(img, M, strength, band_width, k_min, sigma_b):
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
            'out': out}


def build_M(coco, im_info, shape):
    H, W = shape[:2]
    anns = [a for a in coco['annotations'] if a['image_id'] == im_info['id']]
    M = np.zeros((H, W), np.uint8)
    for a in anns:
        seg = a['segmentation']
        if isinstance(seg, list):
            m = coco_mask.decode(coco_mask.frPyObjects(seg, H, W))
            if m.ndim == 3: m = m.any(axis=2)
        else:
            m = coco_mask.decode(seg)
        M |= m.astype(np.uint8)
    return M


def pick(coco, img_root, pid):
    if pid is not None:
        for im in coco['images']:
            if im['id'] == pid: return im
    counts = {}
    for a in coco['annotations']:
        counts[a['image_id']] = counts.get(a['image_id'], 0) + 1
    cands = [(- abs(counts.get(im['id'], 0) - 10), im)
             for im in coco['images']
             if os.path.isfile(os.path.join(img_root, im['file_name']))]
    cands.sort(key=lambda x: -x[0])
    return cands[0][1]


# ---- Vẽ pipeline ----
def draw(ctx, out_path):
    """Layout: cột trái = pipeline block, cột phải = thumbnail thật."""
    fig = plt.figure(figsize=(11, 12))

    # Bố cục: chia dọc thành 6 hàng cho 6 block; mỗi hàng gồm 2 cột
    #   left: label + công thức, right: thumbnail
    n_rows = 6
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(n_rows, 2, figure=fig, width_ratios=[1.2, 1],
                  hspace=0.35, wspace=0.15,
                  left=0.05, right=0.97, top=0.94, bottom=0.03)

    def label_ax(row):
        ax = fig.add_subplot(gs[row, 0])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('off')
        return ax

    def img_ax(row):
        return fig.add_subplot(gs[row, 1])

    def block(ax, num, title, formula, color):
        """Vẽ 1 khối bên cột trái: số bước tròn + tiêu đề + công thức."""
        # Nền
        rect = FancyBboxPatch((0.02, 0.15), 0.96, 0.7,
                              boxstyle='round,pad=0.03',
                              facecolor=color, edgecolor='#333',
                              linewidth=1.4)
        ax.add_patch(rect)
        # Số bước
        circ = plt.Circle((0.11, 0.5), 0.10, facecolor='#1a4d8a',
                          edgecolor='white', linewidth=2, zorder=5)
        ax.add_patch(circ)
        ax.text(0.11, 0.5, str(num), ha='center', va='center',
                fontsize=15, color='white', fontweight='bold', zorder=6)
        # Title
        ax.text(0.25, 0.65, title, fontsize=13, fontweight='bold',
                va='center')
        # Formula
        ax.text(0.25, 0.35, formula, fontsize=11, va='center', style='italic',
                color='#333')

    def arrow_down(row, ax_bottom=None):
        """Vẽ mũi tên đi xuống từ hàng row sang hàng row+1 giữa 2 cột."""
        # Dùng matplotlib figure-level annotation với normalized coords
        # Vị trí: giữa 2 hàng, cột trái
        pass  # đã đủ rõ ràng nhờ thứ tự dọc; không cần mũi tên

    def show_img(ax, im, title, cmap=None, vmin=None, vmax=None):
        if im.ndim == 3:  # RGB
            ax.imshow(im, vmin=vmin, vmax=vmax)
        else:
            ax.imshow(im, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10, fontweight='bold', pad=4)
        ax.set_xticks([]); ax.set_yticks([])
        # Add border
        for spine in ax.spines.values():
            spine.set_edgecolor('#333')
            spine.set_linewidth(1.2)

    # =================================================================
    # BLOCK 1 — INPUT
    # =================================================================
    ax = label_ax(0)
    block(ax, 0, 'Inputs',
          'Image $I$  +  ground-truth mask $M$', '#e3f2fd')
    ax_img = img_ax(0)
    show_img(ax_img, ctx['img_rgb'], 'Image $I$')

    # =================================================================
    # BLOCK 2 — BOUNDARY BAND WEIGHT (từ M)
    # =================================================================
    ax = label_ax(1)
    block(ax, 1, 'Compute band weight $w$',
          'Boundary → distance $d$ → $w = 1 - d/B$, then Gaussian smooth',
          '#fff3e0')
    ax_img = img_ax(1)
    show_img(ax_img, ctx['w'], 'Band weight $w$', cmap='inferno',
             vmin=0, vmax=1)

    # =================================================================
    # BLOCK 3 — LOCAL SURROUND (từ I)
    # =================================================================
    ax = label_ax(2)
    block(ax, 2, 'Compute local surround $L$',
          'Box-mean filter of $I$ with kernel $K_\\ell = 2B{+}1$',
          '#e1f5fe')
    ax_img = img_ax(2)
    show_img(ax_img, cv2.cvtColor(ctx['L'], cv2.COLOR_BGR2RGB),
             'Local surround $L$')

    # =================================================================
    # BLOCK 4 — KEEP COEFFICIENT (từ w)
    # =================================================================
    ax = label_ax(3)
    block(ax, 3, 'Compute keep coefficient $k$',
          '$s \\sim \\mathcal{U}(s_{\\min}, s_{\\max})$;   '
          '$k = \\max(1 - s\\,w,\\ k_{\\min})$',
          '#f3e5f5')
    ax_img = img_ax(3)
    show_img(ax_img, ctx['k'], 'Keep coeff. $k$', cmap='viridis',
             vmin=ctx['k'].min(), vmax=1)

    # =================================================================
    # BLOCK 5 — BLEND (kết hợp I, L với k)
    # =================================================================
    ax = label_ax(4)
    block(ax, 4, 'Blend $I$ and $L$ with $k$',
          "$I' = k \\cdot I + (1 - k) \\cdot L$",
          '#e8f5e9')
    ax_img = img_ax(4)
    show_img(ax_img, cv2.cvtColor(ctx['out'], cv2.COLOR_BGR2RGB),
             "Output $I'$")

    # =================================================================
    # BLOCK 6 — STOCHASTIC BYPASS
    # =================================================================
    ax = label_ax(5)
    block(ax, 5, 'Stochastic bypass',
          'With prob $1 - p_{\\text{BCA}}$: skip everything and return $I$',
          '#fce4ec')
    ax_img = img_ax(5)
    # Panel này: diff heatmap để cho thấy "chỉ vùng biên bị tác động"
    diff = np.abs(ctx['out'].astype(int) - ctx['img_bgr'].astype(int)).mean(
        axis=2)
    vmax = max(np.percentile(diff, 99), 15)
    im = ax_img.imshow(diff, cmap='hot', vmin=0, vmax=vmax)
    ax_img.set_title("$|I' - I|$ diff heatmap", fontsize=10,
                     fontweight='bold', pad=4)
    ax_img.set_xticks([]); ax_img.set_yticks([])
    for spine in ax_img.spines.values():
        spine.set_edgecolor('#333')
        spine.set_linewidth(1.2)

    # Suptitle
    fig.suptitle('BoundaryContrastAugmentation (BCA)  —  '
                 'training-time transform, 6 steps',
                 fontsize=14, fontweight='bold', y=0.98)

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
    ap.add_argument('--out', default='fig_bca_pipeline_v4.pdf')
    ap.add_argument('--strength', type=float, default=0.55)
    ap.add_argument('--band', type=int, default=15)
    ap.add_argument('--k-min', type=float, default=0.35)
    ap.add_argument('--sigma-b', type=float, default=3.0)
    args = ap.parse_args()

    coco = json.load(open(args.ann))
    im_info = pick(coco, args.img_root, args.image_id)
    img_bgr = cv2.imread(os.path.join(args.img_root, im_info['file_name']))
    print(f'Ảnh: {im_info["file_name"]}, shape={img_bgr.shape}')
    M = build_M(coco, im_info, img_bgr.shape)

    r = bca_transform(img_bgr, M, args.strength, args.band, args.k_min,
                      args.sigma_b)
    ctx = {
        'img_bgr': img_bgr,
        'img_rgb': cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
        'M': M, **r
    }
    draw(ctx, args.out)


if __name__ == '__main__':
    main()
