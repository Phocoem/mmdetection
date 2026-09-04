# -*- coding: utf-8 -*-
"""Trực quan hoá các đại lượng THẬT trong IAPC: I, Î, P_l, P̂_l, M_l, D_l.

Script nạp một checkpoint đã huấn luyện, chạy đúng pipeline IAPC trên một ảnh
test thật, và xuất hình cho từng đại lượng trong công thức consistency.

Sinh hai hình:
  fig_iapc_qualitative.pdf  — lưới đầy đủ: hàng = mức pyramid, cột =
                              P_l, P̂_l, |P_l - P̂_l|, M_l, D_l(x)
  fig_iapc_inputs.pdf       — I, Î, |I - Î|, mask gốc

Cách dùng:
    python tools/research/visualize_iapc.py \\
        configs/fair_lettuce/mask_rcnn_r50_iapc.py \\
        work_dirs/research/mask_rcnn_r50_iapc_lam0p25/seed_2024/best_*.pth \\
        --ann mmdet_dataset/lettuce/annotations/test.json \\
        --img-root mmdet_dataset/lettuce/images/test \\
        --image-id 11 \\
        --out-dir figures_iapc/

Ghi chú:
- Đặc trưng có 256 kênh; để hiển thị ta lấy trung bình |activation| theo kênh
  (cách chuẩn trong literature trực quan hoá feature map).
- D_l(x) là bản đồ khoảng cách THEO VỊ TRÍ, trước khi nhân mask và lấy tổng.
  Công thức trong bài lấy tổng bản đồ này có trọng số M_l rồi chuẩn hoá.
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.structures import InstanceData
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from mmdet.structures.mask import BitmapMasks

try:
    from pycocotools import mask as coco_mask
except ImportError:
    coco_mask = None


# =========================================================================
# Chuẩn bị dữ liệu
# =========================================================================
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def load_coco_sample(ann_path, img_root, image_id=None, target_n=10):
    """Chọn ảnh + dựng foreground union từ annotation COCO."""
    coco = json.load(open(ann_path))
    if image_id is not None:
        info = next((im for im in coco['images'] if im['id'] == image_id),
                    None)
        if info is None:
            raise ValueError(f'image_id {image_id} không tồn tại')
    else:
        cnt = {}
        for a in coco['annotations']:
            cnt[a['image_id']] = cnt.get(a['image_id'], 0) + 1
        cands = [(-abs(cnt.get(im['id'], 0) - target_n), im)
                 for im in coco['images']
                 if os.path.isfile(os.path.join(img_root, im['file_name']))]
        cands.sort(key=lambda t: -t[0])
        info = cands[0][1]

    path = os.path.join(img_root, info['file_name'])
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise RuntimeError(f'không đọc được {path}')
    H, W = img_bgr.shape[:2]

    anns = [a for a in coco['annotations'] if a['image_id'] == info['id']]
    per_inst = []
    for a in anns:
        seg = a['segmentation']
        if isinstance(seg, list):
            if coco_mask is None:
                raise ImportError('cần pycocotools cho polygon annotation')
            m = coco_mask.decode(coco_mask.frPyObjects(seg, H, W))
            if m.ndim == 3:
                m = m.any(axis=2)
        elif isinstance(seg, dict):
            m = coco_mask.decode(seg)
        else:
            continue
        per_inst.append(m.astype(np.uint8))

    if not per_inst:
        raise SystemExit('ảnh này không có instance nào')
    return img_bgr, per_inst, info


def preprocess(img_bgr, size=800):
    """Resize giữ tỉ lệ + pad về bội số 32 + chuẩn hoá, giống pipeline train."""
    H, W = img_bgr.shape[:2]
    scale = size / max(H, W)
    nh, nw = int(round(H * scale)), int(round(W * scale))
    img = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)

    ph, pw = (nh + 31) // 32 * 32, (nw + 31) // 32 * 32
    canvas = np.zeros((ph, pw, 3), np.uint8)
    canvas[:nh, :nw] = img

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32)
    norm = (rgb - MEAN) / STD
    tensor = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0)
    return tensor, canvas, (nh, nw), scale


def resize_masks(per_inst, scale, ph, pw, nh, nw):
    """Resize từng instance mask theo cùng phép biến đổi ảnh."""
    out = []
    for m in per_inst:
        rm = cv2.resize(m, (nw, nh), interpolation=cv2.INTER_NEAREST)
        canvas = np.zeros((ph, pw), np.uint8)
        canvas[:nh, :nw] = rm
        out.append(canvas)
    return out


def denorm(tensor):
    """Đưa tensor đã chuẩn hoá về ảnh RGB uint8 để hiển thị."""
    x = tensor[0].permute(1, 2, 0).cpu().numpy()
    x = x * STD + MEAN
    return np.clip(x, 0, 255).astype(np.uint8)


# =========================================================================
# Bản đồ khoảng cách theo vị trí (phiên bản không gộp của D_l)
# =========================================================================
def distance_map(P, P_hat, alpha=0.5, eps=1e-3):
    """Trả về bản đồ (H,W) khoảng cách lai tại từng vị trí không gian.

    Đây chính là biểu thức trong ngoặc vuông của công thức D_l, TRƯỚC khi
    nhân với M_l và lấy tổng.
    """
    cos = F.cosine_similarity(P, P_hat, dim=1, eps=1e-6)      # (1,H,W)
    cos_term = (1.0 - cos)

    l1 = (P - P_hat).abs().mean(dim=1)                         # (1,H,W)
    scale = P.abs().mean(dim=1).clamp_min(eps)
    l1_term = l1 / scale

    d = alpha * cos_term + (1.0 - alpha) * l1_term
    return d[0].cpu().numpy(), cos_term[0].cpu().numpy(), \
        l1_term[0].cpu().numpy()


def feat_to_image(P):
    """Nén tensor đặc trưng (1,C,H,W) thành bản đồ 2D để hiển thị."""
    return P[0].abs().mean(dim=0).cpu().numpy()


# =========================================================================
# Vẽ
# =========================================================================
def show(ax, arr, title, cmap=None, cbar=False, vmin=None, vmax=None,
         fs=9.5):
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=fs, fontweight='bold', pad=4)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor('#333'); sp.set_linewidth(1.0)
    if cbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, shrink=0.85)
    return im


def fig_inputs(img_clean, img_corr, mask_full, out_path, meta):
    """Hình 1: I, Î, |I - Î|, mask gốc."""
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.1))

    show(axes[0], img_clean, 'Clean image  $I$', fs=11)
    show(axes[1], img_corr, r'Corrupted view  $\hat{I} = T(I)$', fs=11)

    diff = np.abs(img_corr.astype(int) - img_clean.astype(int)).mean(axis=2)
    show(axes[2], diff, r'$|I - \hat{I}|$', cmap='magma', cbar=True, fs=11)

    show(axes[3], mask_full, r'Foreground union  $M$ (dilated)',
         cmap='gray', vmin=0, vmax=1, fs=11)

    fig.suptitle(
        f'IAPC inputs — {meta["name"]}   '
        f'({meta["n_inst"]} instances, {meta["shape"][0]}$\\times$'
        f'{meta["shape"][1]} px)',
        fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.savefig(out_path.replace('.pdf', '.png'), bbox_inches='tight', dpi=200)
    plt.close()
    print(f'  saved {out_path}')


def fig_pyramid(records, out_path, D_scalars, meta):
    """Hình 2: lưới mức × (P_l, P̂_l, |ΔP|, M_l, D_l(x))."""
    n = len(records)
    fig, axes = plt.subplots(n, 5, figsize=(16.5, 3.15 * n))
    if n == 1:
        axes = axes[None, :]

    col_titles = [
        r'$P_l$  (clean feature)',
        r'$\hat{P}_l$  (corrupted feature)',
        r'$|P_l - \hat{P}_l|$',
        r'$M_l$  (resampled mask)',
        r'$D_l(x)$  (hybrid distance)',
    ]

    for r, rec in enumerate(records):
        lvl, Pv, Phv, Mv, Dv = (rec['level'], rec['P'], rec['P_hat'],
                                rec['M'], rec['D'])
        h, w = Pv.shape
        vmax_feat = max(Pv.max(), Phv.max())

        show(axes[r, 0], Pv, col_titles[0] if r == 0 else '',
             cmap='viridis', vmin=0, vmax=vmax_feat, cbar=(r == 0))
        show(axes[r, 1], Phv, col_titles[1] if r == 0 else '',
             cmap='viridis', vmin=0, vmax=vmax_feat, cbar=(r == 0))
        show(axes[r, 2], np.abs(Pv - Phv), col_titles[2] if r == 0 else '',
             cmap='magma', cbar=(r == 0))
        show(axes[r, 3], Mv, col_titles[3] if r == 0 else '',
             cmap='Oranges', vmin=0, vmax=1, cbar=(r == 0))
        show(axes[r, 4], Dv, col_titles[4] if r == 0 else '',
             cmap='inferno', cbar=(r == 0))

        # nhãn hàng
        axes[r, 0].set_ylabel(f'$P_{lvl}$\n{h}$\\times${w}',
                              fontsize=11, fontweight='bold', labelpad=8)
        # D_l vô hướng ghi bên phải
        axes[r, 4].text(1.14, 0.5,
                        f'$D_{lvl} = {D_scalars[r]:.3f}$',
                        transform=axes[r, 4].transAxes,
                        rotation=270, va='center', ha='left',
                        fontsize=10.5, fontweight='bold', color='#15803d')

    fig.suptitle(
        'IAPC pyramid quantities on a real LettuceMOTS image.   '
        r'$D_l$ aggregates $D_l(x)$ weighted by $M_l(x)$.',
        fontsize=12.5, fontweight='bold', y=1.0)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=190)
    plt.savefig(out_path.replace('.pdf', '.png'), bbox_inches='tight', dpi=190)
    plt.close()
    print(f'  saved {out_path}')


# =========================================================================
# MAIN
# =========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('checkpoint')
    ap.add_argument('--ann', required=True)
    ap.add_argument('--img-root', required=True)
    ap.add_argument('--image-id', type=int, default=None)
    ap.add_argument('--out-dir', default='figures_iapc')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--size', type=int, default=800)
    ap.add_argument('--seed', type=int, default=0,
                    help='seed cho corruption sampler (tái lập hình)')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    dev = args.device
    if dev.startswith('cuda') and not torch.cuda.is_available():
        print('! CUDA không có, dùng CPU')
        dev = 'cpu'

    # ---- Build model ----
    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmdet'))
    model = MODELS.build(cfg.model)

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state = ckpt.get('state_dict', ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f'Nạp checkpoint: {len(missing)} khoá thiếu, '
          f'{len(unexpected)} khoá thừa')
    model = model.to(dev).eval()

    is_iapc = hasattr(model, '_make_corrupted_view')
    print(f'Model: {type(model).__name__} '
          f'({"IAPC" if is_iapc else "không phải IAPC — dùng sampler nội bộ"})')

    # ---- Dữ liệu ----
    img_bgr, per_inst, info = load_coco_sample(
        args.ann, args.img_root, args.image_id)
    print(f'Ảnh: {info["file_name"]}, {len(per_inst)} instances')

    x_clean, canvas_bgr, (nh, nw), scale = preprocess(img_bgr, args.size)
    ph, pw = canvas_bgr.shape[:2]
    x_clean = x_clean.to(dev)
    masks_r = resize_masks(per_inst, scale, ph, pw, nh, nw)

    # ---- Sinh ảnh nhiễu ----
    torch.manual_seed(args.seed)
    if is_iapc:
        with torch.no_grad():
            x_corr = model._make_corrupted_view(x_clean)
        dilate_px = model.dilate_mask_px
        levels = model.consistency_levels
        alpha = model.cos_weight
        instance_aware = model.instance_aware
    else:
        # fallback: sampler tương đương nếu checkpoint không phải IAPC
        with torch.no_grad():
            b = x_clean
            d = torch.empty(1, 1, 1, 1, device=dev).uniform_(-0.35, 0.35)
            g = torch.empty(1, 1, 1, 1, device=dev).uniform_(0.55, 1.45)
            mu = b.mean(dim=(1, 2, 3), keepdim=True)
            x_corr = (b + d - mu) * g + mu
            x_corr = x_corr + torch.randn_like(x_corr) * 0.15
        dilate_px, levels, alpha, instance_aware = 8, (0, 1, 2, 3), 0.5, True

    # ---- Trích đặc trưng hai nhánh ----
    with torch.no_grad():
        feats_clean = model.extract_feat(x_clean)
        feats_corr = model.extract_feat(x_corr)
    print(f'Pyramid: {len(feats_clean)} mức, '
          f'kích thước {[tuple(f.shape[-2:]) for f in feats_clean]}')

    # ---- Mask theo mức ----
    M_full = torch.from_numpy(
        np.stack(masks_r).max(axis=0).astype(np.float32)
    )[None, None].to(dev)
    if dilate_px > 0:
        k = 2 * dilate_px + 1
        M_full = F.max_pool2d(M_full, k, stride=1, padding=dilate_px)

    # ---- Tính từng mức ----
    records, D_scalars = [], []
    for li in levels:
        P, Ph = feats_clean[li], feats_corr[li]
        if instance_aware:
            Ml = F.interpolate(M_full, size=P.shape[-2:], mode='bilinear',
                               align_corners=False).clamp(0, 1)
        else:
            Ml = torch.ones((1, 1, *P.shape[-2:]), device=dev)

        Dmap, cos_map, l1_map = distance_map(P, Ph, alpha=alpha)
        Mv = Ml[0, 0].cpu().numpy()

        # D_l vô hướng theo đúng công thức trong bài
        D_scalar = float((Dmap * Mv).sum() / max(Mv.sum(), 1.0))

        records.append(dict(level=li + 2,
                            P=feat_to_image(P), P_hat=feat_to_image(Ph),
                            M=Mv, D=Dmap,
                            cos=cos_map, l1=l1_map))
        D_scalars.append(D_scalar)
        print(f'  P{li+2}: D = {D_scalar:.4f}  '
              f'(mask phủ {100*Mv.mean():.1f}%, '
              f'cos {(cos_map*Mv).sum()/max(Mv.sum(),1):.3f}, '
              f'l1 {(l1_map*Mv).sum()/max(Mv.sum(),1):.3f})')

    w = [1.0] * len(records)
    L_cons = sum(wi * di for wi, di in zip(w, D_scalars)) / sum(w)
    print(f'\nL_cons = {L_cons:.4f}')

    # ---- Vẽ ----
    meta = dict(name=info['file_name'], n_inst=len(per_inst),
                shape=(ph, pw))
    print('\nXuất hình:')
    fig_inputs(denorm(x_clean),
               denorm(x_corr),
               M_full[0, 0].cpu().numpy(),
               os.path.join(args.out_dir, 'fig_iapc_inputs.pdf'), meta)
    fig_pyramid(records,
                os.path.join(args.out_dir, 'fig_iapc_qualitative.pdf'),
                D_scalars, meta)

    # ---- Lưu số liệu để dán vào caption ----
    stats = dict(image=info['file_name'], n_instances=len(per_inst),
                 L_cons=L_cons,
                 levels={f'P{r["level"]}': dict(
                     D=D_scalars[i],
                     mask_coverage=float(r['M'].mean()),
                     resolution=list(r['P'].shape))
                     for i, r in enumerate(records)})
    with open(os.path.join(args.out_dir, 'iapc_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    print(f'  saved {args.out_dir}/iapc_stats.json')


if __name__ == '__main__':
    main()
