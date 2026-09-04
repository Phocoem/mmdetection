"""
Ve heatmap grid theo cau truc:
    - COT: cac dieu kien (Clean, Brightness S3, Contrast S3, Noise S3) -
      DUNG CUNG 1 anh goc nhung o 4 phien ban corruption khac nhau.
    - HANG: tung cap (Baseline P_l, IAPC P_l) cho l = 2,3,4,5, xep sat
      nhau de so sanh truc tiep cung tang / cung dieu kien.
    - Cot dau tien: anh Input (moi phien ban corruption).

Muc dich: cot "Contrast S3" lam noi bat dung cho IAPC that bai (Section
5.2 - regularizer discards magnitude info under contrast reduction),
dat ngay canh Baseline cung tang de nguoi doc thay khac biet truc tiep.

*** THANG MAU CHUNG: chuan hoa min/max chung cho tung (tang x dieu kien)
tren CA Baseline lan IAPC, de mau sac so sanh duoc dung cuong do giua 2
he thong - KHONG tu chuan hoa rieng tung o (se gay ao giac). ***

Cach dung:
    python plot_grid_heatmap.py \\
        --baseline-config configs/fair_lettuce/mask_rcnn_r50_fpn.py \\
        --baseline-ckpt "work_dirs/research/mask_rcnn_r50_fpn/seed_2024/best_coco_segm_mAP_epoch_*.pth" \\
        --iapc-config configs/fair_lettuce/mask_rcnn_r50_iapc_lam0p25.py \\
        --iapc-ckpt "work_dirs/research/mask_rcnn_r50_iapc_lam0p25/seed_2024/best_coco_segm_mAP_epoch_*.pth" \\
        --clean-image  mmdet_dataset/lettuce/images/0011_000010.png \\
        --bright-image mmdet_dataset/lettuce_c/images/brightness_s3/0011_000010.png \\
        --contrast-image mmdet_dataset/lettuce_c/images/contrast_s3/0011_000010.png \\
        --noise-image  mmdet_dataset/lettuce_c/images/gaussian_noise_s3/0011_000010.png \\
        --output grid_heatmap.png
"""
import argparse
import glob

import cv2
import numpy as np
import torch

# PyTorch >=2.6: ep weights_only=False (checkpoint MMEngine chua HistoryBuffer)
_orig_load = torch.load
def _patched_load(*a, **k):
    k.setdefault('weights_only', False)
    return _orig_load(*a, **k)
torch.load = _patched_load

from mmdet.apis import init_detector
from mmengine.dataset import Compose, pseudo_collate

PYRAMID_LEVELS = ['P2', 'P3', 'P4', 'P5']


def resolve_ckpt(pattern):
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise SystemExit(f'Khong tim thay checkpoint: {pattern}')
    return matches[-1]


def load_model(cfg, ckpt, device):
    m = init_detector(cfg, resolve_ckpt(ckpt), device=device)
    m.eval()
    return m


def extract_feats(model, image_path):
    cfg = model.cfg.copy()
    pipeline = Compose(cfg.test_pipeline)
    data = pipeline(dict(img_path=image_path, img_id=0))
    batch = pseudo_collate([data])
    with torch.no_grad():
        processed = model.data_preprocessor(batch, False)
        feats = model.extract_feat(processed['inputs'])
    return feats


def raw_heat(feat_level):
    """(1,C,H,W) -> (H,W) magnitude trung binh |activation| theo kenh."""
    fmap = feat_level[0].detach().float().cpu()
    return fmap.abs().mean(dim=0).numpy()


def overlay(heat_norm, orig_bgr):
    h, w = orig_bgr.shape[:2]
    resized = cv2.resize(heat_norm, (w, h))
    color = cv2.applyColorMap((resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(orig_bgr, 0.45, color, 0.55, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-config', required=True)
    parser.add_argument('--baseline-ckpt', required=True)
    parser.add_argument('--iapc-config', required=True)
    parser.add_argument('--iapc-ckpt', required=True)
    parser.add_argument('--clean-image', required=True)
    parser.add_argument('--bright-image', required=True)
    parser.add_argument('--contrast-image', required=True)
    parser.add_argument('--noise-image', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--levels', nargs='+', default=['P4', 'P2'],
                         help='Cac tang pyramid muon ve (mac dinh P4 P2 - '
                              'noi artifact luoi duoi Contrast S3 ro nhat). '
                              'Vd: --levels P4 hoac --levels P4 P2 P5')
    parser.add_argument('--highlight-condition', default='Contrast S3',
                         help='Cot dieu kien duoc lam noi bat (vien do + '
                              'tieu de dam). Mac dinh "Contrast S3".')
    args = parser.parse_args()

    # Chi giu cac tang duoc chon (Phuong an A: thu hep con P4+P2, tranh
    # hinh qua lon va tranh minh hoa qua nhieu cho "IAPC sang hon" - von
    # de gay hieu nham "IAPC tot hon", mau thuan ket luan null cua bai).
    sel_levels = args.levels
    for lv in sel_levels:
        if lv not in PYRAMID_LEVELS:
            raise SystemExit(f'Tang khong hop le: {lv}. Chon trong '
                              f'{PYRAMID_LEVELS}.')
    sel_level_idx = [PYRAMID_LEVELS.index(lv) for lv in sel_levels]

    conditions = [
        ('Clean', args.clean_image),
        ('Brightness S3', args.bright_image),
        ('Contrast S3', args.contrast_image),
        ('Noise S3', args.noise_image),
    ]

    print('Load Baseline ...')
    baseline = load_model(args.baseline_config, args.baseline_ckpt, args.device)
    print('Load IAPC ...')
    iapc = load_model(args.iapc_config, args.iapc_ckpt, args.device)

    # Thu thap raw heat: [system][condition][level] = (H,W)
    origs = {}
    raw = {'Baseline': {}, 'IAPC': {}}
    for cond_name, img_path in conditions:
        orig = cv2.imread(img_path)
        if orig is None:
            raise SystemExit(f'Khong doc duoc anh: {img_path}')
        origs[cond_name] = orig
        for sys_name, model in [('Baseline', baseline), ('IAPC', iapc)]:
            feats = extract_feats(model, img_path)
            raw[sys_name][cond_name] = [raw_heat(feats[j])
                                         for j in range(len(PYRAMID_LEVELS))]
            print(f'  {sys_name} / {cond_name}: xong {len(PYRAMID_LEVELS)} tang')

    # Thang chuan hoa CHUNG cho tung (level, condition) tren CA 2 he thong
    scales = {}  # (level_idx, cond_name) -> (vmin, vmax)
    for j in range(len(PYRAMID_LEVELS)):
        for cond_name, _ in conditions:
            vals = np.concatenate([
                raw['Baseline'][cond_name][j].ravel(),
                raw['IAPC'][cond_name][j].ravel()])
            scales[(j, cond_name)] = (vals.min(), vals.max())

    # ---- Ghep luoi (chi cac tang da chon) ----
    sample_orig = origs['Clean']
    cell_h, cell_w = sample_orig.shape[:2]
    n_cols = 1 + len(conditions)          # + cot Input
    row_pairs = len(sel_level_idx)        # so tang DA CHON
    n_rows = row_pairs * 2                # moi tang 2 hang
    label_h, row_label_w = 46, 200
    pair_gap = 24                          # khoang trong giua cac cap tang

    grid_h = label_h + n_rows * cell_h + row_pairs * pair_gap
    grid_w = row_label_w + n_cols * cell_w
    grid = np.full((grid_h, grid_w, 3), 255, dtype=np.uint8)

    highlight_col = None
    for jc, (cond_name, _) in enumerate(conditions):
        if cond_name == args.highlight_condition:
            highlight_col = jc + 1  # +1 vi cot 0 la Input

    # Tieu de cot (cot highlight in DAM + mau do)
    col_titles = ['Input'] + [c[0] for c in conditions]
    for jc, title in enumerate(col_titles):
        x0 = row_label_w + jc * cell_w
        is_hl = (jc == highlight_col)
        thickness = 3 if is_hl else 2
        scale = 1.0 if is_hl else 0.9
        color = (0, 0, 200) if is_hl else (0, 0, 0)
        (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        cv2.putText(grid, title, (x0 + cell_w // 2 - tw // 2, label_h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

    y = label_h
    for j in sel_level_idx:
        level = PYRAMID_LEVELS[j]
        for sys_name in ['Baseline', 'IAPC']:
            row_label = f'{sys_name} {level}'
            cv2.putText(grid, row_label, (10, y + cell_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2)
            grid[y:y + cell_h, row_label_w:row_label_w + cell_w] = origs['Clean']
            for jc, (cond_name, _) in enumerate(conditions):
                vmin, vmax = scales[(j, cond_name)]
                heat = raw[sys_name][cond_name][j]
                norm = np.clip((heat - vmin) / max(vmax - vmin, 1e-8), 0, 1)
                ov = overlay(norm, origs[cond_name])
                x0 = row_label_w + (jc + 1) * cell_w
                grid[y:y + cell_h, x0:x0 + cell_w] = ov
            y += cell_h
        y += pair_gap

    # Vien do bao quanh toan bo cot highlight (huong mat den cot chan doan)
    if highlight_col is not None:
        hx0 = row_label_w + highlight_col * cell_w
        cv2.rectangle(grid, (hx0 - 2, label_h - 2),
                      (hx0 + cell_w + 2, y), (0, 0, 220), 4)

    cv2.imwrite(args.output, grid)
    print(f'\nDa luu: {args.output}')
    print('\nGoi y caption:')
    print('"Feature-magnitude overlays (channel-averaged |activation|) for the')
    print('plain baseline versus IAPC, paired at each pyramid level P2-P5,')
    print('across four test conditions. Colour is normalized to a SHARED scale')
    print('per (level, condition) so intensity is comparable between the two')
    print('systems. The Contrast S3 column is the diagnostic one: it is the')
    print('condition under which the consistency constraint is expected to')
    print('suppress the magnitude information the detector must retain')
    print('(Section 5.2). Colour reflects activation magnitude, NOT segmentation')
    print('quality, and should be read with the quantitative AP results."')


if __name__ == '__main__':
    main()
