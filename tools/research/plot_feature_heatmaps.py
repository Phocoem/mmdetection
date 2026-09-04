"""
Ve heatmap feature (kieu Grad-CAM) tren cac tang pyramid P2-P5, so sanh
Mask R-CNN baseline voi cac gia tri lambda cua IAPC - TREN CUNG 1 anh dau
vao. Muc dich: truc quan hoa hien tuong Section 5.2 cua bai ("consistency
regularizer discards magnitude information under contrast reduction").

THIET KE (xem giai thich day du trong tin nhan):
    - Hang: baseline, IAPC control (l=0), l=0.1, l=0.25 (full), l=0.5,
      l=1.0 - 6 hang.
    - Cot: P2, P3, P4, P5 (KHONG lay P6 - IAPC chi ap dung consistency
      len P2-P5, dua P6 vao se gay hieu lam).
    - NEN dung anh o dieu kien contrast_s3 (--image tro toi anh trong
      mmdet_dataset/lettuce_c/.../contrast_s3/), khong dung anh sach.
    - Co --diff de them 1 hang "|baseline - IAPC full|" o cuoi.

Cach dung (LUU Y: dung dau "|" de phan cach Ten|Config|Checkpoint, KHONG
PHAI "=" va ":" - vi ten he thong nhu "IAPC l=1.0" da chua san dau "="):
    python plot_feature_heatmaps.py \\
        --image mmdet_dataset/lettuce_c/some_image_contrast_s3.jpg \\
        --system-configs \\
            "Baseline|configs/fair_lettuce/mask_rcnn_r50_fpn.py|work_dirs/research/mask_rcnn_r50_fpn/seed_2024/best_coco_segm_mAP_epoch_*.pth" \\
            "IAPC l=0|configs/fair_lettuce/mask_rcnn_r50_gpuaug.py|work_dirs/research/mask_rcnn_r50_gpuaug/seed_2024/best_coco_segm_mAP_epoch_*.pth" \\
            "IAPC l=0.25|configs/fair_lettuce/mask_rcnn_r50_iapc_lam0p25.py|work_dirs/research/mask_rcnn_r50_iapc_lam0p25/seed_2024/best_coco_segm_mAP_epoch_*.pth" \\
            "IAPC l=1.0|configs/fair_lettuce/mask_rcnn_r50_iapc.py|work_dirs/research/mask_rcnn_r50_iapc/seed_2024/best_coco_segm_mAP_epoch_*.pth" \\
        --output feature_heatmap_lambda.png \\
        --diff Baseline "IAPC l=0.25"

Yeu cau: chay tren may co mmdet + checkpoint that (script nay dung
mmdet.apis.init_detector va model.extract_feat, khong chay duoc trong moi
truong khong co mmdet/torch/GPU).
"""
import argparse
import csv
import glob
from pathlib import Path

import cv2
import numpy as np
import torch

# --- SUA LOI: PyTorch >=2.6 doi mac dinh weights_only=True cho torch.load,
# trong khi checkpoint MMEngine (chua ca HistoryBuffer, khong chi tensor)
# bi chan boi thay doi nay. Vi day la checkpoint TU MINH TRAIN RA (nguon
# dang tin cay), ep weights_only=False truoc khi goi init_detector. ---
_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

from mmdet.apis import init_detector
from mmengine.dataset import Compose, pseudo_collate

PYRAMID_LEVELS = ['P2', 'P3', 'P4', 'P5']  # dung 4 tang IAPC that su dung
                                             # (consistency_levels=(0,1,2,3))


def load_model(config_path, checkpoint_path, device):
    model = init_detector(config_path, checkpoint_path, device=device)
    model.eval()
    return model


def extract_features(model, image_path):
    """Chay anh qua data_preprocessor + backbone/neck, tra ve tuple feature
    map tung tang pyramid (giong het cach consistency_mask_rcnn.py tu goi
    self.extract_feat noi bo)."""
    cfg = model.cfg.copy()
    test_pipeline = Compose(cfg.test_pipeline)
    data = dict(img_path=image_path, img_id=0)
    data = test_pipeline(data)
    data_batch = pseudo_collate([data])
    with torch.no_grad():
        processed = model.data_preprocessor(data_batch, False)
        # SUA LOI: data_preprocessor tra ve DICT {'inputs':..., 'data_samples':...},
        # KHONG PHAI tuple - unpack truc tiep "a, b = processed" se lay nham
        # KEY (chuoi "inputs") thay vi gia tri tensor thuc su.
        batch_img = processed['inputs']
        feats = model.extract_feat(batch_img)
    return feats


def feat_to_heatmap(feat, orig_bgr, fixed_scale=None):
    """1 feature map (1,C,H,W) -> anh overlay mau BGR uint8.
    Neu fixed_scale=(vmin,vmax) duoc truyen vao, chuan hoa theo thang CHUNG
    (can thiet khi so sanh nhieu mo hinh voi nhau - neu moi anh tu chuan
    hoa rieng [0,1] thi khong so sanh CUONG DO giua cac mo hinh duoc)."""
    fmap = feat[0].detach().float().cpu()
    heat = fmap.abs().mean(dim=0).numpy()
    if fixed_scale is None:
        vmin, vmax = heat.min(), heat.max()
    else:
        vmin, vmax = fixed_scale
    heat = (heat - vmin) / max(vmax - vmin, 1e-8)
    heat = np.clip(heat, 0, 1)
    h, w = orig_bgr.shape[:2]
    heat_resized = cv2.resize(heat, (w, h))
    heat_color = cv2.applyColorMap((heat_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(orig_bgr, 0.45, heat_color, 0.55, 0), heat_resized


def diff_heatmap(heat_a, heat_b, orig_bgr):
    """|heat_a - heat_b| (da chuan hoa [0,1] tu truoc) -> overlay."""
    diff = np.abs(heat_a - heat_b)
    diff = diff / max(diff.max(), 1e-8)
    diff_color = cv2.applyColorMap((diff * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(orig_bgr, 0.45, diff_color, 0.55, 0)


def parse_systems(items):
    """Dinh dang: "Ten he thong|duong_dan_config.py|checkpoint_pattern"
    - dung "|" lam dau phan cach vi ten he thong (vd "IAPC l=1.0") co the
    chua dau "=" - dung "=" lam dau phan cach se cat SAI vi tri va con
    lam TRUNG KEY giua cac ten "IAPC l=0", "IAPC l=0.25", "IAPC l=1.0"
    (tat ca deu bat dau "IAPC l" truoc dau "=" dau tien), khien cac muc
    sau GHI DE mat cac muc truoc trong dict."""
    systems = {}
    for item in items:
        parts = item.split('|')
        if len(parts) != 3:
            raise SystemExit(
                f'Sai dinh dang: "{item}"\n'
                f'Can dung dung 3 phan cach boi "|": '
                f'"Ten|duong_dan_config.py|checkpoint_pattern"')
        name, cfg_path, ckpt_pattern = parts
        if name in systems:
            raise SystemExit(f'Ten he thong bi TRUNG: "{name}" xuat hien '
                              f'2 lan trong --system-configs.')
        matches = sorted(glob.glob(ckpt_pattern))
        if not matches:
            raise SystemExit(f'Khong tim thay checkpoint khop: {ckpt_pattern}')
        systems[name] = (cfg_path, matches[-1])
    return systems


def build_grid(systems, image_path, output_path, device, diff_pair=None,
                per_row_normalize=False):
    orig = cv2.imread(image_path)
    if orig is None:
        raise SystemExit(f'Khong doc duoc anh: {image_path}')

    # Buoc 1: chay tat ca model, thu thap RAW heat (chua chuan hoa) de
    # tinh thang chuan hoa CHUNG cho tung tang pyramid (cong bang giua
    # cac hang khi so sanh mau sac).
    raw_heats = {}   # name -> [heat_P2, heat_P3, ...] (chua chuan hoa)
    for name, (cfg_path, ckpt_path) in systems.items():
        print(f'Dang chay: {name} ({cfg_path}) ...')
        model = load_model(cfg_path, ckpt_path, device)
        feats = extract_features(model, image_path)
        heats = []
        for f in feats[:len(PYRAMID_LEVELS)]:
            fmap = f[0].detach().float().cpu()
            heats.append(fmap.abs().mean(dim=0).numpy())
        raw_heats[name] = heats
        del model
        torch.cuda.empty_cache()

    # *** KIEM TRA QUAN TRONG: in ra magnitude TRUC TIEP (chua chuan hoa)
    # de kiem chung xem "Baseline nhin phang" co phai do magnitude thuc su
    # thap hon nhieu so voi lambda cao, hay chi la ao giac do thang mau
    # chung. Neu magnitude tang manh theo lambda, day la dau hieu truc
    # tiep cua hien tuong "magnitude collapse" ma Section 3.5.4 canh bao. ***
    print('\n' + '=' * 70)
    print('MAGNITUDE TRUNG BINH |activation| THUC TE (chua chuan hoa mau)')
    print('=' * 70)
    header = f'{"He thong":<16}' + ''.join(f'{lv:>12}' for lv in PYRAMID_LEVELS)
    print(header)
    for name in systems:
        row = f'{name:<16}'
        for j in range(len(PYRAMID_LEVELS)):
            row += f'{raw_heats[name][j].mean():>12.4f}'
        print(row)
    print('=' * 70)
    print('Neu cot nao co gia tri tang manh theo lambda (vd Baseline < l=0 '
          '< l=0.25 < l=1.0), day la bang chung SO cho hien tuong magnitude '
          'tang theo lambda - khop voi canh bao Section 3.5.4 ve "network '
          'can drive Lcons toward zero by uniformly scaling activations".\n')

    # *** SUA: tu dong luu bang nay ra CSV (cung ten voi anh output, doi
    # duoi .csv) - tranh phai copy tay tu terminal moi lan chay. Dung de
    # ve plot_magnitude_vs_ap.py sau nay. ***
    magnitude_csv_path = str(Path(output_path).with_suffix('')) + '_magnitude.csv'
    with open(magnitude_csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['System'] + PYRAMID_LEVELS)
        for name in systems:
            w.writerow([name] + [f'{raw_heats[name][j].mean():.6f}'
                                  for j in range(len(PYRAMID_LEVELS))])
    print(f'Da luu bang magnitude ra CSV: {magnitude_csv_path}\n')

    # Thang chuan hoa CHUNG cho moi cot (tang pyramid), tinh tren TAT CA
    # cac hang, de mau sac so sanh duoc dung cuong do giua cac mo hinh.
    scales = []
    for j in range(len(PYRAMID_LEVELS)):
        all_vals = np.concatenate([raw_heats[n][j].ravel() for n in systems])
        scales.append((all_vals.min(), all_vals.max()))

    rows_img, row_labels, normed_heats = [], [], {}
    for name in systems:
        row_imgs, row_normed = [], []
        for j in range(len(PYRAMID_LEVELS)):
            heat = raw_heats[name][j]
            if per_row_normalize:
                # SUA: chuan hoa RIENG tung hang (0-1 theo chinh no) - mat
                # kha nang so sanh cuong do TUYET DOI giua cac hang, nhung
                # hien ro cau truc khong gian BEN TRONG tung hang (vd
                # Baseline/l=0 co the dang bi "phang" do thang chung).
                vmin, vmax = heat.min(), heat.max()
            else:
                vmin, vmax = scales[j]
            normed = np.clip((heat - vmin) / max(vmax - vmin, 1e-8), 0, 1)
            h, w = orig.shape[:2]
            resized = cv2.resize(normed, (w, h))
            color = cv2.applyColorMap((resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(orig, 0.45, color, 0.55, 0)
            row_imgs.append(overlay)
            row_normed.append(resized)
        rows_img.append(row_imgs)
        row_labels.append(name)
        normed_heats[name] = row_normed

    if diff_pair:
        name_a, name_b = diff_pair
        diff_row = [diff_heatmap(normed_heats[name_a][j], normed_heats[name_b][j], orig)
                    for j in range(len(PYRAMID_LEVELS))]
        rows_img.append(diff_row)
        row_labels.append(f'|{name_a} - {name_b}|')

    # ---- Ghep luoi - THEM cot "Input" dau tien (anh goc, KHONG overlay),
    # nhan hang kieu (a)/(b)/(c)... giong dung phong cach anh mau ----
    n_rows, n_cols_heat = len(rows_img), len(PYRAMID_LEVELS)
    n_cols = n_cols_heat + 1  # +1 cho cot Input
    cell_h, cell_w = orig.shape[0], orig.shape[1]
    label_h, row_label_w = 44, 190
    grid = np.full((n_rows * (cell_h + label_h) + label_h,
                     row_label_w + n_cols * cell_w, 3), 255, dtype=np.uint8)

    # Tieu de cot: "Input" roi den P2..P5 (giu dung ten tang pyramid -
    # chinh xac hon "Layer 1-4" chung chung, vi day la FPN co y nghia
    # do phan giai cu the tung tang, khop dung Eq. 6-8 cua bai)
    col_titles = ['Input'] + PYRAMID_LEVELS
    for j, title in enumerate(col_titles):
        x0 = row_label_w + j * cell_w
        (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.putText(grid, title, (x0 + cell_w // 2 - tw // 2, label_h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    row_letters = [f'({chr(97 + i)})' for i in range(n_rows)]  # (a),(b),(c)...
    for i, row_imgs in enumerate(rows_img):
        y0 = label_h + i * (cell_h + label_h) + label_h
        row_text = f'{row_letters[i]} {row_labels[i]}'
        cv2.putText(grid, row_text, (8, y0 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        # Cot dau tien: ANH GOC, khong overlay heatmap
        grid[y0:y0 + cell_h, row_label_w:row_label_w + cell_w] = orig
        for j, img in enumerate(row_imgs):
            x0 = row_label_w + (j + 1) * cell_w
            grid[y0:y0 + cell_h, x0:x0 + cell_w] = img

    cv2.imwrite(output_path, grid)
    print(f'Da luu: {output_path}')
    print()
    print('Goi y chu thich (Figure caption) - sua lai chi tiet cho dung ten')
    print('he thong va anh dang dung, theo dung phong cach anh mau ban gui:')
    print()
    print(f'"Grad-style feature-magnitude visualization for {n_rows} matched')
    print(f'IAPC variants on the same test image (contrast severity 3). Each')
    print(f'row shows the input image followed by overlays from successive')
    print(f'pyramid levels (P2-P5). Colour indicates channel-averaged absolute')
    print(f'activation magnitude, normalized to a SHARED scale per column so')
    print(f'that colour intensity is comparable across rows. Heatmaps are')
    print(f'interpreted qualitatively and should be read together with the')
    print(f'quantitative magnitude table (printed above) and the AP comparison')
    print(f'in Table 5."')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True,
                         help='NEN la anh o dieu kien corrupted (vd contrast_s3), '
                              'khong phai anh sach - xem giai thich trong docstring')
    parser.add_argument('--system-configs', nargs='+', required=True,
                         help='"Ten hang|duong_dan_config|duong_dan_checkpoint" '
                              '(checkpoint co the dung wildcard *)')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--diff', nargs=2, default=None, metavar=('NAME_A', 'NAME_B'),
                         help='Them 1 hang |NAME_A - NAME_B| o cuoi (vd '
                              '--diff Baseline "IAPC l=0.25")')
    parser.add_argument('--per-row-normalize', action='store_true',
                         help='Chuan hoa mau RIENG tung hang (0-1 theo '
                              'chinh no) thay vi dung thang chung cho ca '
                              'cot - dung de kiem tra xem hang nao "phang" '
                              'co phai do bi thang chung ep xuong khong. '
                              'MAT kha nang so sanh cuong do tuyet doi '
                              'giua cac hang khi bat option nay.')
    args = parser.parse_args()

    systems = parse_systems(args.system_configs)
    build_grid(systems, args.image, args.output, args.device, args.diff,
               args.per_row_normalize)


if __name__ == '__main__':
    main()
