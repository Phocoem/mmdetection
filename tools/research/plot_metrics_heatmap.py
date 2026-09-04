"""
Ve heatmap AP theo (model x dieu kien), DOC TRUC TIEP tung file
metrics.json / condition_metrics.csv (khong di qua CSV tong hop trung
gian nhu convert_mmdet_logs_to_csv.py) - dam bao khong bo sot hay lam
tron sai lech gi truoc khi ve.

Quet ca 2 dinh dang da phat hien trong work_dirs cua ban:
    A) <root>/<system>/seed_<seed>/evaluation/condition_metrics.csv
    B) <root>/<system>/seed_<seed>/evaluation/conditions/<condition>/metrics.json

KHAC BIET SO VOI HEATMAP CU: ngoai 2 panel "Mean AP" va "Std across seeds",
them PANEL THU 3 "So seed dong gop" - to mau + ghi so ngay trong o, cho
biet moi (he thong, dieu kien) duoc tinh tu bao nhieu seed thuc su (3, 4,
5, hay 0 neu chua co du lieu). Day la thong tin bi AN trong heatmap cu -
"Mean AP (3 seeds)" o tieu de chi la MOT con so chung cho ca bang, trong
khi thuc te moi o co the khac nhau ve so seed dong gop (nhu da phat hien
o buoc audit truoc).

Cach dung:
    python plot_metrics_heatmap.py \\
        --root /home/pc/mmdet_AI/mmdetection/work_dirs/research \\
        --output heatmap_with_seed_count.png

    # Chi ve 1 vai he thong, sap xep theo APcorr thay vi Clean AP
    python plot_metrics_heatmap.py \\
        --root /home/pc/mmdet_AI/mmdetection/work_dirs/research \\
        --systems mask_rcnn_r50_fpn_aug mask_rcnn_r50_iapc_lam0p25 \\
        --sort-by apcorr --output subset.png
"""
import argparse
import csv
import json
import re
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # khong can man hinh, chi xuat file
import matplotlib.pyplot as plt
import numpy as np

METRIC_KEY = 'coco/segm_mAP'

COND_PREFIX = {
    'brightness': 'B', 'contrast': 'C', 'gaussian_noise': 'G',
    'uneven_contrast': 'U', 'dappled_light': 'D',
}
COND_RE = re.compile(
    r'^(brightness|contrast|gaussian_noise|uneven_contrast|dappled_light)_s(\d)$')
SEED_DIR_RE = re.compile(r'^seed_(\d+)$')

CONDITIONS_ORDERED = (
    ['clean'] +
    [f'{fam}_s{s}' for fam in
     ('brightness', 'contrast', 'gaussian_noise',
      'uneven_contrast', 'dappled_light')
     for s in (1, 2, 3)]
)
CONDITION_LABELS = [c for c in CONDITIONS_ORDERED]  # dung nguyen ten day du
UNIFORM_CONDITIONS = [c for c in CONDITIONS_ORDERED
                      if c.split('_s')[0] in
                      ('brightness', 'contrast', 'gaussian_noise')
                      and c != 'clean']


def read_condition_metrics_csv(path: Path) -> dict:
    """DOC TRUC TIEP file CSV -> {condition_full_name: ap_value}."""
    out = {}
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cond = row.get('condition')
            val = row.get(METRIC_KEY)
            if cond in CONDITIONS_ORDERED and val not in (None, ''):
                out[cond] = float(val)
    return out


def read_conditions_dir(conditions_dir: Path) -> dict:
    """DOC TRUC TIEP tung metrics.json trong thu muc conditions/ ->
    {condition_full_name: ap_value}."""
    out = {}
    for cond_dir in conditions_dir.iterdir():
        if not cond_dir.is_dir() or cond_dir.name not in CONDITIONS_ORDERED:
            continue
        json_path = cond_dir / 'metrics.json'
        if not json_path.exists():
            continue
        with open(json_path, encoding='utf-8') as f:
            payload = json.load(f)
        if METRIC_KEY in payload:
            out[cond_dir.name] = float(payload[METRIC_KEY])
    return out


def scan_raw(root: Path, systems=None):
    """Doc TRUC TIEP toan bo cay thu muc, khong qua file tong hop nao.
    Tra ve: {system: {condition: {seed: ap_value}}}."""
    data = {}
    system_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if systems:
        wanted = set(systems)
        system_dirs = [p for p in system_dirs if p.name in wanted]

    for sys_dir in system_dirs:
        seed_dirs = [p for p in sys_dir.iterdir()
                     if p.is_dir() and SEED_DIR_RE.match(p.name)]
        if not seed_dirs:
            continue
        for seed_dir in seed_dirs:
            seed = SEED_DIR_RE.match(seed_dir.name).group(1)
            eval_dir = seed_dir / 'evaluation'
            csv_path = eval_dir / 'condition_metrics.csv'
            conditions_dir = eval_dir / 'conditions'

            if csv_path.exists():
                values = read_condition_metrics_csv(csv_path)
            elif conditions_dir.is_dir():
                values = read_conditions_dir(conditions_dir)
            else:
                continue

            for cond, ap in values.items():
                data.setdefault(sys_dir.name, {}).setdefault(
                    cond, {})[seed] = ap
    return data


def build_matrices(data: dict, systems_order):
    """Tra ve mean_mat, std_mat, n_mat (numpy, shape = len(systems) x 16),
    va danh sach seed thuc te dung cho moi (system,condition) de kiem tra."""
    n_sys = len(systems_order)
    n_cond = len(CONDITIONS_ORDERED)
    mean_mat = np.full((n_sys, n_cond), np.nan)
    std_mat = np.full((n_sys, n_cond), np.nan)
    n_mat = np.zeros((n_sys, n_cond), dtype=int)
    seed_sets = {}

    for i, system in enumerate(systems_order):
        for j, cond in enumerate(CONDITIONS_ORDERED):
            seed_values = data.get(system, {}).get(cond, {})
            n = len(seed_values)
            n_mat[i, j] = n
            seed_sets[(system, cond)] = sorted(seed_values.keys())
            if n == 0:
                continue
            vals = list(seed_values.values())
            mean_mat[i, j] = statistics.mean(vals)
            std_mat[i, j] = statistics.stdev(vals) if n > 1 else 0.0

    return mean_mat, std_mat, n_mat, seed_sets


def sort_systems(data: dict, sort_by: str):
    systems = list(data.keys())

    def apcorr_of(system):
        vals = []
        for cond in UNIFORM_CONDITIONS:
            seed_values = data.get(system, {}).get(cond, {})
            if seed_values:
                vals.append(statistics.mean(seed_values.values()))
        return statistics.mean(vals) if vals else -1.0

    def clean_of(system):
        seed_values = data.get(system, {}).get('clean', {})
        return statistics.mean(seed_values.values()) if seed_values else -1.0

    key_fn = apcorr_of if sort_by == 'apcorr' else clean_of
    return sorted(systems, key=key_fn, reverse=True)


def plot(mean_mat, std_mat, n_mat, systems_order, output_path, title_suffix=''):
    n_sys, n_cond = mean_mat.shape
    fig, axes = plt.subplots(1, 3, figsize=(3.0 * n_cond * 0.42 + 2, 0.42 * n_sys + 2.2))

    def draw_panel(ax, matrix, cmap, vmin, vmax, fmt, title, is_int=False):
        masked = np.ma.masked_invalid(matrix)
        im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_xticks(range(n_cond))
        ax.set_xticklabels(CONDITION_LABELS, rotation=90, fontsize=7)
        ax.set_yticks(range(n_sys))
        ax.set_yticklabels(systems_order, fontsize=8)
        ax.set_title(title, fontsize=11, fontweight='bold')
        for i in range(n_sys):
            for j in range(n_cond):
                v = matrix[i, j]
                if np.isnan(v):
                    ax.text(j, i, '\u2014', ha='center', va='center', fontsize=7,
                            color='gray')
                else:
                    txt = f'{int(v)}' if is_int else fmt.format(v)
                    ax.text(j, i, txt, ha='center', va='center', fontsize=6.5)
        return im

    im0 = draw_panel(axes[0], mean_mat, 'RdYlGn', 0.30, 0.80, '{:.3f}',
                      'Mean AP (theo so seed thuc te)')
    fig.colorbar(im0, ax=axes[0], fraction=0.03, pad=0.02)

    im1 = draw_panel(axes[1], std_mat, 'YlOrRd', 0.0, 0.05, '{:.3f}',
                      'Std across seeds')
    fig.colorbar(im1, ax=axes[1], fraction=0.03, pad=0.02)

    max_n = int(np.nanmax(n_mat)) if n_mat.size else 0
    n_mat_float = n_mat.astype(float)
    n_mat_float[n_mat == 0] = np.nan
    im2 = draw_panel(axes[2], n_mat_float, 'viridis', 1, max(max_n, 1), '{:d}',
                      'So seed dong gop / o (3 vs 5 vs khac)', is_int=True)
    cbar2 = fig.colorbar(im2, ax=axes[2], fraction=0.03, pad=0.02,
                          ticks=range(1, max(max_n, 1) + 1))
    cbar2.set_label('so seed')

    fig.suptitle(f'AP theo (model x dieu kien) - doc truc tiep tung file{title_suffix}',
                 fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Da luu: {output_path}')


def print_seed_count_report(n_mat, systems_order):
    """In danh sach cac (he thong, dieu kien) co so seed KHONG DONG NHAT
    (vd co dieu kien 5 seed, co dieu kien 3 seed, trong CUNG 1 he thong)."""
    print('\n=== KIEM TRA SO SEED KHONG DONG NHAT TRONG CUNG 1 HE THONG ===')
    for i, system in enumerate(systems_order):
        row = n_mat[i]
        nonzero = row[row > 0]
        if len(nonzero) == 0:
            print(f'{system}: CHUA CO DU LIEU O DIEU KIEN NAO')
            continue
        distinct = sorted(set(int(x) for x in nonzero))
        if len(distinct) > 1:
            detail = ', '.join(
                f'{CONDITIONS_ORDERED[j]}={int(row[j])}seed'
                for j in range(len(row)) if row[j] > 0)
            print(f'{system}: KHONG DONG NHAT - cac muc so seed gap: {distinct}')
            print(f'    chi tiet: {detail}')
        else:
            zero_conds = [CONDITIONS_ORDERED[j] for j in range(len(row)) if row[j] == 0]
            note = f' (thieu hoan toan: {zero_conds})' if zero_conds else ''
            print(f'{system}: dong nhat {distinct[0]} seed cho moi dieu kien co du lieu{note}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--systems', nargs='*', default=None)
    parser.add_argument('--output', required=True)
    parser.add_argument('--sort-by', choices=['clean', 'apcorr'], default='clean')
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f'Khong tim thay thu muc: {root}')

    data = scan_raw(root, args.systems)
    if not data:
        raise SystemExit('Khong doc duoc du lieu nao - kiem tra lai --root.')

    systems_order = sort_systems(data, args.sort_by)
    mean_mat, std_mat, n_mat, seed_sets = build_matrices(data, systems_order)

    print_seed_count_report(n_mat, systems_order)
    plot(mean_mat, std_mat, n_mat, systems_order, args.output)


if __name__ == '__main__':
    main()
