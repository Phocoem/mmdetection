"""
Ve bieu do 2 truc: APcorr (truc trai) va magnitude trung binh tai P5
(truc phai, thang log) - CA HAI theo lambda tren CUNG 1 truc X. Day la
hinh CHINH thay the heatmap Grad-CAM-style lam bang chung cho phat hien
"magnitude tang manh theo lambda nhung AP KHONG tang theo tuong xung" -
tranh nguoi doc hieu nham kieu "cang do = cang tot" quen thuoc tu cac
bai Grad-CAM khac.

Cach dung:
    python plot_magnitude_vs_ap.py \\
        --per-seed per_seed_full.csv \\
        --magnitude-csv feature_heatmaps_batch/heatmap_featureconstraint_lambda_magnitude.csv \\
        --output magnitude_vs_ap.png

(File magnitude-csv duoc TU DONG sinh ra khi chay plot_feature_heatmaps.py
ban moi nhat - cung thu muc, cung ten, them "_magnitude.csv")
"""
import argparse
import csv
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

UNIFORM = ['BS1', 'BS2', 'BS3', 'CS1', 'CS2', 'CS3', 'GS1', 'GS2', 'GS3']

LAMBDA_SYSTEMS = [
    ('mask_rcnn_r50_gpuaug', 0.0, 'IAPC control (l=0)'),
    ('mask_rcnn_r50_iapc_lam0p10', 0.1, 'IAPC (l=0.1)'),
    ('mask_rcnn_r50_iapc_lam0p25', 0.25, 'IAPC (l=0.25, full)'),
    ('mask_rcnn_r50_iapc_lam0p50', 0.5, 'IAPC (l=0.5)'),
    ('mask_rcnn_r50_iapc', 1.0, 'IAPC (l=1.0)'),
]


def load_apcorr_by_lambda(per_seed_path):
    data = {}
    with open(per_seed_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            data.setdefault(row['System'], []).append(row)

    result = {}
    for sys_key, lam, _label in LAMBDA_SYSTEMS:
        rows = data.get(sys_key)
        if not rows:
            print(f'CANH BAO: khong co du lieu APcorr cho {sys_key} trong '
                  f'{per_seed_path}, bo qua diem lambda={lam}.')
            continue
        apcorr_per_seed = [statistics.mean(float(r[c]) for c in UNIFORM)
                            for r in rows]
        result[lam] = statistics.mean(apcorr_per_seed)
    return result


def load_magnitude_by_lambda(magnitude_csv_path):
    """Doc file CSV do plot_feature_heatmaps.py tu sinh ra
    (System,P2,P3,P4,P5). Anh xa ten he thong -> lambda qua LAMBDA_SYSTEMS."""
    name_to_lambda = {label: lam for _key, lam, label in LAMBDA_SYSTEMS}
    result = {}  # lambda -> {level: magnitude}
    with open(magnitude_csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        levels = [c for c in reader.fieldnames if c != 'System']
        for row in reader:
            lam = name_to_lambda.get(row['System'])
            if lam is None:
                print(f'CANH BAO: khong nhan dien duoc he thong "{row["System"]}" '
                      f'trong file magnitude - kiem tra ten co khop LAMBDA_SYSTEMS '
                      f'khong (phai dung ten hien thi da dung khi chay '
                      f'plot_feature_heatmaps.py, vd "IAPC (l=0.25, full)").')
                continue
            result[lam] = {lv: float(row[lv]) for lv in levels}
    return result, levels


def plot(apcorr_by_lambda, magnitude_by_lambda, levels, output_path):
    lambdas = sorted(apcorr_by_lambda.keys())
    x_pos = list(range(len(lambdas)))

    fig, ax1 = plt.subplots(figsize=(8, 5))

    apcorr_vals = [apcorr_by_lambda[l] for l in lambdas]
    ax1.plot(x_pos, apcorr_vals, 'o-', color='tab:blue', linewidth=2.5,
              markersize=8, label=r'$AP_{corr}$ (truc trai)', zorder=3)
    ax1.set_xlabel('consistency weight ' + r'$\lambda$')
    ax1.set_ylabel(r'$AP_{corr}$', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([str(l) for l in lambdas])
    for xp in x_pos:
        ax1.axvline(xp, color='gray', linewidth=0.5, linestyle=':', alpha=0.4)

    ax2 = ax1.twinx()
    import numpy as np
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(levels)))
    for lv, color in zip(levels, colors):
        mag_vals = [magnitude_by_lambda.get(l, {}).get(lv, float('nan'))
                    for l in lambdas]
        ax2.plot(x_pos, mag_vals, 's--', color=color, linewidth=1.5,
                  markersize=6, label=f'magnitude {lv} (truc phai, log)')
    ax2.set_yscale('log')
    ax2.set_ylabel('mean |activation| (thang log)', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left',
               fontsize=8, framealpha=0.9)

    ax1.set_title(r'$AP_{corr}$ hau nhu khong doi trong khi magnitude tang '
                  'vot theo ' + r'$\lambda$', fontsize=11, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f'Da luu: {output_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--per-seed', required=True,
                         help='per_seed_full.csv (de lay APcorr theo lambda)')
    parser.add_argument('--magnitude-csv', required=True,
                         help='File _magnitude.csv tu dong sinh boi '
                              'plot_feature_heatmaps.py')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    apcorr_by_lambda = load_apcorr_by_lambda(args.per_seed)
    magnitude_by_lambda, levels = load_magnitude_by_lambda(args.magnitude_csv)

    common_lambdas = set(apcorr_by_lambda) & set(magnitude_by_lambda)
    missing = (set(apcorr_by_lambda) | set(magnitude_by_lambda)) - common_lambdas
    if missing:
        print(f'CANH BAO: cac lambda sau chi co O MOT trong 2 nguon du lieu '
              f'(APcorr hoac magnitude), se bi bo qua khi ve: {sorted(missing)}')

    plot({l: v for l, v in apcorr_by_lambda.items() if l in common_lambdas},
         {l: v for l, v in magnitude_by_lambda.items() if l in common_lambdas},
         levels, args.output)


if __name__ == '__main__':
    main()
