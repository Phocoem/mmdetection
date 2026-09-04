"""
Ve lai Figure 4 cua bai (2 panel: (a) Sensitivity to consistency weight
lambda, (b) Per-family comparison IAPC vs CPU photometric aug) - dung
DUNG du lieu 5-seed that tu per_seed_full.csv, thay cho du lieu 3-seed
cu trong ban thao goc.

Cach dung:
    python plot_figure4.py --input per_seed_full.csv --output figure4.png
"""
import argparse
import csv
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

UNIFORM = ['BS1', 'BS2', 'BS3', 'CS1', 'CS2', 'CS3', 'GS1', 'GS2', 'GS3']
CONTRAST = ['CS1', 'CS2', 'CS3']
FAMILIES = [
    ('Clean', ['Clean']), ('Brightness', ['BS1', 'BS2', 'BS3']),
    ('Contrast', ['CS1', 'CS2', 'CS3']),
    ('Gaussian\nnoise', ['GS1', 'GS2', 'GS3']),
    ('Uneven\ncontrast', ['US1', 'US2', 'US3']),
    ('Dappled\nlight', ['DS1', 'DS2', 'DS3']),
]
LAMBDA_SYSTEMS = [
    ('mask_rcnn_r50_gpuaug', 0.0), ('mask_rcnn_r50_iapc_lam0p10', 0.1),
    ('mask_rcnn_r50_iapc_lam0p25', 0.25), ('mask_rcnn_r50_iapc_lam0p50', 0.5),
    ('mask_rcnn_r50_iapc', 1.0),
]


def load(path):
    by_sys = {}
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            by_sys.setdefault(row['System'], {})[row['Seed']] = row
    return by_sys


def apcorr(by_sys, system, cols):
    seeds = by_sys[system]
    per_seed = [statistics.mean(float(v[c]) for c in cols) for v in seeds.values()]
    return statistics.mean(per_seed)


def plot(by_sys, output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # ---- Panel (a): Gia tri tuyet doi APcorr theo tung lambda ----
    lambdas = [lam for _, lam in LAMBDA_SYSTEMS]
    apc_all9 = [apcorr(by_sys, sys, UNIFORM) for sys, _ in LAMBDA_SYSTEMS]
    apc_contrast = [apcorr(by_sys, sys, CONTRAST) for sys, _ in LAMBDA_SYSTEMS]

    x_pos = list(range(len(lambdas)))  # truc hang muc - moi lambda cach deu
    for xp in x_pos:
        ax1.axvline(xp, color='gray', linewidth=0.6, linestyle=':', alpha=0.5)

    ax1.plot(x_pos, apc_all9, 'o-', color='tab:blue',
             label=r'$AP_{corr}$ (all 9)', linewidth=2, markersize=7)
    ax1.plot(x_pos, apc_contrast, 's--', color='tab:red',
             label='contrast family', linewidth=2, markersize=7)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([str(lam) for lam in lambdas], fontsize=9)
    ax1.set_xlabel('consistency weight ' + r'$\lambda$')
    ax1.set_ylabel('mask AP')
    ax1.set_title('(a) Sensitivity to ' + r'$\lambda$', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=8, loc='lower left')
    ax1.grid(alpha=0.25, axis='y')

    # ---- Panel (b): Per-family comparison ----
    iapc_vals = [apcorr(by_sys, 'mask_rcnn_r50_iapc_lam0p25', cols)
                 for _, cols in FAMILIES]
    aug_vals = [apcorr(by_sys, 'mask_rcnn_r50_fpn_aug', cols)
                for _, cols in FAMILIES]
    labels = [name for name, _ in FAMILIES]
    x = range(len(labels))
    width = 0.35

    bars1 = ax2.bar([i - width / 2 for i in x], iapc_vals, width,
                     label=r'IAPC ($\lambda$ = 0.25)', color='tab:blue')
    bars2 = ax2.bar([i + width / 2 for i in x], aug_vals, width,
                     label='CPU photometric aug', color='lightgray',
                     edgecolor='black')

    for i, (iv, av) in enumerate(zip(iapc_vals, aug_vals)):
        delta = iv - av
        y = max(iv, av) + 0.003
        color = 'tab:green' if delta > 0 else 'tab:red'
        ax2.text(i, y, f'{delta:+.3f}', ha='center', fontsize=8,
                  color=color, fontweight='bold')

    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel('mask AP')
    ax2.set_title('(b) Per-family comparison against the strongest augmentation',
                   fontsize=10, fontweight='bold')
    ax2.legend(fontsize=8, loc='lower right')
    ymin = min(min(iapc_vals), min(aug_vals)) - 0.02
    ymax = max(max(iapc_vals), max(aug_vals)) + 0.02
    ax2.set_ylim(ymin, ymax)
    ax2.grid(alpha=0.25, axis='y')

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f'Da luu: {output_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    by_sys = load(args.input)
    plot(by_sys, args.output)


if __name__ == '__main__':
    main()
