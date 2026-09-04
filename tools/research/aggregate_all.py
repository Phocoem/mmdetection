# -*- coding: utf-8 -*-
"""Tổng hợp toàn bộ kết quả thực nghiệm thành bảng bài báo.

Đọc mọi metrics.json của (model x seed x condition), tính:
  - Clean AP (mean +- std qua seed)
  - AP_corr = mean AP qua các điều kiện corrupted (công thức 7)
  - RD (Robustness Drop) = clean AP - AP_corr
  - SI (Stability Index) = AP_corr / clean AP
  - AP theo từng họ corruption và severity
Xuất:
  - summary_table.csv          (bảng chính mean+-std)
  - summary_per_condition.csv  (chi tiết từng điều kiện)
  - summary_table.tex          (LaTeX sẵn dán vào bài)
  - in ra terminal bảng tóm tắt + xếp hạng

Cách dùng:
    python tools/research/aggregate_all.py \
        --root work_dirs/research \
        --seeds 2024 2025 2026 \
        --eval-subdir evaluation \
        --out-prefix work_dirs/research/FINAL

    # spatially-varying:
    python tools/research/aggregate_all.py \
        --root work_dirs/research --seeds 2024 2025 2026 \
        --eval-subdir evaluation_spatial \
        --out-prefix work_dirs/research/FINAL_spatial
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

METRIC = 'coco/segm_mAP'


def load(root, seeds, eval_subdir):
    """records[model][seed][condition] = value"""
    rec = defaultdict(lambda: defaultdict(dict))
    root = os.path.abspath(root)
    for model_dir in sorted(glob.glob(os.path.join(root, '*'))):
        if not os.path.isdir(model_dir):
            continue
        model = os.path.basename(model_dir)
        for seed in seeds:
            cond_root = os.path.join(model_dir, f'seed_{seed}',
                                     eval_subdir, 'conditions')
            if not os.path.isdir(cond_root):
                continue
            for cond_dir in sorted(glob.glob(os.path.join(cond_root, '*'))):
                mfile = os.path.join(cond_dir, 'metrics.json')
                if not os.path.isfile(mfile):
                    continue
                try:
                    data = json.load(open(mfile))
                except Exception:  # noqa: BLE001
                    continue
                if METRIC in data:
                    cond = os.path.basename(cond_dir)
                    rec[model][seed][cond] = float(data[METRIC])
    return rec


def per_seed_metrics(cond_vals):
    """cond_vals: {condition: value} cho 1 seed -> (clean, ap_corr, rd, si)."""
    clean = cond_vals.get('clean', None)
    corr = [v for c, v in cond_vals.items() if c != 'clean']
    if clean is None or not corr:
        return None
    ap_corr = float(np.mean(corr))
    rd = clean - ap_corr
    si = ap_corr / clean if clean > 0 else 0.0
    return clean, ap_corr, rd, si


def fam_of(cond):
    """Tách họ corruption từ tên điều kiện (bỏ hậu tố _s1/_s2/_s3)."""
    for suf in ('_s1', '_s2', '_s3'):
        if cond.endswith(suf):
            return cond[:-3]
    return cond


def mean_std(xs):
    a = np.asarray(xs, dtype=float)
    return float(a.mean()), float(a.std(ddof=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='work_dirs/research')
    ap.add_argument('--seeds', nargs='+', default=['2024', '2025', '2026'])
    ap.add_argument('--eval-subdir', default='evaluation')
    ap.add_argument('--out-prefix', default='work_dirs/research/FINAL')
    ap.add_argument('--baseline', default='mask_rcnn_r50_fpn',
                    help='model baseline để tính delta AP_corr')
    args = ap.parse_args()

    rec = load(args.root, args.seeds, args.eval_subdir)
    if not rec:
        raise SystemExit('Không đọc được metrics.json nào. Kiểm tra --root, '
                         '--seeds, --eval-subdir.')

    # Tổng hợp mỗi model: mean+-std của clean/ap_corr/rd/si + theo họ
    rows = []
    per_cond_rows = []
    all_conditions = set()
    for model, seeds_data in rec.items():
        clean_l, corr_l, rd_l, si_l = [], [], [], []
        fam_acc = defaultdict(list)  # fam -> list AP qua (seed, severity)
        n_seed_ok = 0
        for seed, cond_vals in seeds_data.items():
            all_conditions.update(cond_vals.keys())
            m = per_seed_metrics(cond_vals)
            if m is None:
                continue
            n_seed_ok += 1
            clean_l.append(m[0]); corr_l.append(m[1])
            rd_l.append(m[2]); si_l.append(m[3])
            for c, v in cond_vals.items():
                if c == 'clean':
                    continue
                fam_acc[fam_of(c)].append(v)
                per_cond_rows.append(
                    {'model': model, 'seed': seed, 'condition': c, 'AP': v})
        if n_seed_ok == 0:
            continue
        row = {'model': model, 'n_seeds': n_seed_ok}
        for key, lst in (('clean', clean_l), ('AP_corr', corr_l),
                         ('RD', rd_l), ('SI', si_l)):
            mu, sd = mean_std(lst)
            row[f'{key}_mean'] = mu
            row[f'{key}_std'] = sd
        for fam, lst in fam_acc.items():
            mu, sd = mean_std(lst)
            row[f'{fam}_mean'] = mu
            row[f'{fam}_std'] = sd
        rows.append(row)

    rows.sort(key=lambda r: r['AP_corr_mean'], reverse=True)

    # Delta AP_corr so với baseline
    base = next((r for r in rows if r['model'] == args.baseline), None)
    base_corr = base['AP_corr_mean'] if base else None

    # ---- In terminal ----
    print(f"\n{'='*92}")
    print(f"TỔNG HỢP {args.eval_subdir} — {len(rows)} model, "
          f"{args.seeds} seed | metric={METRIC}")
    print('='*92)
    hdr = f"{'Model':<40}{'Clean':>12}{'AP_corr':>14}{'RD':>10}{'SI':>10}"
    if base_corr is not None:
        hdr += f"{'ΔAP_corr':>11}"
    print(hdr)
    print('-'*92)
    for r in rows:
        line = (f"{r['model']:<40}"
                f"{r['clean_mean']:.3f}±{r['clean_std']:.3f} "
                f"{r['AP_corr_mean']:.3f}±{r['AP_corr_std']:.3f} "
                f"{r['RD_mean']:.3f}    {r['SI_mean']:.3f}")
        if base_corr is not None:
            d = r['AP_corr_mean'] - base_corr
            line += f"  {d:+.3f}"
        print(line)
    print('-'*92)
    if rows:
        best = rows[0]
        print(f"→ AP_corr cao nhất: {best['model']} "
              f"({best['AP_corr_mean']:.3f})")

    # ---- CSV chính ----
    import csv
    fixed = ['model', 'n_seeds', 'clean_mean', 'clean_std',
             'AP_corr_mean', 'AP_corr_std', 'RD_mean', 'RD_std',
             'SI_mean', 'SI_std']
    fams = sorted({fam_of(c) for c in all_conditions if c != 'clean'})
    fam_cols = []
    for f in fams:
        fam_cols += [f'{f}_mean', f'{f}_std']
    cols = fixed + fam_cols
    csv_path = f'{args.out_prefix}_summary_table.csv'
    with open(csv_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nĐã lưu: {csv_path}")

    # ---- CSV chi tiết từng điều kiện ----
    pc_path = f'{args.out_prefix}_per_condition.csv'
    with open(pc_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['model', 'seed', 'condition', 'AP'])
        w.writeheader()
        w.writerows(per_cond_rows)
    print(f"Đã lưu: {pc_path}")

    # ---- LaTeX ----
    tex_path = f'{args.out_prefix}_summary_table.tex'
    with open(tex_path, 'w') as fh:
        fh.write('% Auto-generated. Clean/AP_corr/RD/SI = mean over seeds.\n')
        fh.write('\\begin{table}[t]\\centering\n')
        fh.write('\\caption{Robustness summary (mean over ' +
                 f'{len(args.seeds)} seeds).}}\n')
        fh.write('\\begin{tabular}{lcccc}\n\\hline\n')
        fh.write('Model & Clean AP & $\\overline{AP}_{corr}$ & RD & SI \\\\\n')
        fh.write('\\hline\n')
        for r in rows:
            name = r['model'].replace('_', '\\_')
            fh.write(f"{name} & "
                     f"{r['clean_mean']:.3f} & "
                     f"{r['AP_corr_mean']:.3f} & "
                     f"{r['RD_mean']:.3f} & "
                     f"{r['SI_mean']:.3f} \\\\\n")
        fh.write('\\hline\n\\end{tabular}\n\\end{table}\n')
    print(f"Đã lưu: {tex_path}")


if __name__ == '__main__':
    main()
