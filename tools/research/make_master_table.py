# -*- coding: utf-8 -*-
"""Gộp AP (per_condition) + Efficiency thành 1 bảng master duy nhất cho bài.

Đọc:
  - FINAL_per_condition.csv     : (model, seed, condition, AP)
  - EFFICIENCY_FINAL.csv        : Params/FLOPs/Latency/train_time...

Xuất:
  - MASTER_TABLE.csv            : mọi thứ cho mỗi model
  - MASTER_TABLE.tex            : bảng LaTeX chính cho bài (Table 6/7/8/E gộp)
  - MASTER_TABLE.md             : phiên bản Markdown dễ đọc
  - MISSING_CHECKLIST.txt       : những cột còn trống + gợi ý cách bổ sung

Cách dùng:
    python tools/research/make_master_table.py \
        --per-condition work_dirs/research/FINAL_per_condition.csv \
        --efficiency   work_dirs/research/EFFICIENCY_FINAL.csv \
        --out-prefix   work_dirs/research/MASTER \
        --focus-models mask_rcnn_r50_fpn mask_rcnn_r50_fpn_aug mask_rcnn_r50_fpn_bca \
                       mask_rcnn_r50_cbam_fpn mask_rcnn_r50_bifpn \
                       mask_rcnn_r50_dgcf_fpn_v2 mask_rcnn_r101_fpn
"""

import argparse
import csv
import os
from collections import defaultdict
from statistics import mean, stdev

FAMILIES = ('brightness', 'contrast', 'gaussian_noise',
            'uneven_contrast', 'dappled_light')


def fam_of(cond):
    for f in FAMILIES:
        if cond.startswith(f):
            return f
    return None


def load_per_condition(path):
    """rec[model][seed][condition] = AP"""
    rec = defaultdict(lambda: defaultdict(dict))
    with open(path) as f:
        for row in csv.DictReader(f):
            m, s, c = row['model'], row['seed'], row['condition']
            try:
                rec[m][s][c] = float(row['AP'])
            except Exception:  # noqa: BLE001
                pass
    return rec


def load_efficiency(path):
    """eff[model] = {params_M, flops_G, latency_ms, fps, train_time_h_mean...}"""
    eff = {}
    if not os.path.isfile(path):
        return eff
    with open(path) as f:
        for row in csv.DictReader(f):
            if not row.get('model'):
                continue
            def num(k):
                v = row.get(k, '')
                if v in ('', None):
                    return None
                try:
                    return float(v)
                except Exception:  # noqa: BLE001
                    return None
            eff[row['model']] = {
                'params_M': num('params_M'),
                'flops_G': num('flops_G'),
                'latency_ms': num('latency_ms'),
                'fps': num('fps'),
                'peak_infer_vram_MB': num('peak_infer_vram_MB'),
                'best_val_mAP_mean': num('best_val_mAP_mean'),
                'best_epoch_mean': num('best_epoch_mean'),
                'stop_epoch_mean': num('stop_epoch_mean'),
                'train_time_h_mean': num('train_time_h_mean'),
                'train_time_h_std': num('train_time_h_std'),
                'peak_train_vram_MB_mean': num('peak_train_vram_MB_mean'),
                'model_size_MB_mean': num('model_size_MB_mean'),
                'note': row.get('note') or '',
            }
    return eff


def per_model_summary(rec, eff=None):
    """Trả về summary[model] = dict với clean/AP_corr/RD/SI + theo họ.

    Nếu per_condition không có 'clean', dùng best_val_mAP_mean từ efficiency
    làm proxy (hoặc bỏ clean/RD/SI nếu cũng không có).
    """
    eff = eff or {}
    summary = {}
    for model, seeds in rec.items():
        clean_l, ap_corr_l, rd_l, si_l = [], [], [], []
        fam_l = {f: [] for f in FAMILIES}
        seen_conds = set()
        # Clean proxy từ efficiency (nếu per_condition không có)
        clean_proxy = eff.get(model, {}).get('best_val_mAP_mean')
        for seed, conds in seeds.items():
            seen_conds.update(conds)
            clean = conds.get('clean', clean_proxy)
            corr = [v for c, v in conds.items() if c != 'clean']
            if not corr:
                continue
            ap_corr = mean(corr)
            ap_corr_l.append(ap_corr)
            if clean is not None:
                clean_l.append(clean)
                rd_l.append(clean - ap_corr)
                si_l.append(ap_corr / clean if clean > 0 else 0)
            for f in FAMILIES:
                vs = [v for c, v in conds.items() if fam_of(c) == f]
                if vs:
                    fam_l[f].append(mean(vs))
        if not ap_corr_l:
            continue

        def ms(x):
            if not x:
                return None, None
            return (mean(x), stdev(x) if len(x) > 1 else 0.0)

        row = {'model': model, 'n_seeds': len(ap_corr_l),
               'n_conditions': len(seen_conds)}
        for k, xs in (('clean', clean_l), ('AP_corr', ap_corr_l),
                      ('RD', rd_l), ('SI', si_l)):
            mv, sv = ms(xs)
            row[f'{k}_mean'] = mv; row[f'{k}_std'] = sv
        for f, xs in fam_l.items():
            mv, sv = ms(xs)
            row[f'{f}_mean'] = mv; row[f'{f}_std'] = sv
        # Đánh dấu clean là proxy nếu dùng
        if not clean_l or all(v == clean_proxy for v in clean_l):
            row['clean_source'] = 'best_val_mAP proxy' if clean_proxy else 'none'
        summary[model] = row
    return summary


def merge(summary, eff):
    for m, r in summary.items():
        r.update({k: v for k, v in eff.get(m, {}).items() if k not in r})
    return summary


def write_csv(rows, path):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f'  saved {path}')


def fmt(v, spec='.3f', fallback='--'):
    try:
        return f'{v:{spec}}'
    except Exception:  # noqa: BLE001
        return fallback


def write_latex(rows, path, focus):
    with open(path, 'w') as f:
        f.write('% Master table for the paper — 3-seed mean±std\n')
        f.write('\\begin{table*}[t]\\centering\\small\n')
        f.write('\\caption{Master results: robustness (mean over 3 seeds) '
                'and efficiency. AP\\textsubscript{corr} = mean AP over the '
                '9 corrupted conditions; RD = clean AP $-$ AP\\textsubscript{corr}; '
                'SI = AP\\textsubscript{corr} / clean AP. FLOPs measured on '
                'backbone+neck (the components that differ across models).}\n')
        f.write('\\label{tab:master}\n')
        f.write('\\begin{tabular}{lcccccccc}\n\\hline\n')
        f.write('Model & Clean AP & AP$_\\text{corr}$ & RD & SI & '
                'Params (M) & FLOPs (G) & Lat.\\ (ms) & Train (h) \\\\\n')
        f.write('\\hline\n')
        selected = [r for r in rows if not focus or r['model'] in focus]
        selected.sort(key=lambda r: -(r.get('AP_corr_mean') or 0))
        for r in selected:
            name = r['model'].replace('_', '\\_')
            f.write(
                f"{name} & "
                f"{fmt(r.get('clean_mean'))}$\\pm${fmt(r.get('clean_std'),'.3f')} & "
                f"{fmt(r.get('AP_corr_mean'))}$\\pm${fmt(r.get('AP_corr_std'),'.3f')} & "
                f"{fmt(r.get('RD_mean'))} & "
                f"{fmt(r.get('SI_mean'))} & "
                f"{fmt(r.get('params_M'),'.1f')} & "
                f"{fmt(r.get('flops_G'),'.1f')} & "
                f"{fmt(r.get('latency_ms'),'.1f')} & "
                f"{fmt(r.get('train_time_h_mean'),'.2f')}"
                f"$\\pm${fmt(r.get('train_time_h_std'),'.2f')} \\\\\n")
        f.write('\\hline\n\\end{tabular}\n\\end{table*}\n')
    print(f'  saved {path}')


def write_markdown(rows, path, focus):
    selected = [r for r in rows if not focus or r['model'] in focus]
    selected.sort(key=lambda r: -(r.get('AP_corr_mean') or 0))
    with open(path, 'w') as f:
        f.write('# Master Experimental Results\n\n')
        f.write('3-seed mean±std. Sorted by AP_corr (descending).\n\n')
        f.write('| Model | Clean AP | AP_corr | RD | SI | Params (M) | FLOPs (G) | Latency (ms) | FPS | Train (h) |\n')
        f.write('|---|---|---|---|---|---|---|---|---|---|\n')
        for r in selected:
            f.write(f"| {r['model']} | "
                    f"{fmt(r.get('clean_mean'))}±{fmt(r.get('clean_std'),'.3f')} | "
                    f"{fmt(r.get('AP_corr_mean'))}±{fmt(r.get('AP_corr_std'),'.3f')} | "
                    f"{fmt(r.get('RD_mean'))} | "
                    f"{fmt(r.get('SI_mean'))} | "
                    f"{fmt(r.get('params_M'),'.1f')} | "
                    f"{fmt(r.get('flops_G'),'.1f')} | "
                    f"{fmt(r.get('latency_ms'),'.1f')} | "
                    f"{fmt(r.get('fps'),'.1f')} | "
                    f"{fmt(r.get('train_time_h_mean'),'.2f')}±"
                    f"{fmt(r.get('train_time_h_std'),'.2f')} |\n")

        # Bảng chi tiết theo họ corruption
        f.write('\n## AP theo họ corruption (mean±std 3 seed)\n\n')
        f.write('| Model | Brightness | Contrast | Gaussian | Uneven contrast | Dappled light |\n')
        f.write('|---|---|---|---|---|---|\n')
        for r in selected:
            def cell(k):
                mv = r.get(f'{k}_mean'); sv = r.get(f'{k}_std')
                if mv is None: return '--'
                return f'{mv:.3f}±{sv:.3f}'
            f.write(f"| {r['model']} | {cell('brightness')} | {cell('contrast')} | "
                    f"{cell('gaussian_noise')} | {cell('uneven_contrast')} | "
                    f"{cell('dappled_light')} |\n")
    print(f'  saved {path}')


def missing_checklist(rows, path, focus):
    checks = []
    focus_set = set(focus) if focus else {r['model'] for r in rows}
    for r in rows:
        if r['model'] not in focus_set:
            continue
        issues = []
        if r.get('flops_G') is None: issues.append('FLOPs')
        if r.get('latency_ms') is None: issues.append('latency/FPS')
        if r.get('train_time_h_mean') is None: issues.append('train_time (log?)')
        if r.get('best_val_mAP_mean') is None: issues.append('best_val_mAP')
        if r.get('n_seeds', 0) < 3: issues.append(f"chỉ {r.get('n_seeds',0)} seed")
        if r.get('AP_corr_mean') is None: issues.append('AP_corr')
        for fam in ('uneven_contrast', 'dappled_light'):
            if r.get(f'{fam}_mean') is None:
                issues.append(f'{fam} (chưa eval spatial)')
        if issues:
            checks.append((r['model'], issues))

    with open(path, 'w') as f:
        f.write('CHECKLIST — thông số còn thiếu cho các model then chốt\n')
        f.write('=' * 70 + '\n\n')
        if not checks:
            f.write('OK — không thiếu gì cho các model trong focus.\n')
        for model, issues in checks:
            f.write(f'{model}\n')
            for i in issues:
                f.write(f'  - THIẾU: {i}\n')
            f.write('\n')

        f.write('\nGỢI Ý BỔ SUNG (các thứ nên có thêm cho bài Q1):\n')
        f.write('-' * 70 + '\n')
        f.write("""\
[1] BOOTSTRAP CI + p-value cho các cặp then chốt (aug vs FPN, aug vs DGCF,
    BCA vs aug). Chạy paired_bootstrap_test.py sau khi dump predictions.
[2] GATE WEIGHT ANALYSIS: nếu giữ DGCF, dump α_o/α_c/α_d để chứng minh gate
    không "adaptive" thật (dump_gate_weights.py, 3 seed).
[3] AugMix efficiency: fpn_augmix hiện chỉ có train_time, thiếu flops/latency.
    Thêm vào --configs khi chạy measure_all_efficiency.py.
[4] SOLO/SOLOv2 efficiency: nếu giữ trong bài, cần đo đầy đủ.
[5] Per-severity table: chi tiết s1/s2/s3 cho từng họ (đã có trong per_condition,
    chưa gộp vào master — thêm nếu cần Table 8 dạng cũ).
[6] Downstream metric (plant count error) — chưa có, nên thêm nếu muốn Q1
    nông nghiệp mạnh.
""")
    print(f'  saved {path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-condition', required=True)
    ap.add_argument('--efficiency', required=True)
    ap.add_argument('--out-prefix', default='MASTER')
    ap.add_argument('--focus-models', nargs='*', default=[])
    args = ap.parse_args()

    rec = load_per_condition(args.per_condition)
    eff = load_efficiency(args.efficiency)
    summary = per_model_summary(rec, eff)
    rows = list(merge(summary, eff).values())
    rows.sort(key=lambda r: -(r.get('AP_corr_mean') or 0))

    print('Ghi các file:')
    write_csv(rows, f'{args.out_prefix}_TABLE.csv')
    write_latex(rows, f'{args.out_prefix}_TABLE.tex', args.focus_models)
    write_markdown(rows, f'{args.out_prefix}_TABLE.md', args.focus_models)
    missing_checklist(rows, f'{args.out_prefix}_MISSING.txt',
                      args.focus_models)

    # In bảng ra terminal
    print('\n' + '=' * 110)
    print(f"{'Model':<38}{'Clean':>13}{'AP_corr':>13}{'RD':>8}{'SI':>7}"
          f"{'FLOPs':>9}{'Lat(ms)':>10}{'Train(h)':>10}")
    print('-' * 110)
    focus = set(args.focus_models) if args.focus_models else None
    for r in rows:
        if focus and r['model'] not in focus:
            continue
        print(f"{r['model']:<38}"
              f"{fmt(r.get('clean_mean')):>7}±{fmt(r.get('clean_std'),'.3f'):<5}"
              f"{fmt(r.get('AP_corr_mean')):>7}±{fmt(r.get('AP_corr_std'),'.3f'):<5}"
              f"{fmt(r.get('RD_mean')):>8}"
              f"{fmt(r.get('SI_mean')):>7}"
              f"{fmt(r.get('flops_G'),'.1f'):>9}"
              f"{fmt(r.get('latency_ms'),'.1f'):>10}"
              f"{fmt(r.get('train_time_h_mean'),'.2f'):>10}")
    print('=' * 110)


if __name__ == '__main__':
    main()
