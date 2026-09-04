"""
Tinh lai TOAN BO Table 5/6/8 (+ cot Worst-case AP moi) truc tiep tu
per_seed_full.csv (dinh dang: System,Seed,Clean,BS1..DS3) - dung DUNG
5-seed data thay vi 3-seed cu cua ban goc.

Cach dung:
    python generate_final_tables.py --input per_seed_full.csv \\
        --out-dir final_tables/

Sinh ra:
    final_tables/table6_full.md       - Table 6 day du (16 he thong x
                                         16 dieu kien), mean +- std, 5 seed
    final_tables/table6_full.csv      - cung du lieu, dang CSV de xu ly tiep
    final_tables/table5_summary.md    - Table 5 (subset dai dien, gia tri
                                         trung binh tung family)
    final_tables/table8_ablation.md   - Table 8 (5 ablation + full IAPC)
"""
import argparse
import csv
import statistics
from pathlib import Path

UNIFORM = ['BS1', 'BS2', 'BS3', 'CS1', 'CS2', 'CS3', 'GS1', 'GS2', 'GS3']
ALL_COND = UNIFORM + ['US1', 'US2', 'US3', 'DS1', 'DS2', 'DS3']
FAMILY_COLS = {
    'Bright.': ['BS1', 'BS2', 'BS3'], 'Contrast': ['CS1', 'CS2', 'CS3'],
    'Gauss.': ['GS1', 'GS2', 'GS3'], 'Uneven': ['US1', 'US2', 'US3'],
    'Dappled': ['DS1', 'DS2', 'DS3'],
}

TABLE5_SYSTEMS = [
    ('mask_rcnn_r50_fpn', 'Mask R-CNN R50 (FPN)'),
    ('mask_rcnn_r101_fpn', 'Mask R-CNN R101'),
    ('mask_rcnn_r50_cbam_fpn', 'CBAM-FPN'),
    ('mask_rcnn_r50_bifpn', 'BiFPN'),
    ('mask_rcnn_r50_fpn_aug', 'CPU photometric aug'),
    ('mask_rcnn_r50_gpuaug', 'IAPC control (l=0)'),
    ('mask_rcnn_r50_iapc_lam0p25', 'IAPC (l=0.25)'),
]

TABLE8_SYSTEMS = [
    ('mask_rcnn_r50_iapc_lam0p25', 'IAPC (full)'),
    ('mask_rcnn_r50_abl_global', 'w/o instance-awareness (global)'),
    ('mask_rcnn_r50_abl_p2only', 'P2 only'),
    ('mask_rcnn_r50_abl_cosonly', 'cosine only (alpha=1)'),
    ('mask_rcnn_r50_abl_l1only', 'l1 only (alpha=0)'),
    ('mask_rcnn_r50_abl_nosg', 'w/o stop-gradient'),
]


def load(path):
    data = {}  # system -> list of seed rows (dict of float)
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            system = row['System']
            vals = {c: float(row[c]) for c in ['Clean'] + ALL_COND
                    if c in row and row[c] not in (None, '')}
            data.setdefault(system, []).append(vals)
    return data


def system_stats(rows):
    """Tra ve dict: cot -> (mean, std), tinh tren cac seed co du lieu."""
    out = {}
    cols = set()
    for r in rows:
        cols.update(r.keys())
    for c in cols:
        vals = [r[c] for r in rows if c in r]
        if not vals:
            continue
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        out[c] = (mean, std, len(vals))
    return out


def apcorr_worst(stats):
    means = {c: stats[c][0] for c in UNIFORM if c in stats}
    if len(means) < 9:
        return None, None, None
    apcorr = statistics.mean(means.values())
    worst_cond = min(means, key=means.get)
    worst_ap = means[worst_cond]
    return apcorr, worst_ap, worst_cond


def write_table6(data, out_dir):
    header = ['System', 'Clean'] + ALL_COND + ['APcorr', 'WorstAP', 'WorstCond', 'RD', 'SI']
    md_lines = ['| ' + ' | '.join(header) + ' |',
                '|' + '---|' * len(header)]
    csv_rows = []

    for system in sorted(data.keys()):
        stats = system_stats(data[system])
        apcorr, worst_ap, worst_cond = apcorr_worst(stats)
        clean = stats.get('Clean', (None,))[0]
        row_vals = [system]
        row_vals.append(f'{clean:.3f}' if clean is not None else '-')
        for c in ALL_COND:
            if c in stats:
                m, s, n = stats[c]
                row_vals.append(f'{m:.3f}')
            else:
                row_vals.append('-')
        if apcorr is not None:
            rd = clean - apcorr if clean is not None else None
            si = apcorr / clean if clean else None
            row_vals += [f'{apcorr:.3f}', f'{worst_ap:.3f}', worst_cond,
                         f'{rd:.3f}' if rd is not None else '-',
                         f'{si:.3f}' if si is not None else '-']
        else:
            row_vals += ['-', '-', '-', '-', '-']
        md_lines.append('| ' + ' | '.join(row_vals) + ' |')
        csv_rows.append(dict(zip(header, row_vals)))

    (out_dir / 'table6_full.md').write_text('\n'.join(md_lines), encoding='utf-8')
    with open(out_dir / 'table6_full.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(csv_rows)
    print(f'Da ghi: {out_dir / "table6_full.md"}  va  {out_dir / "table6_full.csv"}')


def write_table5(data, out_dir):
    header = ['System'] + list(FAMILY_COLS.keys()) + ['APcorr']
    md_lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    for sys_key, label in TABLE5_SYSTEMS:
        if sys_key not in data:
            md_lines.append(f'| {label} | (khong co du lieu: {sys_key}) |' + ' |' * (len(header) - 2))
            continue
        stats = system_stats(data[sys_key])
        row = [label]
        for fam, cols in FAMILY_COLS.items():
            vals = [stats[c][0] for c in cols if c in stats]
            row.append(f'{statistics.mean(vals):.3f}' if vals else '-')
        apcorr, _, _ = apcorr_worst(stats)
        row.append(f'{apcorr:.3f}' if apcorr is not None else '-')
        md_lines.append('| ' + ' | '.join(row) + ' |')
    (out_dir / 'table5_summary.md').write_text('\n'.join(md_lines), encoding='utf-8')
    print(f'Da ghi: {out_dir / "table5_summary.md"}')


def write_table8(data, out_dir):
    header = ['Variant', 'Clean', 'APcorr']
    md_lines = ['| ' + ' | '.join(header) + ' |', '|---|---|---|']
    for sys_key, label in TABLE8_SYSTEMS:
        if sys_key not in data:
            md_lines.append(f'| {label} | (khong co du lieu: {sys_key}) | |')
            continue
        stats = system_stats(data[sys_key])
        clean = stats.get('Clean', (None,))[0]
        apcorr, _, _ = apcorr_worst(stats)
        md_lines.append(
            f'| {label} | {clean:.3f} | {apcorr:.3f} |'
            if clean is not None and apcorr is not None else f'| {label} | - | - |')
    (out_dir / 'table8_ablation.md').write_text('\n'.join(md_lines), encoding='utf-8')
    print(f'Da ghi: {out_dir / "table8_ablation.md"}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load(args.input)

    print(f'Doc duoc {len(data)} he thong tu {args.input}')
    write_table6(data, out_dir)
    write_table5(data, out_dir)
    write_table8(data, out_dir)


if __name__ == '__main__':
    main()
