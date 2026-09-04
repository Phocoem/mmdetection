"""
Doc so lieu PER-SEED (khong phai da gop trung binh nhu Table 6) va tinh
seed-level paired t-test giua 2 he thong bat ky - dung DUNG phuong phap
luan Section 5.4 cua bai (t-test tren cac lan train doc lap, KHONG PHAI
bootstrap tren anh test).

=== DINH DANG INPUT CAN THIET ===
CSV "wide", MOI DONG la 1 (he thong, seed) - tuc 1 he thong xuat hien
NHIEU dong, moi dong ung voi 1 seed:

    System,Seed,Clean,BS1,BS2,BS3,CS1,CS2,CS3,GS1,GS2,GS3,US1,US2,US3,DS1,DS2,DS3

Vi du (FPN+Aug chay 5 seed):
    mask_rcnn_r50_fpn_aug,2024,0.729,...
    mask_rcnn_r50_fpn_aug,2025,0.727,...
    mask_rcnn_r50_fpn_aug,2026,0.731,...
    mask_rcnn_r50_fpn_aug,2027,0.726,...
    mask_rcnn_r50_fpn_aug,2028,0.728,...

Day CHINH LA dinh dang can co trong file #7 da de nghi o
"file_can_de_xac_thuc.txt" (raw per-seed results) - neu ket qua that cua
ban dang o dinh dang khac (JSON tu MMDetection, log file...), noi ro dinh
dang that, toi viet them ham convert.

=== CACH DUNG ===
    # Xem toan bo bang APcorr/std tung he thong (nhu Section 4.9)
    python seed_level_ttest.py --input per_seed.csv --summary

    # So sanh 2 he thong cu the (vd. cau hoi FPN+Aug vs IAPC lambda=0.25)
    python seed_level_ttest.py --input per_seed.csv \\
        --compare mask_rcnn_r50_fpn_aug mask_rcnn_r50_iapc_lam0p25

    # Chi so sanh tren 1 dieu kien cu the thay vi APcorr trung binh
    python seed_level_ttest.py --input per_seed.csv \\
        --compare mask_rcnn_r50_fpn_aug mask_rcnn_r50_iapc_lam0p25 \\
        --condition CS3
"""
import argparse
import csv
import statistics
from collections import defaultdict

from scipy import stats

UNIFORM_COLS = ['BS1', 'BS2', 'BS3', 'CS1', 'CS2', 'CS3', 'GS1', 'GS2', 'GS3']
ALL_COND_COLS = UNIFORM_COLS + ['US1', 'US2', 'US3', 'DS1', 'DS2', 'DS3']


def load(path: str):
    """Tra ve dict: system -> seed -> {condition: AP, ..., 'APcorr': x}."""
    data = defaultdict(dict)
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        missing = [c for c in ['System', 'Seed'] + UNIFORM_COLS
                   if c not in reader.fieldnames]
        if missing:
            raise ValueError(f'Thieu cot bat buoc trong input: {missing}')
        skipped = []
        for row in reader:
            system = row['System']
            seed = row['Seed']
            values = {c: float(row[c]) for c in ALL_COND_COLS if c in row and row[c]}
            if 'Clean' in row and row['Clean']:
                values['Clean'] = float(row['Clean'])

            missing_uniform = [c for c in UNIFORM_COLS if c not in values]
            if missing_uniform:
                skipped.append((system, seed, missing_uniform))
                continue  # khong du 9 dieu kien de tinh APcorr - bo qua dong nay

            values['APcorr'] = statistics.mean(
                values[c] for c in UNIFORM_COLS)
            data[system][seed] = values

        if skipped:
            print('CANH BAO: bo qua cac (he thong, seed) thieu du lieu de '
                  'tinh APcorr (can du 9 dieu kien BS1-3/CS1-3/GS1-3):')
            for system, seed, missing in skipped:
                print(f'  - {system} / seed {seed}: thieu {missing}')
            print()
    return data


def paired_ttest(a, b):
    """Paired t-test thu cong, dung dung cong thuc seed-level cua bai:
    t = mean(diff) / (std(diff)/sqrt(n)), df = n-1."""
    if len(a) != len(b):
        raise ValueError(
            f'So seed khong khop: {len(a)} vs {len(b)} - can 2 he thong '
            f'chay TREN CUNG mot tap seed de ghep cap dung.')
    n = len(a)
    diffs = [x - y for x, y in zip(a, b)]
    mean_diff = statistics.mean(diffs)
    std_diff = statistics.stdev(diffs) if n > 1 else 0.0
    se = std_diff / (n ** 0.5) if n > 1 and std_diff > 0 else float('nan')
    if std_diff == 0:
        t = float('inf') if mean_diff != 0 else 0.0
        p = 0.0 if mean_diff != 0 else 1.0
    else:
        t = mean_diff / se
        df = n - 1
        p = float(stats.t.sf(abs(t), df) * 2)  # two-tailed
    df = n - 1
    return dict(n=n, mean_diff=mean_diff, std_diff=std_diff, t=t, df=df, p=p)


def print_summary(data):
    header = f'{"System":<32}{"n_seed":>7}{"mean APcorr":>13}{"std":>8}'
    print(header)
    print('-' * len(header))
    for system, per_seed in sorted(data.items()):
        apcorrs = [v['APcorr'] for v in per_seed.values()]
        n = len(apcorrs)
        mean = statistics.mean(apcorrs)
        std = statistics.stdev(apcorrs) if n > 1 else 0.0
        print(f'{system:<32}{n:>7}{mean:>13.4f}{std:>8.4f}')


def compare(data, sys_a, sys_b, condition=None):
    for s in (sys_a, sys_b):
        if s not in data:
            raise ValueError(f'Khong tim thay he thong "{s}" trong input. '
                              f'Cac he thong co san: {sorted(data.keys())}')

    seeds_a = set(data[sys_a].keys())
    seeds_b = set(data[sys_b].keys())
    common = sorted(seeds_a & seeds_b)
    if seeds_a != seeds_b:
        print(f'CANH BAO: 2 he thong khong chay tren CUNG tap seed.\n'
              f'  {sys_a}: {sorted(seeds_a)}\n'
              f'  {sys_b}: {sorted(seeds_b)}\n'
              f'  -> chi dung {len(common)} seed chung: {common}\n')

    metric = condition if condition else 'APcorr'
    a_vals = [data[sys_a][s][metric] for s in common]
    b_vals = [data[sys_b][s][metric] for s in common]

    result = paired_ttest(a_vals, b_vals)

    print(f'So sanh: {sys_a}  vs  {sys_b}  (chi so: {metric})')
    print(f'  Seed dung          : {common}')
    print(f'  {sys_a:<28}: {[f"{v:.4f}" for v in a_vals]}')
    print(f'  {sys_b:<28}: {[f"{v:.4f}" for v in b_vals]}')
    print(f'  n (so cap seed)    : {result["n"]}')
    print(f'  Chenh lech trung binh (A-B): {result["mean_diff"]:+.4f}')
    print(f'  Do lech chuan cua chenh lech: {result["std_diff"]:.4f}')
    print(f'  t = {result["t"]:.3f}   (df = {result["df"]})')
    print(f'  p (hai duoi)       : {result["p"]:.4f}')
    if result['df'] < 2:
        print('  *** CANH BAO: df qua nho (< 2), ket qua khong dang tin '
              'cay. Can it nhat 3 seed chung, ly tuong 5+ nhu Nhom A #10. ***')
    alpha = 0.05
    verdict = 'CO Y NGHIA' if result['p'] < alpha else 'KHONG co y nghia'
    print(f'  -> Voi alpha={alpha}: {verdict} (p {"<" if result["p"]<alpha else ">="} {alpha})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True,
                         help='CSV dinh dang wide, 1 dong = 1 (system, seed)')
    parser.add_argument('--summary', action='store_true',
                         help='In bang tong hop mean/std APcorr tung he thong')
    parser.add_argument('--compare', nargs=2, metavar=('SYSTEM_A', 'SYSTEM_B'),
                         help='So sanh 2 he thong bang paired t-test')
    parser.add_argument('--condition', default=None,
                         help='Chi so sanh tren 1 dieu kien (vd CS3) thay vi '
                              'APcorr trung binh 9 dieu kien')
    args = parser.parse_args()

    data = load(args.input)

    if args.summary or not args.compare:
        print_summary(data)
        print()

    if args.compare:
        compare(data, args.compare[0], args.compare[1], args.condition)
