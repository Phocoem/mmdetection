"""
Doc DAY DU tat ca thu muc con dang "evaluation*" duoi moi seed_<seed>/
(evaluation/, evaluation_spatial/, hoac bat ky ten nao khac bat dau bang
"evaluation") - THAY THE convert_mmdet_logs_to_csv.py, vi ban do CHI quet
dung "evaluation/", bo sot hoan toan "evaluation_spatial/" da phat hien.

Ho tro CA 2 dinh dang trong MOI thu muc con:
    A) <eval_dir>/condition_metrics.csv
    B) <eval_dir>/conditions/<condition>/metrics.json

Cach dung:
    python read_full_evaluation.py \\
        --root /home/pc/mmdet_AI/mmdetection/work_dirs/research \\
        --output per_seed_full.csv

Xuat canh bao ro rang neu 1 dieu kien xuat hien o NHIEU thu muc evaluation*
voi gia tri KHAC NHAU (dau hieu du lieu cu/moi lan lon).
"""
import argparse
import csv
import json
import re
from pathlib import Path

METRIC_COL = 'coco/segm_mAP'

COND_PREFIX = {
    'brightness': 'B', 'contrast': 'C', 'gaussian_noise': 'G',
    'uneven_contrast': 'U', 'dappled_light': 'D',
}
COND_RE = re.compile(
    r'^(brightness|contrast|gaussian_noise|uneven_contrast|dappled_light)_s(\d)$')
OUTPUT_COLS = ['Clean', 'BS1', 'BS2', 'BS3', 'CS1', 'CS2', 'CS3',
               'GS1', 'GS2', 'GS3', 'US1', 'US2', 'US3', 'DS1', 'DS2', 'DS3']
SEED_DIR_RE = re.compile(r'^seed_(\d+)$')


def condition_to_col(condition: str):
    if condition == 'clean':
        return 'Clean'
    m = COND_RE.match(condition)
    if not m:
        return None
    family, severity = m.groups()
    return f'{COND_PREFIX[family]}S{severity}'


def parse_condition_metrics_csv(path: Path) -> dict:
    out = {}
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if 'condition' not in reader.fieldnames or METRIC_COL not in reader.fieldnames:
            return out
        for row in reader:
            col = condition_to_col(row['condition'])
            if col is not None and row.get(METRIC_COL):
                out[col] = float(row[METRIC_COL])
    return out


def parse_conditions_dir(conditions_dir: Path) -> dict:
    out = {}
    for cond_dir in conditions_dir.iterdir():
        if not cond_dir.is_dir():
            continue
        json_path = cond_dir / 'metrics.json'
        if not json_path.exists():
            continue
        col = condition_to_col(cond_dir.name)
        if col is None:
            continue
        with open(json_path, encoding='utf-8') as f:
            payload = json.load(f)
        if METRIC_COL in payload:
            out[col] = float(payload[METRIC_COL])
    return out


def parse_one_eval_dir(eval_dir: Path) -> dict:
    """Doc 1 thu muc evaluation* don le (co the la CSV hoac JSON-per-condition
    hoac ca 2). Tra ve {col: value}."""
    out = {}
    csv_path = eval_dir / 'condition_metrics.csv'
    conditions_dir = eval_dir / 'conditions'
    if csv_path.exists():
        out.update(parse_condition_metrics_csv(csv_path))
    if conditions_dir.is_dir():
        for k, v in parse_conditions_dir(conditions_dir).items():
            out.setdefault(k, v)  # neu CSV da co roi thi giu CSV, khong ghi de
    return out


def find_eval_dirs(seed_dir: Path):
    """TAT CA thu muc con ten bat dau bang 'evaluation' - khong chi
    'evaluation/' ma ca 'evaluation_spatial/' va bat ky bien the nao khac."""
    return sorted(p for p in seed_dir.iterdir()
                   if p.is_dir() and p.name.startswith('evaluation'))


def read_full_seed(seed_dir: Path, verbose_path_label: str):
    """Gop dieu kien tu TAT CA thu muc evaluation* duoi 1 seed_dir.
    Tra ve (merged_dict, list_nguon_da_doc)."""
    merged = {}
    sources = {}
    dirs_read = []
    for eval_dir in find_eval_dirs(seed_dir):
        values = parse_one_eval_dir(eval_dir)
        if not values:
            continue
        dirs_read.append(eval_dir.name)
        for col, val in values.items():
            if col in merged:
                if abs(merged[col] - val) > 1e-4:
                    print(f'  CANH BAO MAU THUAN: {verbose_path_label} - dieu kien '
                          f'{col} co o CA "{sources[col]}" ({merged[col]}) VA '
                          f'"{eval_dir.name}" ({val}) - gia tri KHAC NHAU. '
                          f'Dang giu gia tri tu "{sources[col]}".')
                continue
            merged[col] = val
            sources[col] = eval_dir.name
    return merged, dirs_read


def scan(root: Path, systems=None):
    rows = []
    system_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if systems:
        wanted = set(systems)
        system_dirs = [p for p in system_dirs if p.name in wanted]

    for sys_dir in system_dirs:
        seed_dirs = sorted(
            (p for p in sys_dir.iterdir() if p.is_dir() and SEED_DIR_RE.match(p.name)),
            key=lambda p: int(SEED_DIR_RE.match(p.name).group(1)))
        for seed_dir in seed_dirs:
            seed = SEED_DIR_RE.match(seed_dir.name).group(1)
            label = f'{sys_dir.name}/seed_{seed}'
            merged, dirs_read = read_full_seed(seed_dir, label)
            if not dirs_read:
                print(f'  CANH BAO: {seed_dir} khong co thu muc evaluation* nao co du lieu.')
                continue
            print(f'Doc {label}: gop tu {dirs_read} -> {len(merged)}/16 dieu kien')

            row = {'System': sys_dir.name, 'Seed': seed}
            for col in OUTPUT_COLS:
                row[col] = f'{merged[col]:.4f}' if col in merged else ''
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--systems', nargs='*', default=None)
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f'Khong tim thay thu muc: {root}')

    rows = scan(root, args.systems)
    if not rows:
        raise SystemExit('Khong doc duoc du lieu nao.')

    fieldnames = ['System', 'Seed'] + OUTPUT_COLS
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_systems = len({r['System'] for r in rows})
    print(f'\nDa ghi {len(rows)} dong ({n_systems} he thong) vao {args.output}')


if __name__ == '__main__':
    main()
