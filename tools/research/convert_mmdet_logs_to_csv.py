"""
Quet toan bo cay thu muc, tu dong nhan dien 1 trong 2 DINH DANG KHAC NHAU
(da phat hien work_dirs cua ban dang TRON LAN ca 2 kieu giua cac he thong -
xem canh bao "DINH DANG KHONG DONG NHAT" o cuoi neu gap):

DINH DANG A - 1 file CSV gop het dieu kien (vd mask_rcnn_r50_iapc_lam0p25):
    <root>/<system>/seed_<seed>/evaluation/condition_metrics.csv
    Cot bat buoc: condition, coco/segm_mAP

DINH DANG B - 1 file JSON rieng cho MOI dieu kien (vd mask_rcnn_r50_fpn_aug):
    <root>/<system>/seed_<seed>/evaluation/conditions/<condition>/metrics.json
    Vi du noi dung metrics.json:
        {"coco/segm_mAP": 0.716, "coco/segm_mAP_50": 0.947, ...}
    <condition> (ten thu muc con) chinh la ten dieu kien, vd "brightness_s1".

Ca 2 dinh dang deu dung chung quy uoc ten dieu kien:
    clean
    brightness_s1/s2/s3
    contrast_s1/s2/s3
    gaussian_noise_s1/s2/s3
    uneven_contrast_s1/s2/s3
    dappled_light_s1/s2/s3
(dieu kien nao thieu se de trong o output, kem canh bao)

Gop lai thanh 1 CSV "wide" per-seed, dung truc tiep cho:
    - seed_level_ttest.py  (--input file_nay.csv)
    - compute_worst_case_ap.py (sau khi bo cot Seed / gop theo he thong)

=== OUTPUT ===
    System,Seed,Clean,BS1,BS2,BS3,CS1,CS2,CS3,GS1,GS2,GS3,US1,US2,US3,DS1,DS2,DS3

=== CACH DUNG ===
    python convert_mmdet_logs_to_csv.py \\
        --root /home/pc/mmdet_AI/mmdetection/work_dirs/research \\
        --output per_seed_full.csv

    # Chi quet mot vai he thong cu the (bo qua he thong chua co ket qua)
    python convert_mmdet_logs_to_csv.py \\
        --root /home/pc/mmdet_AI/mmdetection/work_dirs/research \\
        --systems mask_rcnn_r50_fpn_aug mask_rcnn_r50_iapc_lam0p25 \\
        --output per_seed_2systems.csv
"""
import argparse
import csv
import json
import re
from pathlib import Path

METRIC_COL = 'coco/segm_mAP'

# ten dieu kien trong condition_metrics.csv -> ma ngan dung trong output
COND_PREFIX = {
    'brightness': 'B',
    'contrast': 'C',
    'gaussian_noise': 'G',
    'uneven_contrast': 'U',
    'dappled_light': 'D',
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
    """DINH DANG A: 1 file CSV gop het dieu kien.
    -> {'BS1': 0.727, 'Clean': 0.739, ...}."""
    out = {}
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if 'condition' not in reader.fieldnames:
            raise ValueError(f'{path}: thieu cot "condition"')
        if METRIC_COL not in reader.fieldnames:
            raise ValueError(f'{path}: thieu cot "{METRIC_COL}"')
        for row in reader:
            col = condition_to_col(row['condition'])
            if col is None:
                print(f'  CANH BAO: khong nhan dien duoc condition '
                      f'"{row["condition"]}" trong {path}, bo qua dong nay.')
                continue
            out[col] = float(row[METRIC_COL])
    return out


def parse_conditions_dir(conditions_dir: Path) -> dict:
    """DINH DANG B: 1 thu muc con cho MOI dieu kien, moi thu muc co
    metrics.json rieng. -> {'BS1': 0.727, 'Clean': 0.739, ...}."""
    out = {}
    cond_dirs = sorted(p for p in conditions_dir.iterdir() if p.is_dir())
    for cond_dir in cond_dirs:
        json_path = cond_dir / 'metrics.json'
        if not json_path.exists():
            print(f'  CANH BAO: khong thay {json_path}, bo qua dieu kien '
                  f'"{cond_dir.name}".')
            continue
        col = condition_to_col(cond_dir.name)
        if col is None:
            print(f'  CANH BAO: khong nhan dien duoc condition '
                  f'"{cond_dir.name}" (thu muc {cond_dir}), bo qua.')
            continue
        with open(json_path, encoding='utf-8') as f:
            payload = json.load(f)
        if METRIC_COL not in payload:
            print(f'  CANH BAO: {json_path} thieu key "{METRIC_COL}", bo qua.')
            continue
        out[col] = float(payload[METRIC_COL])
    return out


def parse_evaluation_dir(eval_dir: Path):
    """Tu dong nhan dien Dinh dang A hay B trong 1 thu muc evaluation/.
    Tra ve (values_dict, format_name) hoac (None, None) neu khong tim thay gi."""
    csv_path = eval_dir / 'condition_metrics.csv'
    conditions_dir = eval_dir / 'conditions'

    has_csv = csv_path.exists()
    has_dir = conditions_dir.is_dir()

    if has_csv and has_dir:
        print(f'  CANH BAO: {eval_dir} co CA HAI dinh dang (condition_metrics.csv '
              f'va conditions/) - dang uu tien dung CSV. Kiem tra xem 2 nguon '
              f'nay co khop nhau khong, tranh du lieu cu con sot lai.')
        return parse_condition_metrics_csv(csv_path), 'CSV'
    if has_csv:
        return parse_condition_metrics_csv(csv_path), 'CSV'
    if has_dir:
        return parse_conditions_dir(conditions_dir), 'JSON-per-condition'
    return None, None


def scan(root: Path, systems=None):
    """Quet <root>/<system>/seed_<seed>/evaluation/condition_metrics.csv.
    Tra ve list[dict] moi dict la 1 dong output (System, Seed, cac dieu kien)."""
    rows = []
    system_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if systems:
        wanted = set(systems)
        system_dirs = [p for p in system_dirs if p.name in wanted]
        found = {p.name for p in system_dirs}
        missing = wanted - found
        if missing:
            print(f'CANH BAO: khong tim thay thu muc he thong: {sorted(missing)}')

    formats_used = {}  # system -> set cac dinh dang gap phai (de canh bao)

    for sys_dir in system_dirs:
        seed_dirs = sorted(
            (p for p in sys_dir.iterdir() if p.is_dir() and SEED_DIR_RE.match(p.name)),
            key=lambda p: int(SEED_DIR_RE.match(p.name).group(1)))
        if not seed_dirs:
            continue
        for seed_dir in seed_dirs:
            seed = SEED_DIR_RE.match(seed_dir.name).group(1)
            eval_dir = seed_dir / 'evaluation'
            if not eval_dir.is_dir():
                print(f'  CANH BAO: khong thay {eval_dir}, bo qua.')
                continue

            values, fmt = parse_evaluation_dir(eval_dir)
            if values is None:
                print(f'  CANH BAO: {eval_dir} khong co condition_metrics.csv '
                      f'lan khong co thu muc conditions/, bo qua.')
                continue
            print(f'Doc ({fmt}): {eval_dir}')
            formats_used.setdefault(sys_dir.name, set()).add(fmt)

            row = {'System': sys_dir.name, 'Seed': seed}
            missing_cols = []
            for col in OUTPUT_COLS:
                if col in values:
                    row[col] = f'{values[col]:.4f}'
                else:
                    row[col] = ''
                    missing_cols.append(col)
            if missing_cols:
                print(f'  CANH BAO: {eval_dir} thieu dieu kien: {missing_cols}')
            rows.append(row)

    inconsistent = {s: f for s, f in formats_used.items() if len(f) > 1}
    mixed_across_systems = len({f for fs in formats_used.values() for f in fs}) > 1
    if inconsistent:
        print('\nCANH BAO: cac he thong sau dung LAN LON nhieu dinh dang giua '
              'cac seed khac nhau (co the la du lieu tu 2 lan chay khac nhau):')
        for s, f in inconsistent.items():
            print(f'  - {s}: {sorted(f)}')
    if mixed_across_systems:
        print('\nGHI CHU: cac he thong khac nhau trong work_dirs nay dang dung '
              'dinh dang luu ket qua khac nhau (CSV gop vs JSON-per-condition). '
              'Neu 2 pipeline danh gia nay khong hoan toan giong nhau (vd khac '
              'phien ban script, khac cach lam tron so, khac cach xu ly '
              'NaN/instance rong), so sanh giua cac he thong co the khong con '
              'apples-to-apples. Nen kiem tra lai 2 script sinh ra 2 dinh dang '
              'nay co cung logic tinh COCO mask AP hay khong.')
        for s, f in sorted(formats_used.items()):
            print(f'  - {s}: {sorted(f)}')

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True,
                         help='Thu muc goc chua cac he thong, vd '
                              'work_dirs/research')
    parser.add_argument('--output', required=True)
    parser.add_argument('--systems', nargs='*', default=None,
                         help='Chi quet cac he thong nay (mac dinh: quet het)')
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f'Khong tim thay thu muc: {root}')

    rows = scan(root, args.systems)
    if not rows:
        raise SystemExit('Khong tim thay ket qua nao - kiem tra lai --root '
                          'va cau truc thu muc.')

    fieldnames = ['System', 'Seed'] + OUTPUT_COLS
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_systems = len({r['System'] for r in rows})
    print(f'\nDa ghi {len(rows)} dong ({n_systems} he thong) vao {args.output}')


if __name__ == '__main__':
    main()
