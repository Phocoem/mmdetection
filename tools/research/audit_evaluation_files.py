"""
Doc SAU tung file ket qua (khong chi gop thanh 1 dong nhu
convert_mmdet_logs_to_csv.py) - kiem tra tinh hop le, day du, nhat quan
CUA TUNG FILE truoc khi dua vao t-test.

Quet ca 2 dinh dang (giong convert_mmdet_logs_to_csv.py):
    A) <root>/<system>/seed_<seed>/evaluation/condition_metrics.csv
    B) <root>/<system>/seed_<seed>/evaluation/conditions/<condition>/metrics.json

Voi MOI file tim thay, kiem tra:
    1. Du 6 key/cot metric khong: segm_mAP, segm_mAP_50, segm_mAP_75,
       segm_mAP_s, segm_mAP_m, segm_mAP_l
    2. Gia tri co nam trong [0,1] khong (AP am hoac >1 la dau hieu loi doc/
       loi don vi)
    3. segm_mAP <= segm_mAP_50 (AP50 la nguong long nhat, phai >= AP trung
       binh) - vi pham la dau hieu du lieu hong hoac doc nham cot
    4. segm_mAP_75 <= segm_mAP_50 - cung ly do tren
    5. Co dieu kien nao bi LAP (xuat hien 2 lan trong 1 file CSV) khong
    6. Thoi gian sua file (mtime) - de phat hien cac file duoc sinh o
       THOI DIEM CACH XA NHAU BAT THUONG trong cung 1 he thong (dau hieu
       co the la 2 lan chay/2 phien ban script khac nhau, khong chi khac
       dinh dang luu)

Xuat ra:
    - Ma tran day du: he thong x seed -> so dieu kien co/16, dinh dang, mtime
    - Danh sach TUNG loi/canh bao cu the, kem duong dan file chinh xac
    - (tuy chon --out-report) ghi toan bo report ra file .txt

Cach dung:
    python audit_evaluation_files.py \\
        --root /home/pc/mmdet_AI/mmdetection/work_dirs/research \\
        --out-report audit_report.txt
"""
import argparse
import csv
import json
import re
import statistics
from datetime import datetime
from pathlib import Path

METRIC_KEYS = ['coco/segm_mAP', 'coco/segm_mAP_50', 'coco/segm_mAP_75',
               'coco/segm_mAP_s', 'coco/segm_mAP_m', 'coco/segm_mAP_l']

COND_PREFIX = {
    'brightness': 'B', 'contrast': 'C', 'gaussian_noise': 'G',
    'uneven_contrast': 'U', 'dappled_light': 'D',
}
COND_RE = re.compile(
    r'^(brightness|contrast|gaussian_noise|uneven_contrast|dappled_light)_s(\d)$')
SEED_DIR_RE = re.compile(r'^seed_(\d+)$')

EXPECTED_CONDITIONS = (
    ['clean'] +
    [f'{fam}_s{s}' for fam in
     ('brightness', 'contrast', 'gaussian_noise',
      'uneven_contrast', 'dappled_light')
     for s in (1, 2, 3)]
)  # 16 dieu kien


def condition_to_col(condition: str):
    if condition == 'clean':
        return 'Clean'
    m = COND_RE.match(condition)
    if not m:
        return None
    family, severity = m.groups()
    return f'{COND_PREFIX[family]}S{severity}'


class Issue:
    def __init__(self, path, severity, message):
        self.path = path
        self.severity = severity  # 'LOI' hoac 'CANH BAO'
        self.message = message

    def __str__(self):
        return f'[{self.severity}] {self.path}\n    {self.message}'


def check_metric_values(payload: dict, path: Path, issues: list):
    """Kiem tra 1 dict metric (tu 1 dieu kien) - dung 5 quy tac dau."""
    missing = [k for k in METRIC_KEYS if k not in payload]
    if missing:
        issues.append(Issue(path, 'LOI', f'Thieu key: {missing}'))
        return

    for k in METRIC_KEYS:
        v = payload[k]
        if v is None:
            issues.append(Issue(path, 'LOI', f'{k} = null'))
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            issues.append(Issue(path, 'LOI', f'{k} khong phai so: {v!r}'))
            continue
        if not (0.0 <= v <= 1.0):
            issues.append(Issue(path, 'LOI',
                                 f'{k} = {v} nam ngoai khoang [0,1]'))

    ap = payload.get('coco/segm_mAP')
    ap50 = payload.get('coco/segm_mAP_50')
    ap75 = payload.get('coco/segm_mAP_75')
    try:
        if ap is not None and ap50 is not None and float(ap) > float(ap50):
            issues.append(Issue(
                path, 'CANH BAO',
                f'segm_mAP ({ap}) > segm_mAP_50 ({ap50}) - bat thuong, '
                f'AP50 (nguong long nhat) thuong phai >= AP trung binh'))
        if ap75 is not None and ap50 is not None and float(ap75) > float(ap50):
            issues.append(Issue(
                path, 'CANH BAO',
                f'segm_mAP_75 ({ap75}) > segm_mAP_50 ({ap50}) - bat thuong'))
    except (TypeError, ValueError):
        pass  # da bao loi khong phai so o tren roi


def audit_csv_file(csv_path: Path, issues: list):
    """Audit dinh dang A (condition_metrics.csv). Tra ve set cac dieu kien
    (ten col ngan) doc duoc thanh cong."""
    found_cols = set()
    seen_conditions = {}
    try:
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if 'condition' not in (reader.fieldnames or []):
                issues.append(Issue(csv_path, 'LOI', 'Thieu cot "condition"'))
                return found_cols
            for row in reader:
                cond = row['condition']
                seen_conditions[cond] = seen_conditions.get(cond, 0) + 1
                col = condition_to_col(cond)
                if col is None:
                    issues.append(Issue(
                        csv_path, 'CANH BAO',
                        f'Ten dieu kien khong nhan dien duoc: "{cond}"'))
                    continue
                payload = {}
                for k in METRIC_KEYS:
                    if k in row and row[k] != '':
                        payload[k] = row[k]
                check_metric_values(payload, csv_path, issues)
                if not any(i.path == csv_path and cond in i.message
                           for i in issues[-1:]):
                    found_cols.add(col)
    except Exception as e:  # noqa: BLE001 - can bao het loi doc file
        issues.append(Issue(csv_path, 'LOI', f'Khong doc duoc file: {e}'))
        return found_cols

    dup = {c: n for c, n in seen_conditions.items() if n > 1}
    if dup:
        issues.append(Issue(csv_path, 'LOI',
                             f'Dieu kien bi LAP trong file: {dup}'))

    missing_expected = set(EXPECTED_CONDITIONS) - set(seen_conditions)
    if missing_expected:
        issues.append(Issue(
            csv_path, 'CANH BAO',
            f'Thieu {len(missing_expected)}/16 dieu kien: '
            f'{sorted(missing_expected)}'))
    return found_cols


def audit_conditions_dir(conditions_dir: Path, issues: list):
    """Audit dinh dang B (conditions/<cond>/metrics.json). Tra ve set cac
    dieu kien doc duoc thanh cong."""
    found_cols = set()
    seen = set()
    for cond_dir in sorted(p for p in conditions_dir.iterdir() if p.is_dir()):
        seen.add(cond_dir.name)
        json_path = cond_dir / 'metrics.json'
        if not json_path.exists():
            issues.append(Issue(json_path, 'LOI', 'File khong ton tai'))
            continue
        try:
            with open(json_path, encoding='utf-8') as f:
                payload = json.load(f)
        except json.JSONDecodeError as e:
            issues.append(Issue(json_path, 'LOI', f'JSON khong hop le: {e}'))
            continue
        except Exception as e:  # noqa: BLE001
            issues.append(Issue(json_path, 'LOI', f'Khong doc duoc file: {e}'))
            continue

        check_metric_values(payload, json_path, issues)
        col = condition_to_col(cond_dir.name)
        if col is None:
            issues.append(Issue(json_path, 'CANH BAO',
                                 f'Ten thu muc dieu kien khong nhan dien '
                                 f'duoc: "{cond_dir.name}"'))
        else:
            found_cols.add(col)

    missing_expected = set(EXPECTED_CONDITIONS) - seen
    if missing_expected:
        issues.append(Issue(
            conditions_dir, 'CANH BAO',
            f'Thieu {len(missing_expected)}/16 dieu kien (thu muc con): '
            f'{sorted(missing_expected)}'))
    return found_cols


def audit_seed_dir(seed_dir: Path, issues: list):
    """Tra ve (n_conditions_found, format_name, mtime_min, mtime_max)."""
    eval_dir = seed_dir / 'evaluation'
    if not eval_dir.is_dir():
        issues.append(Issue(seed_dir, 'LOI', 'Khong co thu muc evaluation/'))
        return 0, None, None, None

    csv_path = eval_dir / 'condition_metrics.csv'
    conditions_dir = eval_dir / 'conditions'
    has_csv, has_dir = csv_path.exists(), conditions_dir.is_dir()

    if not has_csv and not has_dir:
        issues.append(Issue(eval_dir, 'LOI',
                             'Khong co condition_metrics.csv lan khong co '
                             'thu muc conditions/'))
        return 0, None, None, None

    if has_csv and has_dir:
        issues.append(Issue(eval_dir, 'CANH BAO',
                             'Co CA HAI dinh dang cung luc - kiem tra xem '
                             'co phai du lieu cu con sot lai khong'))

    all_files = []
    if has_csv:
        found = audit_csv_file(csv_path, issues)
        fmt = 'CSV'
        all_files.append(csv_path)
    else:
        found = audit_conditions_dir(conditions_dir, issues)
        fmt = 'JSON-per-condition'
        all_files.extend(conditions_dir.glob('*/metrics.json'))

    mtimes = [f.stat().st_mtime for f in all_files if f.exists()]
    mtime_min = min(mtimes) if mtimes else None
    mtime_max = max(mtimes) if mtimes else None

    return len(found), fmt, mtime_min, mtime_max


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--systems', nargs='*', default=None)
    parser.add_argument('--out-report', default=None,
                         help='Ghi toan bo report chi tiet ra file .txt nay')
    parser.add_argument('--mtime-gap-warn-hours', type=float, default=6.0,
                         help='Canh bao neu cac file trong CUNG 1 he thong '
                              '(khac seed) cach nhau qua so gio nay - dau '
                              'hieu co the la 2 lan chay khac nhau')
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f'Khong tim thay thu muc: {root}')

    system_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if args.systems:
        wanted = set(args.systems)
        system_dirs = [p for p in system_dirs if p.name in wanted]

    all_issues = []
    matrix_rows = []  # (system, seed, n_found, fmt, mtime_min, mtime_max)

    for sys_dir in system_dirs:
        seed_dirs = sorted(
            (p for p in sys_dir.iterdir()
             if p.is_dir() and SEED_DIR_RE.match(p.name)),
            key=lambda p: int(SEED_DIR_RE.match(p.name).group(1)))
        if not seed_dirs:
            continue
        for seed_dir in seed_dirs:
            seed = SEED_DIR_RE.match(seed_dir.name).group(1)
            n_found, fmt, mt_min, mt_max = audit_seed_dir(seed_dir, all_issues)
            matrix_rows.append((sys_dir.name, seed, n_found, fmt, mt_min, mt_max))

    lines = []

    def emit(s=''):
        lines.append(s)
        print(s)

    emit('=' * 78)
    emit('MA TRAN DAY DU: he thong x seed')
    emit('=' * 78)
    header = f'{"System":<32}{"Seed":>6}{"Dieu kien":>11}{"Dinh dang":>20}{"Thoi gian file":>34}'
    emit(header)
    emit('-' * len(header))
    for system, seed, n_found, fmt, mt_min, mt_max in matrix_rows:
        cond_str = f'{n_found}/16'
        if mt_min is not None:
            t_min = datetime.fromtimestamp(mt_min).strftime('%Y-%m-%d %H:%M')
            t_max = datetime.fromtimestamp(mt_max).strftime('%Y-%m-%d %H:%M')
            t_str = t_min if t_min == t_max else f'{t_min} -> {t_max}'
        else:
            t_str = '-'
        flag = '  <-- THIEU' if n_found < 16 else ''
        emit(f'{system:<32}{seed:>6}{cond_str:>11}{str(fmt):>20}{t_str:>34}{flag}')

    # canh bao mtime cach xa nhau trong cung 1 he thong
    emit('')
    emit('=' * 78)
    emit('KIEM TRA THOI GIAN FILE GIUA CAC SEED CUNG 1 HE THONG')
    emit('=' * 78)
    by_system = {}
    for system, seed, n_found, fmt, mt_min, mt_max in matrix_rows:
        if mt_min is None:
            continue
        by_system.setdefault(system, []).append((seed, mt_min, mt_max))
    for system, entries in sorted(by_system.items()):
        all_mins = [e[1] for e in entries]
        all_maxs = [e[2] for e in entries]
        span_hours = (max(all_maxs) - min(all_mins)) / 3600.0
        note = ''
        if span_hours > args.mtime_gap_warn_hours:
            note = (f'  <-- CANH BAO: cac seed cach nhau {span_hours:.1f} gio, '
                     f'vuot nguong {args.mtime_gap_warn_hours}h - kiem tra xem '
                     f'co phai 2 lan chay/2 phien ban script khac nhau khong')
        emit(f'{system:<32} khoang thoi gian: {span_hours:6.1f} gio{note}')

    emit('')
    emit('=' * 78)
    emit(f'CHI TIET LOI / CANH BAO TUNG FILE  ({len(all_issues)} muc)')
    emit('=' * 78)
    n_loi = sum(1 for i in all_issues if i.severity == 'LOI')
    n_canhbao = len(all_issues) - n_loi
    emit(f'Tong: {n_loi} LOI, {n_canhbao} CANH BAO\n')
    for issue in all_issues:
        emit(str(issue))
        emit('')

    if args.out_report:
        with open(args.out_report, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f'\nDa ghi report day du vao: {args.out_report}')


if __name__ == '__main__':
    main()
