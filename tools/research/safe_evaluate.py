"""
Wrapper AN TOAN quanh evaluate_benchmark.py - sua loi "chay lettuce_c roi
chay lettuce_d (hoac nguoc lai) cho CUNG 1 seed se XOA MAT ket qua cua lan
truoc" da xac nhan qua so sanh 2 heatmap (bifpn/cbam_fpn/r101_fpn mat het
spatial khi chay uniform; iapc_lam0p25 mat het uniform khi chay spatial;
fpn_aug/iapc_cpuaug/fpn mat 1 seed uniform khi them 1 seed spatial).

CACH SUA: KHONG BAO GIO goi evaluate_benchmark.py voi --output-dir tro
thang vao thu muc that. Thay vao do:
    1. Doc truoc du lieu CU dang co trong <output-dir> (neu co).
    2. Chay evaluate_benchmark.py THAT vao 1 thu muc TAM rieng.
    3. GOP (union) ket qua tam voi du lieu cu: dieu kien nao vua duoc
       danh gia lai thi lay gia tri MOI, dieu kien nao KHONG duoc dong
       toi thi GIU NGUYEN gia tri CU - khong bao gio mat du lieu.
    4. Ghi ket qua da gop vao <output-dir>/condition_metrics.csv (dinh
       dang chuan hoa - 1 file CSV, de merge lai lan sau).
    5. Xoa thu muc tam.

Cach dung - GIONG HET evaluate_benchmark.py, chi doi ten script:
    python safe_evaluate.py \\
        configs/fair_lettuce/mask_rcnn_r50_iapc_lam0p25.py \\
        work_dirs/research/mask_rcnn_r50_iapc_lam0p25/seed_2027/best_coco_segm_mAP_epoch_*.pth \\
        --clean-root /home/pc/mmdet_AI/mmdetection/mmdet_dataset/lettuce \\
        --benchmark-root /home/pc/mmdet_AI/mmdetection/mmdet_dataset/lettuce_d \\
        --output-dir work_dirs/research/mask_rcnn_r50_iapc_lam0p25/seed_2027/evaluation \\
        --seed 2027

Neu can doi duong dan script goc (mac dinh tools/research/evaluate_benchmark.py):
    ... --eval-script tools/research/evaluate_benchmark.py
"""
import argparse
import csv
import glob
import json
import shutil
import subprocess
import sys
from pathlib import Path

METRIC_KEYS = ['coco/segm_mAP', 'coco/segm_mAP_50', 'coco/segm_mAP_75',
               'coco/segm_mAP_s', 'coco/segm_mAP_m', 'coco/segm_mAP_l']


def read_evaluation_dir(eval_dir: Path) -> dict:
    """Doc TOAN BO metric (khong chi segm_mAP) tu 1 thu muc evaluation/,
    tu dong nhan dien dinh dang CSV hay JSON-per-condition.
    Tra ve {condition_full_name: {metric_key: value, ...}}."""
    out = {}
    if not eval_dir.is_dir():
        return out

    csv_path = eval_dir / 'condition_metrics.csv'
    conditions_dir = eval_dir / 'conditions'

    if csv_path.exists():
        with open(csv_path, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                cond = row.get('condition')
                if not cond:
                    continue
                out[cond] = {k: row[k] for k in METRIC_KEYS if k in row and row[k] != ''}

    if conditions_dir.is_dir():
        for cond_dir in conditions_dir.iterdir():
            if not cond_dir.is_dir():
                continue
            json_path = cond_dir / 'metrics.json'
            if not json_path.exists():
                continue
            with open(json_path, encoding='utf-8') as f:
                payload = json.load(f)
            out[cond_dir.name] = {k: payload[k] for k in METRIC_KEYS if k in payload}

    return out


def write_merged_csv(merged: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'condition_metrics.csv'
    fieldnames = ['condition'] + METRIC_KEYS
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cond in sorted(merged.keys()):
            row = {'condition': cond}
            row.update(merged[cond])
            writer.writerow(row)
    return csv_path


def resolve_checkpoint(pattern: str) -> str:
    if '*' not in pattern and '?' not in pattern:
        return pattern
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise SystemExit(f'Khong tim thay checkpoint khop pattern: {pattern}')
    if len(matches) > 1:
        print(f'CANH BAO: co {len(matches)} checkpoint khop pattern, '
              f'dung file moi nhat: {matches[-1]}')
        print(f'  Toan bo: {matches}')
    return matches[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--clean-root', required=True)
    parser.add_argument('--benchmark-root', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seed', required=True)
    parser.add_argument('--eval-script',
                         default='tools/research/evaluate_benchmark.py',
                         help='Duong dan script danh gia THAT (mac dinh '
                              'tools/research/evaluate_benchmark.py)')
    parser.add_argument('--keep-tmp', action='store_true',
                         help='Giu lai thu muc tam de kiem tra (mac dinh '
                              'xoa sau khi gop xong)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    tmp_dir = output_dir.parent / (output_dir.name + '__tmp_safe_eval')
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    checkpoint = resolve_checkpoint(args.checkpoint)

    print(f'[1/4] Doc du lieu CU dang co trong {output_dir} ...')
    old_data = read_evaluation_dir(output_dir)
    print(f'      -> {len(old_data)} dieu kien da co san: {sorted(old_data.keys())}')

    print(f'[2/4] Chay {args.eval_script} vao thu muc TAM {tmp_dir} ...')
    cmd = [sys.executable, args.eval_script, args.config, checkpoint,
           '--clean-root', args.clean_root,
           '--benchmark-root', args.benchmark_root,
           '--output-dir', str(tmp_dir),
           '--seed', str(args.seed)]
    print('      ' + ' '.join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(f'evaluate_benchmark.py that bai (return code '
                          f'{result.returncode}) - KHONG gop du lieu, giu '
                          f'nguyen {output_dir} nhu cu de tranh mat du lieu.')

    print(f'[3/4] Doc ket qua MOI tu {tmp_dir} va gop voi du lieu cu ...')
    new_data = read_evaluation_dir(tmp_dir)
    if not new_data:
        raise SystemExit(f'evaluate_benchmark.py chay xong nhung khong doc '
                          f'duoc ket qua tu {tmp_dir} - kiem tra lai dinh '
                          f'dang output cua script that. KHONG gop du lieu.')
    print(f'      -> {len(new_data)} dieu kien vua danh gia: {sorted(new_data.keys())}')

    overlap = set(old_data) & set(new_data)
    if overlap:
        print(f'      CANH BAO: {len(overlap)} dieu kien duoc danh gia LAI '
              f'(ghi de gia tri cu bang gia tri moi): {sorted(overlap)}')

    merged = dict(old_data)
    merged.update(new_data)  # moi de len cu cho dieu kien trung, cu giu nguyen cho con lai

    csv_path = write_merged_csv(merged, output_dir)
    print(f'[4/4] Da ghi {len(merged)} dieu kien (gop) vao {csv_path}')
    print(f'      Truoc: {len(old_data)} dieu kien -> Sau: {len(merged)} dieu kien '
          f'(+{len(merged) - len(old_data)})')

    if not args.keep_tmp:
        shutil.rmtree(tmp_dir)
    else:
        print(f'(Giu lai thu muc tam theo yeu cau: {tmp_dir})')


if __name__ == '__main__':
    main()
