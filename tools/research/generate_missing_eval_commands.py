"""
Doc gap_fill/partial_eval_needed.csv (xuat boi fill_seed_gaps.py) va sinh
CHINH XAC cac lenh evaluate_benchmark.py con thieu - tu dong chon dung
BENCHMARK_ROOT theo loai dieu kien bi thieu:
    - Thieu brightness/contrast/gaussian_noise (9 dieu kien uniform)
      -> BENCHMARK_ROOT = <lettuce_c> (corruption dong nhat)
    - Thieu uneven_contrast/dappled_light (6 dieu kien spatial)
      -> BENCHMARK_ROOT = <lettuce_d> (corruption bien thien khong gian)
    - Thieu CA HAI loai -> sinh 2 lenh rieng (chay ca lettuce_c va lettuce_d)

Giu dung template lenh that ban da dung:
    python tools/research/evaluate_benchmark.py \\
        <config> \\
        <checkpoint glob best_coco_segm_mAP_epoch_*.pth> \\
        --clean-root ${CLEAN_ROOT} \\
        --benchmark-root ${BENCHMARK_ROOT} \\
        --output-dir <work_dir>/evaluation \\
        --seed <seed>

Gop theo (he thong, loai benchmark) thanh vong lap for giong dung phong
cach ban da viet, thay vi liet ke tung dong rieng le.

Cach dung:
    python generate_missing_eval_commands.py \\
        --gap-csv gap_fill/partial_eval_needed.csv \\
        --clean-root /home/pc/mmdet_AI/mmdetection/mmdet_dataset/lettuce \\
        --benchmark-root-uniform /home/pc/mmdet_AI/mmdetection/mmdet_dataset/lettuce_c \\
        --benchmark-root-spatial /home/pc/mmdet_AI/mmdetection/mmdet_dataset/lettuce_d \\
        --configs-dir configs/fair_lettuce \\
        --output run_missing_eval.sh

    bash run_missing_eval.sh
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

UNIFORM_FAMILIES = ('brightness', 'contrast', 'gaussian_noise')
SPATIAL_FAMILIES = ('uneven_contrast', 'dappled_light')


def condition_family(cond: str) -> str:
    if cond == 'clean':
        return 'clean'
    return cond.rsplit('_s', 1)[0]


def classify_row(missing_conditions):
    families = {condition_family(c) for c in missing_conditions}
    needs_uniform = any(f in UNIFORM_FAMILIES for f in families)
    needs_spatial = any(f in SPATIAL_FAMILIES for f in families)
    return needs_uniform, needs_spatial


def load_gap_csv(path: Path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            missing = row['MissingConditions'].split(';') if row['MissingConditions'] else []
            rows.append((row['System'], row['Seed'], missing))
    return rows


def build_commands(rows, clean_root, bench_uniform, bench_spatial, configs_dir):
    # group[(system, benchmark_kind)] = list of seeds
    groups = defaultdict(list)
    skipped_no_config = set()

    for system, seed, missing in rows:
        needs_uniform, needs_spatial = classify_row(missing)
        config_path = configs_dir / f'{system}.py'
        if not config_path.exists():
            skipped_no_config.add(system)
            continue
        if needs_uniform:
            groups[(system, 'uniform')].append(seed)
        if needs_spatial:
            groups[(system, 'spatial')].append(seed)

    return groups, skipped_no_config


def emit_script(groups, clean_root, bench_uniform, bench_spatial, configs_dir,
                 out_path, safe_eval_script):
    lines = ['#!/bin/bash', '# TU DONG SINH boi generate_missing_eval_commands.py',
             '# Dung safe_evaluate.py (khong goi truc tiep evaluate_benchmark.py)',
             '# de KHONG bi ghi de mat du lieu khi doi benchmark-root cho cung 1 seed.',
             '# Chi chay DANH GIA (khong train lai) - checkpoint da co san.',
             '', f'CLEAN_ROOT={clean_root}',
             'FAILED_LOG=run_missing_eval_failures.log',
             '> "$FAILED_LOG"  # xoa log cu neu co, bat dau moi']

    # sap xep de output on dinh, gom theo he thong
    systems = sorted({s for (s, _kind) in groups})
    for system in systems:
        for kind, bench_root, label in (
                ('uniform', bench_uniform, 'BRIGHTNESS/CONTRAST/GAUSSIAN_NOISE (lettuce_c)'),
                ('spatial', bench_spatial, 'UNEVEN_CONTRAST/DAPPLED_LIGHT (lettuce_d)')):
            seeds = groups.get((system, kind))
            if not seeds:
                continue
            seeds_str = ' '.join(sorted(set(seeds), key=int))
            config_path = configs_dir / f'{system}.py'
            lines.append('')
            lines.append(f'echo "======== {system} - thieu {label} - seeds: {seeds_str} ========"')
            lines.append(f'BENCHMARK_ROOT={bench_root}')
            lines.append(f'for SEED in {seeds_str}; do')
            lines.append(f'  echo "================ EVALUATE SEED ${{SEED}} ================"')
            lines.append(f'  if ! python {safe_eval_script} \\')
            lines.append(f'    {config_path} \\')
            lines.append(
                f'    work_dirs/research/{system}/seed_${{SEED}}/best_coco_segm_mAP_epoch_*.pth \\')
            lines.append(f'    --clean-root ${{CLEAN_ROOT}} \\')
            lines.append(f'    --benchmark-root ${{BENCHMARK_ROOT}} \\')
            lines.append(
                f'    --output-dir work_dirs/research/{system}/seed_${{SEED}}/evaluation \\')
            lines.append(f'    --seed ${{SEED}}; then')
            lines.append(
                f'    echo "THAT BAI: {system} seed=${{SEED}} benchmark={kind}" | tee -a "$FAILED_LOG"')
            lines.append(f'  fi')
            lines.append('done')

    lines.append('')
    lines.append('echo ""')
    lines.append('echo "========================================"')
    lines.append('if [ -s "$FAILED_LOG" ]; then')
    lines.append('  echo "CO LENH THAT BAI - xem chi tiet:"')
    lines.append('  cat "$FAILED_LOG"')
    lines.append('else')
    lines.append('  echo "TAT CA LENH DANH GIA THANH CONG."')
    lines.append('fi')

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gap-csv', required=True,
                         help='File partial_eval_needed.csv xuat boi '
                              'fill_seed_gaps.py')
    parser.add_argument('--clean-root', required=True)
    parser.add_argument('--benchmark-root-uniform', required=True,
                         help='Thu muc chua brightness/contrast/gaussian_noise '
                              '(vd .../lettuce_c)')
    parser.add_argument('--benchmark-root-spatial', required=True,
                         help='Thu muc chua uneven_contrast/dappled_light '
                              '(vd .../lettuce_d)')
    parser.add_argument('--configs-dir', required=True)
    parser.add_argument('--safe-eval-script',
                         default='tools/research/safe_evaluate.py',
                         help='Duong dan safe_evaluate.py TU THU MUC GOC '
                              'project (mac dinh tools/research/safe_evaluate.py '
                              '- phai khop dung noi ban da copy file vao)')
    parser.add_argument('--output', default='run_missing_eval.sh')
    args = parser.parse_args()

    gap_csv = Path(args.gap_csv)
    configs_dir = Path(args.configs_dir)
    if not gap_csv.exists():
        raise SystemExit(f'Khong tim thay {gap_csv} - chay fill_seed_gaps.py '
                          f'truoc de tao file nay.')

    rows = load_gap_csv(gap_csv)
    groups, skipped = build_commands(
        rows, args.clean_root, args.benchmark_root_uniform,
        args.benchmark_root_spatial, configs_dir)

    if skipped:
        print(f'CANH BAO: bo qua {len(skipped)} he thong vi khong tim thay '
              f'config trong {configs_dir}: {sorted(skipped)}')

    n_uniform_runs = sum(len(v) for k, v in groups.items() if k[1] == 'uniform')
    n_spatial_runs = sum(len(v) for k, v in groups.items() if k[1] == 'spatial')
    print(f'Se sinh lenh cho: {n_uniform_runs} lan chay lettuce_c (uniform), '
          f'{n_spatial_runs} lan chay lettuce_d (spatial)')

    emit_script(groups, args.clean_root, args.benchmark_root_uniform,
                args.benchmark_root_spatial, configs_dir, args.output,
                args.safe_eval_script)
    print(f'Da sinh: {args.output}')
    print(f'Chay bang: bash {args.output}')


if __name__ == '__main__':
    main()
