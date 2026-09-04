"""Evaluate one checkpoint on clean test and all Lettuce-C conditions."""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


PRIMARY_METRIC = 'coco/segm_mAP'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument(
        '--clean-root', default='mmdet_dataset/lettuce')
    parser.add_argument('--clean-ann-file', default='annotations/test.json')
    parser.add_argument('--clean-image-prefix', default='images/test/')
    parser.add_argument(
        '--benchmark-root', default='mmdet_dataset/lettuce_d')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--skip-existing', action='store_true')
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def run_and_log(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('w', encoding='utf-8') as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace')
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end='')
            log.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def evaluate_condition(args, name, data_root, ann_file, image_prefix):
    output_dir = Path(args.output_dir).resolve()
    condition_dir = output_dir / 'conditions' / name
    metrics_path = condition_dir / 'metrics.json'
    if args.skip_existing and metrics_path.is_file():
        return json.loads(metrics_path.read_text(encoding='utf-8'))

    command = [
        sys.executable,
        'tools/research/evaluate_one.py',
        args.config,
        args.checkpoint,
        '--data-root',
        str(data_root),
        '--ann-file',
        ann_file,
        '--image-prefix',
        image_prefix,
        '--work-dir',
        str(condition_dir / 'runner'),
        '--metrics-out',
        str(metrics_path),
        '--seed',
        str(args.seed),
    ]
    run_and_log(command, condition_dir / 'console.log')
    return json.loads(metrics_path.read_text(encoding='utf-8'))


def main():
    args = parse_args()
    benchmark_root = Path(args.benchmark_root).resolve()
    manifest_path = benchmark_root / 'manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f'Benchmark manifest not found: {manifest_path}. '
            'Run tools/research/build_corruption_benchmark.py first.')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))

    rows = [{
        'condition': 'clean',
        'corruption': 'clean',
        'severity': 0,
        **evaluate_condition(
            args,
            'clean',
            Path(args.clean_root).resolve(),
            args.clean_ann_file,
            args.clean_image_prefix),
    }]
    for condition in manifest['conditions']:
        corruption = condition['corruption']
        severity = condition['severity']
        name = f'{corruption}_s{severity}'
        rows.append({
            'condition': name,
            'corruption': corruption,
            'severity': severity,
            **evaluate_condition(
                args,
                name,
                benchmark_root,
                manifest['output_annotation'],
                condition['image_prefix']),
        })

    if PRIMARY_METRIC not in rows[0]:
        raise KeyError(
            f'{PRIMARY_METRIC} not found. Available: {sorted(rows[0])}')

    clean_ap = float(rows[0][PRIMARY_METRIC])
    corrupted_rows = rows[1:]
    mpc = sum(float(row[PRIMARY_METRIC]) for row in corrupted_rows) / len(
        corrupted_rows)
    per_corruption = {}
    for corruption in manifest['corruptions']:
        values = [
            float(row[PRIMARY_METRIC]) for row in corrupted_rows
            if row['corruption'] == corruption
        ]
        per_corruption[corruption] = sum(values) / len(values)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_keys = sorted({
        key for row in rows for key, value in row.items()
        if key not in ('condition', 'corruption', 'severity')
        and isinstance(value, (int, float))
    })
    with (output_dir / 'condition_metrics.csv').open(
            'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(
            file,
            fieldnames=['condition', 'corruption', 'severity'] + metric_keys,
            extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        'primary_metric': PRIMARY_METRIC,
        'clean_performance': clean_ap,
        'mean_performance_under_corruption': mpc,
        'relative_performance_under_corruption': (
            mpc / clean_ap if clean_ap else None),
        'absolute_robustness_drop': clean_ap - mpc,
        'per_corruption_mean': per_corruption,
        'num_corrupted_conditions': len(corrupted_rows),
        'seed': args.seed,
        'config': str(Path(args.config).resolve()),
        'config_sha256': sha256_file(Path(args.config).resolve()),
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'checkpoint_sha256': sha256_file(Path(args.checkpoint).resolve()),
        'benchmark_manifest': str(manifest_path),
        'benchmark_manifest_sha256': sha256_file(manifest_path),
    }
    (output_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

