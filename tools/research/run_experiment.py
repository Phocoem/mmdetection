"""Run one reproducible train/evaluate experiment and capture provenance."""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument(
        '--stage', choices=('train', 'evaluate', 'all'), default='all')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--checkpoint')
    parser.add_argument(
        '--work-root', default='work_dirs/research')
    parser.add_argument(
        '--clean-root', default='mmdet_dataset/lettuce')
    parser.add_argument(
        '--benchmark-root', default='mmdet_dataset/lettuce_c')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--skip-existing-eval', action='store_true')
    return parser.parse_args()


def sha256_file(path):
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def command_output(command):
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding='utf-8',
        errors='replace',
        check=False)
    return result.stdout.strip()


def run_and_log(command, log_path):
    start = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8') as log:
        log.write(f'\nCOMMAND: {subprocess.list2cmdline(command)}\n')
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
    return time.perf_counter() - start


def write_timing(run_dir, key, seconds):
    path = run_dir / 'timings.json'
    timings = (
        json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {})
    timings[key] = seconds
    path.write_text(
        json.dumps(timings, indent=2, ensure_ascii=True),
        encoding='utf-8')


def write_provenance(run_dir, args):
    config = Path(args.config).resolve()
    data_root = Path(args.clean_root).resolve()
    annotation_hashes = {}
    for split in ('train', 'val', 'test'):
        path = data_root / 'annotations' / f'{split}.json'
        annotation_hashes[split] = {
            'path': str(path),
            'sha256': sha256_file(path),
        }

    git_prefix = [
        'git',
        '-c',
        f'safe.directory={Path.cwd().resolve().as_posix()}',
    ]
    manifest = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'config': str(config),
        'config_sha256': sha256_file(config),
        'seed': args.seed,
        'python': sys.version,
        'platform': platform.platform(),
        'git_commit': command_output(git_prefix + ['rev-parse', 'HEAD']),
        'git_status': command_output(git_prefix + ['status', '--short']),
        'annotation_hashes': annotation_hashes,
        'early_stopping_policy': {
            'monitor': 'coco/segm_mAP',
            'min_delta': 0.001,
            'patience': 20,
            'safety_cap_epochs': 200,
        },
    }
    (run_dir / 'provenance.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True),
        encoding='utf-8')
    (run_dir / 'pip_freeze.txt').write_text(
        command_output([sys.executable, '-m', 'pip', 'freeze']) + '\n',
        encoding='utf-8')
    source_diff = command_output(git_prefix + [
        'diff', '--', 'configs/fair_lettuce', 'mmdet/models',
        'tools/research', 'requirements_research.txt'
    ])
    (run_dir / 'source_diff.patch').write_text(
        source_diff + '\n', encoding='utf-8')


def find_best_checkpoint(run_dir):
    candidates = sorted(
        run_dir.glob('best_coco_segm_mAP*.pth'),
        key=lambda path: path.stat().st_mtime,
        reverse=True)
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f'No best checkpoint found in {run_dir}. '
        'Training must validate and save coco/segm_mAP.')


def preflight(run_dir, args):
    audit_command = [
        sys.executable,
        'tools/research/audit_dataset.py',
        '--data-root',
        str(Path(args.clean_root).resolve()),
        '--output',
        str(run_dir / 'dataset_audit.json'),
        '--check-dimensions',
        '--hash-images',
    ]
    seconds = run_and_log(audit_command, run_dir / 'preflight_console.log')
    write_timing(run_dir, 'dataset_audit_wall_seconds', seconds)
    if args.stage in ('evaluate', 'all'):
        manifest = Path(args.benchmark_root).resolve() / 'manifest.json'
        if not manifest.is_file():
            raise FileNotFoundError(
                f'Benchmark manifest not found: {manifest}. '
                'Build the benchmark before evaluation.')
        audit = json.loads(
            (run_dir / 'dataset_audit.json').read_text(encoding='utf-8'))
        benchmark = json.loads(manifest.read_text(encoding='utf-8'))
        test_audit = audit['splits']['test']
        if (test_audit['annotation_sha256']
                != benchmark['source_annotation_sha256']):
            raise ValueError(
                'Benchmark annotation hash differs from current clean test.')
        if (test_audit['image_set_sha256']
                != benchmark['source_images_sha256']):
            raise ValueError(
                'Benchmark source-image hash differs from current clean test.')


def summarize_training(run_dir):
    scalar_paths = sorted(
        run_dir.glob('**/vis_data/scalars.json'),
        key=lambda path: path.stat().st_mtime,
        reverse=True)
    if not scalar_paths:
        scalar_paths = sorted(
            run_dir.glob('**/scalars.json'),
            key=lambda path: path.stat().st_mtime,
            reverse=True)
    if not scalar_paths:
        return

    records = []
    for line in scalar_paths[0].read_text(encoding='utf-8').splitlines():
        if line.strip():
            records.append(json.loads(line))
    validation = [
        item for item in records if 'coco/segm_mAP' in item
    ]
    training = [item for item in records if 'loss' in item]
    if not validation:
        return
    best = max(validation, key=lambda item: item['coco/segm_mAP'])
    summary = {
        'scalars_path': str(scalar_paths[0]),
        'best_validation_segm_mAP': best['coco/segm_mAP'],
        'best_validation_step': best.get('step'),
        'validation_records': len(validation),
        'last_logged_epoch': max(
            (item.get('epoch', 0) for item in training), default=None),
        'last_logged_iter': max(
            (item.get('iter', 0) for item in training), default=None),
    }
    (run_dir / 'training_summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding='utf-8')


def main():
    args = parse_args()
    config = Path(args.config).resolve()
    if not config.is_file():
        raise FileNotFoundError(config)

    run_dir = (
        Path(args.work_root).resolve() / config.stem / f'seed_{args.seed}')
    run_dir.mkdir(parents=True, exist_ok=True)
    write_provenance(run_dir, args)
    preflight(run_dir, args)

    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else None
    if args.stage in ('train', 'all'):
        command = [
            sys.executable,
            'tools/train.py',
            str(config),
            '--work-dir',
            str(run_dir),
        ]
        if args.resume:
            command.append('--resume')
        if args.amp:
            command.append('--amp')
        command.extend(['--cfg-options', f'randomness.seed={args.seed}'])
        seconds = run_and_log(command, run_dir / 'train_console.log')
        write_timing(run_dir, 'training_wall_seconds', seconds)
        summarize_training(run_dir)
        checkpoint = find_best_checkpoint(run_dir)
        (run_dir / 'selected_checkpoint.txt').write_text(
            str(checkpoint) + '\n', encoding='utf-8')

    if args.stage in ('evaluate', 'all'):
        if checkpoint is None:
            selected = run_dir / 'selected_checkpoint.txt'
            if selected.is_file():
                checkpoint = Path(
                    selected.read_text(encoding='utf-8').strip()).resolve()
            else:
                checkpoint = find_best_checkpoint(run_dir)
        command = [
            sys.executable,
            'tools/research/evaluate_benchmark.py',
            str(config),
            str(checkpoint),
            '--clean-root',
            str(Path(args.clean_root).resolve()),
            '--benchmark-root',
            str(Path(args.benchmark_root).resolve()),
            '--output-dir',
            str(run_dir / 'evaluation'),
            '--seed',
            str(args.seed),
        ]
        if args.skip_existing_eval:
            command.append('--skip-existing')
        seconds = run_and_log(command, run_dir / 'evaluation_console.log')
        write_timing(run_dir, 'evaluation_wall_seconds', seconds)


if __name__ == '__main__':
    main()
