"""Run a predeclared matrix of fair configs and random seeds sequentially."""

import argparse
import subprocess
import sys
from pathlib import Path


def default_configs():
    return sorted(
        str(path) for path in Path('configs/fair_lettuce').glob('*.py')
        if path.name != 'audit_fairness.py')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', default=default_configs())
    parser.add_argument('--seeds', type=int, nargs='+', default=[2026, 2027, 2028])
    parser.add_argument(
        '--stage', choices=('train', 'evaluate', 'all'), default='all')
    parser.add_argument('--work-root', default='work_dirs/research')
    parser.add_argument('--clean-root', default='mmdet_dataset/lettuce')
    parser.add_argument('--benchmark-root', default='mmdet_dataset/lettuce_c')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--skip-existing-eval', action='store_true')
    parser.add_argument('--continue-on-error', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    failures = []
    total = len(args.configs) * len(args.seeds)
    index = 0
    for config in args.configs:
        for seed in args.seeds:
            index += 1
            print(f'\n===== Study run {index}/{total}: {config}, seed={seed} =====')
            command = [
                sys.executable,
                'tools/research/run_experiment.py',
                config,
                '--stage',
                args.stage,
                '--seed',
                str(seed),
                '--work-root',
                args.work_root,
                '--clean-root',
                args.clean_root,
                '--benchmark-root',
                args.benchmark_root,
            ]
            if args.resume:
                command.append('--resume')
            if args.amp:
                command.append('--amp')
            if args.skip_existing_eval:
                command.append('--skip-existing-eval')
            result = subprocess.run(command, check=False)
            if result.returncode:
                failures.append({'config': config, 'seed': seed})
                if not args.continue_on_error:
                    raise subprocess.CalledProcessError(
                        result.returncode, command)

    if failures:
        raise RuntimeError(f'{len(failures)} study runs failed: {failures}')
    print(f'Completed {total} study runs.')


if __name__ == '__main__':
    main()

