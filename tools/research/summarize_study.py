"""Aggregate clean and corruption metrics across models and seeds."""

import argparse
import csv
import json
import statistics
from pathlib import Path


METRICS = (
    'clean_performance',
    'mean_performance_under_corruption',
    'relative_performance_under_corruption',
    'absolute_robustness_drop',
)
RUN_FIELDS = (
    'best_validation_segm_mAP',
    'last_logged_epoch',
    'training_wall_seconds',
    'evaluation_wall_seconds',
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work-root', default='work_dirs/research')
    parser.add_argument('--output-dir', default='work_dirs/research_summary')
    return parser.parse_args()


def main():
    args = parse_args()
    work_root = Path(args.work_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for path in sorted(work_root.glob('*/seed_*/evaluation/summary.json')):
        summary = json.loads(path.read_text(encoding='utf-8'))
        run_dir = path.parents[1]
        training_summary_path = run_dir / 'training_summary.json'
        timing_path = run_dir / 'timings.json'
        training_summary = (
            json.loads(training_summary_path.read_text(encoding='utf-8'))
            if training_summary_path.is_file() else {})
        timings = (
            json.loads(timing_path.read_text(encoding='utf-8'))
            if timing_path.is_file() else {})
        row = {
            'model': path.parents[2].name,
            'seed': int(path.parents[1].name.removeprefix('seed_')),
            **{metric: summary[metric] for metric in METRICS},
            'summary_path': str(path),
        }
        row.update({
            'best_validation_segm_mAP':
            training_summary.get('best_validation_segm_mAP'),
            'last_logged_epoch': training_summary.get('last_logged_epoch'),
            'training_wall_seconds': timings.get('training_wall_seconds'),
            'evaluation_wall_seconds': timings.get('evaluation_wall_seconds'),
        })
        runs.append(row)
    if not runs:
        raise FileNotFoundError(f'No evaluation summaries found in {work_root}')

    with (output_dir / 'runs.csv').open(
            'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=runs[0].keys())
        writer.writeheader()
        writer.writerows(runs)

    model_names = sorted({row['model'] for row in runs})
    aggregate = []
    for model in model_names:
        model_runs = [row for row in runs if row['model'] == model]
        item = {'model': model, 'num_seeds': len(model_runs)}
        for metric in METRICS:
            values = [float(row[metric]) for row in model_runs]
            item[f'{metric}_mean'] = statistics.mean(values)
            item[f'{metric}_std'] = (
                statistics.stdev(values) if len(values) > 1 else 0.0)
        for field in RUN_FIELDS:
            values = [
                float(row[field]) for row in model_runs
                if row.get(field) is not None
            ]
            if values:
                item[f'{field}_mean'] = statistics.mean(values)
                item[f'{field}_std'] = (
                    statistics.stdev(values) if len(values) > 1 else 0.0)
        aggregate.append(item)

    with (output_dir / 'models.csv').open(
            'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=aggregate[0].keys())
        writer.writeheader()
        writer.writerows(aggregate)
    (output_dir / 'study_summary.json').write_text(
        json.dumps(
            {'runs': runs, 'models': aggregate},
            indent=2,
            ensure_ascii=True),
        encoding='utf-8')
    print(f'Aggregated {len(runs)} runs across {len(model_names)} models.')


if __name__ == '__main__':
    main()
