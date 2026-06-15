"""Evaluate one clean or corrupted test condition and dump metrics."""

import argparse
import json
from pathlib import Path

from mmengine.config import Config
from mmengine.runner import Runner

from mmdet.registry import RUNNERS
from mmdet.utils import setup_cache_size_limit_of_dynamo


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--ann-file', required=True)
    parser.add_argument('--image-prefix', required=True)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--metrics-out', required=True)
    parser.add_argument('--seed', type=int, default=2026)
    return parser.parse_args()


def leaf_dataset(dataset):
    while 'dataset' in dataset:
        dataset = dataset['dataset']
    return dataset


def main():
    args = parse_args()
    setup_cache_size_limit_of_dynamo()

    cfg = Config.fromfile(args.config)
    cfg.work_dir = str(Path(args.work_dir).resolve())
    cfg.load_from = str(Path(args.checkpoint).resolve())
    cfg.resume = False
    cfg.randomness = dict(seed=args.seed, deterministic=True)

    dataset = leaf_dataset(cfg.test_dataloader.dataset)
    dataset.data_root = str(Path(args.data_root).resolve()) + '/'
    dataset.ann_file = args.ann_file
    dataset.data_prefix.img = args.image_prefix
    dataset.test_mode = True
    cfg.test_evaluator.ann_file = str(
        Path(args.data_root).resolve() / args.ann_file)

    if 'backbone' in cfg.model and 'init_cfg' in cfg.model.backbone:
        cfg.model.backbone.init_cfg = None

    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)
    metrics = runner.test()

    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=True, default=float),
        encoding='utf-8')
    print(f'Metrics: {metrics_path}')


if __name__ == '__main__':
    main()

