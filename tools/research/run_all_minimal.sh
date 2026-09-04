#!/usr/bin/env bash
set -e
cd /home/pc/mmdet_AI/mmdetection
export PYTHONPATH=/home/pc/mmdet_AI/mmdetection:$PYTHONPATH
MANIFEST=${1:-tools/research/lettuce_models_manifest.json}
CLEAN_ROOT=${2:-mmdet_dataset/lettuce}
BENCH_ROOT=${3:-mmdet_dataset/lettuce_c}
SEED=${4:-2026}
python tools/research/train_from_manifest.py --manifest "$MANIFEST" --seeds "$SEED" --skip-existing
python tools/research/evaluate_from_manifest.py --manifest "$MANIFEST" --clean-root "$CLEAN_ROOT" --benchmark-root "$BENCH_ROOT" --seed "$SEED"
python tools/research/make_paper_results.py --manifest "$MANIFEST" --eval-root work_dirs/research --out-dir paper_outputs --seeds "$SEED" --main-corruptions brightness contrast gaussian_noise defocus_blur motion_blur
