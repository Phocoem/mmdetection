#!/usr/bin/env bash
set -e

cd /home/pc/mmdet_AI/mmdetection
export PYTHONPATH=/home/pc/mmdet_AI/mmdetection:$PYTHONPATH

MODELS=${1:-tools/research/models_dgcf_simple.json}
CLEAN_ROOT=${2:-mmdet_dataset/lettuce}
BENCH_ROOT=${3:-mmdet_dataset/lettuce_c}

python tools/research/eval_dgcf_existing_ckpts.py \
  --models "$MODELS" \
  --clean-root "$CLEAN_ROOT" \
  --benchmark-root "$BENCH_ROOT" \
  --skip-existing

python tools/research/make_dgcf_tables_figures.py \
  --models "$MODELS" \
  --out-dir paper_outputs_dgcf_simple
