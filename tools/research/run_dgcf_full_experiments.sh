#!/usr/bin/env bash
set -e

cd /home/pc/mmdet_AI/mmdetection
export PYTHONPATH=/home/pc/mmdet_AI/mmdetection:$PYTHONPATH

MANIFEST=${1:-tools/research/lettuce_dgcf_manifest.json}
CLEAN_ROOT=${2:-mmdet_dataset/lettuce}
BENCH_ROOT=${3:-mmdet_dataset/lettuce_c}
SEED=${4:-2026}

echo "[0] Validate manifest"
python tools/research/validate_manifest.py "$MANIFEST" --check-files --seed "$SEED"

echo "[1] Check fairness"
python tools/research/check_fair_configs_dgcf.py \
  --manifest "$MANIFEST" \
  --seed "$SEED" \
  --out paper_outputs_dgcf/tables/fairness_check.csv

echo "[2] Train models"
python tools/research/train_dgcf_suite.py \
  --manifest "$MANIFEST" \
  --seeds "$SEED" \
  --skip-existing

echo "[3] Evaluate clean and visual degradations"
python tools/research/evaluate_dgcf_suite.py \
  --manifest "$MANIFEST" \
  --clean-root "$CLEAN_ROOT" \
  --benchmark-root "$BENCH_ROOT" \
  --seed "$SEED"

echo "[4] Make paper tables and figures"
python tools/research/make_dgcf_paper_outputs.py \
  --manifest "$MANIFEST" \
  --eval-root work_dirs/research \
  --out-dir paper_outputs_dgcf \
  --seeds "$SEED" \
  --main-corruptions brightness contrast gaussian_noise defocus_blur motion_blur

echo "[5] Qualitative visualization"
python tools/research/visualize_dgcf_qualitative.py \
  --manifest "$MANIFEST" \
  --clean-root "$CLEAN_ROOT" \
  --benchmark-root "$BENCH_ROOT" \
  --benchmark-manifest "$BENCH_ROOT/manifest.json" \
  --out-dir paper_outputs_dgcf/qualitative \
  --seed "$SEED" \
  --image-indices 0 5 10 \
  --conditions clean contrast:5 gaussian_noise:5 defocus_blur:5 motion_blur:5

echo "[DONE] paper_outputs_dgcf created."
