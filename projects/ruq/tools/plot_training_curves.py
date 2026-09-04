#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_training_curves.py

Parse MMDetection/MMEngine training logs and export paper-ready training curves
and CSV/XLSX summaries.

Supported log formats:
- work_dir/vis_data/scalars.json       (MMEngine JSON lines)
- work_dir/*.json / *.log.json         (legacy JSON lines)
- work_dir/training_summary.json       (optional metadata)

Examples:
python projects/ruq/tools/plot_training_curves.py \
  --work-dirs \
    work_dirs/mask_rcnn_r50_fpn \
    work_dirs/pointrend_r50_fpn \
    work_dirs/ruq_mask_rcnn_r50_fpn_1x_lettuce \
  --names "Mask R-CNN" "PointRend" "RUQ-Mask R-CNN" \
  --out-dir paper_assets/training_curves

Or discover model/seed structure:
python projects/ruq/tools/plot_training_curves.py \
  --results-root work_dirs/research \
  --seed seed_2026 \
  --out-dir paper_assets/training_curves
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_DISPLAY_NAMES = {
    "mask_rcnn_r50_fpn": "Mask R-CNN R50",
    "mask_rcnn_r101_fpn": "Mask R-CNN R101",
    "pointrend_r50_fpn": "PointRend R50",
    "pointrend_r101_fpn": "PointRend R101",
    "ruq_mask_rcnn_r50_fpn": "RUQ-Mask R-CNN",
    "ruq_mask_rcnn_r50_fpn_1x_lettuce": "RUQ-Mask R-CNN",
    "ruq_mask_rcnn_r101_fpn": "RUQ-Mask R-CNN R101",
}

METRIC_ALIASES = {
    "loss": ["loss"],
    "loss_cls": ["loss_cls", "loss_cls", "roi_head.bbox_head.loss_cls"],
    "loss_bbox": ["loss_bbox", "roi_head.bbox_head.loss_bbox"],
    "loss_mask": ["loss_mask", "roi_head.mask_head.loss_mask"],
    "loss_quality": ["loss_quality", "roi_head.mask_head.loss_quality"],
    "loss_uncertainty": ["loss_uncertainty", "roi_head.mask_head.loss_uncertainty"],
    "segm_mAP": ["coco/segm_mAP", "segm_mAP", "mask_mAP", "segm/AP", "AP_segm"],
    "segm_mAP_50": ["coco/segm_mAP_50", "segm_mAP_50", "mask_mAP_50", "segm/AP50"],
    "segm_mAP_75": ["coco/segm_mAP_75", "segm_mAP_75", "mask_mAP_75", "segm/AP75"],
    "bbox_mAP": ["coco/bbox_mAP", "bbox_mAP", "box_mAP", "bbox/AP"],
}


def display_name(path: Path, fallback: Optional[str] = None) -> str:
    if fallback:
        return fallback
    name = path.name
    if name.startswith("seed_"):
        name = path.parent.name
    return MODEL_DISPLAY_NAMES.get(name, name)


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        y = float(x)
        if np.isnan(y) or np.isinf(y):
            return None
        return y
    except Exception:
        return None


def flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out


def parse_json_lines(path: Path) -> List[Dict[str, Any]]:
    rows = []
    try:
        text = path.read_text(encoding="utf-8-sig", errors="ignore")
    except Exception:
        return rows
    for line in text.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
            rows.append(flatten(obj))
        except Exception:
            continue
    return rows


def parse_log_text(path: Path) -> List[Dict[str, Any]]:
    """Fallback for plain .log files. Extract simple key=value metrics."""
    rows = []
    try:
        lines = path.read_text(encoding="utf-8-sig", errors="ignore").splitlines()
    except Exception:
        return rows
    for line in lines:
        if "Epoch" not in line and "Iter" not in line:
            continue
        row: Dict[str, Any] = {}
        m_epoch = re.search(r"Epoch\s*\[?(\d+)", line)
        if m_epoch:
            row["epoch"] = int(m_epoch.group(1))
        m_iter = re.search(r"iter(?:ation)?[:=\s]+(\d+)", line, re.I)
        if m_iter:
            row["iter"] = int(m_iter.group(1))
        for key in ["loss", "loss_cls", "loss_bbox", "loss_mask", "loss_quality", "loss_uncertainty", "lr"]:
            m = re.search(rf"{re.escape(key)}[:=]\s*([0-9.eE+-]+)", line)
            if m:
                row[key] = float(m.group(1))
        if len(row) > 1:
            rows.append(row)
    return rows


def find_log_files(work_dir: Path) -> List[Path]:
    candidates = []
    vs = work_dir / "vis_data" / "scalars.json"
    if vs.is_file():
        candidates.append(vs)
    candidates.extend(sorted(work_dir.glob("*.json")))
    candidates.extend(sorted(work_dir.glob("*.log.json")))
    # Include .log only as fallback, and avoid huge binary files.
    candidates.extend(sorted(work_dir.glob("*.log")))
    # Remove duplicates while preserving order.
    out, seen = [], set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        if p.name in {"training_summary.json"}:
            continue
        out.append(p)
    return out


def metric_value(row: Dict[str, Any], logical: str) -> Optional[float]:
    keys = METRIC_ALIASES.get(logical, [logical])
    for key in keys:
        if key in row:
            v = safe_float(row[key])
            if v is not None:
                return v
    # suffix fallback
    for key, value in row.items():
        for alias in keys:
            if str(key).endswith(alias):
                v = safe_float(value)
                if v is not None:
                    return v
    return None


def infer_step(row: Dict[str, Any], last_step: int) -> Tuple[float, int]:
    for k in ["step", "iter", "iteration", "global_step"]:
        if k in row and safe_float(row[k]) is not None:
            return float(row[k]), int(float(row[k]))
    if "epoch" in row and safe_float(row["epoch"]) is not None:
        return float(row["epoch"]), last_step + 1
    return float(last_step + 1), last_step + 1


def collect_work_dir(work_dir: Path, model_name: str) -> pd.DataFrame:
    all_rows = []
    last_step = -1
    for log_file in find_log_files(work_dir):
        rows = parse_json_lines(log_file)
        if not rows and log_file.suffix == ".log":
            rows = parse_log_text(log_file)
        for row in rows:
            x, last_step = infer_step(row, last_step)
            rec: Dict[str, Any] = {
                "model": model_name,
                "work_dir": str(work_dir),
                "log_file": str(log_file),
                "x": x,
                "epoch": row.get("epoch", ""),
                "iter": row.get("iter", row.get("iteration", row.get("step", ""))),
            }
            for logical in METRIC_ALIASES:
                v = metric_value(row, logical)
                if v is not None:
                    rec[logical] = v
            if len(rec) > 6:
                all_rows.append(rec)
    return pd.DataFrame(all_rows)


def discover_dirs(results_root: Path, seed: Optional[str]) -> List[Path]:
    if seed:
        return sorted(p for p in results_root.glob(f"*/{seed}") if p.is_dir())
    return sorted(p for p in results_root.glob("*/*") if p.is_dir() and p.name.startswith("seed_"))


def smooth_series(y: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return y
    return y.rolling(window=window, min_periods=max(1, window // 3), center=False).mean()


def plot_metric(df: pd.DataFrame, metric: str, out_dir: Path, smooth: int = 20) -> None:
    sub = df[["model", "x", metric]].dropna() if metric in df.columns else pd.DataFrame()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for model, g in sub.groupby("model", sort=False):
        g = g.sort_values("x")
        y = smooth_series(g[metric], smooth if metric.startswith("loss") else 1)
        ax.plot(g["x"], y, label=model, linewidth=1.8)
    ax.set_xlabel("Training step / epoch")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(metric.replace("_", " ").title())
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"curve_{metric}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if df.empty:
        return pd.DataFrame(rows)
    metrics = [m for m in METRIC_ALIASES if m in df.columns]
    for model, g in df.groupby("model", sort=False):
        row = {"model": model, "num_log_rows": len(g)}
        for m in metrics:
            vals = g[m].dropna()
            if vals.empty:
                continue
            if m.startswith("loss"):
                row[f"final_{m}"] = float(vals.iloc[-1])
                row[f"min_{m}"] = float(vals.min())
            else:
                row[f"best_{m}"] = float(vals.max())
                row[f"final_{m}"] = float(vals.iloc[-1])
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dirs", nargs="*", default=None)
    parser.add_argument("--names", nargs="*", default=None)
    parser.add_argument("--results-root", default=None)
    parser.add_argument("--seed", default="seed_2026")
    parser.add_argument("--out-dir", default="paper_assets/training_curves")
    parser.add_argument("--smooth", type=int, default=20)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.work_dirs:
        work_dirs = [Path(p) for p in args.work_dirs]
    elif args.results_root:
        work_dirs = discover_dirs(Path(args.results_root), args.seed)
    else:
        raise ValueError("Provide --work-dirs or --results-root")

    names = args.names or [None] * len(work_dirs)
    if len(names) != len(work_dirs):
        raise ValueError("--names must have the same length as --work-dirs")

    frames = []
    for wd, nm in zip(work_dirs, names):
        if not wd.is_dir():
            print(f"[WARN] Missing work dir: {wd}")
            continue
        model_name = display_name(wd, nm)
        df = collect_work_dir(wd, model_name)
        if df.empty:
            print(f"[WARN] No logs parsed for {wd}")
            continue
        frames.append(df)
        print(f"[OK] Parsed {len(df)} rows: {model_name} <- {wd}")

    if not frames:
        raise RuntimeError("No training logs parsed.")
    full = pd.concat(frames, ignore_index=True)
    full.to_csv(out_dir / "training_curves_raw.csv", index=False, encoding="utf-8-sig")
    summary = build_summary(full)
    summary.to_csv(out_dir / "training_summary_table.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(out_dir / "training_curves.xlsx", engine="openpyxl") as writer:
        full.to_excel(writer, sheet_name="raw", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)

    for metric in METRIC_ALIASES:
        if metric in full.columns:
            plot_metric(full, metric, out_dir, smooth=args.smooth)

    print(f"[DONE] Training curves and tables saved to: {out_dir}")


if __name__ == "__main__":
    main()
