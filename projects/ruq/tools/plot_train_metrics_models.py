#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_train_metrics_models.py

Plot training curves for multiple MMDetection/MMEngine model runs.
It parses JSON-line logs from each work_dir, including common paths such as:
  - work_dir/*.json
  - work_dir/*.log.json
  - work_dir/vis_data/scalars.json

Outputs:
  - one PNG/PDF curve per metric
  - all_training_metrics.csv
  - training_summary_table.csv
  - training_curves.xlsx, if pandas/openpyxl are available

Examples:
python projects/ruq/tools/plot_train_metrics_models.py \
  --work-dirs work_dirs/mask_rcnn_r50 work_dirs/pointrend_r50 work_dirs/ruq_mask_rcnn \
  --names "Mask R-CNN" "PointRend" "RUQ-Mask R-CNN" \
  --out-dir paper_assets/training_curves

python projects/ruq/tools/plot_train_metrics_models.py \
  --results-root work_dirs/research \
  --seed seed_2026 \
  --out-dir paper_assets/training_curves
"""

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt

MODEL_DISPLAY_NAMES = {
    "mask_rcnn_r50_fpn": "Mask R-CNN R50",
    "mask_rcnn_r101_fpn": "Mask R-CNN R101",
    "pointrend_r50_fpn": "PointRend R50",
    "pointrend_r101_fpn": "PointRend R101",
    "ruq_mask_rcnn_r50_fpn": "RUQ-Mask R-CNN R50",
    "ruq_mask_rcnn_r50_fpn_1x_lettuce": "RUQ-Mask R-CNN R50",
    "mask_rcnn_r50_quality_fpn": "Mask R-CNN + Quality",
    "mask_rcnn_r50_uncertainty_fpn": "Mask R-CNN + Uncertainty",
    "mask_rcnn_r50_ruq_fpn": "RUQ-Mask R-CNN R50",
    "solo_r50": "SOLO R50",
    "yolact_r50": "YOLACT R50",
    "condinst_r50": "CondInst R50",
}

DEFAULT_METRICS = [
    "loss",
    "loss_cls",
    "loss_bbox",
    "loss_mask",
    "loss_quality",
    "loss_uncertainty",
    "loss_consistency",
    "lr",
    "coco/segm_mAP",
    "coco/segm_mAP_50",
    "coco/segm_mAP_75",
    "coco/bbox_mAP",
    "segm_mAP",
    "segm_mAP_50",
    "segm_mAP_75",
    "bbox_mAP",
]

METRIC_DISPLAY = {
    "loss": "Total loss",
    "loss_cls": "Classification loss",
    "loss_bbox": "Box regression loss",
    "loss_mask": "Mask loss",
    "loss_quality": "Quality loss",
    "loss_uncertainty": "Uncertainty loss",
    "loss_consistency": "Consistency loss",
    "lr": "Learning rate",
    "coco/segm_mAP": "Segmentation mAP",
    "coco/segm_mAP_50": "Segmentation AP50",
    "coco/segm_mAP_75": "Segmentation AP75",
    "coco/bbox_mAP": "Bounding-box mAP",
    "segm_mAP": "Segmentation mAP",
    "segm_mAP_50": "Segmentation AP50",
    "segm_mAP_75": "Segmentation AP75",
    "bbox_mAP": "Bounding-box mAP",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def display_model_name(raw: str) -> str:
    return MODEL_DISPLAY_NAMES.get(raw, raw)


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        y = float(x)
        if math.isnan(y) or math.isinf(y):
            return None
        return y
    except Exception:
        return None


def discover_work_dirs(results_root: Path, seed: Optional[str]) -> Tuple[List[Path], List[str]]:
    if seed:
        dirs = sorted([p for p in results_root.glob(f"*/{seed}") if p.is_dir()])
    else:
        dirs = sorted([p for p in results_root.glob("*/*") if p.is_dir() and p.name.startswith("seed_")])
    names = []
    for d in dirs:
        raw = d.parent.name if d.name.startswith("seed_") else d.name
        names.append(display_model_name(raw))
    return dirs, names


def is_log_file(path: Path) -> bool:
    if path.name == "metrics.json":
        return False
    if "evaluation" in path.parts or "paper_assets" in path.parts:
        return False
    if path.name.endswith(".log.json") or path.name == "scalars.json":
        return True
    if path.suffix == ".json" and ("vis_data" in path.parts or re.search(r"\d{8}_\d{6}", path.name)):
        return True
    return False


def find_log_files(work_dir: Path) -> List[Path]:
    candidates = []
    for p in work_dir.rglob("*.json"):
        if is_log_file(p):
            candidates.append(p)
    # Some old mmdet versions use *.log.json
    for p in work_dir.rglob("*.log.json"):
        if p not in candidates:
            candidates.append(p)
    # Prefer scalars.json and log.json files, avoid config dumps.
    return sorted(candidates, key=lambda p: (0 if p.name == "scalars.json" else 1, len(str(p)), str(p)))


def parse_json_lines(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8-sig", errors="ignore")
    except Exception as exc:
        print(f"[WARN] Cannot read {path}: {exc}", file=sys.stderr)
        return rows
    text = text.strip()
    if not text:
        return rows
    # JSON lines are the normal case.
    for line in text.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
        except Exception:
            continue
    if rows:
        return rows
    # Fallback: a single JSON list/dict.
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            rows = [x for x in obj if isinstance(x, dict)]
        elif isinstance(obj, dict):
            if all(isinstance(v, list) for v in obj.values()):
                # dict-of-lists form
                n = max((len(v) for v in obj.values()), default=0)
                for i in range(n):
                    r = {}
                    for k, v in obj.items():
                        if i < len(v):
                            r[k] = v[i]
                    rows.append(r)
            else:
                rows = [obj]
    except Exception:
        pass
    return rows


def normalize_metric_key(key: str) -> str:
    # Common MMEngine prefixes
    key = key.replace("val/", "").replace("train/", "")
    return key


def row_step(row: Dict[str, Any]) -> Optional[float]:
    for k in ["step", "iter", "iteration", "global_step"]:
        if k in row:
            v = safe_float(row.get(k))
            if v is not None:
                return v
    # epoch can be non-integer in some logs; still okay for plotting.
    if "epoch" in row:
        v = safe_float(row.get("epoch"))
        if v is not None:
            return v
    return None


def row_epoch(row: Dict[str, Any]) -> Optional[float]:
    for k in ["epoch", "Epoch"]:
        if k in row:
            v = safe_float(row.get(k))
            if v is not None:
                return v
    return None


def extract_rows(work_dir: Path, model_name: str) -> List[Dict[str, Any]]:
    log_files = find_log_files(work_dir)
    if not log_files:
        print(f"[WARN] No JSON logs found in {work_dir}", file=sys.stderr)
        return []
    all_rows: List[Dict[str, Any]] = []
    for lf in log_files:
        raw_rows = parse_json_lines(lf)
        for i, rr in enumerate(raw_rows):
            r: Dict[str, Any] = {
                "model": model_name,
                "work_dir": str(work_dir),
                "log_file": str(lf),
            }
            step = row_step(rr)
            epoch = row_epoch(rr)
            r["step"] = step if step is not None else i
            r["epoch"] = epoch if epoch is not None else ""
            # Keep numeric metrics only.
            for k, v in rr.items():
                nk = normalize_metric_key(str(k))
                fv = safe_float(v)
                if fv is not None:
                    r[nk] = fv
            # mark train/val mode if available
            if "mode" in rr:
                r["mode"] = rr["mode"]
            elif any(k in r for k in ["coco/segm_mAP", "segm_mAP", "bbox_mAP", "coco/bbox_mAP"]):
                r["mode"] = "val"
            else:
                r["mode"] = "train"
            all_rows.append(r)
    # Deduplicate same model/step/log metrics by keeping last occurrence.
    all_rows.sort(key=lambda x: (float(x.get("step") or 0), str(x.get("mode", ""))))
    return all_rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def smooth_series(values: List[float], window: int) -> List[float]:
    if window <= 1 or len(values) < 3:
        return values
    out = []
    half = window // 2
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def metric_values(rows: List[Dict[str, Any]], model: str, metric: str) -> Tuple[List[float], List[float]]:
    pts = []
    for r in rows:
        if r.get("model") != model:
            continue
        v = safe_float(r.get(metric))
        x = safe_float(r.get("step"))
        if v is None or x is None:
            continue
        pts.append((x, v))
    # Keep order but if duplicate step, keep last.
    latest: Dict[float, float] = {}
    for x, v in pts:
        latest[x] = v
    xs = sorted(latest.keys())
    ys = [latest[x] for x in xs]
    return xs, ys


def is_higher_better(metric: str) -> bool:
    key = metric.lower()
    return any(s in key for s in ["map", "ap", "ar", "acc", "f1", "iou", "precision", "recall"])


def metric_to_filename(metric: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", metric).strip("_")


def plot_metric(rows: List[Dict[str, Any]], models: Sequence[str], metric: str, out_dir: Path, smooth_window: int = 1) -> bool:
    series = []
    for model in models:
        xs, ys = metric_values(rows, model, metric)
        if not xs:
            continue
        ys_plot = smooth_series(ys, smooth_window)
        series.append((model, xs, ys_plot))
    if not series:
        return False
    title = METRIC_DISPLAY.get(metric, metric)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for model, xs, ys in series:
        ax.plot(xs, ys, label=model, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Iteration / step")
    ax.set_ylabel(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fname = metric_to_filename(metric)
    fig.savefig(out_dir / f"curve_{fname}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"curve_{fname}.pdf", bbox_inches="tight")
    plt.close(fig)
    return True


def build_summary(rows: List[Dict[str, Any]], models: Sequence[str], metrics: Sequence[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for model in models:
        row: Dict[str, Any] = {"model": model}
        for metric in metrics:
            xs, ys = metric_values(rows, model, metric)
            if not ys:
                continue
            best = max(ys) if is_higher_better(metric) else min(ys)
            best_idx = ys.index(best)
            last = ys[-1]
            row[f"{metric}__best"] = round(best, 6)
            row[f"{metric}__best_step"] = xs[best_idx]
            row[f"{metric}__last"] = round(last, 6)
            row[f"{metric}__last_step"] = xs[-1]
        out.append(row)
    return out


def export_xlsx(out_dir: Path, rows: List[Dict[str, Any]], summary: List[Dict[str, Any]]) -> None:
    try:
        import pandas as pd
    except Exception:
        return
    xlsx = out_dir / "training_curves.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="all_metrics", index=False)
        pd.DataFrame(summary).to_excel(writer, sheet_name="summary", index=False)
    print(f"[OK] Excel: {xlsx}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training curves for multiple MMDetection runs.")
    parser.add_argument("--work-dirs", nargs="*", default=[], help="List of work_dirs to compare")
    parser.add_argument("--names", nargs="*", default=[], help="Display names corresponding to --work-dirs")
    parser.add_argument("--results-root", default="", help="Root with model/seed dirs, e.g. work_dirs/research")
    parser.add_argument("--seed", default="seed_2026", help="Seed folder name under each model dir")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS)
    parser.add_argument("--smooth-window", type=int, default=1)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    if args.work_dirs:
        work_dirs = [Path(p).resolve() for p in args.work_dirs]
        if args.names:
            if len(args.names) != len(work_dirs):
                raise ValueError("--names must have the same length as --work-dirs")
            names = list(args.names)
        else:
            names = [display_model_name(p.name) for p in work_dirs]
    elif args.results_root:
        work_dirs, names = discover_work_dirs(Path(args.results_root).resolve(), args.seed)
    else:
        raise ValueError("Use either --work-dirs or --results-root")

    if not work_dirs:
        raise FileNotFoundError("No work_dirs found")

    print("[INFO] Runs:")
    for wd, name in zip(work_dirs, names):
        print(f"  - {name}: {wd}")

    all_rows: List[Dict[str, Any]] = []
    for wd, name in zip(work_dirs, names):
        rows = extract_rows(wd, name)
        print(f"[INFO] {name}: {len(rows)} log rows")
        all_rows.extend(rows)

    write_csv(out_dir / "all_training_metrics.csv", all_rows)

    plotted = []
    for metric in args.metrics:
        if plot_metric(all_rows, names, metric, out_dir, smooth_window=args.smooth_window):
            plotted.append(metric)
            print(f"[OK] curve_{metric_to_filename(metric)}.png")

    summary = build_summary(all_rows, names, plotted)
    write_csv(out_dir / "training_summary_table.csv", summary)
    export_xlsx(out_dir, all_rows, summary)

    report = {
        "work_dirs": [str(p) for p in work_dirs],
        "names": names,
        "metrics_requested": args.metrics,
        "metrics_plotted": plotted,
        "smooth_window": args.smooth_window,
    }
    (out_dir / "generation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("==================================================")
    print(f"Done. Training curves saved to: {out_dir.resolve()}")
    print("Main files:")
    print(f"  {out_dir / 'all_training_metrics.csv'}")
    print(f"  {out_dir / 'training_summary_table.csv'}")
    print(f"  {out_dir / 'training_curves.xlsx'}")
    for metric in plotted[:10]:
        print(f"  {out_dir / ('curve_' + metric_to_filename(metric) + '.png')}")
    print("==================================================")


if __name__ == "__main__":
    main()
