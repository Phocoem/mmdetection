#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_paper_tables_v2.py

Paper-ready table/figure generator for lettuce robustness experiments.
Improvements over v1:
- Paper display names for models/corruptions.
- Baselines appear above enhanced models.
- Main 5-corruption tables and full benchmark tables.
- Ablation table for Mask R-CNN variants.
- Severity pivot table S1-S5.
- Heatmaps for mean AP, RD, SI, and severity.
"""

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

MODEL_DISPLAY_NAMES = {
    "mask_rcnn_r50_fpn": "Mask R-CNN R50",
    "mask_rcnn_r101_fpn": "Mask R-CNN R101",
    "solo_r50": "SOLO R50",
    "solov2_r50": "SOLOv2 R50",
    "yolact_r50": "YOLACT R50",
    "mask_rcnn_r50_dgcf_fpn": "DGCF-FPN",
    "mask_rcnn_r50_dgcf_no_detail_fpn": "DGCF-FPN w/o Detail",
    "mask_rcnn_r50_dgcf_no_gate_fpn": "DGCF-FPN w/o Gate",
}

MODEL_ORDER = [
    "Mask R-CNN R50",
    "Mask R-CNN R101",
    "SOLO R50",
    "SOLOv2 R50",
    "YOLACT R50",
    "CondInst R50",
    "DGCF-FPN",
    "DGCF-FPN w/o Detail",
    "DGCF-FPN w/o Gate",
]

BASELINE_MODELS = {
    "Mask R-CNN R50",
    "Mask R-CNN R101",
    "SOLO R50",
    "SOLOv2 R50",
    "YOLACT R50",
}

ABLATION_MODELS = [
    "Mask R-CNN R50",
    "DGCF-FPN",
    "DGCF-FPN w/o Detail",
    "DGCF-FPN w/o Gate",
]

CORRUPTION_DISPLAY_NAMES = {
    "clean": "Clean",
    "brightness": "Brightness",
    "contrast": "Contrast",
    "gaussian_noise": "Gaussian Noise",
}

MAIN_CORRUPTIONS = [
    "brightness",
    "contrast",
    "gaussian_noise",
]

CORRUPTION_ORDER = [
    "brightness",
    "contrast",
    "gaussian_noise",
]

METRIC_ALIASES = {
    "mask_ap": ["coco/segm_mAP", "segm_mAP", "mask_mAP", "segm/AP", "AP_segm"],
    "mask_ap50": ["coco/segm_mAP_50", "segm_mAP_50", "mask_mAP_50", "segm/AP50", "AP50_segm"],
    "mask_ap75": ["coco/segm_mAP_75", "segm_mAP_75", "mask_mAP_75", "segm/AP75", "AP75_segm"],
    "mask_aps": ["coco/segm_mAP_s", "segm_mAP_s", "mask_mAP_s"],
    "mask_apm": ["coco/segm_mAP_m", "segm_mAP_m", "mask_mAP_m"],
    "mask_apl": ["coco/segm_mAP_l", "segm_mAP_l", "mask_mAP_l"],
    "box_ap": ["coco/bbox_mAP", "bbox_mAP", "box_mAP", "bbox/AP", "AP_bbox"],
    "box_ap50": ["coco/bbox_mAP_50", "bbox_mAP_50", "box_mAP_50", "bbox/AP50", "AP50_bbox"],
    "box_ap75": ["coco/bbox_mAP_75", "bbox_mAP_75", "box_mAP_75", "bbox/AP75", "AP75_bbox"],
}


def display_model_name(raw: str) -> str:
    return MODEL_DISPLAY_NAMES.get(raw, raw)


def display_corruption_name(raw: str) -> str:
    return CORRUPTION_DISPLAY_NAMES.get(raw, raw.replace("_", " ").title())


def model_sort_key(name: str) -> Tuple[int, str]:
    return (MODEL_ORDER.index(name), name) if name in MODEL_ORDER else (999, name)


def corruption_sort_key(name: str) -> Tuple[int, str]:
    if name in CORRUPTION_ORDER:
        return (CORRUPTION_ORDER.index(name), name)
    display_to_raw = {display_corruption_name(c): c for c in CORRUPTION_ORDER}
    if name in display_to_raw:
        raw = display_to_raw[name]
        return (CORRUPTION_ORDER.index(raw), raw)
    return (999, name)


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        print(f"[WARN] Cannot read JSON {path}: {exc}", file=sys.stderr)
        return None


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig").strip()
    except Exception:
        return ""


def flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, key))
        else:
            out[key] = v
    return out


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


def round_value(x: Any, ndigits: int = 4) -> Any:
    y = safe_float(x)
    if y is None:
        return "" if x is None else x
    return round(y, ndigits)


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def get_metric(metrics: Dict[str, Any], logical_name: str) -> Optional[float]:
    for key in METRIC_ALIASES.get(logical_name, []):
        if key in metrics:
            return safe_float(metrics[key])
    suffix_map = {
        "mask_ap": "segm_mAP", "mask_ap50": "segm_mAP_50", "mask_ap75": "segm_mAP_75",
        "mask_aps": "segm_mAP_s", "mask_apm": "segm_mAP_m", "mask_apl": "segm_mAP_l",
        "box_ap": "bbox_mAP", "box_ap50": "bbox_mAP_50", "box_ap75": "bbox_mAP_75",
    }
    suffix = suffix_map.get(logical_name)
    if suffix:
        for key, value in metrics.items():
            if str(key).endswith(suffix):
                y = safe_float(value)
                if y is not None:
                    return y
    return None


def discover_model_dirs(results_root: Path, seed: Optional[str]) -> List[Path]:
    if seed:
        dirs = sorted(results_root.glob(f"*/{seed}"))
    else:
        dirs = sorted(p for p in results_root.glob("*/*") if p.is_dir() and p.name.startswith("seed_"))
    return [p for p in dirs if p.is_dir()]


def infer_condition_from_path(metrics_path: Path) -> Tuple[str, Optional[int], str]:
    parts = list(metrics_path.parts)
    if "conditions" in parts:
        idx = parts.index("conditions")
        condition_parts = parts[idx + 1: -1]
        condition_parts = [p for p in condition_parts if p not in {"runner", "eval", "test"}]
    else:
        condition_parts = [metrics_path.parent.name]
    if not condition_parts:
        return "unknown", None, "unknown"
    if condition_parts[0] == "clean":
        return "clean", None, "clean"
    if len(condition_parts) >= 2 and str(condition_parts[1]).isdigit():
        corruption = condition_parts[0]
        severity = int(condition_parts[1])
        return corruption, severity, f"{corruption}_{severity}"
    joined = "_".join(condition_parts)
    match = re.match(r"(.+?)(?:_severity_|_s|-|_)([1-5])$", joined)
    if match:
        corruption = match.group(1)
        severity = int(match.group(2))
        return corruption, severity, f"{corruption}_{severity}"
    return joined, None, joined


def find_metrics_files(eval_dir: Path) -> List[Path]:
    return sorted(eval_dir.rglob("metrics.json")) if eval_dir.is_dir() else []


def collect_evaluation_rows(model_dir: Path) -> List[Dict[str, Any]]:
    raw_model = model_dir.parent.name
    model = display_model_name(raw_model)
    seed = model_dir.name
    eval_dir = model_dir / "evaluation"
    rows: List[Dict[str, Any]] = []
    for metrics_path in find_metrics_files(eval_dir):
        metrics = read_json(metrics_path)
        if not metrics:
            continue
        corruption, severity, condition = infer_condition_from_path(metrics_path)
        row: Dict[str, Any] = {
            "model": model,
            "raw_model": raw_model,
            "seed": seed,
            "condition": condition,
            "corruption": corruption,
            "corruption_display": display_corruption_name(corruption),
            "severity": severity if severity is not None else "",
            "metrics_file": str(metrics_path),
        }
        for logical in METRIC_ALIASES:
            row[logical] = get_metric(metrics, logical)
        rows.append(row)
    unique: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for r in rows:
        key = (r["model"], r["seed"], r["condition"])
        old = unique.get(key)
        if old is None or len(r["metrics_file"]) < len(old["metrics_file"]):
            unique[key] = r
    return list(unique.values())


def collect_training_rows(model_dirs: List[Path]) -> List[Dict[str, Any]]:
    rows = []
    for model_dir in model_dirs:
        raw_model = model_dir.parent.name
        model = display_model_name(raw_model)
        seed = model_dir.name
        row: Dict[str, Any] = {
            "model": model,
            "raw_model": raw_model,
            "seed": seed,
            "group": "Baseline" if model in BASELINE_MODELS else "Enhanced",
            "run_dir": str(model_dir),
        }
        selected = model_dir / "selected_checkpoint.txt"
        if selected.is_file():
            row["selected_checkpoint"] = read_text(selected)
        summary = model_dir / "training_summary.json"
        if summary.is_file():
            data = read_json(summary) or {}
            for k, v in flatten_dict(data).items():
                if isinstance(v, (str, int, float, bool)) or v is None:
                    row[k] = v
        best = sorted(model_dir.glob("best_*.pth"))
        row["best_checkpoint_found"] = best[-1].name if best else ""
        row["latest_checkpoint_found"] = "latest.pth" if (model_dir / "latest.pth").is_file() else ""
        rows.append(row)
    return sorted(rows, key=lambda x: model_sort_key(x["model"]))


def by_model_rows(eval_rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for r in eval_rows:
        by_model.setdefault(r["model"], []).append(r)
    return by_model


def get_clean_ap(rows: List[Dict[str, Any]], metric: str = "mask_ap") -> Optional[float]:
    clean = next((r for r in rows if r["corruption"] == "clean"), None)
    return safe_float(clean.get(metric)) if clean else None


def get_corruption_mean(rows: List[Dict[str, Any]], corruption: str, metric: str = "mask_ap") -> Optional[float]:
    vals = [safe_float(r.get(metric)) for r in rows if r["corruption"] == corruption and r["severity"] != ""]
    return mean(vals)


def get_selected_corrupted_mean(rows: List[Dict[str, Any]], corruptions: List[str], metric: str = "mask_ap") -> Optional[float]:
    vals = [safe_float(r.get(metric)) for r in rows if r["corruption"] in corruptions and r["severity"] != ""]
    return mean(vals)


def build_clean_performance(eval_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in eval_rows:
        if r["corruption"] != "clean":
            continue
        out.append({
            "model": r["model"],
            "group": "Baseline" if r["model"] in BASELINE_MODELS else "Enhanced",
            "seed": r["seed"],
            "mask_AP": round_value(r.get("mask_ap")),
            "mask_AP50": round_value(r.get("mask_ap50")),
            "mask_AP75": round_value(r.get("mask_ap75")),
            "mask_APs": round_value(r.get("mask_aps")),
            "mask_APm": round_value(r.get("mask_apm")),
            "mask_APl": round_value(r.get("mask_apl")),
            "box_AP": round_value(r.get("box_ap")),
            "box_AP50": round_value(r.get("box_ap50")),
            "box_AP75": round_value(r.get("box_ap75")),
        })
    return sorted(out, key=lambda x: model_sort_key(x["model"]))


def build_condition_performance(eval_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in eval_rows:
        out.append({
            "model": r["model"],
            "group": "Baseline" if r["model"] in BASELINE_MODELS else "Enhanced",
            "seed": r["seed"],
            "condition": r["condition"],
            "corruption": r["corruption_display"],
            "severity": r["severity"],
            "mask_AP": round_value(r.get("mask_ap")),
            "mask_AP50": round_value(r.get("mask_ap50")),
            "mask_AP75": round_value(r.get("mask_ap75")),
            "box_AP": round_value(r.get("box_ap")),
            "box_AP50": round_value(r.get("box_ap50")),
            "box_AP75": round_value(r.get("box_ap75")),
        })
    return sorted(out, key=lambda x: (model_sort_key(x["model"]), corruption_sort_key(x["corruption"]), str(x["severity"])))


def build_main_results_table(eval_rows: List[Dict[str, Any]], corruptions: List[str]) -> List[Dict[str, Any]]:
    out = []
    for model, rows in by_model_rows(eval_rows).items():
        clean = get_clean_ap(rows, "mask_ap")
        row: Dict[str, Any] = {"model": model, "group": "Baseline" if model in BASELINE_MODELS else "Enhanced", "Clean": round_value(clean)}
        for c in corruptions:
            row[display_corruption_name(c)] = round_value(get_corruption_mean(rows, c, "mask_ap"))
        mean_corr = get_selected_corrupted_mean(rows, corruptions, "mask_ap")
        rd = clean - mean_corr if clean is not None and mean_corr is not None else None
        si = mean_corr / clean if clean not in (None, 0) and mean_corr is not None else None
        row["Mean Corr."] = round_value(mean_corr)
        row["RD"] = round_value(rd)
        row["SI"] = round_value(si)
        out.append(row)
    return sorted(out, key=lambda x: model_sort_key(x["model"]))


def build_robustness_summary(eval_rows: List[Dict[str, Any]], corruptions: List[str], table_name: str) -> List[Dict[str, Any]]:
    out = []
    for model, rows in by_model_rows(eval_rows).items():
        clean_ap = get_clean_ap(rows, "mask_ap")
        mean_corr = get_selected_corrupted_mean(rows, corruptions, "mask_ap")
        clean_box = get_clean_ap(rows, "box_ap")
        mean_box_corr = get_selected_corrupted_mean(rows, corruptions, "box_ap")
        rd = clean_ap - mean_corr if clean_ap is not None and mean_corr is not None else None
        si = mean_corr / clean_ap if clean_ap not in (None, 0) and mean_corr is not None else None
        box_rd = clean_box - mean_box_corr if clean_box is not None and mean_box_corr is not None else None
        mask_box_gap = mean_box_corr - mean_corr if mean_corr is not None and mean_box_corr is not None else None
        corrupted_rows = [r for r in rows if r["corruption"] in corruptions and r["severity"] != ""]
        worst_row = None
        best_row = None
        for r in corrupted_rows:
            ap = safe_float(r.get("mask_ap"))
            if ap is None:
                continue
            if worst_row is None or ap < safe_float(worst_row.get("mask_ap")):
                worst_row = r
            if best_row is None or ap > safe_float(best_row.get("mask_ap")):
                best_row = r
        out.append({
            "model": model,
            "group": "Baseline" if model in BASELINE_MODELS else "Enhanced",
            "clean_mask_AP": round_value(clean_ap),
            "mean_corrupted_mask_AP": round_value(mean_corr),
            "RD": round_value(rd),
            "SI": round_value(si),
            "clean_box_AP": round_value(clean_box),
            "mean_corrupted_box_AP": round_value(mean_box_corr),
            "box_RD": round_value(box_rd),
            "box_minus_mask_gap": round_value(mask_box_gap),
            "worst_condition": worst_row["condition"] if worst_row else "",
            "worst_mask_AP": round_value(worst_row.get("mask_ap")) if worst_row else "",
            "best_corrupted_condition": best_row["condition"] if best_row else "",
            "best_corrupted_mask_AP": round_value(best_row.get("mask_ap")) if best_row else "",
            "num_corrupted_conditions": len(corrupted_rows),
            "benchmark_subset": table_name,
        })
    return sorted(out, key=lambda x: model_sort_key(x["model"]))


def build_ablation_table(eval_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    main = build_main_results_table(eval_rows, MAIN_CORRUPTIONS)
    return [r for r in main if r["model"] in ABLATION_MODELS]


def build_severity_summary_pivot(eval_rows: List[Dict[str, Any]], corruptions: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    out = []
    for model, rows in by_model_rows(eval_rows).items():
        row: Dict[str, Any] = {"model": model, "group": "Baseline" if model in BASELINE_MODELS else "Enhanced"}
        for sev in range(1, 6):
            vals = []
            for r in rows:
                if r.get("severity") not in (sev, str(sev)):
                    continue
                if corruptions is not None and r["corruption"] not in corruptions:
                    continue
                vals.append(safe_float(r.get("mask_ap")))
            row[f"S{sev}"] = round_value(mean(vals))
        out.append(row)
    return sorted(out, key=lambda x: model_sort_key(x["model"]))


def build_ranking_shift(eval_rows: List[Dict[str, Any]], corruptions: List[str]) -> List[Dict[str, Any]]:
    stats = []
    for model, rows in by_model_rows(eval_rows).items():
        clean_ap = get_clean_ap(rows, "mask_ap")
        mean_corr = get_selected_corrupted_mean(rows, corruptions, "mask_ap")
        sev5_vals = [safe_float(r.get("mask_ap")) for r in rows if r["corruption"] in corruptions and r.get("severity") in (5, "5")]
        stats.append({
            "model": model,
            "group": "Baseline" if model in BASELINE_MODELS else "Enhanced",
            "clean_mask_AP": clean_ap,
            "mean_corrupted_mask_AP": mean_corr,
            "severity5_mean_mask_AP": mean(sev5_vals),
        })
    def rank_map(field: str) -> Dict[str, int]:
        ranked = sorted(stats, key=lambda x: (x[field] is None, -(x[field] or -1)))
        return {r["model"]: i + 1 for i, r in enumerate(ranked)}
    cr, mr, s5r = rank_map("clean_mask_AP"), rank_map("mean_corrupted_mask_AP"), rank_map("severity5_mean_mask_AP")
    out = []
    for r in stats:
        model = r["model"]
        shift_corr = mr[model] - cr[model]
        shift_s5 = s5r[model] - cr[model]
        out.append({
            "model": model,
            "group": r["group"],
            "clean_mask_AP": round_value(r["clean_mask_AP"]),
            "clean_rank": cr[model],
            "mean_corrupted_mask_AP": round_value(r["mean_corrupted_mask_AP"]),
            "mean_corrupted_rank": mr[model],
            "severity5_mean_mask_AP": round_value(r["severity5_mean_mask_AP"]),
            "severity5_rank": s5r[model],
            "rank_shift_clean_to_corrupted": shift_corr,
            "rank_shift_clean_to_severity5": shift_s5,
            "shift_symbol_corrupted": "↑" if shift_corr < 0 else ("↓" if shift_corr > 0 else "="),
            "shift_symbol_severity5": "↑" if shift_s5 < 0 else ("↓" if shift_s5 > 0 else "="),
        })
    return sorted(out, key=lambda x: x["clean_rank"])


def build_model_condition_matrix(eval_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    models = sorted(set(r["model"] for r in eval_rows), key=model_sort_key)
    conditions = sorted(set(r["condition"] for r in eval_rows), key=lambda x: (x != "clean", x))
    lookup = {(r["model"], r["condition"]): safe_float(r.get("mask_ap")) for r in eval_rows}
    out = []
    for model in models:
        row = {"model": model}
        for condition in conditions:
            row[condition] = round_value(lookup.get((model, condition)))
        out.append(row)
    return out


def get_fieldnames(rows: List[Dict[str, Any]]) -> List[str]:
    if not rows:
        return []
    fieldnames = list(rows[0].keys())
    for r in rows[1:]:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    return fieldnames


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = get_fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_markdown(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return ""
    headers = get_fieldnames(rows)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join("" if r.get(h) is None else str(r.get(h, "")) for h in headers) + " |")
    return "\n".join(lines) + "\n"


def to_latex(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return ""
    headers = get_fieldnames(rows)
    def esc(x: Any) -> str:
        s = "" if x is None else str(x)
        return s.replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")
    row_end = r" \\" 
    lines = ["\\begin{tabular}{" + "l" * len(headers) + "}", "\\toprule", " & ".join(esc(h) for h in headers) + row_end, "\\midrule"]
    for r in rows:
        lines.append(" & ".join(esc(r.get(h, "")) for h in headers) + row_end)
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines) + "\n"


def write_table_bundle(out_dir: Path, name: str, rows: List[Dict[str, Any]]) -> None:
    write_csv(out_dir / f"{name}.csv", rows)
    (out_dir / f"{name}.md").write_text(to_markdown(rows), encoding="utf-8")
    (out_dir / f"{name}.tex").write_text(to_latex(rows), encoding="utf-8")


def write_xlsx(out_dir: Path, tables: Dict[str, List[Dict[str, Any]]]) -> None:
    try:
        import pandas as pd
    except Exception:
        print("[WARN] pandas not installed, skip Excel export.", file=sys.stderr)
        return
    xlsx_path = out_dir / "paper_tables.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        for name, rows in tables.items():
            pd.DataFrame(rows).to_excel(writer, sheet_name=name[:31], index=False)
    print(f"[OK] Excel: {xlsx_path}")


def write_manifest_info(out_dir: Path, manifest_path: Optional[Path]) -> None:
    if not manifest_path or not manifest_path.is_file():
        return
    m = read_json(manifest_path)
    if not m:
        return
    rows = [{
        "benchmark_name": m.get("benchmark_name"),
        "suite": m.get("suite"),
        "created_at_utc": m.get("created_at_utc"),
        "generator": m.get("generator"),
        "global_seed": m.get("global_seed"),
        "source_image_count": m.get("source_image_count"),
        "num_corruptions": len(m.get("corruptions", [])),
        "num_severities": len(m.get("severities", [])),
        "num_conditions": len(m.get("conditions", [])),
        "total_corrupted_images": len(m.get("conditions", [])) * int(m.get("source_image_count", 0) or 0),
        "corruptions": ", ".join(m.get("corruptions", [])),
        "dependency_versions": json.dumps(m.get("dependency_versions", {}), ensure_ascii=False),
    }]
    write_table_bundle(out_dir, "table_benchmark_manifest", rows)


def import_plotting():
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        return np, plt
    except Exception as exc:
        print(f"[WARN] matplotlib/numpy unavailable, skip figures: {exc}", file=sys.stderr)
        return None, None


def matrix_from_rows(rows: List[Dict[str, Any]], columns: List[str]):
    ordered = sorted(rows, key=lambda x: model_sort_key(x["model"]))
    ylabels = [r["model"] for r in ordered]
    data = [[safe_float(r.get(c)) for c in columns] for r in ordered]
    return ylabels, columns, data


def plot_heatmap(out_dir: Path, name: str, rows: List[Dict[str, Any]], columns: List[str], title: str, cbar_label: str) -> None:
    np, plt = import_plotting()
    if np is None or plt is None or not rows:
        return
    ylabels, xlabels, data = matrix_from_rows(rows, columns)
    arr = np.array([[np.nan if v is None else v for v in row] for row in data], dtype=float)
    fig_w = max(10, 1.25 * len(xlabels) + 4)
    fig_h = max(4.8, 0.55 * len(ylabels) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(arr, aspect="auto")
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_title(title, pad=12)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if not np.isnan(arr[i, j]):
                ax.text(j, i, f"{arr[i, j]:.3f}", ha="center", va="center", fontsize=9)
    baseline_count = sum(1 for y in ylabels if y in BASELINE_MODELS)
    if 0 < baseline_count < len(ylabels):
        ax.axhline(y=baseline_count - 0.5, color="white", linewidth=3)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(out_dir / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_all_heatmaps(out_dir: Path, tables: Dict[str, List[Dict[str, Any]]]) -> None:
    main_rows = tables.get("table_main_results_5corr", [])
    severity_rows = tables.get("table_severity_pivot_5corr", [])
    if not main_rows:
        return
    cols = [display_corruption_name(c) for c in MAIN_CORRUPTIONS]
    plot_heatmap(out_dir, "fig_mean_ap_heatmap_5corr", main_rows, cols, "Mean Mask AP over Severity 1-5 for Selected Corruptions", "Mean mask AP")
    rd_rows = []
    si_rows = []
    for r in main_rows:
        clean = safe_float(r.get("Clean"))
        rd_row = {"model": r["model"], "group": r.get("group", "")}
        si_row = {"model": r["model"], "group": r.get("group", "")}
        for col in cols:
            val = safe_float(r.get(col))
            rd_row[col] = round_value(clean - val if clean is not None and val is not None else None)
            si_row[col] = round_value(val / clean if clean not in (None, 0) and val is not None else None)
        rd_rows.append(rd_row)
        si_rows.append(si_row)
    plot_heatmap(out_dir, "fig_rd_heatmap_5corr", rd_rows, cols, "Robustness Drop by Selected Corruption", "RD = clean AP - corrupted AP")
    plot_heatmap(out_dir, "fig_si_heatmap_5corr", si_rows, cols, "Stability Index by Selected Corruption", "SI = corrupted AP / clean AP")
    if severity_rows:
        plot_heatmap(out_dir, "fig_severity_heatmap_5corr", severity_rows, [f"S{i}" for i in range(1, 6)], "Mean Mask AP by Severity over Selected Corruptions", "Mean mask AP")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="work_dirs/research")
    parser.add_argument("--seed", default="seed_2026")
    parser.add_argument("--manifest", default="mmdet_dataset/lettuce_c/manifest.json")
    parser.add_argument("--out-dir", default="paper_tables_v2")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()
    results_root = Path(args.results_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dirs = discover_model_dirs(results_root, args.seed)
    if not model_dirs:
        raise FileNotFoundError(f"No model dirs found under {results_root} with seed={args.seed}")
    print(f"[INFO] Found {len(model_dirs)} model dirs.")
    for d in model_dirs:
        print(f"  - {d}")
    eval_rows: List[Dict[str, Any]] = []
    for model_dir in model_dirs:
        eval_rows.extend(collect_evaluation_rows(model_dir))
    if not eval_rows:
        print("[WARN] No evaluation metrics found. Did you run evaluate_benchmark.py?", file=sys.stderr)
    tables = {
        "table_training_summary": collect_training_rows(model_dirs),
        "table_clean_performance": build_clean_performance(eval_rows),
        "table_condition_performance": build_condition_performance(eval_rows),
        "table_main_results_5corr": build_main_results_table(eval_rows, MAIN_CORRUPTIONS),
        "table_robustness_summary_5corr": build_robustness_summary(eval_rows, MAIN_CORRUPTIONS, "main_5corr"),
        "table_severity_pivot_5corr": build_severity_summary_pivot(eval_rows, MAIN_CORRUPTIONS),
        "table_ranking_shift_5corr": build_ranking_shift(eval_rows, MAIN_CORRUPTIONS),
        "table_ablation_5corr": build_ablation_table(eval_rows),
        "table_main_results_full": build_main_results_table(eval_rows, CORRUPTION_ORDER),
        "table_robustness_summary_full": build_robustness_summary(eval_rows, CORRUPTION_ORDER, "full_available"),
        "table_severity_pivot_full": build_severity_summary_pivot(eval_rows, CORRUPTION_ORDER),
        "table_ranking_shift_full": build_ranking_shift(eval_rows, CORRUPTION_ORDER),
        "model_condition_matrix_mask_ap": build_model_condition_matrix(eval_rows),
    }
    for name, rows in tables.items():
        write_table_bundle(out_dir, name, rows)
        print(f"[OK] {name}: {len(rows)} rows")
    manifest_path = Path(args.manifest).resolve() if args.manifest else None
    write_manifest_info(out_dir, manifest_path)
    write_xlsx(out_dir, tables)
    if not args.skip_figures:
        plot_all_heatmaps(out_dir, tables)
        print("[OK] Heatmaps exported.")
    report = {
        "results_root": str(results_root),
        "seed": args.seed,
        "model_dirs": [str(p) for p in model_dirs],
        "num_eval_rows": len(eval_rows),
        "output_dir": str(out_dir),
        "main_corruptions": MAIN_CORRUPTIONS,
        "model_order": MODEL_ORDER,
    }
    (out_dir / "generation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("==================================================")
    print(f"Done. Tables saved to: {out_dir}")
    print("Main files:")
    print(f"  {out_dir / 'paper_tables.xlsx'}")
    print(f"  {out_dir / 'table_main_results_5corr.md'}")
    print(f"  {out_dir / 'table_robustness_summary_5corr.md'}")
    print(f"  {out_dir / 'table_ablation_5corr.md'}")
    print(f"  {out_dir / 'fig_mean_ap_heatmap_5corr.png'}")
    print(f"  {out_dir / 'fig_rd_heatmap_5corr.png'}")
    print(f"  {out_dir / 'fig_si_heatmap_5corr.png'}")
    print(f"  {out_dir / 'fig_severity_heatmap_5corr.png'}")
    print("==================================================")


if __name__ == "__main__":
    main()
