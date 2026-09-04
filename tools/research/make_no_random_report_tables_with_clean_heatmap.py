#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_no_random_report_tables_with_clean_heatmap.py

Collect metrics from:
    work_dirs/research/<model>/<run_name>/evaluation

This version includes Clean test performance in the heatmaps.

Main outputs:
    - table_condition_ap_ap50_ap75.*
    - table_corruption_mean_ap_ap50_ap75.*
    - table_severity_ap_ap50_ap75.*
    - table_combined_clean_corruption_severity_ap_ap50_ap75.*
    - table_training_time_summary.*
    - fig_ap_clean_corruption_mean.png/.pdf
    - fig_rd.png/.pdf
    - fig_si.png/.pdf
    - fig_severity_ap.png/.pdf
    - fig_combined_clean_severity_ap.png/.pdf
    - fig_combined_clean_severity_ap50.png/.pdf
    - fig_combined_clean_severity_ap75.png/.pdf
"""

import argparse
import csv
import json
import math
import re
from pathlib import Path


MODEL_DISPLAY = {
    "mask_rcnn_r50_fpn": "Mask R-CNN R50",
    "mask_rcnn_r101_fpn": "Mask R-CNN R101",
    "mask_rcnn_r50_dgcf_fpn": "DGCF-FPN",
    "mask_rcnn_r50_dgcf_no_context_fpn": "DGCF-FPN w/o Context",
    "mask_rcnn_r50_dgcf_no_detail_fpn": "DGCF-FPN w/o Detail",
    "mask_rcnn_r50_dgcf_no_gate_fpn": "DGCF-FPN w/o Gate",
    "solo_r50": "SOLO R50",
    "solov2_r50": "SOLOv2 R50",
    "condinst_r50": "CondInst R50",
}

MODEL_ORDER = [
    "Mask R-CNN R50",
    "Mask R-CNN R101",
    "SOLO R50",
    "SOLOv2 R50",
    "CondInst R50",
    "DGCF-FPN",
    "DGCF-FPN w/o Context",
    "DGCF-FPN w/o Detail",
    "DGCF-FPN w/o Gate",
]

CORR_DISPLAY = {
    "clean": "Clean",
    "brightness": "Brightness",
    "contrast": "Contrast",
    "gaussian_noise": "Gaussian Noise",
    "defocus_blur": "Defocus Blur",
    "motion_blur": "Motion Blur",
    "jpeg": "JPEG",
    "shadow": "Shadow",
}

COLUMN_DISPLAY = {
    "Clean_AP": "Clean",
    "Clean_AP50": "Clean",
    "Clean_AP75": "Clean",
    "Brightness_AP": "Brightness",
    "Contrast_AP": "Contrast",
    "Gaussian Noise_AP": "Gaussian Noise",
    "Defocus Blur_AP": "Defocus Blur",
    "Motion Blur_AP": "Motion Blur",
    "JPEG_AP": "JPEG",
    "Shadow_AP": "Shadow",
    "MeanCorr_AP": "Mean AP",
    "RD_AP": "RD",
    "SI_AP": "SI",
}

DEFAULT_CORRUPTIONS = [
    "brightness",
    "contrast",
    "gaussian_noise",
]

DEFAULT_EXCLUDE = [
    "yolact_r50",
]

METRIC_KEYS = {
    "mask_AP": [
        "coco/segm_mAP",
        "segm_mAP",
        "mask_mAP",
        "segm/AP",
        "AP_segm",
    ],
    "mask_AP50": [
        "coco/segm_mAP_50",
        "segm_mAP_50",
        "mask_mAP_50",
        "segm/AP50",
        "AP50_segm",
    ],
    "mask_AP75": [
        "coco/segm_mAP_75",
        "segm_mAP_75",
        "mask_mAP_75",
        "segm/AP75",
        "AP75_segm",
    ],
    "box_AP": [
        "coco/bbox_mAP",
        "bbox_mAP",
        "box_mAP",
        "bbox/AP",
        "AP_bbox",
    ],
    "box_AP50": [
        "coco/bbox_mAP_50",
        "bbox_mAP_50",
        "box_mAP_50",
        "bbox/AP50",
        "AP50_bbox",
    ],
    "box_AP75": [
        "coco/bbox_mAP_75",
        "bbox_mAP_75",
        "box_mAP_75",
        "bbox/AP75",
        "AP75_bbox",
    ],
}


def safe_float(value):
    try:
        if value is None or value == "":
            return None
        x = float(value)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except Exception:
        return None


def fmt(value, ndigits=4):
    x = safe_float(value)
    return "" if x is None else round(x, ndigits)


def mean(values):
    arr = [v for v in values if v is not None]
    return sum(arr) / len(arr) if arr else None


def model_name(raw):
    return MODEL_DISPLAY.get(raw, raw)


def corr_name(raw):
    return CORR_DISPLAY.get(raw, raw.replace("_", " ").title())


def compact_name(raw):
    return corr_name(raw).replace(" ", "")


def pretty_col(raw):
    return COLUMN_DISPLAY.get(raw, raw)


def model_sort_key(name):
    return (MODEL_ORDER.index(name), name) if name in MODEL_ORDER else (999, name)


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def get_metric(metrics, logical_name):
    for key in METRIC_KEYS[logical_name]:
        if key in metrics:
            return safe_float(metrics[key])

    suffix = {
        "mask_AP": "segm_mAP",
        "mask_AP50": "segm_mAP_50",
        "mask_AP75": "segm_mAP_75",
        "box_AP": "bbox_mAP",
        "box_AP50": "bbox_mAP_50",
        "box_AP75": "bbox_mAP_75",
    }[logical_name]

    for key, value in metrics.items():
        if str(key).endswith(suffix):
            x = safe_float(value)
            if x is not None:
                return x

    return None


def infer_condition(metrics_path):
    parts = list(metrics_path.parts)

    if "conditions" in parts:
        idx = parts.index("conditions")
        cond_parts = [
            p
            for p in parts[idx + 1:-1]
            if p not in {"runner", "eval", "test"}
        ]
    else:
        cond_parts = [metrics_path.parent.name]

    if not cond_parts:
        return "unknown", None, "unknown"

    if cond_parts[0] == "clean":
        return "clean", None, "clean"

    if len(cond_parts) >= 2 and str(cond_parts[1]).isdigit():
        corruption = cond_parts[0]
        severity = int(cond_parts[1])
        return corruption, severity, f"{corruption}_{severity}"

    joined = "_".join(cond_parts)

    match = re.match(r"(.+?)(?:_severity_|_s|-|_)([1-5])$", joined)
    if match:
        corruption = match.group(1)
        severity = int(match.group(2))
        return corruption, severity, f"{corruption}_{severity}"

    return joined, None, joined


def collect_eval_rows(results_root, run_name, exclude):
    rows = []
    exclude = set(exclude)

    for run_dir in sorted(results_root.glob(f"*/{run_name}")):
        if not run_dir.is_dir():
            continue

        raw_model = run_dir.parent.name

        if raw_model in exclude:
            continue

        eval_dir = run_dir / "evaluation"

        for metrics_path in sorted(eval_dir.rglob("metrics.json")):
            metrics = read_json(metrics_path)
            if not metrics:
                continue

            corruption, severity, condition = infer_condition(metrics_path)

            row = {
                "raw_model": raw_model,
                "model": model_name(raw_model),
                "run": run_name,
                "condition": condition,
                "corruption": corruption,
                "corruption_display": corr_name(corruption),
                "severity": severity if severity is not None else "",
                "metrics_file": str(metrics_path),
            }

            for logical_name in METRIC_KEYS:
                row[logical_name] = get_metric(metrics, logical_name)

            rows.append(row)

    unique = {}
    for row in rows:
        unique[(row["raw_model"], row["condition"])] = row

    return sorted(
        unique.values(),
        key=lambda row: (model_sort_key(row["model"]), str(row["condition"])),
    )


def by_model(rows):
    grouped = {}

    for row in rows:
        grouped.setdefault(row["model"], []).append(row)

    return grouped


def get_clean(rows, metric):
    for row in rows:
        if row["corruption"] == "clean":
            return safe_float(row.get(metric))
    return None


def get_condition_metric(model_rows, corruption, severity, metric):
    for row in model_rows:
        if row["corruption"] != corruption:
            continue
        if row["severity"] == "":
            continue
        if int(row["severity"]) != severity:
            continue
        return safe_float(row.get(metric))
    return None


def build_condition_table(rows):
    table = []

    for row in rows:
        table.append(
            {
                "Model": row["model"],
                "Condition": row["condition"],
                "Corruption": row["corruption_display"],
                "Severity": row["severity"],
                "mask_AP": fmt(row.get("mask_AP")),
                "mask_AP50": fmt(row.get("mask_AP50")),
                "mask_AP75": fmt(row.get("mask_AP75")),
                "box_AP": fmt(row.get("box_AP")),
                "box_AP50": fmt(row.get("box_AP50")),
                "box_AP75": fmt(row.get("box_AP75")),
            }
        )

    return sorted(
        table,
        key=lambda item: (model_sort_key(item["Model"]), str(item["Condition"])),
    )


def build_mean_table(rows, corruptions, max_severity):
    table = []

    for model, model_rows in by_model(rows).items():
        clean_ap = get_clean(model_rows, "mask_AP")
        clean_ap50 = get_clean(model_rows, "mask_AP50")
        clean_ap75 = get_clean(model_rows, "mask_AP75")

        row = {
            "Model": model,
            "Clean_AP": fmt(clean_ap),
            "Clean_AP50": fmt(clean_ap50),
            "Clean_AP75": fmt(clean_ap75),
        }

        all_ap = []
        all_ap50 = []
        all_ap75 = []

        for corruption in corruptions:
            corr_rows = [
                r
                for r in model_rows
                if r["corruption"] == corruption
                and r["severity"] != ""
                and int(r["severity"]) <= max_severity
            ]

            ap = mean(safe_float(r.get("mask_AP")) for r in corr_rows)
            ap50 = mean(safe_float(r.get("mask_AP50")) for r in corr_rows)
            ap75 = mean(safe_float(r.get("mask_AP75")) for r in corr_rows)

            row[f"{corr_name(corruption)}_AP"] = fmt(ap)
            row[f"{corr_name(corruption)}_AP50"] = fmt(ap50)
            row[f"{corr_name(corruption)}_AP75"] = fmt(ap75)

            all_ap.extend(safe_float(r.get("mask_AP")) for r in corr_rows)
            all_ap50.extend(safe_float(r.get("mask_AP50")) for r in corr_rows)
            all_ap75.extend(safe_float(r.get("mask_AP75")) for r in corr_rows)

        mean_ap = mean(all_ap)
        mean_ap50 = mean(all_ap50)
        mean_ap75 = mean(all_ap75)

        row["MeanCorr_AP"] = fmt(mean_ap)
        row["MeanCorr_AP50"] = fmt(mean_ap50)
        row["MeanCorr_AP75"] = fmt(mean_ap75)

        row["RD_AP"] = fmt(
            clean_ap - mean_ap
            if clean_ap is not None and mean_ap is not None
            else None
        )

        row["SI_AP"] = fmt(
            mean_ap / clean_ap
            if clean_ap not in (None, 0) and mean_ap is not None
            else None
        )

        table.append(row)

    return sorted(table, key=lambda item: model_sort_key(item["Model"]))


def build_severity_table(rows, corruptions, max_severity):
    table = []

    for model, model_rows in by_model(rows).items():
        row = {"Model": model}

        for severity in range(1, max_severity + 1):
            severity_rows = [
                r
                for r in model_rows
                if r["corruption"] in corruptions
                and r["severity"] != ""
                and int(r["severity"]) == severity
            ]

            row[f"S{severity}_AP"] = fmt(
                mean(safe_float(r.get("mask_AP")) for r in severity_rows)
            )
            row[f"S{severity}_AP50"] = fmt(
                mean(safe_float(r.get("mask_AP50")) for r in severity_rows)
            )
            row[f"S{severity}_AP75"] = fmt(
                mean(safe_float(r.get("mask_AP75")) for r in severity_rows)
            )

        table.append(row)

    return sorted(table, key=lambda item: model_sort_key(item["Model"]))


def build_combined_clean_corruption_severity_table(rows, corruptions, max_severity):
    table = []

    for model, model_rows in by_model(rows).items():
        row = {"Model": model}

        row["Clean_AP"] = fmt(get_clean(model_rows, "mask_AP"))
        row["Clean_AP50"] = fmt(get_clean(model_rows, "mask_AP50"))
        row["Clean_AP75"] = fmt(get_clean(model_rows, "mask_AP75"))

        for corruption in corruptions:
            cname = compact_name(corruption)

            for severity in range(1, max_severity + 1):
                row[f"{cname}_S{severity}_AP"] = fmt(
                    get_condition_metric(model_rows, corruption, severity, "mask_AP")
                )
                row[f"{cname}_S{severity}_AP50"] = fmt(
                    get_condition_metric(model_rows, corruption, severity, "mask_AP50")
                )
                row[f"{cname}_S{severity}_AP75"] = fmt(
                    get_condition_metric(model_rows, corruption, severity, "mask_AP75")
                )

        table.append(row)

    return sorted(table, key=lambda item: model_sort_key(item["Model"]))


def build_metric_heatmap_table_from_combined(
    combined_table,
    corruptions,
    max_severity,
    metric_suffix,
):
    table = []

    clean_key = f"Clean_{metric_suffix}"

    for row in combined_table:
        new_row = {
            "Model": row["Model"],
            "Clean": row.get(clean_key, ""),
        }

        for corruption in corruptions:
            cname = compact_name(corruption)
            display_name = corr_name(corruption)

            for severity in range(1, max_severity + 1):
                source_col = f"{cname}_S{severity}_{metric_suffix}"
                target_col = f"{display_name} S{severity}"
                new_row[target_col] = row.get(source_col, "")

        table.append(new_row)

    return table


def parse_training_time(run_dir):
    json_logs = sorted(run_dir.glob("*.json"))

    if (run_dir / "train" / "vis_data").is_dir():
        json_logs += sorted((run_dir / "train" / "vis_data").glob("*.json"))
    if (run_dir / "vis_data").is_dir():
        json_logs += sorted((run_dir / "vis_data").glob("*.json"))

    times = []
    last_epoch = ""

    for json_log in json_logs:
        try:
            lines = json_log.read_text(
                encoding="utf-8",
                errors="ignore",
            ).splitlines()
        except Exception:
            continue

        for line in lines:
            line = line.strip()
            if not line.startswith("{"):
                continue

            try:
                item = json.loads(line)
            except Exception:
                continue

            if "epoch" in item:
                last_epoch = item.get("epoch")

            if "time" in item:
                t = safe_float(item.get("time"))
                if t is not None:
                    times.append(t)

    best = sorted(run_dir.glob("best_*.pth"))
    best += sorted((run_dir / "train").glob("best_*.pth")) if (run_dir / "train").is_dir() else []

    logs = sorted(run_dir.glob("*.log"))
    logs += sorted((run_dir / "train").glob("*.log")) if (run_dir / "train").is_dir() else []

    return {
        "run_dir": str(run_dir),
        "best_checkpoint": best[-1].name if best else "",
        "last_epoch": last_epoch,
        "avg_iter_time_sec": fmt(mean(times)),
        "json_time_samples": len(times),
        "num_log_files": len(logs),
    }


def build_training_table(results_root, run_name, exclude):
    table = []
    exclude = set(exclude)

    for run_dir in sorted(results_root.glob(f"*/{run_name}")):
        if not run_dir.is_dir():
            continue

        raw_model = run_dir.parent.name

        if raw_model in exclude:
            continue

        row = {
            "Model": model_name(raw_model),
            "raw_model": raw_model,
        }
        row.update(parse_training_time(run_dir))
        table.append(row)

    return sorted(table, key=lambda item: model_sort_key(item["Model"]))


def table_fields(rows):
    fields = []

    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    return fields


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = table_fields(rows)

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def to_md(rows):
    fields = table_fields(rows)

    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]

    for row in rows:
        lines.append(
            "| "
            + " | ".join(str(row.get(field, "")) for field in fields)
            + " |"
        )

    return "\n".join(lines) + "\n"


def to_tex(rows):
    fields = table_fields(rows)

    def escape(value):
        text = str("" if value is None else value)
        text = text.replace("\\", "\\textbackslash{}")
        text = text.replace("_", "\\_")
        text = text.replace("&", "\\&")
        text = text.replace("%", "\\%")
        return text

    lines = [
        "\\begin{tabular}{" + "l" * len(fields) + "}",
        "\\toprule",
        " & ".join(escape(field) for field in fields) + r" \\",
        "\\midrule",
    ]

    for row in rows:
        lines.append(
            " & ".join(escape(row.get(field, "")) for field in fields) + r" \\"
        )

    lines.extend(["\\bottomrule", "\\end{tabular}"])

    return "\n".join(lines) + "\n"


def write_bundle(out_dir, name, rows):
    write_csv(out_dir / f"{name}.csv", rows)
    (out_dir / f"{name}.md").write_text(to_md(rows), encoding="utf-8")
    (out_dir / f"{name}.tex").write_text(to_tex(rows), encoding="utf-8")


def write_xlsx(out_dir, tables):
    try:
        import pandas as pd

        with pd.ExcelWriter(
            out_dir / "paper_tables_no_random.xlsx",
            engine="openpyxl",
        ) as writer:
            for name, rows in tables.items():
                pd.DataFrame(rows).to_excel(
                    writer,
                    sheet_name=name[:31],
                    index=False,
                )
    except Exception as exc:
        print(f"[WARN] Excel skipped: {exc}")


def text_color_for_value(value, vmin, vmax):
    if value is None or math.isnan(value):
        return "black"

    if vmax <= vmin:
        return "black"

    norm = (value - vmin) / (vmax - vmin)

    return "white" if norm < 0.18 or norm > 0.72 else "black"


def plot_heatmap(
    out_dir,
    name,
    table,
    cols,
    title,
    cmap="coolwarm",
    reverse=False,
    value_label="Value",
    vmin=None,
    vmax=None,
):
    try:
        import numpy as np
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] Plot skipped: {exc}")
        return

    if not table:
        return

    y_labels = [row["Model"] for row in table]

    values = np.array(
        [
            [
                np.nan
                if safe_float(row.get(col)) is None
                else safe_float(row.get(col))
                for col in cols
            ]
            for row in table
        ],
        dtype=float,
    )

    valid = values[~np.isnan(values)]
    if valid.size == 0:
        return

    if vmin is None:
        vmin = float(valid.min())
    if vmax is None:
        vmax = float(valid.max())

    cmap_name = cmap + "_r" if reverse and not cmap.endswith("_r") else cmap

    fig_w = max(9.5, len(cols) * 0.95 + 4.5)
    fig_h = max(4.8, len(y_labels) * 0.62 + 1.8)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    image = ax.imshow(
        values,
        aspect="auto",
        cmap=cmap_name,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(
        [pretty_col(col) for col in cols],
        rotation=45,
        ha="right",
        fontsize=10,
    )

    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=10)

    ax.set_title(title, fontsize=14, pad=14)

    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(y_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if not np.isnan(values[i, j]):
                ax.text(
                    j,
                    i,
                    f"{values[i, j]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color=text_color_for_value(values[i, j], vmin, vmax),
                )

    dgcf_start = None
    for index, label in enumerate(y_labels):
        if "DGCF" in label:
            dgcf_start = index
            break

    if dgcf_start is not None and dgcf_start > 0:
        ax.axhline(dgcf_start - 0.5, color="white", linewidth=5.0)

    for cidx, col in enumerate(cols):
        if col == "Clean":
            ax.axvline(cidx + 0.5, color="white", linewidth=5.0)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.04)
    colorbar.set_label(value_label, fontsize=10)
    colorbar.ax.tick_params(labelsize=9)

    fig.tight_layout()

    fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", dpi=300, bbox_inches="tight")

    plt.close(fig)


def plot_metric_heatmaps(out_dir, mean_table, severity_table, corruptions, max_severity):
    ap_cols = ["Clean_AP"]
    ap_cols += [f"{corr_name(corruption)}_AP" for corruption in corruptions]
    ap_cols += ["MeanCorr_AP"]

    plot_heatmap(
        out_dir,
        "fig_ap_clean_corruption_mean",
        mean_table,
        ap_cols,
        "Clean and Mean Corrupted Mask AP",
        cmap="coolwarm",
        reverse=False,
        value_label="Mask AP",
        vmin=0.0,
        vmax=1.0,
    )

    plot_heatmap(
        out_dir,
        "fig_rd",
        mean_table,
        ["RD_AP"],
        "Robustness Drop",
        cmap="coolwarm",
        reverse=True,
        value_label="RD = Clean AP - Corrupted AP",
        vmin=0.0,
        vmax=None,
    )

    plot_heatmap(
        out_dir,
        "fig_si",
        mean_table,
        ["SI_AP"],
        "Stability Index",
        cmap="coolwarm",
        reverse=False,
        value_label="SI = Corrupted AP / Clean AP",
        vmin=0.0,
        vmax=1.0,
    )

    severity_cols = [f"S{severity}_AP" for severity in range(1, max_severity + 1)]

    plot_heatmap(
        out_dir,
        "fig_severity_ap",
        severity_table,
        severity_cols,
        "Mean Mask AP by Severity",
        cmap="coolwarm",
        reverse=False,
        value_label="Mask AP",
        vmin=0.0,
        vmax=1.0,
    )


def plot_combined_clean_corruption_severity_heatmaps(
    out_dir,
    combined_table,
    corruptions,
    max_severity,
):
    metric_configs = [
        ("AP", "Mask AP", "fig_combined_clean_severity_ap"),
        ("AP50", "Mask AP50", "fig_combined_clean_severity_ap50"),
        ("AP75", "Mask AP75", "fig_combined_clean_severity_ap75"),
    ]

    for metric_suffix, metric_label, fig_name in metric_configs:
        heatmap_table = build_metric_heatmap_table_from_combined(
            combined_table,
            corruptions,
            max_severity,
            metric_suffix,
        )

        cols = ["Clean"]

        for corruption in corruptions:
            display_name = corr_name(corruption)
            for severity in range(1, max_severity + 1):
                cols.append(f"{display_name} S{severity}")

        plot_heatmap(
            out_dir,
            fig_name,
            heatmap_table,
            cols,
            f"{metric_label}: Clean and Corrupted Severity",
            cmap="coolwarm",
            reverse=False,
            value_label=metric_label,
            vmin=0.0,
            vmax=1.0,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="work_dirs/research")
    parser.add_argument("--run-name", default="no_random")
    parser.add_argument("--out-dir", default="paper_tables_no_random")
    parser.add_argument("--corruptions", nargs="+", default=DEFAULT_CORRUPTIONS)
    parser.add_argument("--max-severity", type=int, default=3)
    parser.add_argument("--exclude", nargs="+", default=DEFAULT_EXCLUDE)
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_eval_rows(
        results_root,
        args.run_name,
        args.exclude,
    )

    combined_table = build_combined_clean_corruption_severity_table(
        rows,
        args.corruptions,
        args.max_severity,
    )

    tables = {
        "table_condition_ap_ap50_ap75": build_condition_table(rows),
        "table_corruption_mean_ap_ap50_ap75": build_mean_table(
            rows,
            args.corruptions,
            args.max_severity,
        ),
        "table_severity_ap_ap50_ap75": build_severity_table(
            rows,
            args.corruptions,
            args.max_severity,
        ),
        "table_combined_clean_corruption_severity_ap_ap50_ap75": combined_table,
        "table_training_time_summary": build_training_table(
            results_root,
            args.run_name,
            args.exclude,
        ),
    }

    for name, table in tables.items():
        write_bundle(out_dir, name, table)
        print(f"[OK] {name}: {len(table)} rows")

    write_xlsx(out_dir, tables)

    plot_metric_heatmaps(
        out_dir,
        tables["table_corruption_mean_ap_ap50_ap75"],
        tables["table_severity_ap_ap50_ap75"],
        args.corruptions,
        args.max_severity,
    )

    plot_combined_clean_corruption_severity_heatmaps(
        out_dir,
        tables["table_combined_clean_corruption_severity_ap_ap50_ap75"],
        args.corruptions,
        args.max_severity,
    )

    report = {
        "results_root": str(results_root),
        "run_name": args.run_name,
        "out_dir": str(out_dir),
        "num_eval_rows": len(rows),
        "corruptions": args.corruptions,
        "max_severity": args.max_severity,
        "excluded": args.exclude,
        "clean_included_in_heatmaps": True,
    }

    (out_dir / "generation_report_no_random.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[DONE] Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
