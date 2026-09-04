import argparse
import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime
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
}

MODEL_ORDER = [
    "Mask R-CNN R50",
    "Mask R-CNN R101",
    "SOLO R50",
    "SOLOv2 R50",
    "DGCF-FPN",
    "DGCF-FPN w/o Context",
    "DGCF-FPN w/o Detail",
    "DGCF-FPN w/o Gate",
]


def model_name(raw):
    return MODEL_DISPLAY.get(raw, raw)


def model_sort_key(name):
    return (MODEL_ORDER.index(name), name) if name in MODEL_ORDER else (999, name)


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


def safe_int(value):
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except Exception:
        return None


def mean(values):
    arr = [v for v in values if v is not None]
    return sum(arr) / len(arr) if arr else None


def fmt(value, ndigits=6):
    x = safe_float(value)
    return "" if x is None else round(x, ndigits)


def flatten_dict(obj, prefix=""):
    out = {}

    if not isinstance(obj, dict):
        return out

    for key, value in obj.items():
        new_key = f"{prefix}.{key}" if prefix else str(key)

        if isinstance(value, dict):
            out.update(flatten_dict(value, new_key))
        else:
            out[new_key] = value

    return out


def extract_records_from_json_object(obj):
    records = []

    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                records.append(flatten_dict(item))
        return records

    if not isinstance(obj, dict):
        return records

    for key in ["data", "records", "scalars", "vis_data"]:
        if key in obj and isinstance(obj[key], list):
            for item in obj[key]:
                if isinstance(item, dict):
                    records.append(flatten_dict(item))
            return records

    if "scalars" in obj and isinstance(obj["scalars"], dict):
        for name, values in obj["scalars"].items():
            if isinstance(values, list):
                for item in values:
                    if isinstance(item, dict):
                        row = flatten_dict(item)
                        row["name"] = name
                        records.append(row)
            elif isinstance(values, dict):
                row = flatten_dict(values)
                row["name"] = name
                records.append(row)

        if records:
            return records

    records.append(flatten_dict(obj))
    return records


def parse_json_records(path):
    records = []

    try:
        text = path.read_text(encoding="utf-8-sig", errors="ignore").strip()
    except Exception:
        return records

    if not text:
        return records

    if "\n" in text:
        for line in text.splitlines():
            line = line.strip()
            if not line or not line.startswith("{"):
                continue

            try:
                item = json.loads(line)
            except Exception:
                continue

            if isinstance(item, dict):
                records.extend(extract_records_from_json_object(item))

        return records

    try:
        obj = json.loads(text)
    except Exception:
        return records

    return extract_records_from_json_object(obj)


def normalize_record(record, source_path):
    row = dict(record)
    row["_source_json"] = str(source_path)

    epoch = (
        row.get("epoch")
        or row.get("Epoch")
        or row.get("train/epoch")
        or row.get("val/epoch")
        or row.get("step.epoch")
    )

    iteration = (
        row.get("iter")
        or row.get("iteration")
        or row.get("step")
        or row.get("global_step")
        or row.get("train/iter")
        or row.get("val/iter")
    )

    row["_epoch"] = safe_int(epoch)
    row["_iter"] = safe_int(iteration)

    return row


def collect_training_records(train_dir):
    json_files = []

    vis_dir = train_dir / "vis_data"
    if vis_dir.is_dir():
        json_files.extend(sorted(vis_dir.glob("*.json")))
        json_files.extend(sorted(vis_dir.glob("*.jsonl")))

    json_files.extend(sorted(train_dir.glob("*.json")))
    json_files.extend(sorted(train_dir.glob("*.jsonl")))

    records = []

    for path in json_files:
        for raw in parse_json_records(path):
            row = normalize_record(raw, path)
            if row.get("_iter") is not None or row.get("_epoch") is not None:
                records.append(row)

    records.sort(
        key=lambda r: (
            r["_epoch"] if r.get("_epoch") is not None else 10**18,
            r["_iter"] if r.get("_iter") is not None else 10**18,
        )
    )

    return records


def parse_datetime(line):
    patterns = [
        (r"(\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2})", "%Y/%m/%d %H:%M:%S"),
        (r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})", "%Y-%m-%d %H:%M:%S"),
        (r"(\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2})", "%m/%d %H:%M:%S"),
    ]

    for pattern, fmt_str in patterns:
        match = re.search(pattern, line)
        if not match:
            continue

        try:
            dt = datetime.strptime(match.group(1), fmt_str)
            if fmt_str == "%m/%d %H:%M:%S":
                dt = dt.replace(year=datetime.now().year)
            return dt
        except Exception:
            pass

    return None


def parse_log_files(train_dir):
    log_files = sorted(train_dir.glob("*.log"))
    if not log_files:
        log_files = sorted(train_dir.rglob("*.log"))

    first_time = None
    last_time = None
    num_lines = 0
    last_epoch = None
    best_epoch = None
    best_metric = ""
    best_score = None
    best_message = ""

    for log_path in log_files:
        try:
            lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue

        for line in lines:
            num_lines += 1

            dt = parse_datetime(line)
            if dt is not None:
                if first_time is None:
                    first_time = dt
                last_time = dt

            epoch_matches = re.findall(r"Epoch\((?:train|val)\)\s*\[([0-9]+)\]", line)
            if epoch_matches:
                last_epoch = safe_int(epoch_matches[-1])

            ckpt_match = re.search(r"best_.*?_epoch_([0-9]+)\.pth", line)
            if ckpt_match:
                best_epoch = safe_int(ckpt_match.group(1))
                best_message = line.strip()

            if "best" in line.lower():
                score_patterns = [
                    r"best\s+([A-Za-z0-9_./-]+)\s*[:=]\s*([0-9.]+)",
                    r"([A-Za-z0-9_./-]+)\s*[:=]\s*([0-9.]+)",
                ]

                for pattern in score_patterns:
                    match = re.search(pattern, line, flags=re.IGNORECASE)
                    if not match:
                        continue

                    metric = match.group(1)
                    score = safe_float(match.group(2))

                    if score is None:
                        continue

                    if best_score is None or score > best_score:
                        best_score = score
                        best_metric = metric
                        best_message = line.strip()

    duration_sec = None
    if first_time is not None and last_time is not None and last_time >= first_time:
        duration_sec = (last_time - first_time).total_seconds()

    return {
        "num_log_files": len(log_files),
        "num_log_lines": num_lines,
        "log_first_time": first_time.isoformat(sep=" ") if first_time else "",
        "log_last_time": last_time.isoformat(sep=" ") if last_time else "",
        "train_duration_sec_from_log": fmt(duration_sec),
        "train_duration_hours_from_log": fmt(duration_sec / 3600.0 if duration_sec is not None else None),
        "last_epoch_from_log": last_epoch if last_epoch is not None else "",
        "best_epoch_from_log": best_epoch if best_epoch is not None else "",
        "best_metric_from_log": best_metric,
        "best_score_from_log": fmt(best_score),
        "best_message_from_log": best_message,
    }


def infer_best_checkpoint(run_dir, train_dir):
    candidates = []
    candidates.extend(sorted(run_dir.glob("best_*.pth")))
    candidates.extend(sorted(train_dir.glob("best_*.pth")))

    if not candidates:
        return {
            "best_checkpoint": "",
            "best_epoch_from_checkpoint": "",
            "best_metric_from_checkpoint": "",
        }

    path = candidates[-1]
    name = path.name

    epoch = ""
    metric = ""

    m_epoch = re.search(r"epoch_([0-9]+)\.pth", name)
    if m_epoch:
        epoch = safe_int(m_epoch.group(1))

    m_metric = re.search(r"best_(.+?)_epoch_[0-9]+\.pth", name)
    if m_metric:
        metric = m_metric.group(1)

    return {
        "best_checkpoint": name,
        "best_checkpoint_path": str(path),
        "best_epoch_from_checkpoint": epoch,
        "best_metric_from_checkpoint": metric,
    }


def find_numeric_keys(records):
    keys = set()

    for row in records:
        for key, value in row.items():
            if key.startswith("_"):
                continue
            if safe_float(value) is not None:
                keys.add(key)

    return sorted(keys)


def classify_keys(records):
    keys = find_numeric_keys(records)

    loss_keys = [key for key in keys if "loss" in key.lower()]

    lr_keys = [
        key for key in keys
        if key.lower() in {"lr", "learning_rate"}
        or key.lower().endswith(".lr")
        or "learning_rate" in key.lower()
    ]

    time_keys = [
        key for key in keys
        if key.lower() in {"time", "data_time"}
        or key.lower().endswith(".time")
        or key.lower().endswith(".data_time")
        or "iter_time" in key.lower()
    ]

    memory_keys = [
        key for key in keys
        if "memory" in key.lower()
        or "mem" in key.lower()
    ]

    eval_keys = [
        key for key in keys
        if "map" in key.lower()
        or "ap" in key.lower()
        or "bbox" in key.lower()
        or "segm" in key.lower()
        or "coco" in key.lower()
    ]

    return {
        "loss_keys": loss_keys,
        "lr_keys": lr_keys,
        "time_keys": time_keys,
        "memory_keys": memory_keys,
        "eval_keys": eval_keys,
    }


def group_series_by_epoch(records, key, reducer="mean"):
    grouped = defaultdict(list)

    for row in records:
        epoch = row.get("_epoch")
        y = safe_float(row.get(key))

        if epoch is None or y is None:
            continue

        grouped[epoch].append(y)

    xs = []
    ys = []

    for epoch in sorted(grouped):
        values = grouped[epoch]

        if reducer == "last":
            value = values[-1]
        elif reducer == "min":
            value = min(values)
        elif reducer == "max":
            value = max(values)
        else:
            value = mean(values)

        if value is not None:
            xs.append(epoch)
            ys.append(value)

    return xs, ys


def smooth(ys, window):
    if window <= 1 or len(ys) <= 2:
        return ys

    out = []

    for idx in range(len(ys)):
        start = max(0, idx - window + 1)
        chunk = ys[start:idx + 1]
        out.append(sum(chunk) / len(chunk))

    return out


def plot_lines(out_path, title, xlabel, ylabel, series, smooth_window=1):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] Plot skipped: {exc}")
        return

    series = [item for item in series if item[1] and item[2]]
    if not series:
        return

    fig, ax = plt.subplots(figsize=(10.5, 5.8))

    for label, xs, ys in series:
        ax.plot(xs, smooth(ys, smooth_window), linewidth=1.8, marker="o", markersize=3, label=label)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)

    if len(series) <= 12:
        ax.legend(fontsize=8)
    else:
        ax.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def to_markdown(rows):
    if not rows:
        return ""

    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]

    for row in rows:
        lines.append(
            "| " + " | ".join(str(row.get(field, "")) for field in fields) + " |"
        )

    return "\n".join(lines) + "\n"


def write_xlsx(path, tables):
    try:
        import pandas as pd
    except Exception:
        return

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, rows in tables.items():
            pd.DataFrame(rows).to_excel(writer, sheet_name=name[:31], index=False)


def discover_runs(results_root, run_name, train_folder, exclude):
    runs = []

    for run_dir in sorted(results_root.glob(f"*/{run_name}")):
        if not run_dir.is_dir():
            continue

        raw_model = run_dir.parent.name

        if raw_model in exclude:
            continue

        train_dir = run_dir / train_folder
        if not train_dir.is_dir():
            train_dir = run_dir

        runs.append(
            {
                "raw_model": raw_model,
                "model": model_name(raw_model),
                "run_dir": run_dir,
                "train_dir": train_dir,
            }
        )

    return sorted(runs, key=lambda item: model_sort_key(item["model"]))


def build_epoch_records(records, key_info):
    epoch_rows = defaultdict(dict)

    for key in key_info["loss_keys"]:
        xs, ys = group_series_by_epoch(records, key, reducer="mean")
        for x, y in zip(xs, ys):
            epoch_rows[x]["epoch"] = x
            epoch_rows[x][key] = fmt(y)

    for key in key_info["lr_keys"]:
        xs, ys = group_series_by_epoch(records, key, reducer="last")
        for x, y in zip(xs, ys):
            epoch_rows[x]["epoch"] = x
            epoch_rows[x][key] = fmt(y)

    for key in key_info["time_keys"]:
        xs, ys = group_series_by_epoch(records, key, reducer="mean")
        for x, y in zip(xs, ys):
            epoch_rows[x]["epoch"] = x
            epoch_rows[x][key] = fmt(y)

    for key in key_info["eval_keys"]:
        xs, ys = group_series_by_epoch(records, key, reducer="last")
        for x, y in zip(xs, ys):
            epoch_rows[x]["epoch"] = x
            epoch_rows[x][key] = fmt(y)

    return [epoch_rows[k] for k in sorted(epoch_rows)]


def summarize_losses(records, key_info):
    loss_key = "loss" if "loss" in key_info["loss_keys"] else None

    if loss_key is None and key_info["loss_keys"]:
        loss_key = key_info["loss_keys"][0]

    if loss_key is None:
        return {}

    xs, ys = group_series_by_epoch(records, loss_key, reducer="mean")
    if not ys:
        return {}

    return {
        "main_loss_key": loss_key,
        "first_epoch_loss": fmt(ys[0]),
        "last_epoch_loss": fmt(ys[-1]),
        "min_epoch_loss": fmt(min(ys)),
        "epoch_of_min_loss": xs[ys.index(min(ys))],
    }


def summarize_time(records, key_info):
    preferred = None

    for key in ["time", "data_time"]:
        if key in key_info["time_keys"]:
            preferred = key
            break

    if preferred is None and key_info["time_keys"]:
        preferred = key_info["time_keys"][0]

    if preferred is None:
        return {}

    xs, ys = group_series_by_epoch(records, preferred, reducer="mean")
    if not ys:
        return {}

    total = 0.0

    for row in records:
        y = safe_float(row.get(preferred))
        if y is not None:
            total += y

    return {
        "main_time_key": preferred,
        "avg_epoch_iter_time_sec_from_json": fmt(mean(ys)),
        "total_iter_time_sec_from_json": fmt(total),
        "total_iter_time_hours_from_json": fmt(total / 3600.0),
    }


def choose_main_loss_key(key_info):
    if "loss" in key_info["loss_keys"]:
        return "loss"
    return key_info["loss_keys"][0] if key_info["loss_keys"] else None


def choose_main_lr_key(key_info):
    return key_info["lr_keys"][0] if key_info["lr_keys"] else None


def choose_main_time_key(key_info):
    for key in ["time", "data_time"]:
        if key in key_info["time_keys"]:
            return key
    return key_info["time_keys"][0] if key_info["time_keys"] else None


def choose_main_eval_key(key_info):
    preferred = [
        "coco/segm_mAP",
        "segm_mAP",
        "mask_mAP",
        "coco/bbox_mAP",
        "bbox_mAP",
    ]

    for key in preferred:
        if key in key_info["eval_keys"]:
            return key

    return key_info["eval_keys"][0] if key_info["eval_keys"] else None


def collect_one_run(run_info, out_root, smooth_window):
    raw_model = run_info["raw_model"]
    display_model = run_info["model"]
    run_dir = run_info["run_dir"]
    train_dir = run_info["train_dir"]

    records = collect_training_records(train_dir)
    key_info = classify_keys(records)
    epoch_records = build_epoch_records(records, key_info)

    log_info = parse_log_files(train_dir)
    ckpt_info = infer_best_checkpoint(run_dir, train_dir)

    per_model_dir = out_root / "per_model" / raw_model
    per_model_dir.mkdir(parents=True, exist_ok=True)

    write_csv(per_model_dir / "records.csv", records)
    write_csv(per_model_dir / "epoch_records.csv", epoch_records)

    (per_model_dir / "epoch_records.md").write_text(
        to_markdown(epoch_records),
        encoding="utf-8",
    )

    loss_series = []
    for key in key_info["loss_keys"]:
        xs, ys = group_series_by_epoch(records, key, reducer="mean")
        if xs and ys:
            loss_series.append((key, xs, ys))

    plot_lines(
        per_model_dir / "loss_curve_epoch",
        f"{display_model} - Loss",
        "Epoch",
        "Loss",
        loss_series,
        smooth_window=smooth_window,
    )

    lr_series = []
    for key in key_info["lr_keys"]:
        xs, ys = group_series_by_epoch(records, key, reducer="last")
        if xs and ys:
            lr_series.append((key, xs, ys))

    plot_lines(
        per_model_dir / "lr_curve_epoch",
        f"{display_model} - Learning Rate",
        "Epoch",
        "Learning Rate",
        lr_series,
        smooth_window=1,
    )

    time_series = []
    for key in key_info["time_keys"]:
        xs, ys = group_series_by_epoch(records, key, reducer="mean")
        if xs and ys:
            time_series.append((key, xs, ys))

    plot_lines(
        per_model_dir / "time_curve_epoch",
        f"{display_model} - Iteration Time",
        "Epoch",
        "Seconds",
        time_series,
        smooth_window=smooth_window,
    )

    eval_series = []
    for key in key_info["eval_keys"]:
        xs, ys = group_series_by_epoch(records, key, reducer="last")
        if xs and ys:
            eval_series.append((key, xs, ys))

    plot_lines(
        per_model_dir / "eval_metric_curve_epoch",
        f"{display_model} - Evaluation Metrics",
        "Epoch",
        "Metric",
        eval_series,
        smooth_window=1,
    )

    epochs = sorted(
        {
            row.get("_epoch")
            for row in records
            if row.get("_epoch") is not None
        }
    )

    summary = {
        "model": display_model,
        "raw_model": raw_model,
        "run_dir": str(run_dir),
        "train_dir": str(train_dir),
        "num_records": len(records),
        "num_epoch_records": len(epoch_records),
        "first_epoch": epochs[0] if epochs else "",
        "last_epoch": epochs[-1] if epochs else "",
        "loss_keys": ", ".join(key_info["loss_keys"]),
        "lr_keys": ", ".join(key_info["lr_keys"]),
        "time_keys": ", ".join(key_info["time_keys"]),
        "memory_keys": ", ".join(key_info["memory_keys"]),
        "eval_keys": ", ".join(key_info["eval_keys"]),
    }

    summary.update(log_info)
    summary.update(ckpt_info)
    summary.update(summarize_losses(records, key_info))
    summary.update(summarize_time(records, key_info))

    return summary, records, epoch_records, key_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="work_dirs/research")
    parser.add_argument("--run-name", default="no_random")
    parser.add_argument("--train-folder", default="train")
    parser.add_argument("--out-dir", default="training_log_plots_no_random_epoch")
    parser.add_argument("--smooth-window", type=int, default=1)
    parser.add_argument("--exclude", nargs="+", default=["yolact_r50"])
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(
        results_root,
        args.run_name,
        args.train_folder,
        set(args.exclude),
    )

    summaries = []
    all_records = []
    all_epoch_records = []

    combined_loss = []
    combined_lr = []
    combined_time = []
    combined_eval = []

    for run in runs:
        summary, records, epoch_records, key_info = collect_one_run(
            run,
            out_root,
            smooth_window=args.smooth_window,
        )

        summaries.append(summary)

        for row in records:
            flat = {
                "model": run["model"],
                "raw_model": run["raw_model"],
            }
            flat.update(row)
            all_records.append(flat)

        for row in epoch_records:
            flat = {
                "model": run["model"],
                "raw_model": run["raw_model"],
            }
            flat.update(row)
            all_epoch_records.append(flat)

        key = choose_main_loss_key(key_info)
        if key:
            xs, ys = group_series_by_epoch(records, key, reducer="mean")
            if xs and ys:
                combined_loss.append((run["model"], xs, ys))

        key = choose_main_lr_key(key_info)
        if key:
            xs, ys = group_series_by_epoch(records, key, reducer="last")
            if xs and ys:
                combined_lr.append((run["model"], xs, ys))

        key = choose_main_time_key(key_info)
        if key:
            xs, ys = group_series_by_epoch(records, key, reducer="mean")
            if xs and ys:
                combined_time.append((run["model"], xs, ys))

        key = choose_main_eval_key(key_info)
        if key:
            xs, ys = group_series_by_epoch(records, key, reducer="last")
            if xs and ys:
                combined_eval.append((run["model"], xs, ys))

    summaries = sorted(summaries, key=lambda row: model_sort_key(row["model"]))

    write_csv(out_root / "table_training_summary.csv", summaries)
    (out_root / "table_training_summary.md").write_text(
        to_markdown(summaries),
        encoding="utf-8",
    )

    write_csv(out_root / "table_training_records_all.csv", all_records)
    write_csv(out_root / "table_training_epoch_records_all.csv", all_epoch_records)

    write_xlsx(
        out_root / "training_log_tables.xlsx",
        {
            "summary": summaries,
            "records": all_records,
            "epoch_records": all_epoch_records,
        },
    )

    plot_lines(
        out_root / "combined_loss_curve_epoch",
        "Training Loss Comparison",
        "Epoch",
        "Loss",
        combined_loss,
        smooth_window=args.smooth_window,
    )

    plot_lines(
        out_root / "combined_lr_curve_epoch",
        "Learning Rate Comparison",
        "Epoch",
        "Learning Rate",
        combined_lr,
        smooth_window=1,
    )

    plot_lines(
        out_root / "combined_time_curve_epoch",
        "Iteration Time Comparison",
        "Epoch",
        "Seconds",
        combined_time,
        smooth_window=args.smooth_window,
    )

    plot_lines(
        out_root / "combined_eval_metric_curve_epoch",
        "Evaluation Metric Comparison",
        "Epoch",
        "Metric",
        combined_eval,
        smooth_window=1,
    )

    report = {
        "results_root": str(results_root),
        "run_name": args.run_name,
        "train_folder": args.train_folder,
        "out_dir": str(out_root),
        "num_runs": len(runs),
        "num_summary_rows": len(summaries),
        "num_training_records": len(all_records),
        "num_epoch_records": len(all_epoch_records),
        "smooth_window": args.smooth_window,
        "excluded": args.exclude,
        "x_axis": "epoch",
    }

    (out_root / "generation_report_training_logs.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[OK] Runs found: {len(runs)}")
    print(f"[OK] Output dir: {out_root}")
    print(f"[OK] Summary: {out_root / 'table_training_summary.csv'}")
    print(f"[OK] Combined loss: {out_root / 'combined_loss_curve_epoch.png'}")


if __name__ == "__main__":
    main()

