#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_no_random_report_tables.py

Collect metrics from work_dirs/research/<model>/no_random/evaluation
and export AP/AP50/AP75 by severity, corruption mean, RD, SI, training log summary,
CSV/MD/TEX/XLSX and heatmap PNG/PDF.
"""

import argparse, csv, json, math, re
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterable

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
MODEL_ORDER = ["Mask R-CNN R50","Mask R-CNN R101","SOLO R50","SOLOv2 R50","DGCF-FPN","DGCF-FPN w/o Context","DGCF-FPN w/o Detail","DGCF-FPN w/o Gate"]
CORR_DISPLAY = {"clean":"Clean","brightness":"Brightness","contrast":"Contrast","gaussian_noise":"Gaussian Noise"}
COLUMN_DISPLAY = {
    "Brightness_AP": "Brightness",
    "Contrast_AP": "Contrast",
    "Gaussian Noise_AP": "Gaussian Noise",
    "MeanCorr_AP": "Mean AP",
    "RD_AP": "RD",
    "SI_AP": "SI",
}
DEFAULT_CORRUPTIONS = ["brightness","contrast","gaussian_noise"]
METRIC_KEYS = {
    "mask_AP": ["coco/segm_mAP","segm_mAP","mask_mAP","segm/AP","AP_segm"],
    "mask_AP50": ["coco/segm_mAP_50","segm_mAP_50","mask_mAP_50","segm/AP50","AP50_segm"],
    "mask_AP75": ["coco/segm_mAP_75","segm_mAP_75","mask_mAP_75","segm/AP75","AP75_segm"],
    "box_AP": ["coco/bbox_mAP","bbox_mAP","box_mAP","bbox/AP","AP_bbox"],
    "box_AP50": ["coco/bbox_mAP_50","bbox_mAP_50","box_mAP_50","bbox/AP50","AP50_bbox"],
    "box_AP75": ["coco/bbox_mAP_75","bbox_mAP_75","box_mAP_75","bbox/AP75","AP75_bbox"],
}

def safe_float(x):
    try:
        if x is None or x == "": return None
        y = float(x)
        return None if math.isnan(y) or math.isinf(y) else y
    except Exception:
        return None
def fmt(x, nd=4):
    y = safe_float(x)
    return "" if y is None else round(y, nd)
def mean(vals):
    arr = [v for v in vals if v is not None]
    return sum(arr)/len(arr) if arr else None
def model_name(raw): return MODEL_DISPLAY.get(raw, raw)
def corr_name(raw): return CORR_DISPLAY.get(raw, raw.replace("_"," ").title())
def pretty_col(raw): return COLUMN_DISPLAY.get(raw, raw)
def model_sort_key(m): return (MODEL_ORDER.index(m), m) if m in MODEL_ORDER else (999, m)
def read_json(p: Path):
    try: return json.loads(p.read_text(encoding="utf-8-sig"))
    except Exception: return None

def get_metric(metrics, logical):
    for k in METRIC_KEYS[logical]:
        if k in metrics: return safe_float(metrics[k])
    suffix = {"mask_AP":"segm_mAP","mask_AP50":"segm_mAP_50","mask_AP75":"segm_mAP_75","box_AP":"bbox_mAP","box_AP50":"bbox_mAP_50","box_AP75":"bbox_mAP_75"}[logical]
    for k,v in metrics.items():
        if str(k).endswith(suffix):
            y = safe_float(v)
            if y is not None: return y
    return None

def infer_condition(metrics_path: Path):
    parts = list(metrics_path.parts)
    if "conditions" in parts:
        idx = parts.index("conditions")
        cond_parts = [p for p in parts[idx+1:-1] if p not in {"runner","eval","test"}]
    else:
        cond_parts = [metrics_path.parent.name]
    if not cond_parts: return "unknown", None, "unknown"
    if cond_parts[0] == "clean": return "clean", None, "clean"
    if len(cond_parts) >= 2 and str(cond_parts[1]).isdigit():
        c, s = cond_parts[0], int(cond_parts[1])
        return c, s, f"{c}_{s}"
    joined = "_".join(cond_parts)
    m = re.match(r"(.+?)(?:_severity_|_s|-|_)([1-5])$", joined)
    if m:
        c, s = m.group(1), int(m.group(2))
        return c, s, f"{c}_{s}"
    return joined, None, joined

def collect_eval_rows(results_root, run_name):
    rows = []
    for run_dir in sorted(results_root.glob(f"*/{run_name}")):
        if not run_dir.is_dir(): continue
        raw = run_dir.parent.name
        for mp in sorted((run_dir/"evaluation").rglob("metrics.json")):
            metrics = read_json(mp)
            if not metrics: continue
            corruption, severity, condition = infer_condition(mp)
            r = {"raw_model":raw, "model":model_name(raw), "run":run_name, "condition":condition, "corruption":corruption, "corruption_display":corr_name(corruption), "severity":severity if severity is not None else "", "metrics_file":str(mp)}
            for logical in METRIC_KEYS:
                r[logical] = get_metric(metrics, logical)
            rows.append(r)
    uniq = {}
    for r in rows:
        uniq[(r["raw_model"], r["condition"])] = r
    return sorted(uniq.values(), key=lambda r: (model_sort_key(r["model"]), str(r["condition"])))

def by_model(rows):
    d = {}
    for r in rows: d.setdefault(r["model"], []).append(r)
    return d
def get_clean(rows, metric):
    for r in rows:
        if r["corruption"] == "clean": return safe_float(r.get(metric))
    return None

def build_condition_table(rows):
    return sorted([{"Model":r["model"],"Condition":r["condition"],"Corruption":r["corruption_display"],"Severity":r["severity"],"mask_AP":fmt(r.get("mask_AP")),"mask_AP50":fmt(r.get("mask_AP50")),"mask_AP75":fmt(r.get("mask_AP75")),"box_AP":fmt(r.get("box_AP")),"box_AP50":fmt(r.get("box_AP50")),"box_AP75":fmt(r.get("box_AP75"))} for r in rows], key=lambda x:(model_sort_key(x["Model"]), str(x["Condition"])))

def build_mean_table(rows, corruptions, max_severity):
    out = []
    for model, rs in by_model(rows).items():
        row = {"Model":model}
        clean_ap, clean50, clean75 = get_clean(rs,"mask_AP"), get_clean(rs,"mask_AP50"), get_clean(rs,"mask_AP75")
        row.update({"Clean_AP":fmt(clean_ap),"Clean_AP50":fmt(clean50),"Clean_AP75":fmt(clean75)})
        all_ap, all50, all75 = [], [], []
        for c in corruptions:
            crs = [r for r in rs if r["corruption"] == c and r["severity"] != "" and int(r["severity"]) <= max_severity]
            ap, ap50, ap75 = mean(safe_float(r.get("mask_AP")) for r in crs), mean(safe_float(r.get("mask_AP50")) for r in crs), mean(safe_float(r.get("mask_AP75")) for r in crs)
            row[f"{corr_name(c)}_AP"] = fmt(ap); row[f"{corr_name(c)}_AP50"] = fmt(ap50); row[f"{corr_name(c)}_AP75"] = fmt(ap75)
            all_ap += [safe_float(r.get("mask_AP")) for r in crs]; all50 += [safe_float(r.get("mask_AP50")) for r in crs]; all75 += [safe_float(r.get("mask_AP75")) for r in crs]
        m_ap, m50, m75 = mean(all_ap), mean(all50), mean(all75)
        row["MeanCorr_AP"] = fmt(m_ap); row["MeanCorr_AP50"] = fmt(m50); row["MeanCorr_AP75"] = fmt(m75)
        row["RD_AP"] = fmt(clean_ap - m_ap if clean_ap is not None and m_ap is not None else None)
        row["SI_AP"] = fmt(m_ap/clean_ap if clean_ap not in (None,0) and m_ap is not None else None)
        out.append(row)
    return sorted(out, key=lambda r:model_sort_key(r["Model"]))

def build_severity_table(rows, corruptions, max_severity):
    out=[]
    for model, rs in by_model(rows).items():
        row={"Model":model}
        for s in range(1, max_severity+1):
            srs=[r for r in rs if r["corruption"] in corruptions and r["severity"] != "" and int(r["severity"])==s]
            row[f"S{s}_AP"]=fmt(mean(safe_float(r.get("mask_AP")) for r in srs))
            row[f"S{s}_AP50"]=fmt(mean(safe_float(r.get("mask_AP50")) for r in srs))
            row[f"S{s}_AP75"]=fmt(mean(safe_float(r.get("mask_AP75")) for r in srs))
        out.append(row)
    return sorted(out, key=lambda r:model_sort_key(r["Model"]))

def parse_training_time(run_dir):
    json_logs = sorted(run_dir.glob("*.json"))
    if (run_dir/"vis_data").is_dir(): json_logs += sorted((run_dir/"vis_data").glob("*.json"))
    times=[]; last_epoch=""
    for jp in json_logs:
        try:
            for line in jp.read_text(encoding="utf-8", errors="ignore").splitlines():
                line=line.strip()
                if not line.startswith("{"): continue
                obj=json.loads(line)
                if "epoch" in obj: last_epoch=obj.get("epoch")
                if "time" in obj:
                    t=safe_float(obj.get("time"))
                    if t is not None: times.append(t)
        except Exception: pass
    best = sorted(run_dir.glob("best_*.pth"))
    logs = sorted(run_dir.glob("*.log"))
    return {"run_dir":str(run_dir), "best_checkpoint":best[-1].name if best else "", "last_epoch":last_epoch, "avg_iter_time_sec":fmt(mean(times)), "json_time_samples":len(times), "num_log_files":len(logs)}

def build_training_table(results_root, run_name):
    out=[]
    for run_dir in sorted(results_root.glob(f"*/{run_name}")):
        if not run_dir.is_dir(): continue
        raw=run_dir.parent.name
        row={"Model":model_name(raw), "raw_model":raw}
        row.update(parse_training_time(run_dir))
        out.append(row)
    return sorted(out, key=lambda r:model_sort_key(r["Model"]))

def fields(rows):
    fs=[]
    for r in rows:
        for k in r:
            if k not in fs: fs.append(k)
    return fs
def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fs=fields(rows)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w=csv.DictWriter(f, fieldnames=fs); w.writeheader(); w.writerows(rows)
def to_md(rows):
    fs=fields(rows)
    lines=["| "+" | ".join(fs)+" |", "| "+" | ".join(["---"]*len(fs))+" |"]
    for r in rows: lines.append("| "+" | ".join(str(r.get(f,"")) for f in fs)+" |")
    return "\n".join(lines)+"\n"
def to_tex(rows):
    fs=fields(rows)
    def esc(x): return str("" if x is None else x).replace("\\","\\textbackslash{}").replace("_","\\_").replace("&","\\&").replace("%","\\%")
    lines=["\\begin{tabular}{"+"l"*len(fs)+"}", "\\toprule", " & ".join(esc(f) for f in fs)+r" \\", "\\midrule"]
    for r in rows: lines.append(" & ".join(esc(r.get(f,"")) for f in fs)+r" \\")
    lines += ["\\bottomrule","\\end{tabular}"]
    return "\n".join(lines)+"\n"
def write_bundle(out_dir, name, rows):
    write_csv(out_dir/f"{name}.csv", rows)
    (out_dir/f"{name}.md").write_text(to_md(rows), encoding="utf-8")
    (out_dir/f"{name}.tex").write_text(to_tex(rows), encoding="utf-8")

def write_xlsx(out_dir, tables):
    try:
        import pandas as pd
        with pd.ExcelWriter(out_dir/"paper_tables_no_random.xlsx", engine="openpyxl") as writer:
            for name, rows in tables.items():
                pd.DataFrame(rows).to_excel(writer, sheet_name=name[:31], index=False)
    except Exception as e:
        print(f"[WARN] Excel skipped: {e}")

def _text_color_for_value(value, vmin, vmax):
    if value is None or math.isnan(value):
        return "black"
    if vmax <= vmin:
        return "black"
    norm = (value - vmin) / (vmax - vmin)
    return "white" if norm < 0.35 or norm > 0.78 else "black"

def plot_heatmap(out_dir, name, table, cols, title, cmap="YlGnBu", reverse=False, value_label="Mask AP"):
    """
    Cleaner paper-style heatmap.

    Do not mix AP, RD, and SI in one heatmap because their numeric ranges are
    very different. Export them as separate figures for readability.
    """
    try:
        import numpy as np
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] Plot skipped: {e}")
        return

    if not table:
        return

    ys = [r["Model"] for r in table]
    arr = np.array(
        [[np.nan if safe_float(r.get(c)) is None else safe_float(r.get(c)) for c in cols] for r in table],
        dtype=float
    )

    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return

    vmin, vmax = float(valid.min()), float(valid.max())
    cmap_name = cmap + "_r" if reverse and not cmap.endswith("_r") else cmap

    fig_w = max(8.5, len(cols) * 1.15 + 4.2)
    fig_h = max(4.8, len(ys) * 0.52 + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(arr, aspect="auto", cmap=cmap_name, vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([pretty_col(c) for c in cols], rotation=25, ha="right", fontsize=10)
    ax.set_yticks(np.arange(len(ys)))
    ax.set_yticklabels(ys, fontsize=10)
    ax.set_title(title, fontsize=13, pad=12)

    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ys), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if not np.isnan(arr[i, j]):
                ax.text(
                    j, i, f"{arr[i, j]:.3f}",
                    ha="center", va="center",
                    fontsize=8.5,
                    color=_text_color_for_value(arr[i, j], vmin, vmax)
                )

    # Separate baseline and DGCF group visually.
    dgcf_start = None
    for idx, name_y in enumerate(ys):
        if "DGCF" in name_y:
            dgcf_start = idx
            break
    if dgcf_start is not None and dgcf_start > 0:
        ax.axhline(dgcf_start - 0.5, color="white", linewidth=3)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.035)
    cbar.set_label(value_label, fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_metric_heatmaps(out_dir, mean_table, severity_table, corruptions, max_severity):
    # AP only: easy to read.
    ap_cols = [f"{corr_name(c)}_AP" for c in corruptions] + ["MeanCorr_AP"]
    plot_heatmap(
        out_dir,
        "fig_ap_corruption_mean",
        mean_table,
        ap_cols,
        "Mean Mask AP by Corruption",
        cmap="YlGnBu",
        reverse=False,
        value_label="Mask AP"
    )

    # RD only: lower is better, use reversed warm colormap.
    plot_heatmap(
        out_dir,
        "fig_rd",
        mean_table,
        ["RD_AP"],
        "Robustness Drop (Lower is Better)",
        cmap="YlOrRd",
        reverse=True,
        value_label="RD = Clean AP - Corrupted AP"
    )

    # SI only: higher is better.
    plot_heatmap(
        out_dir,
        "fig_si",
        mean_table,
        ["SI_AP"],
        "Stability Index (Higher is Better)",
        cmap="YlGnBu",
        reverse=False,
        value_label="SI = Corrupted AP / Clean AP"
    )

    # Severity AP only.
    sev_cols = [f"S{s}_AP" for s in range(1, max_severity + 1)]
    plot_heatmap(
        out_dir,
        "fig_severity_ap",
        severity_table,
        sev_cols,
        "Mean Mask AP by Severity",
        cmap="YlGnBu",
        reverse=False,
        value_label="Mask AP"
    )


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--results-root", default="work_dirs/research")
    p.add_argument("--run-name", default="no_random")
    p.add_argument("--out-dir", default="paper_tables_no_random")
    p.add_argument("--corruptions", nargs="+", default=DEFAULT_CORRUPTIONS)
    p.add_argument("--max-severity", type=int, default=3)
    args=p.parse_args()
    rr=Path(args.results_root).resolve(); out_dir=Path(args.out_dir).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    rows=collect_eval_rows(rr,args.run_name)
    tables={
        "table_condition_ap_ap50_ap75":build_condition_table(rows),
        "table_corruption_mean_ap_ap50_ap75":build_mean_table(rows,args.corruptions,args.max_severity),
        "table_severity_ap_ap50_ap75":build_severity_table(rows,args.corruptions,args.max_severity),
        "table_training_time_summary":build_training_table(rr,args.run_name),
    }
    for n,t in tables.items():
        write_bundle(out_dir,n,t); print(f"[OK] {n}: {len(t)} rows")
    write_xlsx(out_dir,tables)
    plot_metric_heatmaps(
        out_dir,
        tables["table_corruption_mean_ap_ap50_ap75"],
        tables["table_severity_ap_ap50_ap75"],
        args.corruptions,
        args.max_severity
    )
    report={"results_root":str(rr),"run_name":args.run_name,"out_dir":str(out_dir),"num_eval_rows":len(rows),"corruptions":args.corruptions,"max_severity":args.max_severity}
    (out_dir/"generation_report_no_random.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] Outputs saved to: {out_dir}")
if __name__=="__main__":
    main()