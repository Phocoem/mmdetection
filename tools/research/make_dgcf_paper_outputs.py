#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect DGCF paper results and generate tables/figures.

Outputs:
- clean performance
- mean corrupted AP by corruption
- severity-wise AP
- RD and SI
- worst AP and S5 AP
- ablation table
- heatmaps and severity curves

The parser scans JSON/CSV files under each model's evaluation directory and tries
to recognize COCO metrics flexibly.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


METRIC_ALIASES = {
    "bbox_mAP": ["bbox_mAP", "coco/bbox_mAP", "bbox_AP", "box_AP"],
    "bbox_mAP_50": ["bbox_mAP_50", "coco/bbox_mAP_50", "bbox_AP50", "box_AP50"],
    "bbox_mAP_75": ["bbox_mAP_75", "coco/bbox_mAP_75", "bbox_AP75", "box_AP75"],
    "segm_mAP": ["segm_mAP", "coco/segm_mAP", "mask_mAP", "mask_AP", "mean_mask_ap"],
    "segm_mAP_50": ["segm_mAP_50", "coco/segm_mAP_50", "mask_AP50"],
    "segm_mAP_75": ["segm_mAP_75", "coco/segm_mAP_75", "mask_AP75"],
    "segm_mAP_s": ["segm_mAP_s", "coco/segm_mAP_s", "mask_APs"],
    "segm_mAP_m": ["segm_mAP_m", "coco/segm_mAP_m", "mask_APm"],
    "segm_mAP_l": ["segm_mAP_l", "coco/segm_mAP_l", "mask_APl"],
}

DISPLAY_CORR = {
    "brightness": "Brightness",
    "contrast": "Contrast",
    "gaussian_noise": "Gaussian Noise",
    "defocus_blur": "Defocus Blur",
    "motion_blur": "Motion Blur",
    "clean": "Clean",
}


def load_manifest(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data


def find_eval_dir(model, seed, eval_root):
    work_dir = Path(model["work_dir"].format(seed=seed))
    candidates = [
        work_dir / "evaluation",
        Path(eval_root) / model["key"] / f"seed_{seed}" / "evaluation",
        Path(eval_root) / model["key"] / "evaluation",
    ]
    for c in candidates:
        if c.exists():
            return c

    root = Path(eval_root)
    if root.exists():
        hits = list(root.glob(f"**/{model['key']}*/**/evaluation"))
        if hits:
            return hits[0]
    return None


def flatten_json(obj):
    rows = []
    if isinstance(obj, dict):
        if any(a in obj for aliases in METRIC_ALIASES.values() for a in aliases):
            rows.append(obj)
        for v in obj.values():
            rows.extend(flatten_json(v))
    elif isinstance(obj, list):
        for v in obj:
            rows.extend(flatten_json(v))
    return rows


def get_metric(d, aliases):
    for a in aliases:
        if a in d:
            try:
                return float(d[a])
            except Exception:
                return np.nan
    return np.nan


def infer_condition(path: Path, row: Dict[str, Any] = None):
    row = row or {}
    corruption = row.get("corruption", row.get("condition", row.get("corruption_type", "")))
    severity = row.get("severity", row.get("level", row.get("sev", None)))

    text = str(path).lower()
    if not corruption:
        if "clean" in text:
            corruption = "clean"
        else:
            for c in DISPLAY_CORR:
                if c != "clean" and c in text:
                    corruption = c
                    break

    if isinstance(corruption, str):
        m = re.match(r"(.+?)[_\-]?s([1-5])$", corruption.lower())
        if m:
            corruption = m.group(1)
            severity = int(m.group(2))

    if severity is None or str(severity).strip() in ["", "nan", "None"]:
        m = re.search(r"(?:severity|sev|s)[_\-]?([1-5])", text)
        if m:
            severity = int(m.group(1))
        else:
            severity = 0 if str(corruption).lower() == "clean" else np.nan

    corruption = str(corruption).lower().replace(" ", "_").replace("-", "_")
    corruption = corruption.replace("jpeg_compression", "jpeg")
    severity = int(severity) if str(severity) not in ["nan", "None"] and not pd.isna(severity) else 0
    return corruption, severity


def rows_from_json(path: Path, model, seed):
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    rows = []
    for d in flatten_json(obj):
        metrics = {k: get_metric(d, aliases) for k, aliases in METRIC_ALIASES.items()}
        if all(pd.isna(v) for v in metrics.values()):
            continue
        corruption, severity = infer_condition(path, d)
        rows.append({
            "model_key": model["key"],
            "model_name": model["name"],
            "role": model.get("role", ""),
            "seed": seed,
            "corruption": corruption,
            "severity": severity,
            "source": str(path),
            **metrics
        })
    return rows


def rows_from_csv(path: Path, model, seed):
    try:
        df = pd.read_csv(path)
    except Exception:
        return []

    rows = []
    for _, r in df.iterrows():
        d = r.to_dict()
        metrics = {k: get_metric(d, aliases) for k, aliases in METRIC_ALIASES.items()}
        if all(pd.isna(v) for v in metrics.values()):
            continue
        corruption, severity = infer_condition(path, d)
        rows.append({
            "model_key": model["key"],
            "model_name": model["name"],
            "role": model.get("role", ""),
            "seed": seed,
            "corruption": corruption,
            "severity": severity,
            "source": str(path),
            **metrics
        })
    return rows


def collect(manifest, eval_root, seeds):
    rows = []
    for seed in seeds:
        for model in manifest["models"]:
            eval_dir = find_eval_dir(model, seed, eval_root)
            if eval_dir is None:
                print(f"[WARN] no evaluation dir: {model['name']} seed={seed}")
                continue

            files = list(eval_dir.rglob("*.json")) + list(eval_dir.rglob("*.csv"))
            print(f"[COLLECT] {model['name']} | {eval_dir} | files={len(files)}")
            for f in files:
                if f.suffix.lower() == ".json":
                    rows.extend(rows_from_json(f, model, seed))
                elif f.suffix.lower() == ".csv":
                    rows.extend(rows_from_csv(f, model, seed))

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No metrics collected. Check evaluator output directory and file format.")
    return df


def aggregate_duplicates(df):
    keys = ["model_key", "model_name", "role", "seed", "corruption", "severity"]
    metric_cols = [c for c in METRIC_ALIASES if c in df.columns]
    return df.groupby(keys, dropna=False)[metric_cols].mean(numeric_only=True).reset_index()


def model_order(manifest):
    return [m["name"] for m in manifest["models"]]


def reindex_models(df, manifest):
    order = model_order(manifest)
    existing = [x for x in order if x in df.index]
    rest = [x for x in df.index if x not in existing]
    return df.loc[existing + rest]


def save_heatmap(pivot, out, title, label, fmt=".3f"):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    pivot = pivot.copy()
    data = pivot.values.astype(float)

    fig_w = max(8.0, 1.25 * len(pivot.columns) + 3.0)
    fig_h = max(4.5, 0.50 * len(pivot.index) + 1.8)
    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(data, aspect="auto")
    plt.colorbar(im, label=label)
    plt.title(title)
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
    plt.yticks(range(len(pivot.index)), pivot.index)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                plt.text(j, i, format(data[i, j], fmt), ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out.with_suffix(".png"), dpi=300)
    plt.savefig(out.with_suffix(".pdf"))
    plt.close()


def save_line(df, out, title, y_label):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    for name, g in df.groupby("model_name"):
        g = g.sort_values("severity")
        plt.plot(g["severity"], g["segm_mAP"], marker="o", label=name)
    plt.title(title)
    plt.xlabel("Severity")
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out.with_suffix(".png"), dpi=300)
    plt.savefig(out.with_suffix(".pdf"))
    plt.close()


def to_display_corruptions(cols):
    return [DISPLAY_CORR.get(str(c), str(c)) for c in cols]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="tools/research/lettuce_dgcf_manifest.json")
    parser.add_argument("--eval-root", default="work_dirs/research")
    parser.add_argument("--out-dir", default="paper_outputs_dgcf")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026])
    parser.add_argument("--main-corruptions", nargs="+", default=[
        "brightness", "contrast", "gaussian_noise", "defocus_blur", "motion_blur"
    ])
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    out = Path(args.out_dir)
    tables = out / "tables"
    figs = out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    raw = collect(manifest, args.eval_root, args.seeds)
    raw.to_csv(tables / "raw_collected_metrics.csv", index=False)

    df = aggregate_duplicates(raw)
    df.to_csv(tables / "metrics_by_condition.csv", index=False)

    # Clean performance
    clean = df[df["corruption"].eq("clean")].copy()
    clean_cols = [
        "model_name", "role", "bbox_mAP", "bbox_mAP_50", "bbox_mAP_75",
        "segm_mAP", "segm_mAP_50", "segm_mAP_75", "segm_mAP_s", "segm_mAP_m", "segm_mAP_l"
    ]
    clean_table = clean.groupby(["model_name", "role"])[[c for c in clean_cols if c not in ["model_name", "role"]]].mean(numeric_only=True).reset_index()
    clean_table.to_csv(tables / "table_clean_performance.csv", index=False)

    # Mean AP by corruption over severities.
    corr_df = df[df["corruption"].isin(args.main_corruptions)].copy()
    by_corr = corr_df.groupby(["model_name", "role", "corruption"])["segm_mAP"].mean().reset_index()
    by_corr.to_csv(tables / "table_mean_ap_by_corruption_long.csv", index=False)

    pivot_corr = by_corr.pivot_table(index="model_name", columns="corruption", values="segm_mAP", aggfunc="mean")
    pivot_corr = pivot_corr.reindex(columns=args.main_corruptions)
    pivot_corr["Mean"] = pivot_corr.mean(axis=1)
    pivot_corr = reindex_models(pivot_corr, manifest)
    pivot_corr.columns = to_display_corruptions(pivot_corr.columns)
    pivot_corr.to_csv(tables / "table_mean_ap_by_corruption.csv")
    save_heatmap(pivot_corr.drop(columns=["Mean"], errors="ignore"), figs / "fig_mean_mask_ap_by_corruption",
                 "Mean Mask AP over Severity 1–5 for Selected Corruptions", "Mean mask AP")

    # Severity-wise.
    sev = corr_df.groupby(["model_name", "role", "severity"])["segm_mAP"].mean().reset_index()
    sev.to_csv(tables / "table_severity_long.csv", index=False)
    pivot_sev = sev.pivot_table(index="model_name", columns="severity", values="segm_mAP", aggfunc="mean")
    pivot_sev = reindex_models(pivot_sev, manifest)
    pivot_sev.columns = [f"S{int(c)}" for c in pivot_sev.columns]
    pivot_sev.to_csv(tables / "table_severity.csv")
    save_heatmap(pivot_sev, figs / "fig_mean_mask_ap_by_severity",
                 "Mean Mask AP by Severity over Selected Corruptions", "Mean mask AP")
    save_line(sev, figs / "fig_severity_curves", "Severity-wise Mean Mask AP", "Mean mask AP")

    # Robustness summary.
    clean_mean = clean.groupby(["model_name", "role"])["segm_mAP"].mean().reset_index().rename(columns={"segm_mAP": "clean_mask_ap"})
    corr_mean = by_corr.groupby(["model_name", "role"])["segm_mAP"].mean().reset_index().rename(columns={"segm_mAP": "mean_corrupted_mask_ap"})
    worst = by_corr.groupby(["model_name", "role"])["segm_mAP"].min().reset_index().rename(columns={"segm_mAP": "worst_corruption_ap"})
    s5 = corr_df[corr_df["severity"].eq(5)].groupby(["model_name", "role"])["segm_mAP"].mean().reset_index().rename(columns={"segm_mAP": "severity5_ap"})

    summary = clean_mean.merge(corr_mean, how="outer").merge(worst, how="outer").merge(s5, how="outer")
    summary["RD"] = summary["clean_mask_ap"] - summary["mean_corrupted_mask_ap"]
    summary["SI"] = summary["mean_corrupted_mask_ap"] / summary["clean_mask_ap"]
    summary["rank_mean_corrupted_ap"] = summary["mean_corrupted_mask_ap"].rank(ascending=False, method="min")
    summary.to_csv(tables / "table_robustness_summary.csv", index=False)

    # RD and SI heatmaps by corruption.
    clean_map = dict(zip(clean_mean["model_name"], clean_mean["clean_mask_ap"]))
    rd_rows = []
    si_rows = []
    for _, r in by_corr.iterrows():
        cap = clean_map.get(r["model_name"], np.nan)
        rd_rows.append({
            "model_name": r["model_name"],
            "corruption": r["corruption"],
            "RD": cap - r["segm_mAP"]
        })
        si_rows.append({
            "model_name": r["model_name"],
            "corruption": r["corruption"],
            "SI": r["segm_mAP"] / cap if cap and np.isfinite(cap) else np.nan
        })

    rd = pd.DataFrame(rd_rows).pivot_table(index="model_name", columns="corruption", values="RD")
    si = pd.DataFrame(si_rows).pivot_table(index="model_name", columns="corruption", values="SI")
    rd = reindex_models(rd.reindex(columns=args.main_corruptions), manifest)
    si = reindex_models(si.reindex(columns=args.main_corruptions), manifest)
    rd.columns = to_display_corruptions(rd.columns)
    si.columns = to_display_corruptions(si.columns)
    rd.to_csv(tables / "table_rd_by_corruption.csv")
    si.to_csv(tables / "table_si_by_corruption.csv")
    save_heatmap(rd, figs / "fig_rd_by_corruption", "Robustness Drop by Corruption", "RD = clean AP - corrupted AP")
    save_heatmap(si, figs / "fig_si_by_corruption", "Stability Index by Corruption", "SI = corrupted AP / clean AP")

    # Ablation table.
    ablation_roles = ["main_baseline", "proposed", "ablation"]
    ablation = summary[summary["role"].isin(ablation_roles)].copy()
    # Put R50, ablations, proposed in readable order.
    ablation_order = ["Mask R-CNN R50", "DGCF w/o Detail", "DGCF w/o Gate", "Enhanced Mask R-CNN (DGCF)"]
    ablation["order"] = ablation["model_name"].apply(lambda x: ablation_order.index(x) if x in ablation_order else 999)
    ablation = ablation.sort_values("order").drop(columns=["order"])
    ablation.to_csv(tables / "table_ablation_dgcf.csv", index=False)

    # Short markdown report.
    report = []
    report.append("# DGCF robustness experiment summary\n")
    report.append("## Main robustness summary\n")
    rep = summary.sort_values("mean_corrupted_mask_ap", ascending=False)
    report.append(rep[["model_name", "role", "clean_mask_ap", "mean_corrupted_mask_ap", "RD", "SI", "worst_corruption_ap", "severity5_ap"]].to_markdown(index=False, floatfmt=".3f"))
    report.append("\n\n## Main claim template\n")
    report.append("Enhanced Mask R-CNN (DGCF) should be compared primarily against Mask R-CNN R50-FPN. "
                  "Do not claim that DGCF is universally best under every corruption. "
                  "Claim robustness improvement over the standard Mask R-CNN baseline.")
    (out / "summary_report.md").write_text("\n".join(report), encoding="utf-8")

    print(f"[DONE] outputs written to: {out}")
    print(f"tables: {tables}")
    print(f"figures: {figs}")


if __name__ == "__main__":
    main()
