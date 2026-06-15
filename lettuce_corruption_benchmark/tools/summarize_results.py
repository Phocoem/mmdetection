"""
Summarize MMDetection metric JSON files and compute robustness metrics.

Expected input filename pattern:
results/raw_json/model__condition.json

Examples:
maskrcnn_r50__clean.json
maskrcnn_r50__hard.json
solov2__motion_blur.json
"""

import argparse
import json
from pathlib import Path

import pandas as pd


PREFERRED_MASK_KEYS = [
    "coco/segm_mAP", "segm_mAP", "mask_mAP",
    "coco/segm_mAP_50", "segm_mAP_50", "mask_mAP_50",
    "coco/segm_mAP_75", "segm_mAP_75", "mask_mAP_75",
]

def metric_get(d, candidates):
    for k in candidates:
        if k in d:
            return d[k]
    return None


def flatten_dict(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}/{k}"
        if isinstance(v, dict):
            out.update(flatten_dict(v, key))
        else:
            out[key] = v
    return out


def read_metric_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    d = flatten_dict(raw) if isinstance(raw, dict) else {}

    row = {
        "box_AP": metric_get(d, ["coco/bbox_mAP", "bbox_mAP", "bbox/mAP"]),
        "box_AP50": metric_get(d, ["coco/bbox_mAP_50", "bbox_mAP_50", "bbox/mAP_50"]),
        "box_AP75": metric_get(d, ["coco/bbox_mAP_75", "bbox_mAP_75", "bbox/mAP_75"]),
        "mask_AP": metric_get(d, ["coco/segm_mAP", "segm_mAP", "mask_mAP", "segm/mAP"]),
        "mask_AP50": metric_get(d, ["coco/segm_mAP_50", "segm_mAP_50", "mask_mAP_50", "segm/mAP_50"]),
        "mask_AP75": metric_get(d, ["coco/segm_mAP_75", "segm_mAP_75", "mask_mAP_75", "segm/mAP_75"]),
        "AR": metric_get(d, ["coco/segm_AR@100", "segm_AR@100", "AR@100", "segm/AR@100"]),
    }
    return row


def compute_robustness(df: pd.DataFrame) -> pd.DataFrame:
    clean = df[df["condition"] == "clean"][["model", "mask_AP"]].rename(columns={"mask_AP": "AP_clean"})
    medium = df[df["condition"] == "medium"][["model", "mask_AP"]].rename(columns={"mask_AP": "AP_medium"})
    hard = df[df["condition"] == "hard"][["model", "mask_AP"]].rename(columns={"mask_AP": "AP_hard"})

    rob = clean.merge(medium, on="model", how="left").merge(hard, on="model", how="left")
    rob["RD_medium"] = rob["AP_clean"] - rob["AP_medium"]
    rob["RD_hard"] = rob["AP_clean"] - rob["AP_hard"]
    rob["SI"] = rob["AP_hard"] / rob["AP_clean"]
    rob["RAS"] = rob[["AP_clean", "AP_medium", "AP_hard"]].mean(axis=1)
    rob["nRAS"] = rob["RAS"] / rob["AP_clean"]

    rob["rank_clean"] = rob["AP_clean"].rank(ascending=False, method="min").astype("Int64")
    rob["rank_hard"] = rob["AP_hard"].rank(ascending=False, method="min").astype("Int64")
    rob["rank_SI"] = rob["SI"].rank(ascending=False, method="min").astype("Int64")
    rob["rank_shift"] = (rob["rank_clean"] - rob["rank_hard"]).abs()

    # Mean Corruption Performance over single corruptions if available.
    single_conditions = ["noise", "gaussian_blur", "motion_blur", "brightness", "contrast", "gamma", "shadow", "jpeg"]
    mcp = (
        df[df["condition"].isin(single_conditions)]
        .groupby("model")["mask_AP"]
        .mean()
        .reset_index()
        .rename(columns={"mask_AP": "mCP"})
    )
    rob = rob.merge(mcp, on="model", how="left")
    return rob.sort_values(["SI", "AP_hard"], ascending=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Folder containing model__condition.json files")
    parser.add_argument("--out-all", default="results/all_results.csv")
    parser.add_argument("--out-robust", default="results/robustness_summary.csv")
    args = parser.parse_args()

    rows = []
    input_dir = Path(args.input_dir)

    for path in sorted(input_dir.glob("*.json")):
        if "__" not in path.stem:
            print(f"[WARN] Skip {path.name}: filename must be model__condition.json")
            continue

        model, condition = path.stem.split("__", 1)
        metrics = read_metric_json(path)
        rows.append({"model": model, "condition": condition, **metrics})

    if not rows:
        raise RuntimeError(f"No valid result JSON files found in {input_dir}")

    df = pd.DataFrame(rows)
    Path(args.out_all).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_all, index=False)

    rob = compute_robustness(df)
    rob.to_csv(args.out_robust, index=False)

    print(f"Saved: {args.out_all}")
    print(f"Saved: {args.out_robust}")
    print(rob.to_string(index=False))


if __name__ == "__main__":
    main()
