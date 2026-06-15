import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SINGLE_CONDITIONS = ["noise", "gaussian_blur", "motion_blur", "brightness", "contrast", "gamma", "shadow", "jpeg"]


def fig_clean_saturation(df, outdir):
    clean = df[df["condition"] == "clean"].sort_values("mask_AP", ascending=False)
    plt.figure(figsize=(12, 5))
    plt.bar(clean["model"], clean["mask_AP"])
    plt.ylabel("Clean Mask AP")
    plt.xlabel("Model")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = outdir / "fig1_clean_saturation.png"
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def fig_decay_curve(df, outdir):
    order = ["clean", "medium", "hard"]
    sub = df[df["condition"].isin(order)].copy()
    sub["condition"] = pd.Categorical(sub["condition"], categories=order, ordered=True)
    sub = sub.sort_values(["model", "condition"])

    plt.figure(figsize=(9, 5))
    for model, g in sub.groupby("model"):
        plt.plot(g["condition"].astype(str), g["mask_AP"], marker="o", label=model)
    plt.ylabel("Mask AP")
    plt.xlabel("Stress level")
    plt.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    path = outdir / "fig2_robustness_decay.png"
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def fig_corruption_heatmap(df, outdir):
    clean = df[df["condition"] == "clean"][["model", "mask_AP"]].rename(columns={"mask_AP": "AP_clean"})
    merged = df.merge(clean, on="model", how="left")
    merged["AP_drop"] = merged["AP_clean"] - merged["mask_AP"]

    keep = SINGLE_CONDITIONS + ["medium", "hard"]
    pivot = merged[merged["condition"].isin(keep)].pivot(index="model", columns="condition", values="AP_drop")
    pivot = pivot[[c for c in keep if c in pivot.columns]]

    plt.figure(figsize=(11, max(4, 0.45 * len(pivot))))
    im = plt.imshow(pivot.values, aspect="auto")
    plt.colorbar(im, label="Mask AP Drop")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    plt.tight_layout()
    path = outdir / "fig3_corruption_sensitivity_heatmap.png"
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def fig_ranking_shift(rob, outdir):
    rob = rob.sort_values("rank_shift", ascending=False)
    plt.figure(figsize=(11, 5))
    plt.bar(rob["model"], rob["rank_shift"])
    plt.ylabel("Ranking Shift")
    plt.xlabel("Model")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = outdir / "fig4_ranking_shift.png"
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def fig_si_bar(rob, outdir):
    rob = rob.sort_values("SI", ascending=False)
    plt.figure(figsize=(11, 5))
    plt.bar(rob["model"], rob["SI"])
    plt.ylabel("Stability Index = AP_hard / AP_clean")
    plt.xlabel("Model")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = outdir / "fig5_stability_index.png"
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-results", default="results/all_results.csv")
    parser.add_argument("--robust-results", default="results/robustness_summary.csv")
    parser.add_argument("--outdir", default="results/figures")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.all_results)
    rob = pd.read_csv(args.robust_results)

    paths = [
        fig_clean_saturation(df, outdir),
        fig_decay_curve(df, outdir),
        fig_corruption_heatmap(df, outdir),
        fig_ranking_shift(rob, outdir),
        fig_si_bar(rob, outdir),
    ]

    for p in paths:
        print(f"Saved: {p}")


if __name__ == "__main__":
    main()
