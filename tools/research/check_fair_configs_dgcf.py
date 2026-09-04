#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check fair training/evaluation protocol among MMDetection configs.

This script compares protocol fields only. Model architecture differences are expected.
"""

import argparse
import json
from pathlib import Path

from mmengine.config import Config


PROTOCOL_KEYS = [
    "train_dataloader",
    "val_dataloader",
    "test_dataloader",
    "val_evaluator",
    "test_evaluator",
    "train_cfg",
    "val_cfg",
    "test_cfg",
    "optim_wrapper",
    "param_scheduler",
    "default_hooks",
    "randomness",
    "auto_scale_lr",
    "env_cfg",
]


def freeze(x):
    return json.dumps(x, sort_keys=True, default=str)


def load_manifest(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["models"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="tools/research/lettuce_dgcf_manifest.json")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", default="paper_outputs_dgcf/tables/fairness_check.csv")
    args = parser.parse_args()

    models = load_manifest(args.manifest)
    cfg_items = []
    for m in models:
        cfg_path = Path(m["config"].format(seed=args.seed))
        if not cfg_path.is_file():
            print(f"[SKIP] missing config: {cfg_path}")
            continue
        cfg_items.append((m, Config.fromfile(str(cfg_path))))

    if len(cfg_items) < 2:
        raise SystemExit("[ERROR] fewer than two valid configs")

    ref_m, ref_cfg = cfg_items[0]
    rows = []
    print(f"[Reference] {ref_m['name']} | {ref_m['config']}")

    for m, cfg in cfg_items:
        row = {"key": m["key"], "name": m["name"], "role": m.get("role", "")}
        neck = cfg.model.get("neck", {})
        row["neck_type"] = neck.get("type", "inherited") if isinstance(neck, dict) else str(neck)
        print("\n" + "=" * 100)
        print(f"{m['name']} | {m['config']}")
        print(f"neck: {row['neck_type']}")
        for k in PROTOCOL_KEYS:
            same = freeze(cfg.get(k, None)) == freeze(ref_cfg.get(k, None))
            row[k] = "OK" if same else "DIFF"
            print(f"{k:22s}: {row[k]}")
        rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[OK] saved fairness report: {out}")
    print("[NOTE] If dataloader/optimizer/scheduler/randomness are DIFF, explain or fix them.")


if __name__ == "__main__":
    main()
