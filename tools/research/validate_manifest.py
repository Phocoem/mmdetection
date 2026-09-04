#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate experiment manifest for lettuce DGCF paper."""

import argparse
import json
from pathlib import Path


REQUIRED_MODEL_KEYS = ["key", "name", "role", "config", "work_dir"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--check-files", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    path = Path(args.manifest)
    if not path.is_file():
        raise SystemExit(f"[ERROR] manifest not found: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SystemExit(f"[ERROR] invalid JSON: {e}")

    if "models" not in data or not isinstance(data["models"], list):
        raise SystemExit("[ERROR] manifest must contain a list field: models")

    keys = []
    for i, model in enumerate(data["models"]):
        for k in REQUIRED_MODEL_KEYS:
            if k not in model:
                raise SystemExit(f"[ERROR] model #{i} missing key: {k}")
        if model["key"] in keys:
            raise SystemExit(f"[ERROR] duplicated model key: {model['key']}")
        keys.append(model["key"])

        cfg = Path(model["config"].format(seed=args.seed))
        work_dir = Path(model["work_dir"].format(seed=args.seed))
        if args.check_files and not cfg.is_file():
            print(f"[WARN] config not found: {cfg}")
        if args.check_files and not work_dir.exists():
            print(f"[WARN] work_dir not found yet: {work_dir}")

        bad = ["aspp", "rgcf"]
        if any(x in model["key"].lower() or x in model["name"].lower() for x in bad):
            raise SystemExit(f"[ERROR] ASPP/RGCF should not be in main DGCF manifest: {model['key']}")

    print("[OK] valid JSON manifest")
    print(f"models: {len(data['models'])}")
    print("model keys:")
    for k in keys:
        print(f"  - {k}")


if __name__ == "__main__":
    main()
