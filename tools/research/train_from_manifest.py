#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train model suite from JSON manifest.

Fixes:
- prints useful message when manifest JSON is invalid
- skips missing configs instead of crashing
- supports {seed} in config/work_dir/checkpoint
"""

import argparse
import json
import subprocess
from pathlib import Path
from json import JSONDecodeError


def load_manifest(path):
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8-sig")
        return json.loads(text)
    except JSONDecodeError as e:
        print("\n[ERROR] Manifest is not valid JSON:", path)
        print(f"        {e}")
        print("\nFirst 30 lines of the file:")
        try:
            lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
            for i, line in enumerate(lines[:30], 1):
                print(f"{i:03d}: {line}")
        except Exception:
            pass
        print("\nCommon causes:")
        print("- JSON has comments using # or //")
        print("- missing comma between model objects")
        print("- trailing comma before ] or }")
        print("- you accidentally passed README.md or a .py file as --manifest")
        raise SystemExit(2)


def has_placeholder(s):
    return "YOUR_" in str(s)


def best_checkpoint_exists(work_dir):
    wd = Path(work_dir)
    return wd.exists() and (list(wd.glob("best_coco_segm_mAP*.pth")) or list(wd.glob("best_*.pth")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026])
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    models = manifest.get("models", [])
    if not models:
        raise SystemExit("[ERROR] Manifest has no 'models' list.")

    for seed in args.seeds:
        for m in models:
            key = m.get("key", "")
            if args.only and key not in args.only:
                continue

            cfg = str(m.get("config", "")).format(seed=seed)
            work_dir = str(m.get("work_dir", "")).format(seed=seed)

            if has_placeholder(cfg) or has_placeholder(work_dir):
                print(f"[SKIP] placeholder entry: {key}")
                continue
            if not Path(cfg).is_file():
                print(f"[SKIP] config not found for {key}: {cfg}")
                continue
            if args.skip_existing and best_checkpoint_exists(work_dir):
                print(f"[SKIP] existing best checkpoint: {key} -> {work_dir}")
                continue

            cmd = [
                "python", "tools/train.py", cfg,
                "--work-dir", work_dir,
                "--cfg-options", f"randomness.seed={seed}",
            ]

            print("\n" + "=" * 100)
            print(f"[TRAIN] {m.get('name', key)} | seed={seed}")
            print(" ".join(cmd))
            if not args.dry_run:
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
