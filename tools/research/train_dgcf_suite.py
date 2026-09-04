#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train all models in lettuce DGCF manifest.

Example:
python tools/research/train_dgcf_suite.py \
  --manifest tools/research/lettuce_dgcf_manifest.json \
  --seeds 2026 \
  --skip-existing
"""

import argparse
import json
import subprocess
from pathlib import Path


def load_manifest(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))["models"]


def find_checkpoint(work_dir):
    wd = Path(work_dir)
    if not wd.exists():
        return None
    patterns = ["best_coco_segm_mAP*.pth", "best_*.pth", "epoch_*.pth", "*.pth"]
    for pat in patterns:
        hits = list(wd.glob(pat))
        if hits:
            return sorted(hits, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="tools/research/lettuce_dgcf_manifest.json")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026])
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()

    models = load_manifest(args.manifest)

    for seed in args.seeds:
        for m in models:
            if args.only and m["key"] not in args.only:
                continue

            cfg = Path(m["config"].format(seed=seed))
            work_dir = Path(m["work_dir"].format(seed=seed))

            if not cfg.is_file():
                print(f"[SKIP] config not found: {cfg}")
                continue

            if args.skip_existing and find_checkpoint(work_dir):
                print(f"[SKIP] checkpoint already exists: {m['name']} | {work_dir}")
                continue

            cmd = [
                "python", "tools/train.py", str(cfg),
                "--work-dir", str(work_dir),
                "--cfg-options", f"randomness.seed={seed}",
            ]
            if args.extra_args:
                cmd.extend(args.extra_args)

            print("\n" + "=" * 100)
            print(f"[TRAIN] {m['name']} | seed={seed}")
            print(" ".join(cmd))
            if not args.dry_run:
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
