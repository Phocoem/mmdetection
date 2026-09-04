#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate clean + corrupted benchmark for all DGCF-paper models.

This script calls your existing:
    tools/research/evaluate_benchmark.py
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
    parser.add_argument("--clean-root", default="mmdet_dataset/lettuce")
    parser.add_argument("--benchmark-root", default="mmdet_dataset/lettuce_c")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--evaluator", default="tools/research/evaluate_benchmark.py")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    models = load_manifest(args.manifest)

    for m in models:
        if args.only and m["key"] not in args.only:
            continue

        cfg = Path(m["config"].format(seed=args.seed))
        work_dir = Path(m["work_dir"].format(seed=args.seed))
        checkpoint = m.get("checkpoint", "").format(seed=args.seed) if m.get("checkpoint") else ""
        checkpoint = Path(checkpoint) if checkpoint else find_checkpoint(work_dir)

        if not cfg.is_file():
            print(f"[SKIP] config not found: {cfg}")
            continue
        if checkpoint is None or not Path(checkpoint).is_file():
            print(f"[SKIP] checkpoint not found: {m['name']} | {work_dir}")
            continue

        out_dir = work_dir / "evaluation"
        cmd = [
            "python", args.evaluator,
            str(cfg), str(checkpoint),
            "--clean-root", args.clean_root,
            "--benchmark-root", args.benchmark_root,
            "--output-dir", str(out_dir),
            "--seed", str(args.seed),
        ]

        print("\n" + "=" * 100)
        print(f"[EVAL] {m['name']} | seed={args.seed}")
        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
