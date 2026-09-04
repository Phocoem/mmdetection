#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple DGCF experiment runner.

Không train. Chỉ dùng checkpoint đã có để evaluate:
- clean test
- 5 corruptions x 5 severities thông qua evaluate_benchmark.py

Main paper models:
- Mask R-CNN R50
- Mask R-CNN R101
- SOLO R50
- YOLACT R50
- Enhanced Mask R-CNN (DGCF)
- DGCF w/o Detail
- DGCF w/o Gate
- SOLOv2 R50

Bỏ: ASPP, RGCFPN.
"""

import argparse
import json
import subprocess
from pathlib import Path


def load_models(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))["models"]


def find_ckpt(work_dir):
    work_dir = Path(work_dir)
    patterns = [
        "best_coco_segm_mAP*.pth",
        "best_*.pth",
        "epoch_*.pth",
        "*.pth",
    ]
    for pat in patterns:
        hits = list(work_dir.glob(pat))
        if hits:
            return sorted(hits, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="tools/research/models_dgcf_simple.json")
    p.add_argument("--clean-root", default="mmdet_dataset/lettuce")
    p.add_argument("--benchmark-root", default="mmdet_dataset/lettuce_c")
    p.add_argument("--evaluator", default="tools/research/evaluate_benchmark.py")
    p.add_argument("--seed", default="2026")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    for m in load_models(args.models):
        cfg = Path(m["config"])
        work_dir = Path(m["work_dir"])
        ckpt = find_ckpt(work_dir)
        out_dir = work_dir / "evaluation"

        if not cfg.is_file():
            print(f"[SKIP] missing config: {cfg}")
            continue
        if ckpt is None:
            print(f"[SKIP] missing checkpoint: {work_dir}")
            continue
        if args.skip_existing and out_dir.exists() and any(out_dir.rglob("*.json")):
            print(f"[SKIP] existing evaluation: {m['name']} -> {out_dir}")
            continue

        cmd = [
            "python", args.evaluator,
            str(cfg), str(ckpt),
            "--clean-root", args.clean_root,
            "--benchmark-root", args.benchmark_root,
            "--output-dir", str(out_dir),
            "--seed", str(args.seed),
        ]

        print("\n" + "=" * 90)
        print(f"[EVAL] {m['name']}")
        print(f"checkpoint: {ckpt}")
        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
