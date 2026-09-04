#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check whether selected MMDetection configs share the same training protocol.

Run:
    cd /home/pc/mmdet_AI/mmdetection
    export PYTHONPATH=/home/pc/mmdet_AI/mmdetection:$PYTHONPATH
    python tools/research/check_fair_configs.py configs/fair_lettuce/*.py
"""

import argparse
import json
from mmengine.config import Config

KEYS = [
    "train_dataloader",
    "val_dataloader",
    "test_dataloader",
    "optim_wrapper",
    "param_scheduler",
    "train_cfg",
    "val_cfg",
    "test_cfg",
    "default_hooks",
    "randomness",
    "auto_scale_lr",
]


def freeze(obj):
    try:
        return json.dumps(obj, sort_keys=True, default=str)
    except Exception:
        return str(obj)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+")
    args = parser.parse_args()

    records = []
    for cfg_path in args.configs:
        cfg = Config.fromfile(cfg_path)
        rec = {"config": cfg_path}
        for k in KEYS:
            rec[k] = freeze(cfg.get(k, None))
        rec["work_dir"] = cfg.get("work_dir", "")
        rec["model_type"] = cfg.model.get("type", "Inherited")
        rec["neck_type"] = cfg.model.get("neck", {}).get("type", "Inherited/default")
        records.append(rec)

    ref = records[0]
    print(f"[Reference] {ref['config']}")
    print()

    for rec in records:
        print("=" * 100)
        print(rec["config"])
        print(f"work_dir: {rec['work_dir']}")
        print(f"neck_type: {rec['neck_type']}")
        for k in KEYS:
            same = (rec[k] == ref[k])
            print(f"{k:20s}: {'OK' if same else 'DIFF'}")
        print()

    print("Note: Different architecture fields are expected. DIFF in optimizer/scheduler/dataloader/seed indicates unfair protocol.")


if __name__ == "__main__":
    main()
