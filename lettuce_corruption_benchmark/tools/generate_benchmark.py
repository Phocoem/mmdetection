"""
Generate benchmark folders.

Main protocol:
- clean
- noise
- gaussian_blur
- motion_blur
- brightness
- contrast
- gamma
- shadow
- jpeg
- medium
- hard

Detailed protocol:
- severity 1/2/3 for each single corruption
- clean, medium, hard
"""

import argparse
import subprocess
from pathlib import Path


MAIN_CONDITIONS = [
    ("clean", 1, "clean"),
    ("noise", 2, "noise"),
    ("gaussian_blur", 2, "gaussian_blur"),
    ("motion_blur", 2, "motion_blur"),
    ("brightness", 2, "brightness"),
    ("contrast", 2, "contrast"),
    ("gamma", 2, "gamma"),
    ("shadow", 2, "shadow"),
    ("jpeg", 2, "jpeg"),
    ("medium", 1, "medium"),
    ("hard", 3, "hard"),
]

DETAILED_CONDITIONS = [
    ("clean", 1, "clean"),
    ("medium", 1, "medium"),
    ("hard", 3, "hard"),
]

for cond in ["noise", "gaussian_blur", "motion_blur", "brightness", "contrast", "gamma", "shadow", "jpeg"]:
    for sev in [1, 2, 3]:
        DETAILED_CONDITIONS.append((cond, sev, f"{cond}_s{sev}"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Clean test image folder")
    parser.add_argument("--root-output", required=True, help="Root output folder, e.g. stress")
    parser.add_argument("--mode", choices=["main", "detailed"], default="main")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    conditions = MAIN_CONDITIONS if args.mode == "main" else DETAILED_CONDITIONS
    root = Path(args.root_output)
    root.mkdir(parents=True, exist_ok=True)

    for condition, severity, folder_name in conditions:
        out_dir = root / folder_name
        meta_csv = root / "_metadata" / f"{folder_name}.csv"
        cmd = [
            "python", "tools/stress_engine.py",
            "--input", args.input,
            "--output", str(out_dir),
            "--condition", condition,
            "--severity", str(severity),
            "--seed", str(args.seed),
            "--metadata-csv", str(meta_csv),
        ]
        print(" ".join(cmd))
        subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
