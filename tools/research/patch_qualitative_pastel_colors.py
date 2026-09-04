#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patch qualitative visualization colors to use a softer pastel palette.

It modifies:
  tools/research/visualize_dgcf_qualitative_gt.py

Backup:
  tools/research/visualize_dgcf_qualitative_gt.py.bak_pastel
"""

from pathlib import Path


PASTEL_BLOCK = r"""
PASTEL_COLORS = [
    (102, 194, 165),  # teal
    (252, 141, 98),   # salmon
    (141, 160, 203),  # soft blue
    (231, 138, 195),  # pink
    (166, 216, 84),   # green
    (255, 217, 47),   # yellow
    (229, 196, 148),  # beige
    (179, 179, 179),  # gray
    (128, 177, 211),  # sky blue
    (253, 180, 98),   # orange
    (179, 222, 105),  # light green
    (188, 128, 189),  # purple
    (190, 186, 218),  # lavender
    (251, 128, 114),  # coral
    (204, 235, 197),  # mint
    (255, 237, 111),  # pale yellow
]


def get_pastel_color(i):
    return PASTEL_COLORS[i % len(PASTEL_COLORS)]
"""


def main():
    path = Path("tools/research/visualize_dgcf_qualitative_gt.py")
    if not path.exists():
        raise SystemExit("[ERROR] tools/research/visualize_dgcf_qualitative_gt.py not found. Run at MMDetection root.")

    text = path.read_text(encoding="utf-8")

    backup = path.with_suffix(path.suffix + ".bak_pastel")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
        print(f"[OK] backup saved: {backup}")

    if "PASTEL_COLORS" not in text:
        marker = "import matplotlib.pyplot as plt\n"
        if marker not in text:
            raise SystemExit("[ERROR] Cannot find matplotlib import marker.")
        text = text.replace(marker, marker + "\n" + PASTEL_BLOCK + "\n", 1)

    text = text.replace(
        "rng = np.random.default_rng(2026)\n    colors = rng.integers(40, 255, size=(max(1, len(anns)), 3), dtype=np.uint8)",
        "colors = [get_pastel_color(i) for i in range(max(1, len(anns)))]"
    )
    text = text.replace(
        "rng = np.random.default_rng(123)\n    colors = rng.integers(40, 255, size=(max(1, len(inst)), 3), dtype=np.uint8)",
        "colors = [get_pastel_color(i) for i in range(max(1, len(inst)))]"
    )

    text = text.replace(
        "out[mask] = (0.55 * out[mask] + 0.45 * color[mask]).astype(np.uint8)",
        "out[mask] = (0.72 * out[mask] + 0.28 * color[mask]).astype(np.uint8)"
    )

    text = text.replace(
        "color = np.zeros_like(out, dtype=np.uint8)\n        color[:] = colors[i]",
        "color = np.zeros_like(out, dtype=np.uint8)\n        color[:] = np.array(colors[i], dtype=np.uint8)"
    )
    text = text.replace(
        "color = np.zeros_like(out, dtype=np.uint8)\n            color[:] = colors[i]",
        "color = np.zeros_like(out, dtype=np.uint8)\n            color[:] = np.array(colors[i], dtype=np.uint8)"
    )

    text = text.replace("colors[i].tolist()", "tuple(int(x) for x in colors[i])")

    path.write_text(text, encoding="utf-8")
    print(f"[OK] patched pastel colors: {path}")
    print("[OK] overlay alpha changed to 0.28")


if __name__ == "__main__":
    main()
