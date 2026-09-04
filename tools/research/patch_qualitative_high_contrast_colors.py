#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patch qualitative visualization to use high-contrast colors against soil/green background.

It modifies:
  tools/research/visualize_dgcf_qualitative_gt.py

Backup:
  tools/research/visualize_dgcf_qualitative_gt.py.bak_highcontrast

Recommended for lettuce/soil images:
- cyan
- magenta
- yellow
- red
- blue
- lime
- orange
- white

These colors stand out more clearly than pastel colors.
"""

from pathlib import Path


PALETTE_BLOCK = r"""
HIGH_CONTRAST_COLORS = [
    (0, 255, 255),     # cyan
    (255, 0, 255),     # magenta
    (255, 255, 0),     # yellow
    (255, 60, 60),     # red
    (0, 120, 255),     # blue
    (80, 255, 80),     # lime
    (255, 165, 0),     # orange
    (255, 255, 255),   # white
    (180, 0, 255),     # violet
    (0, 255, 160),     # aqua green
    (255, 100, 180),   # hot pink
    (120, 220, 255),   # light cyan-blue
]


def get_vis_color(i):
    return HIGH_CONTRAST_COLORS[i % len(HIGH_CONTRAST_COLORS)]
"""


def main():
    path = Path("tools/research/visualize_dgcf_qualitative_gt.py")
    if not path.exists():
        raise SystemExit("[ERROR] tools/research/visualize_dgcf_qualitative_gt.py not found. Run at MMDetection root.")

    text = path.read_text(encoding="utf-8")

    backup = path.with_suffix(path.suffix + ".bak_highcontrast")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
        print(f"[OK] backup saved: {backup}")

    # Remove previous pastel block if it exists only by overriding function names.
    if "HIGH_CONTRAST_COLORS" not in text:
        marker = "import matplotlib.pyplot as plt\n"
        if marker not in text:
            raise SystemExit("[ERROR] Cannot find matplotlib import marker.")
        text = text.replace(marker, marker + "\n" + PALETTE_BLOCK + "\n", 1)

    # Replace old random color generation.
    text = text.replace(
        "rng = np.random.default_rng(2026)\n    colors = rng.integers(40, 255, size=(max(1, len(anns)), 3), dtype=np.uint8)",
        "colors = [get_vis_color(i) for i in range(max(1, len(anns)))]"
    )
    text = text.replace(
        "rng = np.random.default_rng(123)\n    colors = rng.integers(40, 255, size=(max(1, len(inst)), 3), dtype=np.uint8)",
        "colors = [get_vis_color(i) for i in range(max(1, len(inst)))]"
    )

    # Replace pastel patched color generation if already patched.
    text = text.replace(
        "colors = [get_pastel_color(i) for i in range(max(1, len(anns)))]",
        "colors = [get_vis_color(i) for i in range(max(1, len(anns)))]"
    )
    text = text.replace(
        "colors = [get_pastel_color(i) for i in range(max(1, len(inst)))]",
        "colors = [get_vis_color(i) for i in range(max(1, len(inst)))]"
    )

    # Use stronger overlay than pastel but still preserve image detail.
    text = text.replace(
        "out[mask] = (0.55 * out[mask] + 0.45 * color[mask]).astype(np.uint8)",
        "out[mask] = (0.62 * out[mask] + 0.38 * color[mask]).astype(np.uint8)"
    )
    text = text.replace(
        "out[mask] = (0.72 * out[mask] + 0.28 * color[mask]).astype(np.uint8)",
        "out[mask] = (0.62 * out[mask] + 0.38 * color[mask]).astype(np.uint8)"
    )

    # Make sure tuple colors are converted to np array when filling color image.
    text = text.replace(
        "color = np.zeros_like(out, dtype=np.uint8)\n        color[:] = colors[i]",
        "color = np.zeros_like(out, dtype=np.uint8)\n        color[:] = np.array(colors[i], dtype=np.uint8)"
    )
    text = text.replace(
        "color = np.zeros_like(out, dtype=np.uint8)\n            color[:] = colors[i]",
        "color = np.zeros_like(out, dtype=np.uint8)\n            color[:] = np.array(colors[i], dtype=np.uint8)"
    )

    # Support both original numpy colors and tuple colors for cv2 drawing.
    text = text.replace("colors[i].tolist()", "tuple(int(x) for x in colors[i])")

    path.write_text(text, encoding="utf-8")
    print(f"[OK] patched high-contrast colors: {path}")
    print("[OK] overlay alpha set to 0.38")
    print("[Palette] cyan, magenta, yellow, red, blue, lime, orange, white")


if __name__ == "__main__":
    main()
