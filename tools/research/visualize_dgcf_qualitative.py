#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qualitative predictions for Mask R-CNN R50 vs Enhanced Mask R-CNN (DGCF).

By default it visualizes only:
- Mask R-CNN R50
- Enhanced Mask R-CNN (DGCF)

because these are the key comparison for the paper.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt


def patch_torch_load():
    if getattr(torch.load, "_lettuce_patch_weights_only_false", False):
        return
    old = torch.load
    def patched(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return old(*args, **kwargs)
    patched._lettuce_patch_weights_only_false = True
    torch.load = patched


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def find_checkpoint(work_dir):
    wd = Path(work_dir)
    if not wd.exists():
        return None
    for pat in ["best_coco_segm_mAP*.pth", "best_*.pth", "epoch_*.pth", "*.pth"]:
        hits = list(wd.glob(pat))
        if hits:
            return sorted(hits, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    return None


def coco_image(root, ann, prefix, idx):
    data = read_json(Path(root) / ann)
    imgs = sorted(data["images"], key=lambda x: x.get("id", 0))
    img = imgs[max(0, min(idx, len(imgs)-1))]
    return Path(root) / prefix / img["file_name"]


def condition_path(clean_root, bench_root, bench_manifest, condition, idx):
    if condition == "clean":
        return coco_image(clean_root, "annotations/test.json", "images/test", idx)

    corruption, severity = condition.split(":")
    severity = int(severity)
    man = read_json(bench_manifest)
    ann = man.get("output_annotation", "annotations/test_png.json")
    for item in man.get("conditions", []):
        if item.get("corruption") == corruption and int(item.get("severity")) == severity:
            return coco_image(bench_root, ann, item["image_prefix"].rstrip("/"), idx)
    raise RuntimeError(f"condition not found in manifest: {condition}")


def load_detector(config, checkpoint, device):
    patch_torch_load()
    from mmdet.utils import register_all_modules
    from mmdet.apis import init_detector
    register_all_modules(init_default_scope=True)
    return init_detector(str(config), str(checkpoint), device=device)


def overlay(image_bgr, result, score_thr):
    out = image_bgr.copy()
    inst = getattr(result, "pred_instances", None)
    if inst is None or len(inst) == 0:
        return out

    scores = getattr(inst, "scores", None)
    masks = getattr(inst, "masks", None)
    bboxes = getattr(inst, "bboxes", None)
    if scores is not None:
        keep = scores.detach().cpu().numpy() >= score_thr
    else:
        keep = np.ones(len(inst), dtype=bool)

    rng = np.random.default_rng(2026)
    colors = rng.integers(30, 255, size=(max(1, len(inst)), 3), dtype=np.uint8)

    if masks is not None:
        mn = masks.detach().cpu().numpy().astype(bool)
        for i, m in enumerate(mn):
            if not keep[i]:
                continue
            color = np.zeros_like(out, dtype=np.uint8)
            color[:] = colors[i]
            out[m] = (0.50 * out[m] + 0.50 * color[m]).astype(np.uint8)

    if bboxes is not None:
        boxes = bboxes.detach().cpu().numpy()
        for i, box in enumerate(boxes):
            if not keep[i]:
                continue
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(out, (x1, y1), (x2, y2), colors[i].tolist(), 2)
            if scores is not None:
                cv2.putText(out, f"{scores[i].item():.2f}", (x1, max(0, y1-4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors[i].tolist(), 1)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="tools/research/lettuce_dgcf_manifest.json")
    p.add_argument("--clean-root", default="mmdet_dataset/lettuce")
    p.add_argument("--benchmark-root", default="mmdet_dataset/lettuce_c")
    p.add_argument("--benchmark-manifest", default="mmdet_dataset/lettuce_c/manifest.json")
    p.add_argument("--out-dir", default="paper_outputs_dgcf/qualitative")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--image-indices", nargs="+", type=int, default=[0, 5, 10])
    p.add_argument("--conditions", nargs="+", default=["clean", "contrast:5", "gaussian_noise:5", "defocus_blur:5", "motion_blur:5"])
    p.add_argument("--only", nargs="+", default=["mask_rcnn_r50_fpn", "mask_rcnn_r50_dgcf_fpn"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--score-thr", type=float, default=0.30)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = read_json(args.manifest)
    models = []
    for m in data["models"]:
        if args.only and m["key"] not in args.only:
            continue
        cfg = Path(m["config"].format(seed=args.seed))
        wd = Path(m["work_dir"].format(seed=args.seed))
        ckpt = Path(m["checkpoint"].format(seed=args.seed)) if m.get("checkpoint") else find_checkpoint(wd)
        if not cfg.is_file() or ckpt is None or not Path(ckpt).is_file():
            print(f"[SKIP] missing cfg/ckpt: {m['name']}")
            continue
        models.append((m, load_detector(cfg, ckpt, args.device)))

    if not models:
        raise RuntimeError("No valid models loaded.")

    from mmdet.apis import inference_detector

    for idx in args.image_indices:
        for cond in args.conditions:
            img_path = condition_path(args.clean_root, args.benchmark_root, args.benchmark_manifest, cond, idx)
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[WARN] cannot read: {img_path}")
                continue

            panels = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB)]
            titles = [f"{cond}\ninput"]

            for m, model in models:
                with torch.no_grad():
                    result = inference_detector(model, str(img_path))
                vis = overlay(img, result, args.score_thr)
                panels.append(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
                titles.append(m["name"])

            plt.figure(figsize=(4 * len(panels), 4))
            for i, (panel, title) in enumerate(zip(panels, titles), 1):
                plt.subplot(1, len(panels), i)
                plt.imshow(panel)
                plt.title(title, fontsize=9)
                plt.axis("off")
            plt.tight_layout()
            out = out_dir / f"qual_idx{idx}_{cond.replace(':','_')}.png"
            plt.savefig(out, dpi=300)
            plt.close()
            print(f"[OK] {out}")


if __name__ == "__main__":
    main()
