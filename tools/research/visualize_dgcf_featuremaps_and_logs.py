#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Feature map visualization and train log plotting for DGCF paper."""

import argparse
import json
import re
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


def coco_image(root, ann, prefix, idx):
    data = read_json(Path(root) / ann)
    imgs = sorted(data["images"], key=lambda x: x.get("id", 0))
    img = imgs[max(0, min(idx, len(imgs)-1))]
    return Path(root) / prefix / img["file_name"]


def collect_conditions(clean_root, benchmark_root, benchmark_manifest, idx, only, severities):
    out = [("clean", coco_image(clean_root, "annotations/test.json", "images/test", idx))]
    man = read_json(benchmark_manifest)
    ann = man.get("output_annotation", "annotations/test_png.json")
    for item in man.get("conditions", []):
        c = item["corruption"]
        s = int(item["severity"])
        if only and c not in only:
            continue
        if severities and s not in severities:
            continue
        img = coco_image(benchmark_root, ann, item["image_prefix"].rstrip("/"), idx)
        out.append((f"{c}_s{s}", img))
    return out


def load_model(config, checkpoint, device):
    patch_torch_load()
    from mmdet.utils import register_all_modules
    from mmdet.apis import init_detector
    register_all_modules(init_default_scope=True)
    return init_detector(str(config), str(checkpoint), device=device)


class CaptureHook:
    def __init__(self, module):
        self.output = None
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.output = output

    def close(self):
        self.handle.remove()


def module_by_name(model, layer):
    if layer == "backbone":
        return model.backbone
    if layer == "neck":
        return model.neck
    if layer == "rpn_head":
        return model.rpn_head
    raise ValueError(layer)


def normalize(x):
    x = np.nan_to_num(x.astype(np.float32))
    x -= x.min()
    return x / (x.max() + 1e-8)


def feature_list(output):
    if isinstance(output, torch.Tensor):
        return [output]
    if isinstance(output, (list, tuple)):
        return list(output)
    if isinstance(output, dict):
        return list(output.values())
    return []


def save_featuremap(image_bgr, fmap, out_prefix):
    h, w = image_bgr.shape[:2]
    fmap = cv2.resize(normalize(fmap), (w, h))
    heat = cv2.applyColorMap(np.uint8(255 * fmap), cv2.COLORMAP_JET)
    overlay = np.uint8(0.55 * image_bgr + 0.45 * heat)
    cv2.imwrite(str(out_prefix) + "_overlay.png", overlay)

    plt.figure(figsize=(5, 5))
    plt.imshow(fmap)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(str(out_prefix) + "_heatmap.png", dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


def visualize_features(args):
    if not args.config or not args.checkpoint:
        print("[SKIP] feature maps require --config and --checkpoint")
        return

    from mmdet.apis import inference_detector

    model = load_model(args.config, args.checkpoint, args.device)
    hook = CaptureHook(module_by_name(model, args.layer))
    out_root = Path(args.out_dir) / "featuremaps"
    out_root.mkdir(parents=True, exist_ok=True)

    conds = collect_conditions(
        args.clean_root, args.benchmark_root, args.benchmark_manifest,
        args.image_index, args.only_corruptions, args.severities
    )

    for name, img_path in conds:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        hook.output = None
        with torch.no_grad():
            _ = inference_detector(model, str(img_path))

        cdir = out_root / name
        cdir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(cdir / "input.png"), img)

        feats = feature_list(hook.output)
        for level, feat in enumerate(feats):
            if level not in args.levels:
                continue
            if not isinstance(feat, torch.Tensor) or feat.ndim != 4:
                continue
            fmap = feat.detach().float().abs().mean(dim=1)[0].cpu().numpy()
            save_featuremap(img, fmap, cdir / f"{args.layer}_P{level+2}")
        print(f"[OK] featuremaps: {name}")

    hook.close()


def parse_log(path):
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    data = {"loss": [], "lr": [], "segm_mAP": [], "bbox_mAP": [], "segm_mAP_50": [], "segm_mAP_75": []}
    x = 0.0
    for line_no, line in enumerate(text.splitlines(), 1):
        mt = re.search(r"Epoch\(train\)\s+\[(\d+)\]\[(\d+)/(\d+)\]", line)
        if mt:
            ep, it, total = map(int, mt.groups())
            x = ep + it / max(total, 1)
        mv = re.search(r"Epoch\(val\)\s+\[(\d+)\]", line)
        vx = float(mv.group(1)) if mv else x if x else float(line_no)

        m = re.search(r"(?:^|\s)loss:\s*([0-9.eE+-]+)", line)
        if m:
            data["loss"].append((x if x else line_no, float(m.group(1))))

        m = re.search(r"\blr:\s*([0-9.eE+-]+)", line)
        if m:
            data["lr"].append((x if x else line_no, float(m.group(1))))

        patterns = {
            "segm_mAP": r"coco/segm_mAP:\s*([0-9.eE+-]+)",
            "bbox_mAP": r"coco/bbox_mAP:\s*([0-9.eE+-]+)",
            "segm_mAP_50": r"coco/segm_mAP_50:\s*([0-9.eE+-]+)",
            "segm_mAP_75": r"coco/segm_mAP_75:\s*([0-9.eE+-]+)",
        }
        for key, pat in patterns.items():
            mm = re.search(pat, line)
            if mm:
                data[key].append((vx, float(mm.group(1))))
    return data


def plot_series(points, out, title, ylabel):
    if not points:
        return
    x, y = zip(*points)
    plt.figure(figsize=(8, 4.5))
    plt.plot(x, y, marker="o", markersize=2)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out.with_suffix(".png"), dpi=300)
    plt.savefig(out.with_suffix(".pdf"))
    plt.close()


def visualize_log(args):
    if not args.log_file:
        return
    out = Path(args.out_dir) / "training_curves"
    out.mkdir(parents=True, exist_ok=True)
    data = parse_log(args.log_file)
    plot_series(data["loss"], out / "train_loss", "Training Loss", "Loss")
    plot_series(data["lr"], out / "learning_rate", "Learning Rate", "LR")
    plot_series(data["segm_mAP"], out / "val_mask_ap", "Validation Mask AP", "AP")
    plot_series(data["bbox_mAP"], out / "val_bbox_ap", "Validation BBox AP", "AP")
    plot_series(data["segm_mAP_50"], out / "val_mask_ap50", "Validation Mask AP50", "AP50")
    plot_series(data["segm_mAP_75"], out / "val_mask_ap75", "Validation Mask AP75", "AP75")
    (out / "parsed_log.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"[OK] training curves: {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--clean-root", default="mmdet_dataset/lettuce")
    p.add_argument("--benchmark-root", default="mmdet_dataset/lettuce_c")
    p.add_argument("--benchmark-manifest", default="mmdet_dataset/lettuce_c/manifest.json")
    p.add_argument("--out-dir", default="paper_outputs_dgcf/featuremaps")
    p.add_argument("--layer", choices=["backbone", "neck", "rpn_head"], default="neck")
    p.add_argument("--levels", nargs="+", type=int, default=[0, 1, 2, 3])
    p.add_argument("--image-index", type=int, default=0)
    p.add_argument("--only-corruptions", nargs="+", default=["contrast", "gaussian_noise", "defocus_blur", "motion_blur"])
    p.add_argument("--severities", nargs="+", type=int, default=[5])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--log-file", default="")
    args = p.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    visualize_features(args)
    visualize_log(args)


if __name__ == "__main__":
    main()
