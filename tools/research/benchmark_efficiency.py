# -*- coding: utf-8 -*-
"""Đo hiệu năng tính toán cho một hoặc nhiều config (Major Comment 12).

Báo cáo: Params, FLOPs (input cố định), inference latency mean±std, FPS,
peak VRAM, model size (dung lượng file .pth nếu cung cấp checkpoint).

Giao thức đo (ghi rõ trong bài, mục Experimental setup):
- GPU đơn, batch size 1, input 800x800x3 (khớp test pipeline scale (800,800)).
- 50 lần warm-up, 200 lần đo, torch.cuda.synchronize() trước/sau mỗi lần.
- FP32, model.eval(), torch.no_grad().
- FLOPs đo bằng mmengine.analysis trên cùng input size.

Cách dùng:
    python tools/research/benchmark_efficiency.py \
        configs/fair_lettuce/mask_rcnn_r50_fpn.py \
        configs/fair_lettuce/mask_rcnn_r50_dgcf_fpn_v2.py \
        --checkpoint-glob "work_dirs/research/{name}/seed_2026_3/best_*.pth" \
        --out efficiency_report.csv
"""

import argparse
import glob
import os
import time

import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmdet.registry import MODELS


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def measure_flops(model, input_shape=(3, 800, 800)):
    try:
        from mmengine.analysis import get_model_complexity_info
        res = get_model_complexity_info(
            model, input_shape=input_shape, show_table=False,
            show_arch=False)
        return res['flops'], res['flops_str']
    except Exception as e:  # noqa: BLE001
        return None, f'N/A ({e})'


def measure_latency(model, device, input_shape=(1, 3, 800, 800),
                    warmup=50, iters=200):
    x = torch.randn(*input_shape, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(warmup):
            _ = model.backbone(x) if hasattr(model, 'backbone') else model(x)
        torch.cuda.synchronize(device)
        for _ in range(iters):
            t0 = time.perf_counter()
            feats = model.backbone(x)
            if getattr(model, 'with_neck', False):
                feats = model.neck(feats)
            torch.cuda.synchronize(device)
            times.append(time.perf_counter() - t0)
    peak_mem = torch.cuda.max_memory_allocated(device) / 1024 ** 2
    times = np.array(times) * 1000.0  # ms
    return times.mean(), times.std(), 1000.0 / times.mean(), peak_mem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('configs', nargs='+')
    parser.add_argument('--checkpoint-glob', default=None,
                        help="Pattern chứa {name} để tìm .pth đo model size, "
                             "ví dụ 'work_dirs/research/{name}/seed_2026_3/best_*.pth'")
    parser.add_argument('--out', default='efficiency_report.csv')
    parser.add_argument('--input-size', type=int, default=800)
    args = parser.parse_args()

    device = 'cuda:0'
    rows = []
    for cfg_path in args.configs:
        name = os.path.splitext(os.path.basename(cfg_path))[0]
        print(f'\n=== {name} ===')
        cfg = Config.fromfile(cfg_path)
        init_default_scope(cfg.get('default_scope', 'mmdet'))
        model = MODELS.build(cfg.model).to(device)

        n_params = count_params(model)
        flops, flops_str = measure_flops(
            model, (3, args.input_size, args.input_size))
        lat_mean, lat_std, fps, peak_mem = measure_latency(
            model, device, (1, 3, args.input_size, args.input_size))

        size_mb = None
        if args.checkpoint_glob:
            hits = glob.glob(args.checkpoint_glob.format(name=name))
            if hits:
                size_mb = os.path.getsize(hits[0]) / 1024 ** 2

        row = dict(model=name,
                   params_M=n_params / 1e6,
                   flops=flops_str,
                   latency_ms=f'{lat_mean:.1f}±{lat_std:.1f}',
                   fps=f'{fps:.1f}',
                   peak_vram_MB=f'{peak_mem:.0f}',
                   model_size_MB=f'{size_mb:.0f}' if size_mb else 'N/A')
        print(row)
        rows.append(row)
        del model
        torch.cuda.empty_cache()

    import csv
    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nĐã lưu: {args.out}')


if __name__ == '__main__':
    main()
