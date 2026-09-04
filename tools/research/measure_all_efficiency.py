# -*- coding: utf-8 -*-
"""Đọc toàn bộ thông số training + efficiency của mọi model.

Xuất bảng đầy đủ cho bài Q1:
  - Params (M), FLOPs (G) — đo bằng mmengine.analysis (thay thế tool cũ lỗi)
  - Latency (ms), FPS — đo trực tiếp bằng inference thật
  - VRAM peak (MB)
  - Model size (MB) từ file .pth
  - Train time mỗi seed (h), tổng, mean±std qua các seed
  - Best epoch, stop epoch (early stopping), avg iter time, peak train VRAM
  - Train time PER EPOCH (h) để công bằng khi so sánh khác epoch dừng

Cách dùng:
    python tools/research/measure_all_efficiency.py \
        --root work_dirs/research \
        --seeds 2024 2025 2026 \
        --configs configs/fair_lettuce/mask_rcnn_r50_fpn.py \
                  configs/fair_lettuce/mask_rcnn_r101_fpn.py \
                  configs/fair_lettuce/mask_rcnn_r50_cbam_fpn.py \
                  configs/fair_lettuce/mask_rcnn_r50_bifpn.py \
                  configs/fair_lettuce/mask_rcnn_r50_dgcf_fpn_v2.py \
        --input-size 800 \
        --out work_dirs/research/EFFICIENCY_FINAL.csv

Ghi chú:
- fpn_aug / fpn_bca dùng chung kiến trúc FPN => Params/FLOPs/latency GIỐNG HỆT
  mask_rcnn_r50_fpn. Bạn không cần đo lại; script sẽ tự copy nếu thấy trong
  --alias (mặc định gợi ý fpn -> fpn_aug, fpn_bca).
- Nếu FLOPs vẫn ra 'N/A', xem cột `flops_error` trong output để biết lý do.
"""

import argparse
import csv
import glob
import json
import os
import re
import time
from statistics import mean, stdev

# ------------------------------ TRAIN LOG ------------------------------------
ITERS_PER_EPOCH_DEFAULT = 239  # ước lượng: 477 ảnh train / batch 2 ≈ 239


def find_train_log(work_dir):
    """Tìm file train log. MMEngine 3.x đặt trong <work_dir>/<timestamp>/*.log."""
    cands = glob.glob(os.path.join(work_dir, '*.log'))
    cands += glob.glob(os.path.join(work_dir, '*', '*.log'))
    cands += glob.glob(os.path.join(work_dir, 'train_log.txt'))
    # Loại eval_log.txt
    cands = [c for c in cands if 'eval' not in os.path.basename(c).lower()]
    return cands[0] if cands else None


def parse_train_log(log_path):
    """Trích các số liệu training từ 1 file log."""
    if not log_path or not os.path.isfile(log_path):
        return {}
    with open(log_path, 'r', errors='ignore') as f:
        text = f.read()

    m = {}
    # Seed và deterministic
    s = re.search(r'seed:\s*(\d+)', text)
    if s:
        m['seed_confirmed'] = int(s.group(1))
    d = re.search(r'deterministic:\s*(\w+)', text)
    if d:
        m['deterministic'] = d.group(1)

    # Best epoch từ tên checkpoint
    be = re.findall(r'best_coco_segm_mAP_epoch_(\d+)', text)
    if be:
        m['best_epoch'] = int(be[-1])
    # Best val mAP
    bv = re.findall(r'The best checkpoint with ([\d.]+)', text)
    if bv:
        m['best_val_mAP'] = float(bv[-1])
    # Epoch dừng (early stop) — dòng cuối "Epoch(train) [N]"
    stops = re.findall(r'Epoch\(train\)\s*\[(\d+)\]', text)
    if stops:
        m['stop_epoch'] = int(stops[-1])

    # Iter time trung bình
    iters = re.findall(r'\btime:\s*([\d.]+)', text)
    if iters:
        vals = [float(x) for x in iters if 0.01 < float(x) < 5.0]
        if vals:
            m['avg_iter_time_s'] = sum(vals) / len(vals)

    # Peak VRAM lúc train
    mems = re.findall(r'memory:\s*(\d+)', text)
    if mems:
        m['peak_train_vram_MB'] = max(int(x) for x in mems)

    # Ước lượng tổng thời gian train (giờ)
    if 'avg_iter_time_s' in m and 'stop_epoch' in m:
        m['train_time_h'] = (m['avg_iter_time_s'] * ITERS_PER_EPOCH_DEFAULT *
                             m['stop_epoch']) / 3600.0
        m['time_per_epoch_min'] = (m['avg_iter_time_s'] *
                                   ITERS_PER_EPOCH_DEFAULT) / 60.0

    return m


def gather_train_stats(root, model_name, seeds):
    """Tổng hợp thống kê training của 1 model qua các seed."""
    per_seed = []
    for seed in seeds:
        wd = os.path.join(root, model_name, f'seed_{seed}')
        stats = parse_train_log(find_train_log(wd))
        # Model size từ checkpoint
        ckpts = glob.glob(os.path.join(wd, 'best_coco_segm_mAP_epoch_*.pth'))
        if ckpts:
            stats['model_size_MB'] = os.path.getsize(ckpts[0]) / 1024 ** 2
        stats['seed'] = seed
        per_seed.append(stats)

    # Aggregate mean±std
    agg = {'per_seed': per_seed}
    for k in ('best_epoch', 'stop_epoch', 'avg_iter_time_s',
             'peak_train_vram_MB', 'train_time_h', 'time_per_epoch_min',
             'best_val_mAP', 'model_size_MB'):
        vals = [s[k] for s in per_seed if k in s]
        if vals:
            agg[f'{k}_mean'] = mean(vals)
            agg[f'{k}_std'] = stdev(vals) if len(vals) > 1 else 0.0
    return agg


# ------------------------------ FLOPS + LATENCY ------------------------------
def measure_flops_params(cfg_path, input_size=800):
    """Đo Params + FLOPs của backbone+neck (đủ để so sánh vì head giống nhau).

    Ưu tiên fvcore.FlopCountAnalysis (hoạt động ổn định với mọi kiến trúc);
    fallback qua mmengine.analysis; fallback cuối chỉ đếm Params.
    """
    try:
        import torch
        from mmengine.config import Config
        from mmengine.registry import init_default_scope
        from mmdet.registry import MODELS
    except Exception as e:  # noqa: BLE001
        return {'params_M': None, 'flops_G': None,
                'flops_error': f'import: {e}'}

    try:
        cfg = Config.fromfile(cfg_path)
        init_default_scope(cfg.get('default_scope', 'mmdet'))
        model = MODELS.build(cfg.model)
        model.eval()

        # Params trên TOÀN model (backbone + neck + head + rpn) — đây là con số
        # bạn muốn báo trong bài.
        params_M = sum(p.numel() for p in model.parameters()) / 1e6

        # FLOPs: đo trên (backbone + neck) vì đây là phần khác nhau giữa các
        # baseline. Head (RPN/RoI/mask) giống nhau nên loại khỏi so sánh giúp
        # con số công bằng và tránh lỗi forward do Mask R-CNN cần dict input.
        class BackboneNeck(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.backbone = m.backbone
                self.with_neck = getattr(m, 'with_neck', False)
                self.neck = m.neck if self.with_neck else None

            def forward(self, x):
                feats = self.backbone(x)
                if self.with_neck:
                    feats = self.neck(feats)
                return feats

        sub = BackboneNeck(model).eval()
        x = torch.randn(1, 3, input_size, input_size)

        flops_G, err = None, None
        # 1) fvcore
        try:
            from fvcore.nn import FlopCountAnalysis
            fa = FlopCountAnalysis(sub, x)
            fa.unsupported_ops_warnings(False)
            fa.uncalled_modules_warnings(False)
            flops_G = float(fa.total()) / 1e9
        except ImportError:
            err = 'fvcore not installed (pip install fvcore)'
        except Exception as e:  # noqa: BLE001
            err = f'fvcore: {type(e).__name__}: {e}'

        # 2) fallback mmengine.analysis nếu fvcore fail
        if flops_G is None:
            try:
                from mmengine.analysis import get_model_complexity_info
                info = get_model_complexity_info(
                    sub, input_shape=(3, input_size, input_size),
                    show_table=False, show_arch=False)
                f = info.get('flops', None)
                if isinstance(f, (int, float)) and f > 0:
                    flops_G = float(f) / 1e9
                elif isinstance(f, str):
                    nums = re.findall(r'[\d.]+', f)
                    if nums:
                        mult = 1e9 if 'G' in f else (1e6 if 'M' in f else 1)
                        flops_G = float(nums[0]) * mult / 1e9
            except Exception as e:  # noqa: BLE001
                if err is None:
                    err = f'mmengine: {type(e).__name__}: {e}'

        note = '(backbone+neck; head chung nên loại khỏi so sánh)'
        return {'params_M': params_M,
                'flops_G': flops_G,
                'flops_error': err,
                'flops_note': note if flops_G is not None else None}
    except Exception as e:  # noqa: BLE001
        return {'params_M': None, 'flops_G': None,
                'flops_error': f'build: {type(e).__name__}: {e}'}


def measure_latency(cfg_path, input_size=800, warmup=30, iters=100):
    """Đo latency + FPS + peak inference VRAM."""
    try:
        import torch
        import numpy as np
        from mmengine.config import Config
        from mmengine.registry import init_default_scope
        from mmdet.registry import MODELS
    except Exception as e:  # noqa: BLE001
        return {'latency_ms': None, 'fps': None, 'peak_infer_vram_MB': None,
                'latency_error': str(e)}
    if not torch.cuda.is_available():
        return {'latency_ms': None, 'fps': None, 'peak_infer_vram_MB': None,
                'latency_error': 'no CUDA'}
    try:
        cfg = Config.fromfile(cfg_path)
        init_default_scope(cfg.get('default_scope', 'mmdet'))
        device = 'cuda:0'
        model = MODELS.build(cfg.model).to(device).eval()
        x = torch.randn(1, 3, input_size, input_size, device=device)
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            for _ in range(warmup):
                feats = model.backbone(x)
                if getattr(model, 'with_neck', False):
                    _ = model.neck(feats)
            torch.cuda.synchronize()
            ts = []
            for _ in range(iters):
                t0 = time.perf_counter()
                feats = model.backbone(x)
                if getattr(model, 'with_neck', False):
                    _ = model.neck(feats)
                torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) * 1000.0)
        peak = torch.cuda.max_memory_allocated(device) / 1024 ** 2
        del model, x
        torch.cuda.empty_cache()
        return {'latency_ms': float(np.mean(ts)),
                'latency_std_ms': float(np.std(ts)),
                'fps': 1000.0 / float(np.mean(ts)),
                'peak_infer_vram_MB': float(peak),
                'latency_error': None}
    except Exception as e:  # noqa: BLE001
        return {'latency_ms': None, 'fps': None, 'peak_infer_vram_MB': None,
                'latency_error': f'{type(e).__name__}: {e}'}


# ------------------------------ MAIN -----------------------------------------
def main():
    global ITERS_PER_EPOCH_DEFAULT
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='work_dirs/research')
    p.add_argument('--seeds', nargs='+', default=['2024', '2025', '2026'])
    p.add_argument('--configs', nargs='+', required=True,
                   help='Đường dẫn các file .py config cần đo')
    p.add_argument('--input-size', type=int, default=800)
    p.add_argument('--iters-per-epoch', type=int,
                   default=ITERS_PER_EPOCH_DEFAULT,
                   help='Số iter/epoch (mặc định 239 = 477 ảnh / batch 2)')
    p.add_argument('--out', default='EFFICIENCY_FINAL.csv')
    p.add_argument('--skip-latency', action='store_true',
                   help='Bỏ qua đo latency (nếu chỉ cần train stats + FLOPs)')
    p.add_argument('--alias', nargs='*', default=[
        'mask_rcnn_r50_fpn:mask_rcnn_r50_fpn_aug',
        'mask_rcnn_r50_fpn:mask_rcnn_r50_fpn_bca'],
        help='Model dùng chung kiến trúc: SRC:DEST — DEST sẽ copy FLOPs/latency của SRC')
    args = p.parse_args()

    ITERS_PER_EPOCH_DEFAULT = args.iters_per_epoch

    # Đo FLOPs + latency cho các config được liệt kê
    arch_info = {}
    for cfg in args.configs:
        name = os.path.splitext(os.path.basename(cfg))[0]
        print(f'\n=== {name} ===')
        print('  đo FLOPs/Params...')
        fp = measure_flops_params(cfg, args.input_size)
        print(f"    params_M={fp['params_M']}, flops_G={fp['flops_G']}"
              f"{', LỖI: '+fp['flops_error'] if fp.get('flops_error') else ''}")
        if args.skip_latency:
            lat = {}
        else:
            print('  đo latency...')
            lat = measure_latency(cfg, args.input_size)
            print(f"    latency_ms={lat.get('latency_ms')}, "
                  f"fps={lat.get('fps')}")
        arch_info[name] = {**fp, **lat}

    # Áp dụng alias
    for a in args.alias:
        if ':' in a:
            src, dst = a.split(':', 1)
            if src in arch_info:
                arch_info.setdefault(dst, {}).update({
                    'params_M': arch_info[src].get('params_M'),
                    'flops_G': arch_info[src].get('flops_G'),
                    'latency_ms': arch_info[src].get('latency_ms'),
                    'fps': arch_info[src].get('fps'),
                    'peak_infer_vram_MB': arch_info[src].get('peak_infer_vram_MB'),
                    'note': f'shared architecture with {src}',
                })

    # Đọc train stats cho MỌI model có work_dir
    all_models = set(arch_info.keys())
    if os.path.isdir(args.root):
        for d in os.listdir(args.root):
            if os.path.isdir(os.path.join(args.root, d)):
                all_models.add(d)

    rows = []
    for model in sorted(all_models):
        ts = gather_train_stats(args.root, model, args.seeds)
        row = {'model': model}
        row.update({k: v for k, v in arch_info.get(model, {}).items()
                    if not isinstance(v, dict)})
        # Chỉ lấy các trường aggregate mean, không dump per_seed vào CSV chính
        for k, v in ts.items():
            if k == 'per_seed':
                continue
            row[k] = v
        rows.append(row)

    # ---- Xuất CSV ----
    fields = [
        'model',
        # Kiến trúc
        'params_M', 'flops_G',
        'latency_ms', 'latency_std_ms', 'fps', 'peak_infer_vram_MB',
        # Train
        'best_epoch_mean', 'best_epoch_std',
        'stop_epoch_mean', 'stop_epoch_std',
        'avg_iter_time_s_mean',
        'time_per_epoch_min_mean',
        'train_time_h_mean', 'train_time_h_std',
        'peak_train_vram_MB_mean',
        'best_val_mAP_mean', 'best_val_mAP_std',
        'model_size_MB_mean',
        # Debug
        'flops_error', 'latency_error', 'note',
    ]
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'\nĐã lưu bảng đầy đủ: {args.out}')

    # ---- In bảng gọn ra terminal ----
    print('\n' + '=' * 110)
    print(f"{'Model':<35}{'Params(M)':>10}{'FLOPs(G)':>10}{'Lat(ms)':>10}"
          f"{'FPS':>8}{'Train(h)':>10}{'Stop_ep':>9}{'Best_ep':>9}")
    print('-' * 110)
    for r in rows:
        def fmt(v, spec='.1f'):
            return f'{v:{spec}}' if isinstance(v, (int, float)) else '-'
        print(f"{r['model']:<35}"
              f"{fmt(r.get('params_M')):>10}"
              f"{fmt(r.get('flops_G')):>10}"
              f"{fmt(r.get('latency_ms')):>10}"
              f"{fmt(r.get('fps')):>8}"
              f"{fmt(r.get('train_time_h_mean'), '.2f'):>10}"
              f"{fmt(r.get('stop_epoch_mean'), '.0f'):>9}"
              f"{fmt(r.get('best_epoch_mean'), '.0f'):>9}")
    print('=' * 110)


if __name__ == '__main__':
    main()
