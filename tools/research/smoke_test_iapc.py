# -*- coding: utf-8 -*-
"""Smoke test cho IAPCMaskRCNN — chạy TRƯỚC khi train thật.

Kiểm tra 6 điều:
    1. Config load được, model build được
    2. Forward loss() chạy không lỗi
    3. loss_consistency có trong output và là số hữu hạn
    4. Warmup hoạt động (iter <= warmup -> loss_consistency == 0)
    5. Instance mask downsample đúng shape từng level
    6. backward() chạy được, gradient không NaN

Cách dùng:
    python tools/research/smoke_test_iapc.py \
        configs/fair_lettuce/mask_rcnn_r50_iapc.py
"""

import argparse
import sys

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.structures import InstanceData

from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from mmdet.structures.mask import BitmapMasks


def build_fake_batch(batch_size=2, h=320, w=320, n_inst=3, device='cuda'):
    """Sinh batch giả có đủ gt_instances (bboxes, labels, masks)."""
    imgs = torch.randn(batch_size, 3, h, w, device=device)
    samples = []
    for _ in range(batch_size):
        ds = DetDataSample()
        ds.set_metainfo(dict(
            img_shape=(h, w), ori_shape=(h, w),
            pad_shape=(h, w), scale_factor=(1.0, 1.0),
            batch_input_shape=(h, w)))
        gt = InstanceData()
        # bbox hợp lệ, không chồng biên
        boxes = []
        masks = []
        for i in range(n_inst):
            x1 = 20 + i * 60
            y1 = 30 + i * 40
            x2, y2 = x1 + 50, y1 + 45
            boxes.append([x1, y1, x2, y2])
            m = torch.zeros(h, w, dtype=torch.uint8)
            m[y1:y2, x1:x2] = 1
            masks.append(m.numpy())
        gt.bboxes = torch.tensor(boxes, dtype=torch.float32, device=device)
        gt.labels = torch.zeros(n_inst, dtype=torch.long, device=device)
        gt.masks = BitmapMasks(masks, height=h, width=w)
        ds.gt_instances = gt
        # ignored instances (một số head yêu cầu)
        ig = InstanceData()
        ig.bboxes = torch.zeros((0, 4), dtype=torch.float32, device=device)
        ig.labels = torch.zeros((0,), dtype=torch.long, device=device)
        ds.ignored_instances = ig
        samples.append(ds)
    return imgs, samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--h', type=int, default=320)
    ap.add_argument('--w', type=int, default=320)
    args = ap.parse_args()

    dev = args.device
    if dev.startswith('cuda') and not torch.cuda.is_available():
        print('! CUDA không khả dụng, chuyển sang CPU (sẽ chậm)')
        dev = 'cpu'

    print('=' * 70)
    print('SMOKE TEST — IAPCMaskRCNN')
    print('=' * 70)

    # --- 1. Build model ---
    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmdet'))
    model = MODELS.build(cfg.model).to(dev)
    model.train()
    print(f'[1/6] Build model OK: {type(model).__name__}')
    assert type(model).__name__ == 'IAPCMaskRCNN', \
        'Config không trỏ tới IAPCMaskRCNN'
    print(f'      consistency_weight = {model.consistency_weight}')
    print(f'      consistency_levels = {model.consistency_levels}')
    print(f'      instance_aware     = {model.instance_aware}')
    print(f'      detach_clean       = {model.detach_clean}')
    print(f'      warmup_iters       = {model.warmup_iters}')

    imgs, samples = build_fake_batch(2, args.h, args.w, device=dev)

    # --- 2 & 4. Warmup: iteration đầu phải có loss_consistency == 0 ---
    model._iter.zero_()
    losses = model.loss(imgs, samples)
    print(f'\n[2/6] forward loss() OK, các key: {sorted(losses.keys())}')
    assert 'loss_consistency' in losses, 'Thiếu loss_consistency'
    lc0 = float(losses['loss_consistency'])
    print(f'[3/6] loss_consistency (trong warmup) = {lc0:.6f}')
    if model.warmup_iters > 0:
        assert lc0 == 0.0, 'Trong warmup loss_consistency phải bằng 0'
        print('      -> warmup hoạt động đúng')

    # --- 4b. Sau warmup phải > 0 ---
    model._iter.fill_(model.warmup_iters + 1)
    losses = model.loss(imgs, samples)
    lc1 = float(losses['loss_consistency'])
    print(f'[4/6] loss_consistency (sau warmup)   = {lc1:.6f}')
    assert torch.isfinite(losses['loss_consistency']), \
        'loss_consistency không hữu hạn'
    if model.consistency_weight > 0:
        assert lc1 > 0.0, 'Sau warmup loss_consistency phải > 0'
        print('      -> consistency đang được áp dụng')
    else:
        print('      -> consistency_weight=0 (đây là config đối chứng)')

    # --- 5. Kiểm tra shape mask từng level ---
    with torch.no_grad():
        feats = model.extract_feat(imgs)
    sel = [feats[i] for i in model.consistency_levels]
    masks = model._foreground_masks(samples, sel, (args.h, args.w))
    print('\n[5/6] Mask foreground theo level:')
    for lvl, f, m in zip(model.consistency_levels, sel, masks):
        assert m.shape[-2:] == f.shape[-2:], \
            f'Shape mask không khớp feature ở level {lvl}'
        cov = float(m.mean())
        print(f'      P{lvl+2}: feat={tuple(f.shape)}  '
              f'mask={tuple(m.shape)}  phủ={100*cov:5.1f}%')
        assert 0.0 <= cov <= 1.0
    if model.instance_aware:
        cov_all = float(masks[0].mean())
        assert cov_all < 0.999, \
            'instance_aware=True nhưng mask phủ toàn ảnh — kiểm tra gt_masks'
        print('      -> instance-aware hoạt động (mask KHÔNG phủ toàn ảnh)')

    # --- 6. backward ---
    total = sum(v for v in losses.values()
                if isinstance(v, torch.Tensor) and v.numel() == 1)
    total.backward()
    n_none, n_nan, n_ok = 0, 0, 0
    for p in model.parameters():
        if p.grad is None:
            n_none += 1
        elif not torch.isfinite(p.grad).all():
            n_nan += 1
        else:
            n_ok += 1
    print(f'\n[6/6] backward OK — grad: {n_ok} hợp lệ, '
          f'{n_none} None, {n_nan} NaN/Inf')
    assert n_nan == 0, 'Có gradient NaN/Inf'

    # --- Ước lượng chi phí ---
    if dev.startswith('cuda'):
        import time
        torch.cuda.synchronize()
        model.consistency_weight_backup = model.consistency_weight

        def timeit(n=10):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(n):
                model.zero_grad(set_to_none=True)
                ls = model.loss(imgs, samples)
                s = sum(v for v in ls.values()
                        if isinstance(v, torch.Tensor) and v.numel() == 1)
                s.backward()
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / n * 1000

        model._iter.fill_(model.warmup_iters + 1)
        t_on = timeit()
        w = model.consistency_weight
        model.consistency_weight = 0.0
        t_off = timeit()
        model.consistency_weight = w
        print(f'\n[chi phí] iteration có consistency : {t_on:.1f} ms')
        print(f'          iteration không           : {t_off:.1f} ms')
        print(f'          phụ trội                  : '
              f'{100*(t_on-t_off)/max(t_off,1e-6):.0f}%')

    print('\n' + '=' * 70)
    print('TẤT CẢ PASS — có thể bắt đầu train')
    print('=' * 70)


if __name__ == '__main__':
    sys.exit(main())
