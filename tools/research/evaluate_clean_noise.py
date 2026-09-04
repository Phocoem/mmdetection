#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluate MMDetection models on clean and noise datasets.

This script is designed for your lettuce robustness setup:
    clean root:      mmdet_dataset/lettuce
    benchmark root:  mmdet_dataset/lettuce_c
    conditions:      clean, gaussian_noise:1, gaussian_noise:2, gaussian_noise:3

For clean, it runs tools/test.py with the original config.
For noise, it creates a temporary COCO test dataset by symlinking corrupted
images with the same file names as the clean COCO annotation, then dumps a
temporary config and runs tools/test.py.

Why symlink dataset?
    It avoids changing your original annotations and works with standard COCO
    evaluation because file names remain identical.
"""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def safe_name(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', s).strip('_')


def load_models(args):
    if args.models_json:
        data = json.loads(Path(args.models_json).read_text(encoding='utf-8'))
        return data['models']
    if not args.config:
        raise ValueError('Provide --models-json or --config.')
    return [{
        'name': args.name or Path(args.config).stem,
        'config': args.config,
        'checkpoint': args.checkpoint,
        'work_dir': args.work_dir,
    }]


def find_checkpoint(model):
    if model.get('checkpoint'):
        ckpt = Path(model['checkpoint'])
        if ckpt.exists():
            return str(ckpt)
        raise FileNotFoundError(f'Checkpoint not found: {ckpt}')

    work_dir = model.get('work_dir')
    if not work_dir:
        raise ValueError(f'Model {model.get("name")} has no checkpoint or work_dir.')

    wd = Path(work_dir)
    patterns = [
        'best_coco_segm_mAP*.pth',
        'best*.pth',
        'epoch_*.pth',
        '*.pth',
    ]
    files = []
    for pat in patterns:
        files = sorted(wd.glob(pat), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
        if files:
            break
    if not files:
        raise FileNotFoundError(f'No checkpoint found in {wd}')
    return str(files[0])


def parse_condition(cond: str):
    if cond == 'clean':
        return 'clean', None
    if ':' in cond:
        c, s = cond.split(':', 1)
        return c, s
    return cond, None


def severity_match(path_lower: str, severity):
    if severity is None:
        return True
    s = str(severity).lower()
    tokens = [
        f'severity_{s}', f'severity-{s}', f'severity{s}',
        f'sev_{s}', f'sev-{s}', f'sev{s}',
        f'/s{s}/', f'_s{s}_', f'-s{s}-',
        f'/{s}/', f'_{s}/', f'/{s}_',
    ]
    return any(t in path_lower for t in tokens)


def build_image_index(root: Path, condition: str, severity=None):
    condition_l = condition.lower()
    all_imgs = []
    for p in root.rglob('*'):
        if p.suffix.lower() in IMG_EXTS:
            pl = str(p).lower()
            if condition_l in pl and severity_match(pl.replace('\\', '/'), severity):
                all_imgs.append(p)

    # Fallback: condition only, if severity filtering fails because directory naming is unknown.
    if not all_imgs and severity is not None:
        for p in root.rglob('*'):
            if p.suffix.lower() in IMG_EXTS and condition_l in str(p).lower():
                all_imgs.append(p)

    index = {}
    suffix_index = {}
    for p in all_imgs:
        index.setdefault(p.name, p)
        suffix_index[str(p).replace('\\', '/').lower()] = p
    return index, suffix_index, all_imgs


def get_dataset_cfg(cfg):
    ds = cfg.test_dataloader.dataset
    # common wrappers: dataset=dict(dataset=...)
    while isinstance(ds, dict) and 'dataset' in ds:
        ds = ds['dataset']
    return ds


def resolve_ann_file(ds, clean_root: Path):
    ann_file = Path(ds.get('ann_file', ''))
    data_root = Path(ds.get('data_root', '')) if ds.get('data_root') else clean_root

    candidates = []
    if ann_file.is_absolute():
        candidates.append(ann_file)
    else:
        candidates.append(data_root / ann_file)
        candidates.append(clean_root / ann_file)
        candidates.append(Path.cwd() / ann_file)

    for c in candidates:
        if c.exists():
            return c.resolve()

    # fallback: search common annotation names
    pats = ['*test*.json', '*val*.json', '*.json']
    for pat in pats:
        hits = sorted((clean_root / 'annotations').glob(pat)) if (clean_root/'annotations').exists() else []
        if hits:
            return hits[0].resolve()
    raise FileNotFoundError(f'Cannot resolve annotation file from ann_file={ann_file}, clean_root={clean_root}')


def get_img_prefix(ds):
    prefix = ds.get('data_prefix', {}).get('img', '')
    return prefix or ''


def symlink_or_copy(src: Path, dst: Path, mode='symlink'):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if mode == 'copy':
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def make_noise_dataset(config_path, clean_root, benchmark_root, condition, out_dir, mode='symlink'):
    from mmengine.config import Config

    cfg = Config.fromfile(config_path)
    ds = get_dataset_cfg(cfg)
    ann_path = resolve_ann_file(ds, clean_root)
    img_prefix = get_img_prefix(ds)

    coco = json.loads(ann_path.read_text(encoding='utf-8'))
    cond_name, severity = parse_condition(condition)
    img_index, suffix_index, all_imgs = build_image_index(benchmark_root, cond_name, severity)
    if not all_imgs:
        raise FileNotFoundError(f'No images found for condition={condition} under {benchmark_root}')

    temp_root = out_dir / 'temp_datasets' / safe_name(condition)
    temp_ann = temp_root / 'annotations' / ann_path.name
    temp_ann.parent.mkdir(parents=True, exist_ok=True)
    temp_ann.write_text(json.dumps(coco), encoding='utf-8')

    missing = []
    for img in coco.get('images', []):
        file_name = img.get('file_name')
        if not file_name:
            continue
        base = Path(file_name).name
        src = img_index.get(base)
        if src is None:
            # try suffix match if corrupted set preserves subfolders
            suffix = file_name.replace('\\', '/').lower()
            src = None
            for pl, pp in suffix_index.items():
                if pl.endswith(suffix):
                    src = pp
                    break
        if src is None:
            missing.append(file_name)
            continue
        dst = temp_root / img_prefix / file_name
        symlink_or_copy(src, dst, mode=mode)

    if missing:
        miss_path = temp_root / 'missing_images.txt'
        miss_path.write_text('\n'.join(missing[:2000]), encoding='utf-8')
        print(f'[WARN] Missing {len(missing)} images for {condition}. See {miss_path}')

    # Modify config for temp dataset
    def set_dataset(ds_obj):
        while isinstance(ds_obj, dict) and 'dataset' in ds_obj:
            ds_obj = ds_obj['dataset']
        ds_obj['data_root'] = str(temp_root) + '/'
        ds_obj['ann_file'] = f'annotations/{ann_path.name}'
        ds_obj.setdefault('data_prefix', {})
        ds_obj['data_prefix']['img'] = img_prefix

    set_dataset(cfg.test_dataloader.dataset)
    if hasattr(cfg, 'test_evaluator'):
        if isinstance(cfg.test_evaluator, dict):
            cfg.test_evaluator['ann_file'] = str(temp_ann)
        elif isinstance(cfg.test_evaluator, (list, tuple)):
            for ev in cfg.test_evaluator:
                if isinstance(ev, dict) and 'ann_file' in ev:
                    ev['ann_file'] = str(temp_ann)

    cfg.work_dir = str(out_dir / 'work_dirs' / safe_name(condition))
    temp_cfg_dir = out_dir / 'temp_configs'
    temp_cfg_dir.mkdir(parents=True, exist_ok=True)
    temp_cfg = temp_cfg_dir / f'{Path(config_path).stem}_{safe_name(condition)}.py'
    cfg.dump(str(temp_cfg))
    return temp_cfg


def run_test(py, config, checkpoint, work_dir, log_path):
    cmd = [py, 'tools/test.py', str(config), str(checkpoint), '--work-dir', str(work_dir)]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print('[RUN]', ' '.join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log_path.write_text(proc.stdout, encoding='utf-8')
    if proc.returncode != 0:
        print(proc.stdout[-4000:])
        raise RuntimeError(f'Command failed. See log: {log_path}')
    return proc.stdout


def parse_metrics(text):
    m = {}
    patterns = {
        'segm_mAP': [r"coco/segm_mAP['\":\s]+([0-9.]+)", r"segm_mAP\s*[:=]\s*([0-9.]+)"],
        'bbox_mAP': [r"coco/bbox_mAP['\":\s]+([0-9.]+)", r"bbox_mAP\s*[:=]\s*([0-9.]+)"],
        'segm_mAP_50': [r"coco/segm_mAP_50['\":\s]+([0-9.]+)", r"segm_mAP_50\s*[:=]\s*([0-9.]+)"],
        'segm_mAP_75': [r"coco/segm_mAP_75['\":\s]+([0-9.]+)", r"segm_mAP_75\s*[:=]\s*([0-9.]+)"],
    }
    for k, pats in patterns.items():
        for pat in pats:
            hit = re.findall(pat, text)
            if hit:
                m[k] = hit[-1]
                break
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models-json', default=None)
    ap.add_argument('--config', default=None)
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--work-dir', default=None)
    ap.add_argument('--name', default=None)
    ap.add_argument('--clean-root', default='mmdet_dataset/lettuce')
    ap.add_argument('--benchmark-root', default='mmdet_dataset/lettuce_c')
    ap.add_argument('--conditions', nargs='+', default=['clean', 'gaussian_noise:1', 'gaussian_noise:2', 'gaussian_noise:3'])
    ap.add_argument('--out-dir', default='paper_outputs_clean_noise_eval')
    ap.add_argument('--copy-mode', choices=['symlink', 'copy'], default='symlink')
    ap.add_argument('--python', default=sys.executable)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_root = Path(args.clean_root)
    benchmark_root = Path(args.benchmark_root)
    models = load_models(args)

    rows = []
    for model in models:
        name = model.get('name') or Path(model['config']).stem
        config = model['config']
        ckpt = find_checkpoint(model)
        print(f'\n=== Model: {name} ===')
        print(f'Config: {config}')
        print(f'Checkpoint: {ckpt}')
        for cond in args.conditions:
            cond_safe = safe_name(cond)
            model_safe = safe_name(name)
            log_path = out_dir / 'logs' / model_safe / f'{cond_safe}.log'
            work_dir = out_dir / 'work_dirs' / model_safe / cond_safe
            try:
                if cond == 'clean':
                    test_cfg = config
                else:
                    test_cfg = make_noise_dataset(config, clean_root, benchmark_root, cond, out_dir, mode=args.copy_mode)
                text = run_test(args.python, test_cfg, ckpt, work_dir, log_path)
                metrics = parse_metrics(text)
                row = dict(model=name, condition=cond, status='ok', config=config, checkpoint=ckpt, log=str(log_path))
                row.update(metrics)
                rows.append(row)
            except Exception as e:
                row = dict(model=name, condition=cond, status='failed', error=str(e), config=config, checkpoint=ckpt, log=str(log_path))
                rows.append(row)
                print('[FAILED]', name, cond, e)

    csv_path = out_dir / 'clean_noise_metrics.csv'
    keys = sorted({k for r in rows for k in r.keys()})
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    print(f'\nSaved: {csv_path}')


if __name__ == '__main__':
    main()
