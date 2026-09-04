#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check fairness among MMDetection configs."""
import argparse, json, csv
from pathlib import Path
from mmengine.config import Config

KEYS = [
    'train_dataloader','val_dataloader','test_dataloader',
    'val_evaluator','test_evaluator','train_cfg','val_cfg','test_cfg',
    'optim_wrapper','param_scheduler','default_hooks','randomness',
    'auto_scale_lr','env_cfg'
]

def freeze(x):
    return json.dumps(x, sort_keys=True, default=str)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('configs', nargs='+')
    p.add_argument('--out', default='')
    args = p.parse_args()
    cfgs=[]
    for path in args.configs:
        if not Path(path).is_file():
            print(f'[SKIP] not found: {path}')
            continue
        cfgs.append((path, Config.fromfile(path)))
    if not cfgs:
        raise SystemExit('No valid configs')
    ref_path, ref_cfg = cfgs[0]
    rows=[]
    print(f'[Reference] {ref_path}')
    for path,cfg in cfgs:
        print('='*100)
        print(path)
        neck = cfg.model.get('neck', {})
        neck_type = neck.get('type', 'Inherited/default') if isinstance(neck, dict) else str(neck)
        row={'config':path,'work_dir':cfg.get('work_dir',''),'neck':neck_type}
        print('work_dir:',row['work_dir'])
        print('neck:',neck_type)
        for k in KEYS:
            row[k]='OK' if freeze(cfg.get(k,None)) == freeze(ref_cfg.get(k,None)) else 'DIFF'
            print(f'{k:22s}: {row[k]}')
        rows.append(row)
    if args.out and rows:
        out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
        with out.open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f'[OK] saved {out}')
    print('\nNOTE: DIFF in architecture is expected. DIFF in dataloader/optimizer/scheduler/seed is problematic.')
if __name__=='__main__': main()
