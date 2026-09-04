#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate clean + robustness benchmark from manifest."""
import argparse, json, subprocess
from pathlib import Path

def load(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def placeholder(s): return ('YOUR_' in str(s)) or (not str(s).strip())
def find_ckpt(wd):
    wd=Path(wd)
    if not wd.exists(): return None
    for pat in ['best_coco_segm_mAP*.pth','best_*.pth','epoch_*.pth','*.pth']:
        c=list(wd.glob(pat))
        if c: return str(sorted(c,key=lambda p:p.stat().st_mtime,reverse=True)[0])
    return None

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True); p.add_argument('--clean-root',required=True); p.add_argument('--benchmark-root',required=True)
    p.add_argument('--seed',type=int,default=2026); p.add_argument('--only',nargs='*',default=None)
    p.add_argument('--dry-run',action='store_true'); p.add_argument('--evaluator',default='tools/research/evaluate_benchmark.py')
    args=p.parse_args(); man=load(args.manifest)
    for m in man['models']:
        if args.only and m['key'] not in args.only: continue
        cfg=m['config'].format(seed=args.seed); wd=m['work_dir'].format(seed=args.seed)
        ckpt=m.get('checkpoint','').format(seed=args.seed) if m.get('checkpoint') else ''
        if placeholder(cfg) or placeholder(wd): print('[SKIP] placeholder',m['key']); continue
        if not Path(cfg).is_file(): print('[SKIP] config missing',cfg); continue
        if not ckpt: ckpt=find_ckpt(wd)
        if not ckpt or not Path(ckpt).is_file(): print('[SKIP] checkpoint missing',m['key'],wd); continue
        out=str(Path(wd)/'evaluation')
        cmd=['python',args.evaluator,cfg,ckpt,'--clean-root',args.clean_root,'--benchmark-root',args.benchmark_root,'--output-dir',out,'--seed',str(args.seed)]
        print('\n'+'='*100); print('[EVAL]',m['name']); print(' '.join(cmd))
        if not args.dry_run: subprocess.run(cmd,check=True)
if __name__=='__main__': main()
