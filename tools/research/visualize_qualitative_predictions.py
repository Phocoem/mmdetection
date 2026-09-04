#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualize qualitative predictions from manifest."""
import argparse,json,re
from pathlib import Path
import cv2, numpy as np, torch, matplotlib.pyplot as plt

def patch():
    if getattr(torch.load,'_patched_weights_only_false',False): return
    old=torch.load
    def f(*a,**k): k.setdefault('weights_only',False); return old(*a,**k)
    f._patched_weights_only_false=True; torch.load=f

def loadj(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def find_ckpt(wd):
    wd=Path(wd)
    for pat in ['best_coco_segm_mAP*.pth','best_*.pth','epoch_*.pth','*.pth']:
        c=list(wd.glob(pat)) if wd.exists() else []
        if c: return str(sorted(c,key=lambda p:p.stat().st_mtime,reverse=True)[0])
    return None

def img_from_coco(root,ann,prefix,idx):
    coco=loadj(Path(root)/ann); imgs=sorted(coco['images'],key=lambda x:x.get('id',0)); idx=max(0,min(idx,len(imgs)-1)); return Path(root)/prefix/imgs[idx]['file_name']
def cond_img(clean,bench,man,cond,idx):
    if cond=='clean': return img_from_coco(clean,'annotations/test.json','images/test',idx)
    c,s=cond.split(':'); s=int(s); m=loadj(man); ann=m.get('output_annotation','annotations/test_png.json')
    for it in m.get('conditions',[]):
        if it['corruption']==c and int(it['severity'])==s: return img_from_coco(bench,ann,it['image_prefix'].rstrip('/'),idx)
    raise ValueError(cond)

def load_model(cfg,ckpt,dev):
    patch(); from mmdet.apis import init_detector; from mmdet.utils import register_all_modules
    register_all_modules(init_default_scope=True); return init_detector(cfg,ckpt,device=dev)

def overlay(img,result,thr):
    out=img.copy(); inst=getattr(result,'pred_instances',None)
    if inst is None or len(inst)==0: return out
    masks=getattr(inst,'masks',None); boxes=getattr(inst,'bboxes',None); scores=getattr(inst,'scores',None)
    keep=scores.detach().cpu().numpy()>=thr if scores is not None else np.ones(len(inst),bool)
    rng=np.random.default_rng(42); colors=rng.integers(0,255,size=(max(1,len(inst)),3),dtype=np.uint8)
    if masks is not None:
        ms=masks.detach().cpu().numpy().astype(bool)
        for i,m in enumerate(ms):
            if not keep[i]: continue
            col=np.zeros_like(out); col[:]=colors[i].tolist(); out[m]=(0.55*out[m]+0.45*col[m]).astype(np.uint8)
    if boxes is not None:
        for i,b in enumerate(boxes.detach().cpu().numpy()):
            if keep[i]: cv2.rectangle(out,tuple(b[:2].astype(int)),tuple(b[2:].astype(int)),colors[i].tolist(),2)
    return out

def main():
    p=argparse.ArgumentParser(); p.add_argument('--manifest',required=True); p.add_argument('--clean-root',required=True); p.add_argument('--benchmark-root',required=True); p.add_argument('--benchmark-manifest',required=True); p.add_argument('--out-dir',required=True); p.add_argument('--seed',type=int,default=2026); p.add_argument('--image-indices',nargs='+',type=int,default=[0,5,10]); p.add_argument('--conditions',nargs='+',default=['clean','contrast:5','gaussian_noise:5','defocus_blur:5','motion_blur:5']); p.add_argument('--only',nargs='*',default=None); p.add_argument('--device',default='cuda:0'); p.add_argument('--score-thr',type=float,default=.3); args=p.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True); man=loadj(args.manifest); models=[]
    for m in man['models']:
        if args.only and m['key'] not in args.only: continue
        if 'YOUR_' in m['config'] or 'YOUR_' in m['work_dir']: continue
        cfg=m['config'].format(seed=args.seed); ck=m.get('checkpoint','').format(seed=args.seed) if m.get('checkpoint') else ''
        if not ck: ck=find_ckpt(m['work_dir'].format(seed=args.seed))
        if ck and Path(ck).is_file() and Path(cfg).is_file(): models.append((m,load_model(cfg,ck,args.device)))
        else: print('[SKIP]',m['key'])
    from mmdet.apis import inference_detector
    for idx in args.image_indices:
        for cond in args.conditions:
            ip=cond_img(args.clean_root,args.benchmark_root,args.benchmark_manifest,cond,idx); img=cv2.imread(str(ip)); panels=[cv2.cvtColor(img,cv2.COLOR_BGR2RGB)]; titles=[cond+'\ninput']
            for m,model in models:
                with torch.no_grad(): res=inference_detector(model,str(ip))
                panels.append(cv2.cvtColor(overlay(img,res,args.score_thr),cv2.COLOR_BGR2RGB)); titles.append(m['name'])
            plt.figure(figsize=(max(8,4*len(panels)),4))
            for i,(pa,t) in enumerate(zip(panels,titles),1): plt.subplot(1,len(panels),i); plt.imshow(pa); plt.title(t,fontsize=9); plt.axis('off')
            plt.tight_layout(); op=out/f"qual_idx{idx}_{cond.replace(':','_')}.png"; plt.savefig(op,dpi=300); plt.close(); print('[OK]',op)
if __name__=='__main__': main()
