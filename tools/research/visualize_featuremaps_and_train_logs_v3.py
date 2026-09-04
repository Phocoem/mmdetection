#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Feature maps + train log plots, PyTorch 2.6 compatible."""
import argparse,json,re
from pathlib import Path
import cv2,numpy as np,torch,matplotlib.pyplot as plt

def patch():
    if getattr(torch.load,'_patched_mmdet',False): return
    old=torch.load
    def f(*a,**k): k.setdefault('weights_only',False); return old(*a,**k)
    f._patched_mmdet=True; torch.load=f

def loadj(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def mkdir(p): Path(p).mkdir(parents=True,exist_ok=True)
def img(root,ann,prefix,idx):
    coco=loadj(Path(root)/ann); ims=sorted(coco['images'],key=lambda x:x.get('id',0)); idx=max(0,min(idx,len(ims)-1)); return Path(root)/prefix/ims[idx]['file_name']
def conds(args):
    out=[('clean',img(args.clean_root,'annotations/test.json','images/test',args.image_index))]
    man=loadj(args.manifest); ann=man.get('output_annotation','annotations/test_png.json')
    for it in man.get('conditions',[]):
        c=it['corruption']; s=int(it['severity'])
        if args.only and c not in args.only: continue
        if args.severities and s not in args.severities: continue
        out.append((f'{c}_s{s}',img(args.benchmark_root,ann,it['image_prefix'].rstrip('/'),args.image_index)))
    return out

def model(cfg,ck,dev):
    patch(); from mmdet.apis import init_detector; from mmdet.utils import register_all_modules
    register_all_modules(init_default_scope=True); return init_detector(cfg,ck,device=dev)
class Hook:
    def __init__(self,m): self.feat=None; self.handle=m.register_forward_hook(self.fn)
    def fn(self,m,i,o): self.feat=o
    def close(self): self.handle.remove()
def module(m,layer): return {'backbone':m.backbone,'neck':m.neck,'rpn_head':m.rpn_head}[layer]
def norm(x): x=np.nan_to_num(x.astype(np.float32)); x=x-x.min(); return x/(x.max()+1e-8)
def maps(f,levels):
    fs=[f] if isinstance(f,torch.Tensor) else list(f) if isinstance(f,(tuple,list)) else list(f.values()) if isinstance(f,dict) else []
    out=[]
    for i,t in enumerate(fs):
        if i in levels and isinstance(t,torch.Tensor) and t.ndim==4: out.append((i,t.detach().float().abs().mean(1)[0].cpu().numpy()))
    return out
def save_overlay(im,fm,out):
    h,w=im.shape[:2]; hm=cv2.resize(norm(fm),(w,h)); heat=cv2.applyColorMap(np.uint8(255*hm),cv2.COLORMAP_JET); cv2.imwrite(str(out),np.uint8(.55*im+.45*heat))
def feature(args):
    if args.skip_featuremaps: return
    from mmdet.apis import inference_detector
    mo=model(args.config,args.checkpoint,args.device); hk=Hook(module(mo,args.layer)); root=Path(args.out_dir)/'featuremaps'; mkdir(root)
    for name,ip in conds(args):
        im=cv2.imread(str(ip)); hk.feat=None
        with torch.no_grad(): inference_detector(mo,str(ip))
        if hk.feat is None: continue
        cd=root/name; mkdir(cd); cv2.imwrite(str(cd/'input.png'),im)
        for lv,fm in maps(hk.feat,args.levels):
            plt.figure(figsize=(5,5)); plt.imshow(norm(fm)); plt.axis('off'); plt.tight_layout(pad=0); plt.savefig(cd/f'{args.layer}_level{lv}_heatmap.png',dpi=300,bbox_inches='tight',pad_inches=0); plt.close(); save_overlay(im,fm,cd/f'{args.layer}_level{lv}_overlay.png')
    hk.close()
def parse_log(p):
    d={'loss':[],'lr':[],'segm_mAP':[],'bbox_mAP':[],'segm_mAP_50':[],'segm_mAP_75':[]}; last=0
    for n,line in enumerate(Path(p).read_text(encoding='utf-8',errors='ignore').splitlines(),1):
        mt=re.search(r'Epoch\(train\)\s+\[(\d+)\]\[(\d+)/(\d+)\]',line); x=last if last else n
        if mt: e,it,total=map(int,mt.groups()); x=e+it/max(1,total); last=x
        for k,pat in {'loss':r'(?:^|\s)loss:\s*([0-9.eE+-]+)','lr':r'\blr:\s*([0-9.eE+-]+)','segm_mAP':r'coco/segm_mAP:\s*([0-9.eE+-]+)','bbox_mAP':r'coco/bbox_mAP:\s*([0-9.eE+-]+)','segm_mAP_50':r'coco/segm_mAP_50:\s*([0-9.eE+-]+)','segm_mAP_75':r'coco/segm_mAP_75:\s*([0-9.eE+-]+)'}.items():
            mm=re.search(pat,line)
            if mm: d[k].append((x,float(mm.group(1))))
    return d
def plot(points,out,title,ylabel):
    if not points: return
    x=[p[0] for p in points]; y=[p[1] for p in points]; plt.figure(figsize=(8,4.5)); plt.plot(x,y,marker='o',markersize=2); plt.title(title); plt.xlabel('Epoch'); plt.ylabel(ylabel); plt.grid(True,alpha=.3); plt.tight_layout(); plt.savefig(out,dpi=300); plt.close()
def logs(args):
    if args.skip_log or not args.log_file: return
    out=Path(args.out_dir)/'training_curves'; mkdir(out); d=parse_log(args.log_file)
    for k,t,y in [('loss','Training Loss','Loss'),('lr','Learning Rate','LR'),('segm_mAP','Validation Mask AP','AP'),('bbox_mAP','Validation BBox AP','AP'),('segm_mAP_50','Validation Mask AP50','AP50'),('segm_mAP_75','Validation Mask AP75','AP75')]: plot(d[k],out/f'{k}.png',t,y)
    (out/'parsed_log.json').write_text(json.dumps({k:[{'x':x,'y':y} for x,y in v] for k,v in d.items()},indent=2),encoding='utf-8')
def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--checkpoint',required=True); p.add_argument('--clean-root',default='mmdet_dataset/lettuce'); p.add_argument('--benchmark-root',default='mmdet_dataset/lettuce_c'); p.add_argument('--manifest',default='mmdet_dataset/lettuce_c/manifest.json'); p.add_argument('--out-dir',required=True); p.add_argument('--layer',default='neck',choices=['backbone','neck','rpn_head']); p.add_argument('--levels',nargs='+',type=int,default=[0,1,2,3]); p.add_argument('--device',default='cuda:0'); p.add_argument('--image-index',type=int,default=0); p.add_argument('--only',nargs='+',default=None); p.add_argument('--severities',nargs='+',type=int,default=None); p.add_argument('--log-file',default=None); p.add_argument('--skip-featuremaps',action='store_true'); p.add_argument('--skip-log',action='store_true'); args=p.parse_args(); mkdir(args.out_dir); feature(args); logs(args)
if __name__=='__main__': main()
