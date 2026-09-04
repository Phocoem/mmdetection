#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_single_image_features.py

Exports input, output prediction, C2-C5, P2-P6, and P2'-P6' overlays.
For normal FPN, P and P' are usually identical.
For DGCFPN inheriting FPN, P = original FPN output, P' = enhanced neck output.
"""

import argparse
from pathlib import Path
import cv2, numpy as np, torch
from mmengine.config import Config
from mmengine.dataset import Compose, pseudo_collate
from mmdet.apis import init_detector
from mmdet.models.necks.fpn import FPN

def norm(x):
    x=x.astype(np.float32); lo,hi=np.percentile(x,1),np.percentile(x,99)
    return np.clip((x-lo)/(hi-lo+1e-6),0,1)
def heat(feat, hw):
    if feat.dim()==4: feat=feat[0]
    fmap=feat.detach().float().abs().mean(0).cpu().numpy()
    fmap=cv2.resize(norm(fmap),(hw[1],hw[0]),interpolation=cv2.INTER_LINEAR)
    return cv2.applyColorMap((fmap*255).astype(np.uint8), cv2.COLORMAP_JET)
def save_pair(out_dir,name,feat,img,alpha):
    h=heat(feat,img.shape[:2]); ov=cv2.addWeighted(img,1-alpha,h,alpha,0)
    cv2.imwrite(str(out_dir/f"{name}_heatmap.png"),h)
    cv2.imwrite(str(out_dir/f"{name}_overlay.png"),ov)
def draw_pred(img, result, thr):
    out=img.copy(); pred=result.pred_instances
    if not hasattr(pred,"scores"): return out
    scores=pred.scores.detach().cpu().numpy(); keep=scores>=thr
    if hasattr(pred,"masks"):
        masks=pred.masks.detach().cpu().numpy()
        for m in masks[keep]:
            mask=m.astype(bool); color=np.array([0,255,0],dtype=np.uint8)
            out[mask]=(0.55*out[mask]+0.45*color).astype(np.uint8)
    if hasattr(pred,"bboxes"):
        bboxes=pred.bboxes.detach().cpu().numpy()
        for box,s in zip(bboxes[keep],scores[keep]):
            x1,y1,x2,y2=box.astype(int).tolist()
            cv2.rectangle(out,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(out,f"{s:.2f}",(x1,max(0,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0),1,cv2.LINE_AA)
    return out
def batch(cfg,img_path,model):
    pipeline=Compose(cfg.test_dataloader.dataset.pipeline)
    data=pipeline(dict(img_path=str(img_path), img_id=0))
    return model.data_preprocessor(pseudo_collate([data]), False)
def grid(paths,out_path,tile_w=320):
    imgs=[]
    for p in paths:
        im=cv2.imread(str(p))
        if im is None: continue
        h,w=im.shape[:2]; im=cv2.resize(im,(tile_w,int(h*tile_w/max(w,1))))
        label=p.stem
        cv2.putText(im,label,(8,22),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(im,label,(8,22),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,0,0),1,cv2.LINE_AA)
        imgs.append(im)
    if not imgs: return
    mh=max(i.shape[0] for i in imgs); pads=[]
    for im in imgs:
        if im.shape[0]<mh: im=np.vstack([im,np.zeros((mh-im.shape[0],im.shape[1],3),dtype=np.uint8)])
        pads.append(im)
    cv2.imwrite(str(out_path),np.hstack(pads))
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",required=True); p.add_argument("--checkpoint",required=True); p.add_argument("--image",required=True); p.add_argument("--out-dir",required=True)
    p.add_argument("--device",default="cuda:0"); p.add_argument("--score-thr",type=float,default=0.3); p.add_argument("--alpha",type=float,default=0.45)
    args=p.parse_args(); out_dir=Path(args.out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    cfg=Config.fromfile(args.config); model=init_detector(cfg,args.checkpoint,device=args.device); model.eval()
    img=cv2.imread(str(args.image)); 
    if img is None: raise FileNotFoundError(args.image)
    cv2.imwrite(str(out_dir/"input.png"),img)
    data=batch(cfg,Path(args.image),model)
    with torch.no_grad():
        inp=data["inputs"]
        if isinstance(inp,list): inp=torch.stack(inp,0)
        inp=inp.to(args.device)
        c_feats=model.backbone(inp)
        try: p_before=FPN.forward(model.neck,c_feats)
        except Exception: p_before=model.neck(c_feats)
        p_after=model.neck(c_feats)
        result=model.test_step(data)[0]
    c_paths=[]
    for name,feat in zip(["C2","C3","C4","C5"],c_feats):
        save_pair(out_dir,name,feat,img,args.alpha); c_paths.append(out_dir/f"{name}_overlay.png")
    p_paths=[]
    for name,feat in zip(["P2","P3","P4","P5","P6"],p_before):
        save_pair(out_dir,name,feat,img,args.alpha); p_paths.append(out_dir/f"{name}_overlay.png")
    pp_paths=[]
    for name,feat in zip(["P2_prime","P3_prime","P4_prime","P5_prime","P6_prime"],p_after):
        save_pair(out_dir,name,feat,img,args.alpha); pp_paths.append(out_dir/f"{name}_overlay.png")
    cv2.imwrite(str(out_dir/"output_prediction.png"),draw_pred(img,result,args.score_thr))
    grid([out_dir/"input.png"]+c_paths,out_dir/"grid_C_features.png")
    grid([out_dir/"input.png"]+p_paths,out_dir/"grid_P_before_features.png")
    grid([out_dir/"input.png"]+pp_paths,out_dir/"grid_P_prime_features.png")
    grid([out_dir/"input.png",out_dir/"output_prediction.png"],out_dir/"grid_input_output.png")
    print(f"[OK] Saved to {out_dir}")
if __name__=="__main__":
    main()
