#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect evaluation outputs and create paper tables/figures."""
import argparse, csv, json, re
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt

ALIASES={
 'segm_mAP':['segm_mAP','coco/segm_mAP','mask_mAP','mask_AP','AP','ap','mean_mask_ap'],
 'bbox_mAP':['bbox_mAP','coco/bbox_mAP','box_mAP','bbox_AP'],
 'segm_mAP_50':['segm_mAP_50','coco/segm_mAP_50','mask_AP50','AP50'],
 'segm_mAP_75':['segm_mAP_75','coco/segm_mAP_75','mask_AP75','AP75'],
 'bbox_mAP_50':['bbox_mAP_50','coco/bbox_mAP_50','bbox_AP50'],
 'bbox_mAP_75':['bbox_mAP_75','coco/bbox_mAP_75','bbox_AP75']}
CORRS=['clean','brightness','contrast','gaussian_noise','defocus_blur','motion_blur','fog','jpeg','jpeg_compression','shot_noise','impulse_noise','zoom_blur','snow','frost','pixelate']

def load(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def val(d,aliases):
    for a in aliases:
        if a in d:
            try: return float(d[a])
            except Exception: return None
    return None

def infer(path):
    s=str(path).lower(); corr=''
    for c in CORRS:
        if c in s: corr=c; break
    if not corr and 'clean' in s: corr='clean'
    sev=None
    m=re.search(r'(?:severity|sev|s)[_\-]?([1-5])',s)
    if m: sev=int(m.group(1))
    else:
        for p in path.parts:
            if p in ['1','2','3','4','5']: sev=int(p)
    if corr=='clean': sev=0
    return corr,sev

def flatten(o):
    rows=[]
    if isinstance(o,dict):
        if any(any(a in o for a in al) for al in ALIASES.values()): rows.append(o)
        for v in o.values(): rows+=flatten(v)
    elif isinstance(o,list):
        for v in o: rows+=flatten(v)
    return rows

def row_from_dict(d,path,m,seed):
    mets={k:val(d,a) for k,a in ALIASES.items()}
    if all(v is None for v in mets.values()): return None
    corr=d.get('corruption',d.get('condition',d.get('corruption_type','')))
    sev=d.get('severity',d.get('level',None))
    if not corr:
        corr,sev2=infer(path); sev=sev if sev is not None else sev2
    if isinstance(corr,str):
        mm=re.match(r'(.+)_s([1-5])$',corr)
        if mm: corr=mm.group(1); sev=int(mm.group(2))
    if sev is None: _,sev=infer(path)
    try: sev=int(sev) if sev is not None else 0
    except Exception: sev=0
    return {'model_key':m['key'],'model_name':m['name'],'group':m.get('group',''),'seed':seed,'source':str(path),'corruption':str(corr).replace('jpeg_compression','jpeg'),'severity':sev,**mets}

def collect_file(path,m,seed):
    rows=[]
    if path.suffix.lower()=='.csv':
        try: df=pd.read_csv(path)
        except Exception: return rows
        for _,r in df.iterrows():
            rr=row_from_dict(r.to_dict(),path,m,seed)
            if rr: rows.append(rr)
    elif path.suffix.lower()=='.json':
        try: obj=load(path)
        except Exception: return rows
        for d in flatten(obj):
            rr=row_from_dict(d,path,m,seed)
            if rr: rows.append(rr)
    return rows

def find_eval_dir(m,seed,eval_root):
    wd=Path(m['work_dir'].format(seed=seed))
    cands=[wd/'evaluation',Path(eval_root)/m['key']/f'seed_{seed}'/'evaluation',Path(eval_root)/m['key']/'evaluation']
    for c in cands:
        if c.exists(): return c
    matches=list(Path(eval_root).glob(f"**/{m['key']}*/**/evaluation")) if Path(eval_root).exists() else []
    return matches[0] if matches else None

def collect(man,eval_root,seeds):
    rows=[]
    for seed in seeds:
        for m in man['models']:
            if 'YOUR_' in m['config'] or 'YOUR_' in m['work_dir']: continue
            ed=find_eval_dir(m,seed,eval_root)
            if not ed: print('[WARN] eval dir not found',m['key']); continue
            files=list(ed.rglob('*.csv'))+list(ed.rglob('*.json'))
            print('[COLLECT]',m['name'],ed,'files',len(files))
            for f in files: rows+=collect_file(f,m,seed)
    df=pd.DataFrame(rows)
    if df.empty: raise RuntimeError('No metrics collected. Check evaluator outputs.')
    return df

def dedup(df):
    keys=['model_key','model_name','group','seed','corruption','severity']
    mets=[k for k in ALIASES if k in df.columns]
    return df.groupby(keys,dropna=False)[mets].mean(numeric_only=True).reset_index()

def heat(pivot,path,title,label):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    data=pivot.values.astype(float)
    plt.figure(figsize=(max(7,1.2*len(pivot.columns)+3),max(4,0.45*len(pivot.index)+1.5)))
    im=plt.imshow(data,aspect='auto'); plt.colorbar(im,label=label)
    plt.xticks(range(len(pivot.columns)),pivot.columns,rotation=35,ha='right'); plt.yticks(range(len(pivot.index)),pivot.index)
    plt.title(title)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i,j]): plt.text(j,i,f'{data[i,j]:.3f}',ha='center',va='center',fontsize=8)
    plt.tight_layout(); plt.savefig(path.with_suffix('.png'),dpi=300); plt.savefig(path.with_suffix('.pdf')); plt.close()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--manifest',required=True); p.add_argument('--eval-root',default='work_dirs/research'); p.add_argument('--out-dir',required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=[2026]); p.add_argument('--main-corruptions',nargs='+',default=['brightness','contrast','gaussian_noise','defocus_blur','motion_blur'])
    args=p.parse_args(); out=Path(args.out_dir); td=out/'tables'; fd=out/'figures'; td.mkdir(parents=True,exist_ok=True); fd.mkdir(parents=True,exist_ok=True)
    man=load(args.manifest); raw=collect(man,args.eval_root,args.seeds); raw.to_csv(td/'raw_collected_metrics.csv',index=False)
    df=dedup(raw); df.to_csv(td/'deduplicated_metrics.csv',index=False)
    metric='segm_mAP'
    clean=df[df.corruption.eq('clean')]
    corr=df[df.corruption.isin(args.main_corruptions)]
    bycorr=corr.groupby(['model_key','model_name','group','seed','corruption'])[metric].mean().reset_index()
    bycorr.to_csv(td/'mean_ap_by_corruption.csv',index=False)
    clean_ap=clean.groupby(['model_key','model_name','group','seed'])[metric].mean().reset_index().rename(columns={metric:'clean_ap'})
    mean_corr=bycorr.groupby(['model_key','model_name','group','seed'])[metric].mean().reset_index().rename(columns={metric:'mean_corr_ap'})
    worst=bycorr.groupby(['model_key','model_name','group','seed'])[metric].min().reset_index().rename(columns={metric:'worst_corr_ap'})
    s5=corr[corr.severity.eq(5)].groupby(['model_key','model_name','group','seed'])[metric].mean().reset_index().rename(columns={metric:'severity5_ap'})
    summ=clean_ap.merge(mean_corr,how='outer').merge(worst,how='outer').merge(s5,how='outer')
    summ['RD']=summ.clean_ap-summ.mean_corr_ap; summ['SI']=summ.mean_corr_ap/summ.clean_ap
    summ.to_csv(td/'robustness_summary_by_seed.csv',index=False)
    agg=summ.groupby(['model_key','model_name','group']).mean(numeric_only=True).reset_index().sort_values('mean_corr_ap',ascending=False)
    agg.to_csv(td/'robustness_summary.csv',index=False)
    pivot=bycorr.groupby(['model_name','corruption'])[metric].mean().reset_index().pivot(index='model_name',columns='corruption',values=metric).reindex(columns=args.main_corruptions)
    pivot.to_csv(td/'heatmap_mean_ap.csv'); heat(pivot,fd/'fig_mean_ap_heatmap','Mean Mask AP over Selected Corruptions','Mean mask AP')
    clean_map=df[df.corruption.eq('clean')].groupby('model_name')[metric].mean()
    rd=[]; si=[]
    for _,r in bycorr.groupby(['model_name','corruption'])[metric].mean().reset_index().iterrows():
        ca=clean_map.get(r.model_name,np.nan); ap=r[metric]
        rd.append({'model_name':r.model_name,'corruption':r.corruption,'RD':ca-ap})
        si.append({'model_name':r.model_name,'corruption':r.corruption,'SI':ap/ca if ca else np.nan})
    rdp=pd.DataFrame(rd).pivot(index='model_name',columns='corruption',values='RD').reindex(columns=args.main_corruptions)
    sip=pd.DataFrame(si).pivot(index='model_name',columns='corruption',values='SI').reindex(columns=args.main_corruptions)
    rdp.to_csv(td/'heatmap_rd.csv'); sip.to_csv(td/'heatmap_si.csv')
    heat(rdp,fd/'fig_rd_heatmap','Robustness Drop by Selected Corruption','RD = clean AP - corrupted AP')
    heat(sip,fd/'fig_si_heatmap','Stability Index by Selected Corruption','SI = corrupted AP / clean AP')
    sev=corr.groupby(['model_name','severity'])[metric].mean().reset_index().pivot(index='model_name',columns='severity',values=metric)
    sev.to_csv(td/'heatmap_severity.csv'); heat(sev,fd/'fig_severity_heatmap','Mean Mask AP by Severity','Mean mask AP')
    plt.figure(figsize=(8,5))
    sev2=corr.groupby(['model_name','severity'])[metric].mean().reset_index()
    for model,g in sev2.groupby('model_name'):
        g=g.sort_values('severity'); plt.plot(g.severity,g[metric],marker='o',label=model)
    plt.xlabel('Severity'); plt.ylabel('Mean mask AP'); plt.title('Severity-wise Mean Mask AP'); plt.grid(True,alpha=.3); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(fd/'severity_curves.png',dpi=300); plt.savefig(fd/'severity_curves.pdf'); plt.close()
    md=['# Robustness summary','',agg[['model_name','group','clean_ap','mean_corr_ap','worst_corr_ap','severity5_ap','RD','SI']].to_markdown(index=False,floatfmt='.3f'),'','RD lower is better. SI higher is better.']
    (out/'summary_report.md').write_text('\n'.join(md),encoding='utf-8')
    print('[DONE]',out)
if __name__=='__main__': main()
