"""Champion predictions for each blind class's drafted pool. No vault.score calls (no outcomes touched)."""
import sys,os,json,numpy as np,pandas as pd
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,ROOT); os.chdir(ROOT)
src=open(f"{ROOT}/r7/evolve3.py").read(); exec(compile(src.split("lineage=json.load(open(LIN))")[0],"defs","exec"))
B=json.load(open(f"{ROOT}/BEST_MODEL.json")); g=B["genome"]
for k,v in {"el":0,"wk":0,"ridge_alpha":30.0,"icl_topk":0,"icl_ctx":"all","resid":0,"knn":0,"cat":0,"ixc":0,"pss":0,"icl_norm":"default","icl_outlier":4.0,"icl_shuffle":"latin","labelmix":"single","hz":0,"covw":0,"risk":"mean","ridge":0,"icl_n":8}.items(): g.setdefault(k,v)
n=len(models_of(g)); wts={"rich":tuple([1/n]*n),"thin":tuple([1/n]*n)}
print("champion:",{k:v for k,v in g.items() if v not in(0,"none","default","all","single","mean",8,3,"latin",4.0,30.0,"base")},flush=True)
OPTS=opts_of(g); base=H.cols_for(OPTS); cols=cols_of(g,base); RATE=[c for c in base if c.startswith(("col_","intl_")) and not c.startswith("col_gl_")]
pr,cq=H.fit_prior(W); Btr0=H.build(W,pr,cq,OPTS); HZ=Vt.horizons(); out=[]
for y,d0 in sorted(Vt.blind_inputs().items()):
    kk=HZ[y]; Btr,Bte=feat_tx(Btr0,H.build(d0,pr,cq,OPTS),g["fx"],W,d0,RATE); st=stats_of(Btr); Btr,Bte=add_feats(Btr,g["fx"],st),add_feats(Bte,g["fx"],st)
    P=member_preds(g,Btr,Bte,cols,W,kk if g["hz"] else 5); s=stack(P,Bte,g,wts)
    out.append(pd.DataFrame(dict(pid=d0.pid.values,season=y,k=kk,score=s))); print(y,"scored",len(d0),flush=True)
pd.concat(out).to_csv(f"{ROOT}/r7/board_preds.csv",index=False); print("wrote r7/board_preds.csv")
