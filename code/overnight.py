"""OVERNIGHT QUEUE. Every experiment = one delta vs the frozen BEST model, judged on walk-forward
(gain >= 0.005 AND >= 4/5 folds). Passers get ONE sealed blind scoring (ledgered). For every
experiment we also log the 2023 model rank of pid P144ed94c77 (the diagnostic pid) as a DIAGNOSTIC ONLY.
Ends with a greedy forward-combination of passers -> FINAL genome -> sealed scoring -> board preds."""
import sys,os,json,time,itertools,numpy as np,pandas as pd
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,ROOT); os.chdir(ROOT)
exec(compile(open(f"{ROOT}/r7/evolve3.py").read().split("lineage=json.load(open(LIN))")[0],"defs","exec"))
DEF={"el":0,"wk":0,"ridge_alpha":30.0,"icl_topk":0,"icl_ctx":"all","resid":0,"knn":0,"cat":0,"ixc":0,"pss":0,"icl_norm":"default","icl_outlier":4.0,"icl_shuffle":"latin","labelmix":"single","hz":0,"covw":0,"risk":"mean","ridge":0,"icl_n":8}
BASE=json.load(open("BEST_MODEL.json"))["genome"]; [BASE.setdefault(k,v) for k,v in DEF.items()]
WEMBY="P144ed94c77"; OUT="r7/overnight_results.json"; LOG="r7/overnight_runs.jsonl"
res={"styles":{}}; 
def save(): json.dump(res,open(OUT,"w"),indent=1)
def wemby_rank(g,wts):
    """2023 drafted pool: model rank of the diagnostic pid. Predictions only, no outcomes."""
    OPTS=opts_of(g); base=H.cols_for(OPTS); cols=cols_of(g,base); RATE=[c for c in base if c.startswith(("col_","intl_")) and not c.startswith("col_gl_")]
    pr,cq=H.fit_prior(W); d0=Vt.blind_inputs()[2023]; kk=Vt.horizons()[2023]
    Btr,Bte=feat_tx(H.build(W,pr,cq,OPTS),H.build(d0,pr,cq,OPTS),g["fx"],W,d0,RATE); st=stats_of(Btr); Btr,Bte=add_feats(Btr,g["fx"],st),add_feats(Bte,g["fx"],st)
    P=member_preds(g,Btr,Bte,cols,W,kk if g["hz"] else 5); s=stack(P,Bte,g,wts)
    order=list(d0.pid.values[np.argsort(-s)]); return (order.index(WEMBY)+1) if WEMBY in order else None
t0=time.time(); bf,bw=fitness(BASE); bmean=float(np.mean(list(bf.values())))
print(f"BASE WF {bmean:+.4f} folds {[round(v,3) for v in bf.values()]}  wemby2023 #{wemby_rank(BASE,bw)}  ({time.time()-t0:.0f}s)",flush=True)
Q=[("data wk",{"wk":1}),("data el",{"el":1}),("data wk+el",{"wk":1,"el":1}),("data wk,intl off",{"wk":1,"intl":0}),
   ("label cum3",{"hw":"cum3"}),("label cum4",{"hw":"cum4"}),("label uniform",{"hw":"uniform"}),("label disc70",{"hw":"disc70"}),("label front",{"hw":"front"}),("label top2",{"hw":"top2"}),("label rank(not gauss)",{"label":"rank"}),("label noclip",{"clip":"none"}),("label log1p",{"clip":"log1p"}),
   ("shrink M250",{"M":250.0}),
   ("icl topk100",{"icl_topk":100}),("icl topk60",{"icl_topk":60}),("icl ctx drafted",{"icl_ctx":"drafted"}),("icl n32",{"icl_n":32}),("icl norm none",{"icl_norm":"none"}),("icl norm power",{"icl_norm":"power"}),("icl norm quantile",{"icl_norm":"quantile"}),("icl outlier2",{"icl_outlier":2.0}),("icl outlier8",{"icl_outlier":8.0}),("icl shuffle random",{"icl_shuffle":"random"}),
   ("ridge a30",{"ridge":1,"ridge_alpha":30.0}),("ridge a100",{"ridge":1,"ridge_alpha":100.0}),("ridge a300",{"ridge":1,"ridge_alpha":300.0}),("ridge a1000",{"ridge":1,"ridge_alpha":1000.0}),("knn",{"knn":1}),("catboost",{"cat":1}),("cat+ridge300",{"cat":1,"ridge":1,"ridge_alpha":300.0}),("cat+knn",{"cat":1,"knn":1}),
   ("resid stack",{"resid":1}),("interaction constraints",{"ixc":1}),("per-season target",{"pss":1}),("horizon models",{"hz":1}),("cov weights",{"covw":1}),("hz+covw",{"hz":1,"covw":1}),("labelmix2",{"labelmix":"mix2"}),("labelmix3",{"labelmix":"mix3"}),
   ("thin mild",{"thin":"mild"}),("thin strong",{"thin":"strong"}),("anchor top1",{"anchor":"top1"}),("anchor top3",{"anchor":"top3"}),("cons .1",{"cons":0.1}),("cons .2",{"cons":0.2}),
   ("mono off",{"mono":"off"}),("hurdle",{"hurdle":1}),("bag9",{"bag":9}),("mingp10",{"mingp":10}),("risk q25",{"risk":"q25"}),
   ("fx shoot",{"fx":"shoot"}),("fx shoot+age",{"fx":"shoot+age"}),("fx posz",{"fx":"posz"}),("fx eraz",{"fx":"eraz"})]
results=[]
def run_exp(name,delta,seal=True):
    g=dict(BASE); g.update(delta); t=time.time()
    try: mf,wts=fitness(g)
    except Exception as e: print(f"  fail {name}: {repr(e)[:100]}",flush=True); return None
    gain=float(np.mean(list(mf.values()))-bmean); wins=int(sum(mf[y]>bf[y] for y in mf)); ok=(gain>=0.005 and wins>=4)
    try: wr=wemby_rank(g,wts)
    except Exception: wr=None
    rec=dict(name=name,delta=delta,wf=round(float(np.mean(list(mf.values()))),4),gain=round(gain,4),folds=wins,passed=ok,wemby_2023_rank=wr,secs=round(time.time()-t))
    if ok and seal:
        rows=blind(g,f"OVERNIGHT {name}",wts); bl=[r for r in rows if r["kind"]=="BLIND"]; e=[r["stack"]-r["draft"] for r in bl]
        rec.update(blind_edge=round(float(np.mean(e)),4),ge02=int(sum(x>=0.2 for x in e)),edges={str(r["season"]):round(x,3) for r,x in zip(bl,e)})
        res["styles"][f"OVERNIGHT {name}"]=dict(weights=wts,rows=rows,secs=rec["secs"],phase="OVERNIGHT",genome=g); save()
    results.append(rec); open(LOG,"a").write(json.dumps(rec)+"\n")
    print(f"[{time.strftime('%H:%M:%S')}] {'PASS' if ok else 'fail'} {name:<26} WF {rec['wf']:+.4f} gain {gain:+.4f} folds {wins}/5 wemby#{wr}"+(f" | blind edge {rec['blind_edge']:+.3f} ({rec['ge02']}/7)" if ok else ""),flush=True)
    return rec
for name,delta in Q: run_exp(name,delta)
# seed-stability of the base (how noisy is one fitness estimate?)
sds=[]
for seeds in [(21,22,23),(31,32,33),(41,42,43)]:
    g=dict(BASE); g["bag"]=3
    import honest as _H
    old=_H.predict
    def patched(model,Xtr,ytr,Xte,cfg,_s=seeds,_old=old): return _old(model,Xtr,ytr,Xte,dict(cfg,seeds=_s))
    _H.predict=patched
    try: mf,_=fitness(g); sds.append(float(np.mean(list(mf.values()))))
    finally: _H.predict=old
print(f"seed-stability: base WF over seed sets {np.round(sds,4).tolist()}  sd {np.std(sds):.4f}",flush=True)
# greedy forward combination of passers
passers=[r for r in results if r["passed"]]; passers.sort(key=lambda r:-r["gain"])
print("passers:",[p["name"] for p in passers],flush=True)
cur=dict(BASE); cf,cw=bf,bw; cmean=bmean; chosen=[]
for p in passers:
    g=dict(cur); g.update(p["delta"])
    try: mf,wts=fitness(g)
    except Exception: continue
    gain=float(np.mean(list(mf.values()))-cmean); wins=int(sum(mf[y]>cf[y] for y in mf))
    if gain>=0.003 and wins>=3: cur,cf,cw,cmean=g,mf,wts,float(np.mean(list(mf.values()))); chosen.append(p["name"]); print(f"  combine + {p['name']}: WF {cmean:+.4f}",flush=True)
rows=blind(cur,"OVERNIGHT FINAL",cw); bl=[r for r in rows if r["kind"]=="BLIND"]; e=[r["stack"]-r["draft"] for r in bl]
res["styles"]["OVERNIGHT FINAL"]=dict(weights=cw,rows=rows,secs=0,phase="OVERNIGHT",genome=cur); save()
json.dump(dict(genome=cur,chosen=chosen,wf=cmean,blind_edge=float(np.mean(e)),edges={str(r["season"]):round(x,3) for r,x in zip(bl,e)},ge02=int(sum(x>=0.2 for x in e)),wemby_2023_rank=wemby_rank(cur,cw),ledger=Vt.ledger_count()),open("r7/FINAL_OVERNIGHT.json","w"),indent=1)
print(f"FINAL: +{chosen} | WF {cmean:+.4f} | blind edge {np.mean(e):+.3f} ({sum(x>=0.2 for x in e)}/7) | wemby2023 #{wemby_rank(cur,cw)} | ledger {Vt.ledger_count()}\nOVERNIGHT_DONE",flush=True)
