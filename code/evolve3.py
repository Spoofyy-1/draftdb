"""EVOLUTION v3 — seeded from v2 gen17. New genes (the six mutations):
  labelmix single|mix2|mix3   stack models trained on DIFFERENT labels (own | +cum-rank | +career-year z)
  hz       0|1                horizon-matched: per-k models; fitness scores every fold at k=1..5
  covw     0|1                coverage-conditional stack weights (rich rows vs thin rows)
  risk     mean|q25           xgb member ranks by the 25th-percentile projection (bust-averse)
  ridge    0|1                diverse third learner (ridge on standardized features)
  icl_n    8|32               TabICL internal ensemble size
Monotone acceptance (gain>=0.005 & 4/5 folds). Fitness = walk-forward only, averaged over horizons k=1..5.
No synthetic. Window 2010-2018 (G8). Vault only on accepted champions."""
import sys,os,time,json,itertools,numpy as np,pandas as pd
from scipy.stats import spearmanr,norm
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,ROOT); R7=f"{ROOT}/r7"
import honest as H, vault as Vt
RUNS=f"{R7}/runs.jsonl"; LIN=f"{R7}/lineage.json"; OUT=f"{R7}/r7_results.json"; MIN_GAIN=0.005; MIN_FOLD_WINS=4
S=[f"y_s{i}_war" for i in range(1,6)]; WIN=(2010,2018); TR=H.TRAIN.copy()
for c in S: TR[c]=pd.to_numeric(TR[c],errors="coerce")
W=TR[(TR.draft_year>=WIN[0])&(TR.draft_year<=WIN[1])]
W=W.reset_index(drop=True)   # training pool = ALL labeled rows (drafted + undrafted)
H.audit_window(W.draft_year.values); print("training pool: all labeled rows,",len(W),flush=True)
MU=[float(W[c].fillna(0).mean()) for c in S]; SD=[float(W[c].fillna(0).std())+1e-9 for c in S]
GENES={"hw":["uniform","disc85","disc70","front","top2","cum3","cum4"],"clip":["none","clip40","log1p"],"label":["rank","gaussrank"],
       "fx":["base","shoot","shoot+age","posz","eraz"],"M":[250.0,400.0],"thin":["none","mild","strong"],"anchor":["none","top1","top3"],
       "cons":[0.0,0.1,0.2],"intl":[0,1],"bag":[3,9],"mono":["on","off"],"hurdle":[0,1],
       "el":[0,1],"wk":[0,1],"labelmix":["single","mix2","mix3"],"hz":[0,1],"covw":[0,1],"risk":["mean","q25"],"ridge":[0,1],"icl_n":[8,32],"ridge_alpha":[30.0,100.0,300.0,1000.0],"icl_topk":[0,60,100],"icl_ctx":["all","drafted"],"resid":[0,1],"knn":[0,1],"cat":[0,1],"ixc":[0,1],"pss":[0,1],"icl_norm":["default","none","power","quantile"],"icl_outlier":[2.0,4.0,8.0],"icl_shuffle":["latin","random"]}
HW={"uniform":[1,1,1,1,1],"cum3":[1,1,1,0,0],"cum4":[1,1,1,1,0],"disc85":[.85**i for i in range(5)],"disc70":[.7**i for i in range(5)],"front":[1,.8,.6,.4,.2]}
THIN={"none":0.0,"mild":0.15,"strong":0.35}; ANCH={"none":0,"top1":1,"top3":3}
def label(df,spec,k=5):
    hw,norm_,clip,lab=spec; X=np.stack([df[S[i]].fillna(0).values for i in range(k)],1).astype(float)
    if norm_=="zseason": X=(X-np.array(MU[:k]))/np.array(SD[:k])
    v=np.sort(X,1)[:,-2:].mean(1) if hw=="top2" else (X*np.array(HW[hw][:k])).sum(1)
    if clip=="clip40": v=np.clip(v,-40,40)
    if clip=="log1p": v=np.sign(v)*np.log1p(np.abs(v))
    o=np.zeros(len(df))
    for y in df.draft_year.unique():
        m=(df.draft_year==y).values; r=pd.Series(v[m]).rank(pct=True).values
        o[m]=norm.ppf(np.clip((r*len(r)-0.5)/len(r),0.01,0.99)) if lab=="gaussrank" else r
    return o
def specs(g):
    own=(g["hw"],"none",g["clip"],g["label"]); out=[own]
    if g["labelmix"] in("mix2","mix3"): out.append(("uniform","none","none","rank"))
    if g["labelmix"]=="mix3": out.append(("uniform","zseason","none","gaussrank"))
    return out
def cum(df,k): return df[S[:k]].fillna(0).sum(axis=1).values
def n_(d,c): return pd.to_numeric(d.get(c),errors="coerce")
def z_(s,mu,sd): return (s-mu)/(sd if sd and sd>0 else 1)
def stats_of(B): return {k:(float(n_(B,c).mean()),float(n_(B,c).std())) for k,c in [("ft","col_ft_pct"),("vol","col_fg3a_per40"),("proj","col_shot_proj_3p"),("age","bio_age_at_draft")]}
def add_feats(B,fx,st):
    B=B.copy(); ft,p3,vol,age,proj=n_(B,"col_ft_pct"),n_(B,"col_fg3_pct"),n_(B,"col_fg3a_per40"),n_(B,"bio_age_at_draft"),n_(B,"col_shot_proj_3p")
    if "shoot" in fx: B["x_ft_vol"]=ft*vol; B["x_proj_vol"]=proj*vol; B["x_ft_gap"]=ft-p3; B["x_shoot_score"]=z_(ft,*st["ft"])+z_(vol,*st["vol"])+z_(proj,*st["proj"])
    if "age" in fx: B["x_age_x_shoot"]=(-z_(age,*st["age"]))*(z_(ft,*st["ft"])+z_(vol,*st["vol"])); B["x_old_shooter"]=((age>=21.5)&(ft>=0.78)&(vol>=4)).astype(float)
    return B
def feat_tx(Btr,Bte,fx,trmeta,temeta,RATE):
    Btr,Bte=Btr.copy(),Bte.copy()
    if "eraz" in fx:
        for B,meta in((Btr,trmeta),(Bte,temeta)):
            gg=B[RATE].groupby(meta["draft_year"].values); B[RATE]=(B[RATE]-gg.transform("mean"))/(gg.transform("std").replace(0,np.nan))
    if "posz" in fx:
        ptr=trmeta["bio_pos_code"].fillna(-1).values; pte=temeta["bio_pos_code"].fillna(-1).values
        for p in np.unique(ptr):
            m=ptr==p; mu=Btr.loc[m,RATE].mean(); sd=Btr.loc[m,RATE].std().replace(0,np.nan); Btr.loc[m,RATE]=(Btr.loc[m,RATE]-mu)/sd
            mt=pte==p
            if mt.any(): Bte.loc[mt,RATE]=(Bte.loc[mt,RATE]-mu)/sd
    return Btr,Bte
def models_of(g): return ["xgb","tabicl"]+(["ridge"] if g["ridge"] else [])+(["knn"] if g.get("knn") else [])+(["cat"] if g.get("cat") else [])
def opts_of(g): return dict(shrink=True,intl=bool(g["intl"]),cov=True,M=g["M"],el=bool(g.get("el",0)),wk=bool(g.get("wk",0)))
def cols_of(g,base): return base+(["x_ft_vol","x_proj_vol","x_ft_gap","x_shoot_score"] if "shoot" in g["fx"] else [])+(["x_age_x_shoot","x_old_shooter"] if "age" in g["fx"] else [])
def ixc_groups(cols):
    size=[i for i,c in enumerate(cols) if c.startswith(("bio_height","bio_weight","bio_combine","bio_bmi"))]
    prod=[i for i,c in enumerate(cols) if c.startswith(("col_","intl_","el_","wk_","x_","shr_","adj_"))]
    other=[i for i in range(len(cols)) if i not in size]
    return [sorted(set(size+prod)),sorted(set(other))]
def cfg_of(m,g,cols=None):
    c=dict(seeds=tuple(range(11,11+g["bag"])),mono=(g["mono"]=="on" and "age" not in g["fx"]),icl_n=g["icl_n"],ridge_alpha=g.get("ridge_alpha",30.0),**({"q":0.25} if (m=="xgb" and g["risk"]=="q25") else {}))
    if m=="xgb" and g.get("ixc") and cols is not None: c["ixc"]=ixc_groups(cols)
    if m=="tabicl":
        kw={}
        if g.get("icl_norm","default")!="default": kw["norm_methods"]=g["icl_norm"]
        if g.get("icl_outlier",4.0)!=4.0: kw["outlier_threshold"]=float(g["icl_outlier"])
        if g.get("icl_shuffle","latin")!="latin": kw["feat_shuffle_method"]=g["icl_shuffle"]
        c["icl_kwargs"]=kw
    return c
def season_label(df,i,k):
    v=df[S[i]].fillna(0).values; o=np.zeros(len(df))
    for y in df.draft_year.unique():
        m=(df.draft_year==y).values; o[m]=pd.Series(v[m]).rank(pct=True).values
    return o
def member_preds(g,Btr,Bte,cols,trdf,k):
    """dict model -> rank-averaged prediction over the genome's label set."""
    out={}; topk=None
    if g.get("icl_topk",0):
        import xgboost as xgb
        imp=xgb.XGBRegressor(max_depth=3,n_estimators=300,learning_rate=0.05,subsample=0.8,colsample_bytree=0.6,n_jobs=8,random_state=11).fit(Btr[cols],label(trdf,specs(g)[0],k)).feature_importances_
        topk=[c for _,c in sorted(zip(imp,cols),reverse=True)[:int(g["icl_topk"])]]
    ctx=np.ones(len(Btr),bool)
    if g.get("icl_ctx","all")=="drafted" and "was_drafted" in trdf.columns: ctx=(pd.to_numeric(trdf["was_drafted"],errors="coerce").fillna(0).values==1)
    if ctx.sum()<150: ctx=np.ones(len(Btr),bool)
    cc=topk or cols
    def labels_for(spec):
        if g.get("pss"): return [(HW["disc85"][i],season_label(trdf,i,k)) for i in range(k)]
        return [(1.0,label(trdf,spec,k))]
    def icl_pred(yl): return H.predict("tabicl",Btr[cc][ctx],yl[ctx],Bte[cc],cfg_of("tabicl",g,cols))
    for m in models_of(g):
        rs=[]
        for spec in specs(g):
            acc=np.zeros(len(Bte))
            for w,yl in labels_for(spec):
                if m=="tabicl": p=icl_pred(yl)
                elif m=="xgb" and g.get("resid"):
                    oof=np.zeros(len(Btr)); idx=np.arange(len(Btr)); np.random.default_rng(3).shuffle(idx)
                    for part in np.array_split(idx,3):
                        mask=np.zeros(len(Btr),bool); mask[part]=True
                        oof[mask]=H.predict("tabicl",Btr[cc][~mask],yl[~mask],Btr[cc][mask],cfg_of("tabicl",g,cols))
                    p=pd.Series(icl_pred(yl)).rank(pct=True).values+H.predict("xgb",Btr[cols],pd.Series(yl).rank(pct=True).values-pd.Series(oof).rank(pct=True).values,Bte[cols],cfg_of("xgb",g,cols))
                else: p=H.predict(m,Btr[cols],yl,Bte[cols],cfg_of(m,g,cols))
                if g["hurdle"]:
                    import xgboost as xgb
                    MNc=[f"y_s{i}_minutes" for i in range(1,6)]; played=(trdf[MNc[:k]].apply(pd.to_numeric,errors="coerce").fillna(0).sum(axis=1)>=500).astype(int)
                    p=xgb.XGBClassifier(max_depth=3,learning_rate=0.03,n_estimators=500,subsample=0.7,colsample_bytree=0.5,min_child_weight=8,reg_lambda=5.0,n_jobs=8,random_state=11).fit(Btr[cols],played).predict_proba(Bte[cols])[:,1]*pd.Series(p).rank(pct=True).values
                acc+=w*pd.Series(p).rank(pct=True).values
            rs.append(acc)
        out[m]=np.mean(rs,0)
    return out

def stack(preds,Bte,g,wts):
    """wts: {'rich':(w...), 'thin':(w...)} per model order."""
    models=models_of(g); cov=n_(Bte,"feat_coverage").fillna(0).values; thin=cov<np.median(cov) if g["covw"] else np.zeros(len(cov),bool)
    r=np.zeros(len(cov))
    for i,m in enumerate(models): r+=np.where(thin,wts["thin"][i],wts["rich"][i])*preds[m]
    r=pd.Series(r).rank(pct=True).values-THIN[g["thin"]]*(1-cov)
    mock=n_(Bte,"cons_mock_consensus_rank"); rsci=n_(Bte,"hs_rsci_rank")
    if g["cons"]>0: r=(1-g["cons"])*r+g["cons"]*(-mock).rank(pct=True).fillna(0.5).values
    N=ANCH[g["anchor"]]
    if N: r=np.where(((mock<=N)&((rsci<=10)|(mock<=2))).fillna(False).values,np.maximum(r,0.88),r)
    return r
def icw(p,d,k):
    dd=d.dropna(subset=["actual_pick"]); return float(spearmanr(p[dd.index],cum(dd,k)).statistic)
def wf_folds(g):
    OPTS=opts_of(g); base=H.cols_for(OPTS); cols=cols_of(g,base); RATE=[c for c in base if c.startswith(("col_","intl_")) and not c.startswith("col_gl_")]
    yrs=sorted(W.draft_year.unique()); F={}
    for y in yrs[4:]:
        trm=(W.draft_year.values<y); tem=(W.draft_year.values==y)
        H.audit_forward(W.draft_year.values[trm],y); H.audit_disjoint(W.pid.values[trm],W.pid.values[tem])
        pr,cq=H.fit_prior(W[trm]); Btr,Bte=feat_tx(H.build(W[trm],pr,cq,OPTS),H.build(W[tem],pr,cq,OPTS),g["fx"],W[trm],W[tem],RATE)
        st=stats_of(Btr); Btr,Bte=add_feats(Btr,g["fx"],st),add_feats(Bte,g["fx"],st)
        d=Bte.merge(H.PICKS,on="pid",how="left",suffixes=("","_pk"))
        if "actual_pick_pk" in d: d["actual_pick"]=d["actual_pick_pk"]
        P={}
        if g["hz"]:
            for k in range(1,6): P[k]=member_preds(g,Btr,Bte,cols,W[trm],k)
        else:
            p5=member_preds(g,Btr,Bte,cols,W[trm],5); P={k:p5 for k in range(1,6)}
        F[int(y)]=(P,Bte,d)
    return F
def learn_weights(g,F):
    models=models_of(g); n=len(models); eq=tuple([1/n]*n)
    if not g["covw"]: return {"rich":eq,"thin":eq}
    grid=[w for w in itertools.product(*[np.arange(0,1.01,0.25)]*n) if abs(sum(w)-1)<1e-6]
    best=None
    for wr in grid:
        for wt in grid:
            wts={"rich":wr,"thin":wt}
            sc=np.mean([np.mean([icw(stack(F[y][0][k],F[y][1],g,wts),F[y][2],k) for k in range(1,6)]) for y in F])
            if best is None or sc>best[0]: best=(sc,wts)
    return best[1]
def fitness(g):
    F=wf_folds(g); wts=learn_weights(g,F)
    return {str(y):float(np.mean([icw(stack(F[y][0][k],F[y][1],g,wts),F[y][2],k) for k in range(1,6)])) for y in F},wts
def blind(g,tag,wts):
    OPTS=opts_of(g); base=H.cols_for(OPTS); cols=cols_of(g,base); RATE=[c for c in base if c.startswith(("col_","intl_")) and not c.startswith("col_gl_")]
    pr,cq=H.fit_prior(W); Btr0=H.build(W,pr,cq,OPTS); HZ=Vt.horizons(); rows=[]; models=models_of(g)
    for y,d0 in sorted(Vt.blind_inputs().items()):
        H.audit_disjoint(W.pid.values,d0.pid.values); kk=HZ[y]
        Btr,Bte=feat_tx(Btr0,H.build(d0,pr,cq,OPTS),g["fx"],W,d0,RATE); st=stats_of(Btr); Btr,Bte=add_feats(Btr,g["fx"],st),add_feats(Bte,g["fx"],st)
        P=member_preds(g,Btr,Bte,cols,W,kk if g["hz"] else 5)
        r=dict(style=tag,k=kk,season=int(y),kind="BLIND")
        for m in models: r[m]=Vt.score(dict(zip(d0.pid,P[m])),y,kk,f"{tag}:{m}","evolve3")["ic"]
        s=Vt.score(dict(zip(d0.pid,stack(P,Bte,g,wts))),y,kk,f"{tag}:stack","evolve3"); r["stack"],r["draft"]=s["ic"],s["draft_ic"]; rows.append(r)
    return rows
def review_dump(g,wts,lineage):
    """Every 10 gens: walk-forward residuals of the champion for the LLM critic. Training-era outcomes only."""
    try:
        F=wf_folds(g); folds={}
        for y,(P,Bte,d) in F.items():
            s=stack(P[5],Bte,g,wts); dd=d.dropna(subset=["actual_pick"]).copy()
            dd["model_score"]=s[dd.index]; dd["true_war"]=cum(dd,5)
            dd["model_rank"]=dd["model_score"].rank(ascending=False).astype(int); dd["true_rank"]=dd["true_war"].rank(ascending=False).astype(int)
            dd["league_pick"]=dd["actual_pick"].astype(int)
            cols=["pid","league_pick","true_rank","model_rank","true_war"]
            worst_missed=dd.sort_values("true_rank").head(12)[cols]           # the real good picks: where did they fall?
            worst_over=dd.sort_values("model_rank").head(12)[cols]            # the model's top: were they good?
            folds[str(y)]=dict(ic=float(spearmanr(dd["model_score"],dd["true_war"]).statistic),
                               top_true=worst_missed.to_dict("records"),top_model=worst_over.to_dict("records"),
                               coverage={r.pid:round(float(c),2) for r,c in zip(dd.itertuples(),n_(dd,"feat_coverage").fillna(0).values)})
        json.dump(dict(gen=lineage["gen"],champ=g,fit=lineage["fit"],recent=lineage["history"][-10:],genes={k:v for k,v in GENES.items()},folds=folds,
                       ts=time.strftime("%Y-%m-%d %H:%M:%S")),open(f"{R7}/review_request.json","w"),indent=1)
        print("   >> review_request.json written for the critic",flush=True)
    except Exception as e: print("review dump failed",repr(e)[:100],flush=True)
def next_suggestion(tried):
    """Consume the critic's suggestions (gene deltas) before random mutation."""
    fp=f"{R7}/suggestions.json"
    if not os.path.exists(fp): return None
    try: S_=json.load(open(fp))
    except Exception: return None
    for d in S_.get("deltas",[]):
        ok={k:v for k,v in d.items() if k in GENES and v in GENES[k]}
        if ok and ok!={} and json.dumps(ok,sort_keys=True) not in S_.get("consumed",[]):
            S_.setdefault("consumed",[]).append(json.dumps(ok,sort_keys=True)); json.dump(S_,open(fp,"w"),indent=1); return ok
    return None
def gname(g): return "/".join(f"{k}={g[k]}" for k in GENES)
def neighbors(champ):
    names=set()
    for k in GENES:
        for v in GENES[k]:
            if v!=champ[k]: g=dict(champ); g[k]=v; names.add(gname(g))
    for k1,k2 in itertools.combinations(GENES,2):
        for v1 in GENES[k1]:
            for v2 in GENES[k2]:
                if v1!=champ[k1] and v2!=champ[k2]: g=dict(champ); g[k1]=v1; g[k2]=v2; names.add(gname(g))
    return names
def nat(v): return int(v) if isinstance(v,np.integer) else (float(v) if isinstance(v,np.floating) else (str(v) if isinstance(v,np.str_) else v))
lineage=json.load(open(LIN)) if os.path.exists(LIN) else None
if lineage and "icl_norm" not in lineage["champ"]:
    lineage["champ"].update({"icl_norm":"default","icl_outlier":4.0,"icl_shuffle":"latin"})
    for h in lineage["history"]:
        if "/icl_norm=" not in h["name"]: h["name"]=h["name"]+"/icl_norm=default/icl_outlier=4.0/icl_shuffle=latin"
    json.dump(lineage,open(LIN,"w"),indent=1)
if lineage and "pss" not in lineage["champ"]:
    lineage["champ"].update({"resid":0,"knn":0,"cat":0,"ixc":0,"pss":0})
    for h in lineage["history"]:
        if "/pss=" not in h["name"]: h["name"]=h["name"]+"/resid=0/knn=0/cat=0/ixc=0/pss=0"
    json.dump(lineage,open(LIN,"w"),indent=1)
if lineage and "icl_ctx" not in lineage["champ"]:
    lineage["champ"].update({"ridge_alpha":30.0,"icl_topk":0,"icl_ctx":"all"})
    for h in lineage["history"]:
        if "/ridge_alpha=" not in h["name"]: h["name"]=h["name"]+"/ridge_alpha=30.0/icl_topk=0/icl_ctx=all"
    json.dump(lineage,open(LIN,"w"),indent=1)
if lineage and "wk" not in lineage["champ"]:
    lineage["champ"]["wk"]=0
    for h in lineage["history"]:
        if "/wk=" not in h["name"]: h["name"]=h["name"].replace("/labelmix=","/wk=0/labelmix=")
    json.dump(lineage,open(LIN,"w"),indent=1)
rng=np.random.default_rng(int(time.time())%99991)
if lineage is None and os.path.exists(f"{R7}/seed.json"):
    champ=json.load(open(f"{R7}/seed.json"))["seed_champ"]
    for k,v in {"el":0,"wk":0,"ridge_alpha":30.0,"icl_topk":0,"icl_ctx":"all","resid":0,"knn":0,"cat":0,"ixc":0,"pss":0,"icl_norm":"default","icl_outlier":4.0,"icl_shuffle":"latin","labelmix":"single","hz":0,"covw":0,"risk":"mean","ridge":0,"icl_n":8}.items(): champ.setdefault(k,v)
    cf,wts=fitness(champ); rows=blind(champ,"EVO gen0 (BEST MODEL)",wts); bl=[r for r in rows if r["kind"]=="BLIND"]; e=[r["stack"]-r["draft"] for r in bl]
    lineage=dict(gen=0,champ=champ,fit=cf,wts=wts,history=[dict(gen=0,name=gname(champ),mean=float(np.mean(list(cf.values()))),accepted=True)])
    json.dump(lineage,open(LIN,"w"),indent=1); json.dump({"styles":{"EVO gen0 (BEST MODEL)":dict(weights=wts,rows=rows,secs=0,phase="EVO3d",genome=champ)}},open(OUT,"w"),indent=1)
    print(f"gen0 (seeded from BEST MODEL) WF {np.mean(list(cf.values())):+.4f} blind edge {np.mean(e):+.3f} 2019 {e[0]:+.3f} 2020 {e[1]:+.3f} ({sum(x>=0.2 for x in e)}/7)",flush=True)
if lineage is None:
    champ={"hw":"disc85","clip":"clip40","label":"gaussrank","fx":"base","M":400.0,"thin":"none","anchor":"none","cons":0.0,"intl":1,"bag":3,"mono":"on","hurdle":0,
           "labelmix":"single","hz":0,"covw":0,"risk":"mean","ridge":0,"icl_n":8}
    cf,wts=fitness(champ); rows=blind(champ,"EVO3 gen0",wts); bl=[r for r in rows if r["kind"]=="BLIND"]; e=[r["stack"]-r["draft"] for r in bl]
    lineage=dict(gen=0,champ=champ,fit=cf,wts=wts,history=[dict(gen=0,name=gname(champ),mean=float(np.mean(list(cf.values()))),accepted=True)])
    json.dump(lineage,open(LIN,"w"),indent=1); json.dump({"styles":{"EVO3 gen0":dict(weights=wts,rows=rows,secs=0,phase="EVO3",genome=champ)}},open(OUT,"w"),indent=1)
    print(f"gen0 (v2 gen17 seed, horizon-averaged fitness) WF {np.mean(list(cf.values())):+.4f} blind edge {np.mean(e):+.3f} 2019 {e[0]:+.3f} 2020 {e[1]:+.3f} ({sum(x>=0.2 for x in e)}/7)",flush=True)
res=json.load(open(OUT)); rejects=0
print("== EVOLUTION v3 == ledger",Vt.ledger_count(),flush=True)
NEW=["icl_norm","icl_outlier","icl_shuffle","resid","knn","cat","ixc","pss","ridge_alpha","icl_topk","icl_ctx","el","wk","labelmix","hz","covw","risk","ridge","icl_n"]
while True:
    champ,cf=lineage["champ"],lineage["fit"]; tried={h["name"] for h in lineage["history"]}
    if not(neighbors(champ)-tried): print("CONVERGED",gname(champ),"\nDONE_EVO3",flush=True); json.dump(dict(champion=champ,wts=lineage["wts"],fit=cf),open(f"{R7}/FINAL_CHAMPION.json","w"),indent=1); break
    g=dict(champ); nmut=1 if rejects<8 else 2
    sug=next_suggestion(tried)
    if sug:
        g.update(sug); print(f"   (critic suggestion) {sug}",flush=True)
    else:
        pool=NEW if (rng.random()<0.6 and rejects<12) else list(GENES)
        for k in rng.choice(pool,size=min(nmut,len(pool)),replace=False):
            opts=[v for v in GENES[str(k)] if v!=champ[str(k)]]; g[str(k)]=nat(opts[int(rng.integers(len(opts)))])
    name=gname(g)
    if name in tried: rejects=max(rejects,8); continue
    t0=time.time()
    try: mf,wts=fitness(g)
    except AssertionError as e: print("GUARD BLOCKED",e,flush=True); continue
    except Exception as e: print("fail",repr(e)[:120],"|",name[-90:],flush=True); continue
    gain=float(np.mean(list(mf.values()))-np.mean(list(cf.values()))); wins=sum(mf[y]>cf[y] for y in mf); acc=(gain>=MIN_GAIN and wins>=MIN_FOLD_WINS)
    lineage["gen"]+=1; gen=lineage["gen"]; short="/".join(f"{k}={g[k]}" for k in NEW if g[k]!=champ[k]) or "/".join(f"{k}={g[k]}" for k in GENES if g[k]!=champ[k])
    lineage["history"].append(dict(gen=gen,name=name,mean=float(np.mean(list(mf.values()))),gain=round(gain,4),fold_wins=int(wins),accepted=bool(acc),secs=round(time.time()-t0)))
    open(RUNS,"a").write(json.dumps(lineage["history"][-1])+"\n")
    print(f"[{time.strftime('%H:%M:%S')}] gen{gen:<3} {'KEEP' if acc else 'die '} WF {np.mean(list(mf.values())):+.4f} gain {gain:+.4f} folds {wins}/{len(mf)}  mut: {short}  ({time.time()-t0:.0f}s)",flush=True)
    if acc:
        rejects=0; lineage["champ"]=g; lineage["fit"]=mf; lineage["wts"]=wts
        rows=blind(g,f"EVO gen{gen}",wts); bl=[r for r in rows if r["kind"]=="BLIND"]; e=[r["stack"]-r["draft"] for r in bl]
        res["styles"][f"EVO gen{gen}"]=dict(weights=wts,rows=rows,secs=0,phase="EVO3",genome=g); json.dump(res,open(OUT,"w"),indent=1)
        print(f"     -> NEW CHAMPION: blind edge {np.mean(e):+.3f} | 2019 {e[0]:+.3f} 2020 {e[1]:+.3f} | {sum(x>=0.2 for x in e)}/7 >=.2 | ledger {Vt.ledger_count()}",flush=True)
    else: rejects+=1
    json.dump(lineage,open(LIN,"w"),indent=1)
    if gen%10==0: review_dump(lineage["champ"],lineage["wts"],lineage)
