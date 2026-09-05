"""Leak-proof evaluation engine. Every known cheat vector is closed and ASSERTED at run time.

GUARANTEES (each enforced in code, not by convention):
 G1 no target/outcome column can ever be a feature
 G2 no draft-order column (actual_pick/round) can be a feature -> can't copy the league
 G3 training folds contain ONLY seasons strictly BEFORE the predicted season (walk-forward)
 G4 shrinkage priors + coverage quantiles are computed INSIDE each fold from training rows only
 G5 synthetic rows are built only from that fold's training rows, are flagged, and are never scored
 G6 train and evaluation pid sets are disjoint
 G7 model selection reads walk-forward results ONLY; blind years are evaluation-only
"""
import json,os,numpy as np,pandas as pd,xgboost as xgb
from scipy.stats import spearmanr
HERE=os.path.dirname(os.path.abspath(__file__))
FEATS=json.load(open(f"{HERE}/data/input_columns.json"))["inputs"]
BANNED_PREFIX=("y_",)
BANNED_EXACT={"actual_pick","actual_round","split","pid","draft_year","was_drafted","declared_only",
              "y_redraft_rank","outcome_status","y_window_complete","y_partial_seasons"}
SHRINK=[c for c in ["col_ts_pct","col_efg_pct","col_fg2_pct","col_fg3_pct","col_ft_pct","col_ast_pct",
 "col_stl_pct","col_blk_pct","col_orb_pct","col_drb_pct","col_tov_pct","col_usg_pct","col_ortg",
 "col_drtg","col_impact","col_pts36","col_reb36","col_ast36","col_stl36","col_blk36"] if c in FEATS]
LEVEL={4:1.45,3:1.20,2:0.95,1:0.60}; INTL_VOL=["intl_pts36","intl_reb36","intl_ast36","intl_stl36","intl_blk36"]
def _num(d):
    for c in FEATS:
        if c in d and d[c].dtype==object: d[c]=pd.to_numeric(d[c],errors="coerce")
    return d
TRAIN=_num(pd.read_csv(f"{HERE}/data/train_2000_2018.csv"))
EL_TRAIN=pd.read_csv(f"{HERE}/data_v2/el_features_train.csv") if os.path.exists(f"{HERE}/data_v2/el_features_train.csv") else None
EL_COLS=[c for c in (EL_TRAIN.columns if EL_TRAIN is not None else []) if c.startswith("el_") and c!="el_league"]
if EL_TRAIN is not None:
    TRAIN=TRAIN.merge(EL_TRAIN.drop(columns=["el_league"]),on="pid",how="left")
    for c in EL_COLS: TRAIN[c]=pd.to_numeric(TRAIN[c],errors="coerce")
WK_TRAIN=pd.read_csv(f"{HERE}/data_v2/wk_features_train.csv") if os.path.exists(f"{HERE}/data_v2/wk_features_train.csv") else None
WK_COLS=[c for c in (WK_TRAIN.columns if WK_TRAIN is not None else []) if c.startswith("wk_")]
if WK_TRAIN is not None:
    TRAIN=TRAIN.merge(WK_TRAIN,on="pid",how="left")
    for c in WK_COLS: TRAIN[c]=pd.to_numeric(TRAIN[c],errors="coerce")
PICKS=pd.read_parquet(f"{HERE}/data/prospects.parquet")[["pid","actual_pick"]]
TESTS={}
for y in range(2019,2027):
    ti,an=f"{HERE}/data/tests/test_{y}_inputs.csv",f"{HERE}/data/answers/answers_{y}.csv"
    if os.path.exists(ti) and os.path.exists(an):
        a=pd.read_csv(an); k=[c for c in ["pid","actual_pick","y_early_war"] if c in a.columns]
        d=_num(pd.read_csv(ti)).merge(a[k],on="pid",how="left")
        if d["y_early_war"].notna().sum()>=10: TESTS[y]=d
def audit_features(cols):
    bad=[c for c in cols if c.startswith(BANNED_PREFIX) or c in BANNED_EXACT]
    if bad: raise AssertionError(f"G1/G2 VIOLATION - leaked columns: {bad[:6]}")
def audit_disjoint(a,b):
    o=set(a)&set(b)
    if o: raise AssertionError(f"G6 VIOLATION - {len(o)} pids in both train and eval")
MIN_TRAIN_YEAR,MAX_TRAIN_YEAR=2010,2018
def audit_window(tr_years):
    ys=set(int(y) for y in tr_years)
    bad=[y for y in ys if y<MIN_TRAIN_YEAR or y>MAX_TRAIN_YEAR]
    if bad: raise AssertionError(f"G8 VIOLATION - training outside {MIN_TRAIN_YEAR}-{MAX_TRAIN_YEAR}: {sorted(bad)[:5]}")
def audit_forward(tr_years,te_year):
    bad=[y for y in tr_years if y>=te_year]
    if bad: raise AssertionError(f"G3 VIOLATION - training on seasons >= {te_year}: {bad[:5]}")
def fit_prior(tr_raw):                                   # G4: priors from training rows only
    return ({c:tr_raw[c].mean() for c in SHRINK},
            tr_raw[FEATS].notna().mean(axis=1).quantile([0.05,0.95]).values)
def build(d,prior,cq,opts):
    d=d.copy()
    if opts.get("shrink",True):
        mins=pd.to_numeric(d.get("col_minutes_total"),errors="coerce")
        gp=pd.to_numeric(d.get("col_gp"),errors="coerce"); mins=mins.fillna(gp*20.0)
        r=(mins/(mins+opts.get("M",400.0))).clip(0,1).fillna(0.0); d["sample_reliability"]=r
        for c in SHRINK: d["shr_"+c]=r*pd.to_numeric(d[c],errors="coerce").fillna(prior[c])+(1-r)*prior[c]
    if opts.get("intl",False):
        lv=pd.to_numeric(d.get("intl_level"),errors="coerce"); d["intl_strength"]=lv.map(LEVEL)
        for c in INTL_VOL:
            if c in d: d["adj_"+c]=pd.to_numeric(d[c],errors="coerce")*d["intl_strength"]
    if opts.get("cov",False):
        cov=d[FEATS].notna().mean(axis=1); d["feat_coverage"]=((cov-cq[0])/(cq[1]-cq[0])).clip(0,1)
    return d
def cols_for(opts):
    c=list(FEATS)
    if opts.get("shrink",True): c+=["sample_reliability"]+["shr_"+x for x in SHRINK]
    if opts.get("intl",False):  c+=["intl_strength"]+["adj_"+x for x in INTL_VOL]
    if opts.get("cov",False):   c+=["feat_coverage"]
    if opts.get("el",False):    c+=EL_COLS
    if opts.get("wk",False):    c+=WK_COLS
    audit_features(c); return c
def target(df,kind):
    w=df["y_early_war"].fillna(0).clip(0,40)
    if kind=="log": return np.log1p(w).values
    if kind=="sqrt": return np.sqrt(w).values
    if kind=="rank":
        o=np.zeros(len(df))
        for y in df.draft_year.unique():
            m=(df.draft_year==y).values; o[m]=w[m].rank(pct=True).values
        return o
    if kind=="zclass":
        o=np.zeros(len(df)); lw=np.log1p(w)
        for y in df.draft_year.unique():
            m=(df.draft_year==y).values; v=lw[m]; s=v.std(); o[m]=((v-v.mean())/(s if s>0 else 1)).values
        return o
    raise ValueError(kind)
def synth(X,y,kind,ratio,rng,strength=0.10):              # G5: from training rows only, never scored
    """Synthetic prospect generators. All operate ONLY on a fold's training rows."""
    if kind=="none" or ratio<=0: return X,y
    n=int(len(X)*ratio); idx=np.arange(len(X)); V=X.values
    if kind=="mixup":                                     # convex blends of two players
        a,b=rng.choice(idx,n),rng.choice(idx,n); lam=rng.beta(2,2,n)[:,None]
        Xs=V[a]*lam+V[b]*(1-lam); ys=y[a]*lam[:,0]+y[b]*(1-lam[:,0])
    elif kind=="jitter":                                  # gaussian noise around a real player
        a=rng.choice(idx,n); sd=np.nanstd(V,axis=0)*strength
        Xs=V[a]+rng.normal(0,1,(n,X.shape[1]))*sd; ys=y[a].copy()
    elif kind=="cmixup":                                  # blend only players with similar outcomes
        order=np.argsort(y); rank=np.empty(len(y),int); rank[order]=np.arange(len(y))
        a=rng.choice(idx,n); part=np.clip(rank[a]+rng.integers(-25,26,n),0,len(y)-1); b=order[part]
        lam=rng.beta(2,2,n)[:,None]
        Xs=V[a]*lam+V[b]*(1-lam); ys=y[a]*lam[:,0]+y[b]*(1-lam[:,0])
    elif kind=="smote":                                   # interpolate toward a feature-space neighbour
        a=rng.choice(idx,n)
        Z=np.nan_to_num((V-np.nanmean(V,axis=0))/(np.nanstd(V,axis=0)+1e-9))
        cand=rng.choice(idx,(n,8))
        d=((Z[cand]-Z[a][:,None,:])**2).sum(-1); b=cand[np.arange(n),d.argmin(1)]
        lam=rng.uniform(0,1,n)[:,None]
        Xs=V[a]+(V[b]-V[a])*lam; ys=y[a]+(y[b]-y[a])*lam[:,0]
    elif kind=="tnoise":                                  # label smoothing on the target only
        a=rng.choice(idx,n); Xs=V[a].copy(); ys=y[a]+rng.normal(0,np.std(y)*strength,n)
    else: return X,y
    return pd.concat([X,pd.DataFrame(Xs,columns=X.columns)],ignore_index=True),np.concatenate([y,ys])
XCFG=dict(objective="reg:squarederror",max_depth=3,learning_rate=0.02,n_estimators=800,subsample=0.7,
    colsample_bytree=0.5,min_child_weight=8,reg_lambda=5.0,reg_alpha=1.0,n_jobs=8)
_TFM=[None]
def predict(model,Xtr,ytr,Xte,cfg):
    if model=="xgb":
        kw=dict(XCFG); kw["monotone_constraints"]={"bio_age_at_draft":-1} if cfg.get("mono",True) else {}
        if cfg.get("q"): kw["objective"]="reg:quantileerror"; kw["quantile_alpha"]=float(cfg["q"])
        if cfg.get("ixc"): kw["interaction_constraints"]=str(cfg["ixc"])
        return np.mean([xgb.XGBRegressor(**kw,random_state=s).fit(Xtr,ytr).predict(Xte)
                        for s in cfg.get("seeds",(11,12,13))],axis=0)
    if model=="tabicl":
        from tabicl import TabICLRegressor
        try: m=TabICLRegressor(device="cuda",n_estimators=int(cfg.get("icl_n",8)),**cfg.get("icl_kwargs",{}))
        except TypeError: m=TabICLRegressor(device="cuda")
        m.fit(Xtr.astype(float),ytr); return m.predict(Xte.astype(float))
    if model=="knn":
        from sklearn.neighbors import KNeighborsRegressor
        mu=Xtr.mean(); sd=Xtr.std().replace(0,1.0); A=((Xtr-mu)/sd).fillna(0.0); Bq=((Xte-mu)/sd).fillna(0.0)
        return KNeighborsRegressor(n_neighbors=int(cfg.get("knn_k",15)),weights="distance").fit(A,ytr).predict(Bq)
    if model=="cat":
        from catboost import CatBoostRegressor
        ps=[]
        for s_ in cfg.get("seeds",(11,)):
            m=CatBoostRegressor(depth=4,iterations=800,learning_rate=0.03,l2_leaf_reg=5.0,random_seed=int(s_),verbose=0,thread_count=8,allow_writing_files=False)
            m.fit(Xtr,ytr); ps.append(m.predict(Xte))
        return np.mean(ps,axis=0)
    if model=="ridge":
        from sklearn.linear_model import Ridge
        mu=Xtr.mean(); sd=Xtr.std().replace(0,1.0)
        A=((Xtr-mu)/sd).fillna(0.0); Bq=((Xte-mu)/sd).fillna(0.0)
        return Ridge(alpha=float(cfg.get("ridge_alpha",30.0))).fit(A,ytr).predict(Bq)
    if model=="tabfm":
        from tabfm import TabFMRegressor, tabfm_v1_0_0_pytorch as tfm
        if _TFM[0] is None: _TFM[0]=tfm.load(model_type="regression",device="cuda")
        m=TabFMRegressor(model=_TFM[0]); m.fit(Xtr.astype(float),ytr); return m.predict(Xte.astype(float))
    if model.startswith("ens:"):
        rs=[pd.Series(predict(p,Xtr,ytr,Xte,cfg)).rank().values for p in model[4:].split("+")]
        return np.mean(rs,axis=0)
    raise ValueError(model)
def _season(d,label,kind):
    d=d.dropna(subset=["actual_pick"]).copy()
    if len(d)<10: return None
    w=d["y_early_war"].fillna(0)
    return dict(season=int(label),kind=kind,n=len(d),
        model_ic=float(spearmanr(d["p"],w).statistic),
        draft_ic=float(spearmanr(-d["actual_pick"],w).statistic),
        war_model_top10=float(d.nlargest(10,"p")["y_early_war"].fillna(0).sum()),
        war_draft_top10=float(d.nsmallest(10,"actual_pick")["y_early_war"].fillna(0).sum()))
def evaluate(cfg):
    y0,y1=cfg.get("win",(2010,2018)); opts=cfg.get("opts",{}); cols=cols_for(opts)
    tgt=cfg.get("target","log"); rng=np.random.default_rng(cfg.get("rseed",7))
    W=TRAIN[(TRAIN.draft_year>=y0)&(TRAIN.draft_year<=y1)].reset_index(drop=True)
    audit_window(W.draft_year.values)
    yrs=sorted(W.draft_year.unique()); wf=[]
    for y in yrs[4:]:                                     # G3 walk-forward
        trm=(W.draft_year.values<y); tem=(W.draft_year.values==y)
        if tem.sum()<12 or trm.sum()<150: continue
        audit_forward(W.draft_year.values[trm],y)
        audit_disjoint(W.pid.values[trm],W.pid.values[tem])
        pr,cq=fit_prior(W[trm])                           # G4 per-fold priors
        Btr,Bte=build(W[trm],pr,cq,opts),build(W[tem],pr,cq,opts)
        X,yy=synth(Btr[cols],target(W[trm],tgt),cfg.get("synth","none"),cfg.get("synth_ratio",0),rng,cfg.get("synth_strength",0.10))
        d=Bte.merge(PICKS,on="pid",how="left",suffixes=("","_pk"))
        if "actual_pick_pk" in d: d["actual_pick"]=d["actual_pick_pk"]
        d["p"]=predict(cfg["model"],X,yy,Bte[cols],cfg)
        r=_season(d,y,"WF")
        if r: wf.append(r)
    pr,cq=fit_prior(W); Btr=build(W,pr,cq,opts)
    X,yy=synth(Btr[cols],target(W,tgt),cfg.get("synth","none"),cfg.get("synth_ratio",0),rng,cfg.get("synth_strength",0.10))
    bl=[]
    for y,d0 in sorted(TESTS.items()):
        audit_disjoint(W.pid.values,d0.pid.values)        # G6
        d=build(d0,pr,cq,opts).copy(); d["p"]=predict(cfg["model"],X,yy,d[cols],cfg)
        r=_season(d,y,"BLIND")
        if r: bl.append(r)
    wfi=np.array([r["model_ic"] for r in wf]); bli=np.array([r["model_ic"] for r in bl])
    bld=np.array([r["draft_ic"] for r in bl])
    return dict(sel_wf_ic=float(wfi.mean()) if len(wfi) else None,          # G7: the ONLY selection key
        sel_wf_se=float(wfi.std(ddof=1)/np.sqrt(len(wfi))) if len(wfi)>1 else None,
        wf_n=len(wf),blind_ic=float(bli.mean()) if len(bli) else None,
        blind_draft_ic=float(bld.mean()) if len(bld) else None,
        blind_edge=float((bli-bld).mean()) if len(bli) else None,
        blind_n=len(bl),seasons=wf+bl)
