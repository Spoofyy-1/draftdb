"""engine2 EXPANDING holdout (box only): for each holdout class y >= 2020, earlier test classes join the context with the NBA seasons
they had completed by draft night y (from the sealed vault's expanding_rows: calendar-correct, never class y or later), then his
layer-1 scores class y. 2019 is taken from the strict holdout run. usage (from ~/nba/engine2, venv active):
  python bridge/expanding_holdout.py <variant table parquet> <tag> [--ctx-start-catboost 2008]
Writes outputs/bridge/holdout_expanding_<tag>.csv (pid, draft_year, score) and restores data/processed afterwards."""
import sys,os,shutil,subprocess,json,numpy as np,pandas as pd
HAND="/home/ubuntu/nba/handoff"; sys.path.insert(0,HAND); os.chdir(HAND); import vault as Vt; os.chdir("/home/ubuntu/nba/engine2")
table_path,tag=sys.argv[1],sys.argv[2]; extra=sys.argv[3:]
PROC="data/processed"; base_t=pd.read_parquet(table_path); base_s=pd.read_parquet(f"{PROC}/season_war.parquet")
shutil.copy(table_path,f"{PROC}/draft_table.parquet"); shutil.copy(f"{PROC}/season_war.parquet",f"{PROC}/season_war_base.parquet")
preds=[]
try:
    # 2019: strict (no earlier test class has outcomes) -- run it on the base table
    for y in range(2019,2026):
        t=base_t.copy(); s=base_s.copy()
        if y>2019:
            X=Vt.expanding_rows(y); rows=[]
            wcols=[c for c in X.columns if c.startswith("y_s") and c.endswith("_war")]
            for pid,c,r in zip(X.pid.values,X.draft_year.values,X[wcols].values):
                known=[(int(wcols[i].split("_")[1][1:]),float(v)) for i,v in enumerate(r) if not np.isnan(v)]
                if not known: continue
                for i,v in known: rows.append(dict(bbref_id=pid,season=int(c)+i,war=v))
                m=t.bbref_id==pid
                if m.any():
                    t.loc[m,"labelled"]=True; t.loc[m,"war5"]=sum(v for _,v in known); t.loc[m,"seasons_played"]=len(known)
            s=pd.concat([s,pd.DataFrame(rows)],ignore_index=True).drop_duplicates(["bbref_id","season"],keep="last")
        t.to_parquet(f"{PROC}/draft_table.parquet",index=False); s.to_parquet(f"{PROC}/season_war.parquet",index=False)
        cmd=[sys.executable,"-m","bridge.run_winner","--window","holdout","--years",str(y),"--cutoff","causal","--device","cuda","--tag",f"exp_{tag}_{y}"]+extra
        print("[expanding]",y,"context test classes:",(0 if y==2019 else int(X.draft_year.nunique())),flush=True)
        subprocess.run(cmd,check=True,stdout=open(f"outputs/bridge/run_exp_{tag}_{y}.log","w"),stderr=subprocess.STDOUT)
        p=pd.read_csv("outputs/bridge/predictions_holdout.csv"); p=p[(p.cutoff=="causal")&(p.draft_year==y)][["pid","draft_year","score"]]; preds.append(p)
finally:
    shutil.copy(f"{PROC}/season_war_base.parquet",f"{PROC}/season_war.parquet"); shutil.copy(table_path,f"{PROC}/draft_table.parquet")
out=pd.concat(preds,ignore_index=True); out.to_csv(f"outputs/bridge/holdout_expanding_{tag}.csv",index=False); print("wrote",f"outputs/bridge/holdout_expanding_{tag}.csv",len(out),"rows")
