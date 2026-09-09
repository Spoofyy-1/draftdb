"""Walk-forward blend score for engine2 variants: per-class Spearman of the blend score vs y_early_war (classes 2012-2018, drafted rows),
plain mean and recency-weighted mean (weights 1..7). usage: python3 wf_score.py <predictions_walkforward csv> [more csvs]"""
import sys,pandas as pd,numpy as np
from scipy.stats import spearmanr
tr=pd.read_csv("/Users/kennakao/nba/datarebuild/v4_build/staging_v419/train_2000_2018.csv",usecols=["pid","draft_year","was_drafted","y_early_war"])
lab=tr[tr.was_drafted==1]
for f in sys.argv[1:]:
    d=pd.read_csv(f); d=d[d.cutoff=="causal"] if "cutoff" in d else d; m=d.merge(lab,on=["pid","draft_year"]); out={}
    for y,g in m.groupby("draft_year"):
        if len(g)>=12: out[int(y)]=spearmanr(g.score,g.y_early_war).correlation
    ys=sorted(out); w=np.array([y-2011 for y in ys],float)
    print(f"{f.split('/')[-1]:<40} mean {np.mean([out[y] for y in ys]):+.4f} wmean {np.average([out[y] for y in ys],weights=w):+.4f} | "+" ".join(f"{y%100:02d}:{out[y]:+.3f}" for y in ys))
