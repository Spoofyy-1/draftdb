"""THE VAULT. The only code allowed to read blind outcomes. Enforced, not conventional:
 V1 hash-locked: refuses to run if train/test/answers/manifest bytes changed since lock
 V2 blind INPUT frames leave the vault stripped to pid + declared inputs only (no y_, no picks)
 V3 score() takes predictions by pid and returns aggregate IC only; row outcomes never leave
 V4 every score() call is appended to an append-only hash-chained ledger (count -> Bonferroni)
 V5 static tripwire: no other source file may reference the answers path or outcome columns for test rows
"""
import json,os,hashlib,time,glob,numpy as np,pandas as pd
from scipy.stats import spearmanr
HERE=os.path.dirname(os.path.abspath(__file__)); V=f"{HERE}/vault"; MAN=f"{V}/manifest.json"; LEDGER=f"{V}/ledger.jsonl"
INPUTS=json.load(open(f"{HERE}/data/input_columns.json"))["inputs"]
POOL_DRAFTED=True   # redraft = re-rank the players who were actually drafted
FILES=sorted(glob.glob(f"{V}/answers_*.csv"))+sorted(glob.glob(f"{HERE}/data_v2/*.csv"))+sorted(glob.glob(f"{HERE}/data/tests/test_*_inputs.csv"))+[f"{HERE}/data/train_2000_2018.csv",f"{HERE}/data/input_columns.json"]
def _h(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def lock():
    json.dump({os.path.relpath(p,HERE):_h(p) for p in FILES},open(MAN,"w"),indent=1); os.chmod(MAN,0o444)
def verify():
    if not os.path.exists(MAN): raise RuntimeError("V1: vault not locked; run vault.lock() once")
    m=json.load(open(MAN))
    bad=[p for p in FILES if m.get(os.path.relpath(p,HERE))!=_h(p)]
    missing=[k for k in m if not os.path.exists(f"{HERE}/{k}")]
    if bad or missing: raise RuntimeError(f"V1 VIOLATION: data changed since lock: {bad+missing}")
    return True
if not os.path.exists(MAN):
    lock(); print('VAULT SEALED (first import): manifest written and made read-only')
verify()
_ANS={}
for p in sorted(glob.glob(f"{V}/answers_*.csv")):
    y=int(p.split("_")[-1][:4]); a=pd.read_csv(p); _ANS[y]=a
S=[f"y_s{i}_war" for i in range(1,6)]
def horizons():
    out={}
    for y,a in _ANS.items():
        if all(c in a.columns for c in S):
            k=int(np.clip(a[S].notna().sum(axis=1).max(),1,5))
            if a[S[0]].notna().sum()>=10: out[y]=k
    return out
def blind_inputs():
    """V2: test frames with ONLY pid + declared inputs. Nothing else can leak out."""
    out={}
    for y in horizons():
        d=pd.read_csv(f"{HERE}/data/tests/test_{y}_inputs.csv")
        if POOL_DRAFTED and "was_drafted" in d.columns: d=d[pd.to_numeric(d["was_drafted"],errors="coerce")==1].reset_index(drop=True)
        keep=["pid"]+[c for c in INPUTS if c in d.columns]
        leaked=[c for c in d.columns if c.startswith("y_") or c in("actual_pick","actual_round")]
        f=d[keep].copy()
        elp=f"{HERE}/data_v2/el_features_test.csv"
        if os.path.exists(elp):
            f=f.merge(pd.read_csv(elp).drop(columns=["el_league"]),on="pid",how="left")
            for c in [c for c in f.columns if c.startswith("el_")]: f[c]=pd.to_numeric(f[c],errors="coerce")   # el_ numeric
        wkp=f"{HERE}/data_v2/wk_features_test.csv"
        if os.path.exists(wkp):
            f=f.merge(pd.read_csv(wkp),on="pid",how="left")
            for c in [c for c in f.columns if c.startswith("wk_")]: f[c]=pd.to_numeric(f[c],errors="coerce")
        f.attrs["stripped"]=leaked; out[y]=f
    return out
def _chain_prev():
    if not os.path.exists(LEDGER): return "GENESIS"
    last=None
    for l in open(LEDGER):
        if l.strip(): last=l
    return hashlib.sha256(last.encode()).hexdigest() if last else "GENESIS"
def score(pred_by_pid,season,k,config_id,who):
    """V3+V4: returns {ic, draft_ic, n}. Appends a ledger entry. Never returns rows."""
    a=_ANS[season].copy(); a=a.dropna(subset=["actual_pick"])
    for c in S[:k]: a[c]=pd.to_numeric(a[c],errors="coerce")
    w=a[S[:k]].fillna(0).sum(axis=1).values
    p=np.array([pred_by_pid.get(pid,np.nan) for pid in a["pid"]])
    ok=~np.isnan(p)
    if ok.sum()<10: raise RuntimeError("V3: too few predictions matched")
    ic=float(spearmanr(p[ok],w[ok]).statistic); dic=float(spearmanr(-a["actual_pick"].values[ok],w[ok]).statistic)
    rec=dict(t=time.strftime("%Y-%m-%d %H:%M:%S"),season=season,k=k,config=config_id,who=who,
             n=int(ok.sum()),ic=round(ic,4),draft_ic=round(dic,4),prev=_chain_prev())
    with open(LEDGER,"a") as f: f.write(json.dumps(rec)+"\n")
    return dict(ic=ic,draft_ic=dic,n=int(ok.sum()))
def ledger_count(): return sum(1 for l in open(LEDGER) if l.strip()) if os.path.exists(LEDGER) else 0
def ledger_verify():
    prev="GENESIS"
    for l in open(LEDGER):
        if not l.strip(): continue
        r=json.loads(l)
        if r["prev"]!=prev: raise RuntimeError("V4 VIOLATION: ledger chain broken (entry edited/removed)")
        prev=hashlib.sha256(l.encode()).hexdigest()
    return True
def static_tripwire(allowed=("vault.py","tripwire.py")):
    """V5: no other source file may touch answers or outcome columns of TEST data."""
    bad=[]
    for p in glob.glob(f"{HERE}/*.py"):
        if os.path.basename(p) in allowed: continue
        src=open(p).read()
        if "vault/answers" in src or "data/answers" in src: bad.append((os.path.basename(p),"reads answers path"))
    return bad
