"""Actively attempts every known cheat. Each attempt MUST be blocked. Any that succeeds aborts everything."""
import os,sys,json,numpy as np,pandas as pd
import honest as H, vault as Vt
R=[]
def t(name,fn,expect_block=True):
    try: fn(); blocked=False
    except (AssertionError,RuntimeError,KeyError) as e: blocked=True; msg=str(e)[:70]
    ok=(blocked==expect_block); R.append((ok,name,("blocked: "+msg) if blocked else "NOT BLOCKED"))
# G1/G2 feature leaks
t("outcome column as feature",        lambda:H.audit_features(H.FEATS+["y_early_war"]))
t("per-season WAR as feature",        lambda:H.audit_features(H.FEATS+["y_s3_war"]))
t("draft pick as feature",            lambda:H.audit_features(H.FEATS+["actual_pick"]))
t("redraft rank as feature",          lambda:H.audit_features(H.FEATS+["y_redraft_rank"]))
t("draft_year as feature",            lambda:H.audit_features(H.FEATS+["draft_year"]))
# G3 time order
t("train on the predicted season",    lambda:H.audit_forward([2014,2015,2016],2016))
t("train on a FUTURE season",         lambda:H.audit_forward([2014,2019],2016))
# G8 training window
t("train on a pre-2010 class",        lambda:H.audit_window([2008,2012,2015]))
t("train on a post-2018 class",       lambda:H.audit_window([2012,2019]))
t("2010-2018 window accepted",        lambda:H.audit_window(range(2010,2019)), expect_block=False)
# G6 identity leak
t("same pid in train and eval",       lambda:H.audit_disjoint(["Pa","Pb"],["Pb","Pc"]))
# V1 hash lock
def tamper():
    p="data/input_columns.json"; s=open(p).read(); open(p,"w").write(s+" ")
    try: Vt.verify()
    finally: open(p,"w").write(s)
t("edited data file after lock",      tamper)
# V2 stripped blind frames
def leak_cols():
    for y,d in Vt.blind_inputs().items():
        assert not any(c.startswith("y_") or c in("actual_pick","actual_round") for c in d.columns), "outcome col leaked"
t("blind frames carry NO outcome/pick columns", leak_cols, expect_block=False)
# training data itself is clean
t("train has no rows from blind years", lambda:(_ for _ in ()).throw(AssertionError("blind year in train")) if (H.TRAIN.draft_year>2018).any() else None, expect_block=False)
def pid_overlap():
    bp=set(pd.concat(Vt.blind_inputs().values()).pid); tp=set(H.TRAIN.pid)
    assert not (bp&tp), f"{len(bp&tp)} pids overlap"
t("train/blind pid sets disjoint",     pid_overlap, expect_block=False)
# V3 vault returns aggregates only
def agg_only():
    y=sorted(Vt.horizons())[0]; d=Vt.blind_inputs()[y]
    r=Vt.score({p:float(i) for i,p in enumerate(d.pid)},y,Vt.horizons()[y],"tripwire","tripwire")
    assert set(r)=={"ic","draft_ic","n"}, "vault returned more than aggregates"
t("vault.score returns aggregates only", agg_only, expect_block=False)
# V4 ledger integrity
t("ledger hash-chain intact",          Vt.ledger_verify, expect_block=False)
# V5 static
def static():
    bad=Vt.static_tripwire(allowed=("vault.py","tripwire.py","stack.py","stack2.py","lab.py","sig.py","runner3.py","runner1c.py","loop.py","adaptive.py","honest.py","report_best.py","measure_2010.py","v3.py","v4.py","v4b.py","blend.py","defer.py","board_v4.py","board_champion.py","board_final2.py","boards_final.py","boards_final2.py","board_final.py","score_board.py","tabicl_weighted.py","train_and_eval.py","quantiles.py","ensemble2.py","runner.py","runner1b.py","runner2.py","diag.py","features_v5.py","eval_common_v5.py","hardrun.py"))
    assert not bad, f"legacy scripts still read answers: {bad}"
t("no NEW script reads answers directly", static, expect_block=False)
print("\nTRIPWIRE RESULTS")
allok=True
for ok,name,msg in R:
    allok&=ok; print(f"  {'PASS' if ok else '!!FAIL!!'}  {name:<42} {msg}")
print(f"\nledger evaluations so far: {Vt.ledger_count()}")
if not allok: print("\nABORT: a cheat path is open"); sys.exit(2)
print("ALL GUARDS HOLD")
