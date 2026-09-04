"""One experiment run: validation (causal + pooled), optional locked holdout, and an out-of-sample redraft of
every draft class that had at least two earlier classes to learn from on draft night.

Protocols (which draft classes the model may see as labelled context when scoring class Y):
  causal : classes < Y, labelled with the seasons played up to Y -- exactly what was knowable on draft night Y
  pooled : every labelled non-holdout class != Y with today's labels -- leave-one-year-out, non-causal upper bound

Every run is appended to outputs/ledger.jsonl; holdout evaluations are counted separately in
outputs/holdout_ledger.jsonl so repeated peeking stays visible.

Usage: python -m nbadraft.run [--models tabfm,lgbm,...] [--holdout] [--tag name]
"""

import argparse
import concurrent.futures as cf
import hashlib
import json
import multiprocessing as mp
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from nbadraft import config as C
from nbadraft.dataset import FEATURES
from nbadraft.models import MODELS, TARGET, fit_predict
from nbadraft.war import peak_war

TOP_K = (14, 30)
NORTH_STAR = "war_captured_pct@14"
SUMMARY_KEYS = ["spearman", "spearman_nba", "war_captured_pct@14", "war_captured_pct@30", "ndcg@14", "war_ours@14", "war_actual@14", "war_oracle@14"]


# --------------------------------------------------------------------------- splits

def context_years(y, protocol, labelled):
    if protocol == "causal":
        return [c for c in labelled if c < y]
    return [c for c in labelled if c != y and c not in C.HOLDOUT_YEARS]


def split_table():
    rows = []
    for y in C.DRAFT_YEARS:
        if y < C.FIRST_LABELED_DRAFT:
            role = "no_features"
        elif y in C.HOLDOUT_YEARS:
            role = "holdout"
        elif y in C.VAL_YEARS:
            role = "validation"
        elif y <= C.LAST_LABELED_DRAFT:
            role = "context"
        else:
            role = "unlabelled"
        rows.append({"year": y, "role": role, "labelled": y <= C.LAST_LABELED_DRAFT,
                     "causal_context": list(range(C.FIRST_LABELED_DRAFT, y))})
    return rows


# --------------------------------------------------------------------------- metrics

def _dcg(v):
    return float(np.sum(v / np.log2(np.arange(2, len(v) + 2))))


def year_metrics(df, score):
    """df: one draft class (modelled + labelled rows); score: higher = better prospect."""
    y = df[TARGET].values
    ours = y[np.argsort(-score, kind="stable")]
    actual = y[np.argsort(df["pick"].values, kind="stable")]
    oracle = np.sort(y)[::-1]
    # spearman_nba: how well the real draft order predicted realised WAR. Same scale as ours, so the pair answers
    # "did we get the order more right than the NBA did?". Pick is negated because pick 1 is the best slot.
    out = {"n": int(len(y)), "spearman": float(spearmanr(score, y).correlation),
           "spearman_nba": float(spearmanr(-df["pick"].values, y).correlation)}
    for k in TOP_K:
        k_ = min(k, len(y))
        o, a, best = ours[:k_].sum(), actual[:k_].sum(), oracle[:k_].sum()
        out[f"war_ours@{k}"], out[f"war_actual@{k}"], out[f"war_oracle@{k}"] = float(o), float(a), float(best)
        # share of the available improvement over the real draft that we captured: 100 = perfect, 0 = same as the NBA, <0 = worse
        out[f"war_captured_pct@{k}"] = float(100 * (o - a) / (best - a)) if best > a else 0.0
        out[f"ndcg@{k}"] = _dcg(np.clip(ours[:k_], 0, None)) / max(_dcg(np.clip(oracle[:k_], 0, None)), 1e-9)
    return out


# --------------------------------------------------------------------------- job pool

_TABLE, _SEASONS, _DEVICE = None, None, "cpu"


def _gpus():
    """Usable GPUs. device_count() still reports them when CUDA cannot initialise, so gate on is_available()."""
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def _n_workers():
    # One process per GPU. On CPU, TabFM on a few hundred rows barely scales past 4 threads, so favour processes over threads.
    return _gpus() or max(1, (os.cpu_count() or 8) // 4)


def _init(gpu_queue, threads):
    global _TABLE, _SEASONS, _DEVICE
    _TABLE = pd.read_parquet(C.PROC / "draft_table.parquet")
    _SEASONS = pd.read_parquet(C.PROC / "season_war.parquet")
    if _gpus():
        _DEVICE = f"cuda:{gpu_queue.get()}"
    else:
        _DEVICE = "cpu"
        torch.set_num_threads(threads)


def _job(key, ctx_years, seed):
    model, year, protocol = key
    t = _TABLE
    ctx = t[t.modelled & t.labelled & t.draft_year.isin(ctx_years)]
    if protocol == "causal":  # relabel with only the seasons that had been played by draft night `year`
        ctx = ctx.assign(**{TARGET: peak_war(ctx, _SEASONS, through=year)[TARGET].values})
    test = t[t.modelled & (t.draft_year == year)]
    t0 = time.time()
    score = fit_predict(model, ctx, test, device=_DEVICE, seed=seed)
    return key, test.bbref_id.tolist(), score.tolist(), len(ctx), time.time() - t0


def run_jobs(jobs, seed):
    """jobs: {(model, year, protocol): context_years}. One worker process per GPU, or several CPU workers when there is no usable GPU."""
    n_workers = _n_workers()
    threads = max(1, (os.cpu_count() or 8) // n_workers)
    ctx = mp.get_context("spawn")
    q = ctx.Manager().Queue()
    for g in range(n_workers):
        q.put(g)
    with cf.ProcessPoolExecutor(n_workers, mp_context=ctx, initializer=_init, initargs=(q, threads)) as ex:
        futs = [ex.submit(_job, k, cy, seed) for k, cy in jobs.items()]
        return {r[0]: r[1:] for r in (f.result() for f in cf.as_completed(futs))}


# --------------------------------------------------------------------------- run

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS), help="first model listed is used for the redrafts")
    ap.add_argument("--holdout", action="store_true", help="evaluate the locked holdout years (logged)")
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    models = a.models.split(",")
    redraft_model = models[0]

    table = pd.read_parquet(C.PROC / "draft_table.parquet")
    labelled = sorted(set(table.loc[table.modelled & table.labelled, "draft_year"]))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + a.tag

    jobs = {}
    for m in models:
        for y in C.VAL_YEARS:
            for p in ("causal", "pooled"):
                jobs[(m, y, p)] = context_years(y, p, labelled)
        if a.holdout:
            for y in C.HOLDOUT_YEARS:
                jobs[(m, y, "causal")] = context_years(y, "causal", labelled)
    # Redrafts are causal only: a class is redrafted iff at least two earlier classes had played by its draft night.
    redraft_years = []
    for y in C.DRAFT_YEARS:
        cy = context_years(y, "causal", labelled)
        if len(cy) >= 2 and (table.modelled & (table.draft_year == y)).any():
            redraft_years.append(y)
            jobs.setdefault((redraft_model, y, "causal"), cy)

    t0 = time.time()
    res = run_jobs(jobs, a.seed)
    gpus = _gpus()
    print(f"{len(jobs)} jobs, {_n_workers()} {'GPU' if gpus else 'CPU'} worker(s), {time.time() - t0:.0f}s")

    metrics = []
    for (m, y, p), (ids, score, n_ctx, secs) in res.items():
        if y in C.HOLDOUT_YEARS and not a.holdout:
            continue
        split = "holdout" if y in C.HOLDOUT_YEARS else "validation" if y in C.VAL_YEARS else None
        if split is None:
            continue
        df = table.set_index("bbref_id").loc[ids].reset_index()
        metrics.append({"split": split, "protocol": p, "model": m, "year": y, "n_context": n_ctx, "context_years": jobs[(m, y, p)],
                        "seconds": secs, **year_metrics(df, np.array(score))})

    redrafts = []
    for y in redraft_years:
        key = (redraft_model, y, "causal")
        ids, score = res[key][:2]
        cls = table[table.draft_year == y].sort_values("pick")
        redrafts.append({"year": y, "model": redraft_model, "protocol": "causal", "context_years": jobs[key],
                         "picks": _picks(cls, pd.Series(score, index=ids))})

    run = {
        "run_id": run_id, "created": datetime.now(timezone.utc).isoformat(), "tag": a.tag, "models": models, "redraft_model": redraft_model,
        "features": FEATURES, "feature_hash": hashlib.md5(",".join(FEATURES).encode()).hexdigest()[:8],
        "target": f"{TARGET} = mean over best {C.PEAK_SEASONS} NBA seasons of {C.WAR_PER_MIN} * (rating + {C.WAR_REPLACEMENT}) * minutes; never played = {C.NEVER_PLAYED_WAR}",
        "north_star": NORTH_STAR, "split": split_table(), "val_years": list(C.VAL_YEARS), "holdout_years": list(C.HOLDOUT_YEARS),
        "holdout_evaluated": a.holdout, "gpus": gpus, "metrics": metrics, "summary": summarize(metrics).to_dict("records"), "redrafts": redrafts,
    }
    out_dir = C.OUT / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run.json").write_text(json.dumps(run, indent=1, default=_json))

    _ledger(C.OUT / "ledger.jsonl", run, "validation")
    if a.holdout:
        _ledger(C.OUT / "holdout_ledger.jsonl", run, "holdout")
        n = sum(1 for _ in open(C.OUT / "holdout_ledger.jsonl"))
        print(f"!! holdout evaluated -- {n} holdout evaluation(s) logged so far; every extra look leaks information")
    pd.set_option("display.width", 200)
    print(summarize(metrics).round(3).to_string(index=False))
    print("run written:", out_dir / "run.json")
    from nbadraft.db import rebuild
    rebuild()


def _picks(cls, score):
    """Every pick of one draft class, with the model's prediction and re-ranked slot for the players it could score."""
    rows = []
    for r in cls.itertuples():
        pred = float(score[r.bbref_id]) if r.bbref_id in score.index else None
        rows.append({"player": r.player, "bbref_id": r.bbref_id, "actual_pick": int(r.pick), "team": r.team, "college": r.college,
                     "modelled": bool(r.modelled), "pred": pred, "new_pick": None,
                     "peak_war": float(r.peak_war) if r.labelled else None, "seasons_played": int(r.seasons_played), "labelled": bool(r.labelled)})
    for i, p in enumerate(sorted((p for p in rows if p["pred"] is not None), key=lambda p: -p["pred"])):
        p["new_pick"] = i + 1
    return rows


def summarize(metrics):
    if not metrics:
        return pd.DataFrame(columns=["split", "protocol", "model"] + SUMMARY_KEYS)
    return pd.DataFrame(metrics).groupby(["split", "protocol", "model"])[SUMMARY_KEYS].mean().reset_index()


def _ledger(path, run, split):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps({"run_id": run["run_id"], "created": run["created"], "split": split, "models": run["models"], "feature_hash": run["feature_hash"],
                            "summary": [s for s in run["summary"] if s["split"] == split]}, default=_json) + "\n")


def _json(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


if __name__ == "__main__":
    main()
