"""Layer 1: raw AI predictions for the walk-forward years, causal protocol only, TabFM / EXAONE on one GPU.

Reuses infra's table, causal relabelling and metrics exactly, so a number here is the number `validation.run` would
report. Adds knobs `run.py` does not have (label transform, feature groups from tournament.contract, classification
bins, stacks, per-population segmentation, context recency) and writes every per-player prediction -- point score
and, for classifiers, the full bin distribution -- to outputs/layer_1/<tag>.parquet so layer 2 can build rules on top
without touching a GPU. All features come from draft_table.parquet; nothing from a draft's own year or later is
visible, and labels come from `war_target(..., through=year)` exactly as in run.py.

Config keys: name, model (tabfm | tabfm_cls | exaone | exaone_cls | ridge), features (group/column list, '-group'
removes), label (raw | zscore | rank), bins, n_estimators, seeds, segment (bool), population (all | college | intl),
ctx_last, ctx_min_seasons, stack ([member configs], rank-averaged, optional weight per member).

Usage: python -m tournament.layer1 --device cuda:0 --years val --tag sweep1 --configs '[{"name": ...}, ...]'
"""

import argparse
import hashlib
import json
import os
import sys
import threading
import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from infra import config as C
from infra.models import TARGET, predict
from validation.run import context_years, year_metrics
from infra.war import war_target
from tournament import contract as F

# Model selection happens on the walk-forward years just before the test window; the test window is looked at once.
YEARS = {"val": list(range(2013, 2018)), "val7": list(range(2011, 2018)), "val8": list(range(2011, 2019)), "test": list(C.VAL_YEARS)}
YEARS["all"] = YEARS["val"] + YEARS["test"]


def transform_labels(ctx: pd.DataFrame, how: str) -> pd.DataFrame:
    """Rewrite TARGET on the context rows. zscore/rank are within each draft class, so classes observed for one season
    and for ten sit on one scale (the metric is a within-class rank correlation)."""
    if how == "raw":
        return ctx
    g = ctx.groupby("draft_year")[TARGET]
    if how == "rank":
        return ctx.assign(**{TARGET: g.rank(pct=True).values})
    if how == "zscore":
        return ctx.assign(**{TARGET: ((ctx[TARGET] - g.transform("mean")) / g.transform("std").replace(0, 1)).values})
    raise ValueError(how)


# --------------------------------------------------------------------------- cache, progress log

def config_sig(cfg: dict, years: str, seed: int, table: pd.DataFrame, G: dict) -> str:
    """Cache key: the config, the columns it resolves to and their actual values, the years and the seed. Adding an
    unrelated column to the table (a new source) therefore does not invalidate results that never used it."""
    resolved = sorted(set(sum((F.resolve(m.get("features", cfg.get("features")), G, table.columns) for m in (cfg.get("stack") or [cfg])), [])))
    data = pd.util.hash_pandas_object(table[["key", "draft_year", "modelled", "labelled", "pick", TARGET] + resolved], index=False).values
    body = json.dumps({k: v for k, v in cfg.items() if k != "name"}, sort_keys=True) + json.dumps(resolved) + f"{years}{seed}{TARGET}" + hashlib.md5(data.tobytes()).hexdigest()
    return hashlib.md5(body.encode()).hexdigest()[:12]


def log_progress(tag: str, i: int, n: int, name: str, summ: dict):
    """Append one line per finished config to <repo>/logs.txt with the best score so far for this sweep tag.
    Runs on a daemon thread so the GPU loop never waits on the disk."""
    threading.Thread(target=_log_progress, args=(tag, i, n, name, dict(summ), time.strftime("%H:%M:%S")), daemon=True).start()


def _log_progress(tag, i, n, name, summ, stamp):
    log = C.ROOT / "logs.txt"
    sweep = tag.rsplit("_gpu", 1)[0]
    best_name, best = name, summ["spearman"]
    if log.exists():
        for line in log.read_text().splitlines():
            if line.startswith(f"[{sweep} ") and " spearman " in line:
                try:
                    s = float(line.split(" spearman ")[1].split()[0])
                    if s > best:
                        best, best_name = s, line.split("| ")[1].split(" spearman ")[0].strip()
                except (IndexError, ValueError):
                    pass
    with open(log, "a") as f:
        f.write(f"[{sweep} {stamp}] {tag[-4:]} {i}/{n} done | {name} spearman {summ['spearman']:.3f} "
                f"(scouts {summ['spearman_nba']:.3f}, wc14 {summ['wc14']:+.1f}, wins {summ['wins']}) | best so far: {best_name} spearman {best:.3f}\n")


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--configs", required=True, help="JSON list of configs (see module docstring), or a path to a .json file")
    ap.add_argument("--years", default="val", choices=list(YEARS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True, help="outputs/layer_1/<tag>.parquet")
    a = ap.parse_args()
    configs = json.load(open(a.configs)) if a.configs.endswith(".json") else json.loads(a.configs)
    years = YEARS[a.years]

    table = pd.read_parquet(C.PROC / "draft_table.parquet")
    seasons = pd.read_parquet(C.PROC / "season_war.parquet")
    G = F.validate(table, verbose=False)
    labelled = sorted(set(table.loc[table.modelled & table.labelled, "draft_year"]))
    # The pool is the players actually taken on draft night -- as candidates AND as context. Nobody else.
    assert table["pick"].notna().all(), "draft_table contains a row without an actual pick"
    cache_dir = C.OUT / "layer_1" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    preds, rows = [], []
    for i_cfg, cfg in enumerate(configs, 1):
        # cache: the same config (resolved to the same columns) on the same table and years is never computed twice
        hit = cache_dir / f"{config_sig(cfg, a.years, a.seed, table, G)}.parquet"
        if hit.exists():
            cached = pd.read_parquet(hit).assign(config=cfg["name"])
            preds.append(cached)
            per_year = []
            for y in years:
                scored = cached[(cached.member == -1) & (cached.draft_year == y)][["bbref_id", "draft_year", "score"]]
                scored = scored.merge(table[["bbref_id", "draft_year", TARGET, "pick"]], on=["bbref_id", "draft_year"])
                per_year.append({**year_metrics(scored, scored.score.values), "year": y})
            df = pd.DataFrame(per_year)
            rows.append({"tag": a.tag, "years": a.years, "config": cfg["name"], "model": "cached", "n_feats": -1, "spearman": df.spearman.mean(), "spearman_nba": df.spearman_nba.mean(),
                         "wc14": df["war_captured_pct@14"].mean(), "wc30": df["war_captured_pct@30"].mean(), "wins": int((df.spearman > df.spearman_nba).sum()), "secs": 0.0})
            print(f"[{a.years}] {cfg['name']:40s} (cached)          spearman {df.spearman.mean():.3f} vs nba {df.spearman_nba.mean():.3f}", flush=True)
            log_progress(a.tag, i_cfg, len(configs), cfg["name"], rows[-1])
            continue
        t = table if cfg.get("population", "all") == "all" else table[table.source == cfg["population"]]
        members = cfg.get("stack") or [cfg]
        for i, m in enumerate(members):
            m.setdefault("model", cfg.get("model", "tabfm"))
            m.setdefault("label", cfg.get("label", "raw"))
            m.setdefault("bins", cfg.get("bins", 5))
            m["_feats"] = F.resolve(m.get("features", cfg.get("features")), G, t.columns)
        model = "+".join(m["model"] for m in members) if "stack" in cfg else members[0]["model"]
        n_feats = len(set(sum((m["_feats"] for m in members), [])))
        per_year, t0 = [], time.time()
        for y in years:
            cy = context_years(y, "causal", labelled)
            if cfg.get("ctx_last"):
                cy = cy[-cfg["ctx_last"]:]
            if cfg.get("ctx_min_seasons"):  # drop the newest classes, whose labels rest on very few NBA seasons
                cy = [c for c in cy if y - c >= cfg["ctx_min_seasons"]]
            ctx0 = t[t.modelled & t.labelled & t.draft_year.isin(cy)]
            ctx0 = ctx0.assign(**{TARGET: war_target(ctx0, seasons, through=y)[TARGET].values})
            test = t[t.modelled & (t.draft_year == y)]
            assert test["pick"].notna().all(), "candidate without an actual pick -- the pool must be the real draftees"
            parts = []
            for i, m in enumerate(members):
                # member-level horizon filter: a "long-horizon" member learns only from classes with >= k seasons of
                # outcomes (who eventually became good), to complement members that lean toward early production
                ctx_m = ctx0[ctx0.draft_year <= y - m["ctx_min_seasons"]] if m.get("ctx_min_seasons") else ctx0
                ctx = transform_labels(ctx_m, m["label"])
                seeds = m.get("seeds", cfg.get("seeds", 1))
                # segment: one in-context fit per population (college / intl); z-scored labels put both on one scale
                segs = sorted(test.source.unique()) if cfg.get("segment") else [None]
                score, proba = np.zeros(len(test)), None
                for sg in segs:
                    ci = ctx if sg is None else ctx[ctx.source == sg]
                    ti = np.ones(len(test), bool) if sg is None else (test.source == sg).values
                    if ti.sum() == 0 or len(ci) < 10:
                        continue
                    outs = [predict(m["model"], ci, test[ti], m["_feats"], a.device, a.seed + s, m["bins"], m.get("n_estimators", cfg.get("n_estimators", 32))) for s in range(seeds)]
                    score[ti] = np.mean([o[0] for o in outs], axis=0)
                    if outs[0][1] is not None:
                        proba = np.zeros((len(test), m["bins"])) if proba is None else proba
                        proba[ti] = np.mean([o[1] for o in outs], axis=0)
                parts.append(rankdata(score) * m.get("weight", 1.0))
                rec = test[["bbref_id", "player", "draft_year", "pick", TARGET, "labelled", "source", "age_at_draft", "height_in", "class_year", "rec_rank", "role"]].copy()
                rec["config"], rec["member"], rec["model"], rec["years"], rec["score"] = cfg["name"], i, m["model"], a.years, score
                for b in range(m["bins"] if proba is not None else 0):
                    rec[f"p{b}"] = proba[:, b]
                preds.append(rec)
            stacked = np.sum(parts, axis=0) / sum(m.get("weight", 1.0) for m in members)
            rec = test[["bbref_id", "draft_year"]].assign(config=cfg["name"], member=-1, model=model, years=a.years, score=stacked)
            preds.append(rec)
            mt = year_metrics(test, np.asarray(stacked, dtype=float))
            mt["year"] = y
            per_year.append(mt)
        df = pd.DataFrame(per_year)
        pd.concat([p for p in preds if (p.config == cfg["name"]).all()], ignore_index=True).drop(columns=["config"]).to_parquet(hit, index=False)
        summ = {"tag": a.tag, "years": a.years, "config": cfg["name"], "model": model, "n_feats": n_feats, "spearman": df.spearman.mean(),
                "spearman_nba": df.spearman_nba.mean(), "wc14": df["war_captured_pct@14"].mean(), "wc30": df["war_captured_pct@30"].mean(),
                "wins": int((df.spearman > df.spearman_nba).sum()), "secs": time.time() - t0}
        rows.append(summ)
        print(json.dumps({"config": cfg["name"], "years": a.years, "per_year": df[["year", "spearman", "spearman_nba", "war_captured_pct@14"]].round(3).to_dict("records")}), file=sys.stderr)
        print(f"[{a.years}] {cfg['name']:40s} {model:16s} f={n_feats:<3d} spearman {summ['spearman']:.3f} vs nba {summ['spearman_nba']:.3f}  "
              f"wc14 {summ['wc14']:+6.1f}  wc30 {summ['wc30']:+6.1f}  wins {summ['wins']}/{len(years)}  {summ['secs']:.0f}s", flush=True)
        log_progress(a.tag, i_cfg, len(configs), cfg["name"], summ)

    out = C.OUT / "layer_1"
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(preds, ignore_index=True).to_parquet(out / f"{a.tag}.parquet", index=False)
    log = out / "summary.csv"
    pd.DataFrame(rows).to_csv(log, mode="a", header=not log.exists(), index=False)
    print("layer_1 written:", out / f"{a.tag}.parquet")


if __name__ == "__main__":
    main()
