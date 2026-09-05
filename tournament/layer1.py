"""Layer 1: raw AI predictions for the walk-forward years, using the complete real-draftee pool.

Reuses infra's table, causal relabelling and metrics exactly, so a number here is the number `pipeline.run` would
report. Adds knobs `run.py` does not have (label transform, feature groups from tournament.contract, classification
bins, stacks, per-population segmentation, context recency) and writes every per-player prediction -- point score
and, for classifiers, the full bin distribution -- to outputs/layer_1/<tag>.parquet so layer 2 can build rules on top
without touching a GPU. All features come from draft_table.parquet; nothing from a draft's own year or later is
visible. Labels are causal by default; `label_cutoff=full` is an explicitly named retrospective protocol where
already-completed context classes use their full five-season outcomes.

Config keys: name, model (tabfm | tabfm_cls | exaone | exaone_cls | xgb | tabicl | tabldm | tabpfn26 | tabpfn3 | ridge),
features (group/column list, '-group' removes), label (raw | zscore | rank | gaussrank | disc85_gaussrank),
label_cutoff (causal | full), label_horizon (full | match | short3 | 1..5 -- how many NBA seasons a training label
spans; "match" mirrors what the scored year's own label can cover, "short3" trains 1-2 season classes on 3-season
labels), bins, n_estimators, model_options, seeds, seed_start, segment (bool), ctx_start,
ctx_end, ctx_last, ctx_min_seasons, stack ([member configs], rank-averaged, optional weight per member).

Usage: python -m tournament.layer1 --device cuda:0 --years context --tag sweep1 --configs '[{"name": ...}, ...]'
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
from scipy.stats import norm, rankdata

from infra import config as C
from infra.models import TARGET, predict
from pipeline.run import context_years, year_metrics
from infra.war import war_target
from tournament import contract as F

# Model selection happens inside the context years; the holdout window is looked at once.
YEARS = {"context": list(range(2013, 2018)), "context6": list(range(2013, 2019)),
         "context7": list(range(2011, 2018)), "context8": list(range(2011, 2019)), "holdout": list(C.HOLDOUT_YEARS)}
YEARS["all"] = YEARS["context"] + YEARS["holdout"]


def label_horizon_for(year: int, horizon) -> int:
    """Seasons of NBA outcome a label may span. "match" mirrors what the evaluation label for `year` can cover:
    a 2025 class judged after the 2026 season has exactly one season in it, so training on five-season labels asks
    the model to rank something the scorer cannot see."""
    if horizon in (None, "full"):
        return C.TARGET_SEASONS
    visible = max(1, min(C.TARGET_SEASONS, C.LAST_SEASON - year))
    if horizon == "match":
        return visible
    if horizon == "short3":  # tournament.horizon: 1-2 season labels are rookie noise; 3-season labels rank them best
        return 3 if visible <= 2 else C.TARGET_SEASONS
    return max(1, min(C.TARGET_SEASONS, int(horizon)))


def horizon_target(ctx: pd.DataFrame, seasons: pd.DataFrame, horizon: int, through: int) -> np.ndarray:
    """WAR over each row's first `horizon` NBA seasons, counting only seasons completed by `through`."""
    rows = ctx[["bbref_id", "draft_year"]].reset_index(drop=True)
    rows["_row"] = np.arange(len(rows))
    played = rows.merge(seasons[["bbref_id", "season", "war"]], on="bbref_id", how="left")
    played = played[(played.season > played.draft_year) & (played.season <= through)].sort_values(["_row", "season"])
    played["_n"] = played.groupby("_row").cumcount()
    played = played[played["_n"] < horizon]
    score = played.groupby("_row").war.sum()
    values = pd.Series(C.NEVER_PLAYED_WAR, index=np.arange(len(ctx)), dtype=float)
    values.loc[score.index] = score.values
    return values.to_numpy()


def transform_labels(ctx: pd.DataFrame, how: str, seasons: pd.DataFrame | None = None, through: int | None = None,
                     horizon: int = C.TARGET_SEASONS) -> pd.DataFrame:
    """Rewrite TARGET on context rows, including discounted and Gaussian-ranked variants."""
    if how == "raw":
        return ctx
    if how == "disc85_gaussrank":
        if seasons is None or through is None:
            raise ValueError("disc85_gaussrank needs season WAR and a label cutoff")
        rows = ctx[["bbref_id", "draft_year"]].reset_index(drop=True)
        rows["_row"] = np.arange(len(rows))
        played = rows.merge(seasons[["bbref_id", "season", "war"]], on="bbref_id", how="left")
        played = played[(played.season > played.draft_year) & (played.season <= through)].sort_values(["_row", "season"])
        played["_n"] = played.groupby("_row").cumcount()
        played = played[played["_n"] < horizon]
        score = (played.war * np.power(0.85, played["_n"])).groupby(played["_row"]).sum()
        values = pd.Series(0.0, index=np.arange(len(ctx)))
        values.loc[score.index] = score.clip(-40, 40)
        ctx = ctx.assign(**{TARGET: values.values})
    g = ctx.groupby("draft_year")[TARGET]
    if how == "rank":
        return ctx.assign(**{TARGET: g.rank(pct=True).values})
    if how == "zscore":
        return ctx.assign(**{TARGET: ((ctx[TARGET] - g.transform("mean")) / g.transform("std").replace(0, 1)).values})
    if how in {"gaussrank", "disc85_gaussrank"}:
        rank = g.rank(method="average")
        n = g.transform("count")
        q = ((rank - 0.5) / n).clip(0.01, 0.99)
        return ctx.assign(**{TARGET: norm.ppf(q)})
    raise ValueError(how)


def select_features(ctx: pd.DataFrame, feats: list[str], min_coverage: float = 0.0, topk: int | None = None) -> list[str]:
    """Fold-local coverage and univariate selection; evaluation rows never influence the chosen columns."""
    keep = [f for f in feats if ctx[f].notna().mean() >= min_coverage and ctx[f].nunique(dropna=True) > 1]
    if not topk or len(keep) <= topk:
        return keep
    numeric = ctx[keep].apply(pd.to_numeric, errors="coerce")
    signal = numeric.corrwith(ctx[TARGET], method="spearman").abs().fillna(-1)
    return signal.sort_values(ascending=False, kind="stable").head(topk).index.tolist()


# --------------------------------------------------------------------------- cache, progress log

def config_sig(cfg: dict, years: str, seed: int, table: pd.DataFrame, G: dict) -> str:
    """Cache key: the config, the columns it resolves to and their actual values, the years and the seed. Adding an
    unrelated column to the table (a new source) therefore does not invalidate results that never used it."""
    resolved = sorted(set(sum((F.resolve(m.get("features", cfg.get("features")), G, table.columns) for m in (cfg.get("stack") or [cfg])), [])))
    data = pd.util.hash_pandas_object(table[["key", "draft_year", "modelled", "labelled", "pick", TARGET] + resolved], index=False).values
    body = "all-drafted-v1|" + json.dumps({k: v for k, v in cfg.items() if k != "name"}, sort_keys=True) + json.dumps(resolved) + f"{years}{seed}{TARGET}" + hashlib.md5(data.tobytes()).hexdigest()
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
    ap.add_argument("--years", default="context", choices=list(YEARS))
    ap.add_argument("--year-list", help="comma-separated subset of --years, for sharding one config across GPUs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True, help="outputs/layer_1/<tag>.parquet")
    a = ap.parse_args()
    configs = json.load(open(a.configs)) if a.configs.endswith(".json") else json.loads(a.configs)
    years = [int(y) for y in a.year_list.split(",")] if a.year_list else YEARS[a.years]
    if not set(years) <= set(YEARS[a.years]):
        raise SystemExit(f"--year-list must be a subset of --years {a.years}: {YEARS[a.years]}")
    years_key = a.years if not a.year_list else f"{a.years}:{','.join(map(str, years))}"

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
        hit = cache_dir / f"{config_sig(cfg, years_key, a.seed, table, G)}.parquet"
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
            if cfg.get("ctx_start"):
                cy = [c for c in cy if c >= cfg["ctx_start"]]
            if cfg.get("ctx_end"):
                cy = [c for c in cy if c <= cfg["ctx_end"]]
            if cfg.get("ctx_last"):
                cy = cy[-cfg["ctx_last"]:]
            if cfg.get("ctx_min_seasons"):  # drop the newest classes, whose labels rest on very few NBA seasons
                cy = [c for c in cy if y - c >= cfg["ctx_min_seasons"]]
            ctx0 = t[t.modelled & t.labelled & t.draft_year.isin(cy)]
            label_through = C.LAST_SEASON if cfg.get("label_cutoff", "causal") == "full" else y
            horizon = label_horizon_for(y, cfg.get("label_horizon"))
            if horizon == C.TARGET_SEASONS:
                ctx0 = ctx0.assign(**{TARGET: war_target(ctx0, seasons, through=label_through)[TARGET].values})
            else:
                ctx0 = ctx0.assign(**{TARGET: horizon_target(ctx0, seasons, horizon, label_through)})
            # Candidate pool is every player drafted that year. Rows without a complete feature source remain candidates;
            # the tabular models score their NaNs through the same fitted missing-value handling as any sparse prospect.
            pool = t[t.draft_year == y]
            assert pool["pick"].notna().all(), "candidate without an actual pick -- the pool must be the real draftees"
            parts = []
            for i, m in enumerate(members):
                # member-level horizon filter: a "long-horizon" member learns only from classes with >= k seasons of
                # outcomes (who eventually became good), to complement members that lean toward early production
                ctx_m = ctx0[ctx0.draft_year <= y - m["ctx_min_seasons"]] if m.get("ctx_min_seasons") else ctx0
                ctx = transform_labels(ctx_m, m["label"], seasons, label_through, horizon)
                seeds = m.get("seeds", cfg.get("seeds", 1))
                seed_start = m.get("seed_start", cfg.get("seed_start", 0))
                # segment: one in-context fit per population (college / intl); z-scored labels put both on one scale
                segs = sorted(pool.source.unique()) if cfg.get("segment") else [None]
                score, proba = np.zeros(len(pool)), None
                for sg in segs:
                    ci = ctx if sg is None else ctx[ctx.source == sg]
                    ti = np.ones(len(pool), bool) if sg is None else (pool.source == sg).values
                    if ti.sum() == 0 or len(ci) < 10:
                        continue
                    selected = select_features(
                        ci, m["_feats"], m.get("min_coverage", cfg.get("min_coverage", 0.0)),
                        m.get("feature_topk", cfg.get("feature_topk")),
                    )
                    outs = [predict(m["model"], ci, pool[ti], selected, a.device, a.seed + seed_start + s, m["bins"],
                                    m.get("n_estimators", cfg.get("n_estimators", 32)),
                                    m.get("model_options", cfg.get("model_options"))) for s in range(seeds)]
                    score[ti] = np.mean([o[0] for o in outs], axis=0)
                    if outs[0][1] is not None:
                        proba = np.zeros((len(pool), m["bins"])) if proba is None else proba
                        proba[ti] = np.mean([o[1] for o in outs], axis=0)
                parts.append(rankdata(score) * m.get("weight", 1.0))
                rec = pool[["bbref_id", "player", "draft_year", "pick", TARGET, "labelled", "source", "age_at_draft", "height_in", "class_year", "rec_rank", "role"]].copy()
                rec["config"], rec["member"], rec["model"], rec["years"], rec["score"] = cfg["name"], i, m["model"], years_key, score
                for b in range(m["bins"] if proba is not None else 0):
                    rec[f"p{b}"] = proba[:, b]
                preds.append(rec)
            stacked = np.sum(parts, axis=0) / sum(m.get("weight", 1.0) for m in members)
            rec = pool[["bbref_id", "draft_year"]].assign(config=cfg["name"], member=-1, model=model, years=years_key, score=stacked)
            preds.append(rec)
            mt = year_metrics(pool, np.asarray(stacked, dtype=float))
            mt["year"] = y
            per_year.append(mt)
        df = pd.DataFrame(per_year)
        pd.concat([p for p in preds if (p.config == cfg["name"]).all()], ignore_index=True).drop(columns=["config"]).to_parquet(hit, index=False)
        summ = {"tag": a.tag, "years": years_key, "config": cfg["name"], "model": model, "n_feats": n_feats, "spearman": df.spearman.mean(),
                "spearman_nba": df.spearman_nba.mean(), "wc14": df["war_captured_pct@14"].mean(), "wc30": df["war_captured_pct@30"].mean(),
                "wins": int((df.spearman > df.spearman_nba).sum()), "secs": time.time() - t0}
        rows.append(summ)
        print(json.dumps({"config": cfg["name"], "years": years_key, "per_year": df[["year", "spearman", "spearman_nba", "war_captured_pct@14"]].round(3).to_dict("records")}), file=sys.stderr)
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
