"""Layer 2: deterministic selection rules on top of layer-1 predictions. No GPU, no training.

A rule maps the layer-1 outputs for one draft class (point scores and bin distributions) to a final
order. Rules are kept simple and explainable -- at most two parameters, monotone in their inputs -- and are chosen on
the context years only. The holdout years are evaluated behind --holdout, and every such look is appended to
outputs/layer_2/holdout_looks.jsonl so the number of peeks stays visible (same idea as the repo's run ledger).

Rules
  blend      rank-average of several layer-1 configs (weights)
  upside     E[bin] + lam * P(top bin): reward a high ceiling on top of the expected outcome (classifier members)
  floor      E[bin] - lam * P(bottom bin): penalise bust risk
  tilt       (1 - |b|) * rank(model) + b * rank(pre-draft column): games played, coach / program pedigree, market disagreement
  stackctx   ridge over model ranks + fixed pre-draft covariate ranks, weights fit on the context years only (--stack-context)

Usage: python -m tournament.layer2 --layer1 outputs/layer_1/<tag>.parquet [--holdout]
"""

# BRIDGE: `X | None` in signatures needs Python 3.10+; this makes the module importable on the 3.9 that is
# BRIDGE: the only interpreter on the porting Mac. No-op on the box (3.12).
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from infra import config as C
from infra.config import TARGET


# --------------------------------------------------------------------------- metrics

def evaluate(df: pd.DataFrame, col: str) -> dict:
    """Mean over draft years of Spearman(model, truth), Spearman(scouts, truth), WAR captured @14, years won."""
    sp, nba, wc, per_year = [], [], [], {}
    for year, g in df.groupby("draft_year"):
        g = g[g.labelled.astype(bool) & g[TARGET].notna()]
        if len(g) < 3:
            continue
        sp.append(spearmanr(g[col], g[TARGET]).correlation)
        per_year[int(year)] = float(sp[-1])
        nba.append(spearmanr(-g["pick"], g[TARGET]).correlation)
        ours = g.sort_values(col, ascending=False)[TARGET].head(14).sum()
        act = g.sort_values("pick")[TARGET].head(14).sum()
        best = g[TARGET].nlargest(14).sum()
        wc.append(100 * (ours - act) / (best - act) if best > act else 0.0)
    return {"spearman": float(np.mean(sp)), "spearman_nba": float(np.mean(nba)), "gap": float(np.mean(sp) - np.mean(nba)),
            "wc14": float(np.mean(wc)), "wins": int(np.sum(np.array(sp) > np.array(nba))), "n_years": len(sp),
            "_per_year": per_year}


def bootstrap_gap(df: pd.DataFrame, col: str, n=2000, seed=0) -> tuple[float, float, float]:
    """95% CI and P(model > scouts) for the mean Spearman gap, resampling years and players within years."""
    rng = np.random.default_rng(seed)
    years = sorted(df.draft_year.unique())
    by = {y: g for y, g in df.groupby("draft_year")}
    gaps = []
    for _ in range(n):
        ys = rng.choice(years, size=len(years), replace=True)
        g_ = []
        for y in ys:
            g = by[y].sample(frac=1, replace=True, random_state=int(rng.integers(1 << 31)))
            g = g[g.labelled.astype(bool)]
            g_.append(spearmanr(g[col], g[TARGET]).correlation - spearmanr(-g["pick"], g[TARGET]).correlation)
        gaps.append(np.mean(g_))
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    return float(lo), float(hi), float(np.mean(np.array(gaps) > 0))


# --------------------------------------------------------------------------- rules

def _rank_within_year(df, col):
    return df.groupby("draft_year")[col].rank(pct=True)


def wide(layer1: pd.DataFrame, years: str) -> pd.DataFrame:
    """One row per (draft_year, bbref_id) with one column per config (stacked score, member -1) plus P(top)/P(bottom)
    of the first classifier member of each config."""
    l1 = layer1[layer1.years == years]
    base = l1[l1.member == -1].pivot_table(index=["draft_year", "bbref_id"], columns="config", values="score")
    meta = l1[l1.member >= 0].drop_duplicates(["draft_year", "bbref_id"]).set_index(["draft_year", "bbref_id"])[["player", "pick", TARGET, "labelled", "age_at_draft", "class_year"]]
    out = meta.join(base)
    # population and the deterministic international score come from the table (layer 1 does not carry them)
    tab = pd.read_parquet(C.PROC / "draft_table.parquet").set_index(["draft_year", "bbref_id"])
    extra = dict.fromkeys(("source", "iz_young_x_eff", "iz_eff_36", "mock_rank_consensus", "mock_n_sources", *TILT_COLUMNS, *STACK_RAW))
    out = out.join(tab[[c for c in extra if c in tab.columns]])
    pcols = [c for c in l1.columns if c.startswith("p") and c[1:].isdigit()]
    for cfg, g in l1[(l1.member >= 0) & l1[pcols].notna().any(axis=1)].groupby("config"):
        first = g[g.member == g.member.min()].set_index(["draft_year", "bbref_id"])
        pk = [c for c in pcols if first[c].notna().any()]
        top, bot = max(pk, key=lambda c: int(c[1:])), min(pk, key=lambda c: int(c[1:]))
        out[f"{cfg}::ptop"], out[f"{cfg}::pbot"] = first[top], first[bot]
        out[f"{cfg}::ebin"] = sum(first[c] * int(c[1:]) for c in pk)
    return out.reset_index()


def rule_blend(w: pd.DataFrame, configs: list[str], weights=None) -> pd.Series:
    weights = weights or [1.0] * len(configs)
    return sum(wt * _rank_within_year(w, c) for c, wt in zip(configs, weights)) / sum(weights)


def rule_upside(w: pd.DataFrame, cfg: str, lam: float) -> pd.Series:
    return w[f"{cfg}::ebin"] + lam * w[f"{cfg}::ptop"]


def rule_floor(w: pd.DataFrame, cfg: str, lam: float) -> pd.Series:
    return w[f"{cfg}::ebin"] - lam * w[f"{cfg}::pbot"]


def rule_consensus(w: pd.DataFrame, score: pd.Series, wc: float, skip_unranked: bool = False) -> pd.Series:
    """(1 - wc) * rank(model) + wc * rank(pre-draft consensus mock). Players absent from every mock rank last.
    skip_unranked: a player no mock ranked has no consensus information -- keep the model's rank for him instead."""
    w = w.assign(_s=score, _m=-w["mock_rank_consensus"].fillna(61))
    rm, rc = _rank_within_year(w, "_s"), _rank_within_year(w, "_m")
    blended = (1 - wc) * rm + wc * rc
    if skip_unranked and "mock_n_sources" in w:
        blended = blended.where(w["mock_n_sources"].fillna(0) > 0, rm)
    return blended


def rule_intl(w: pd.DataFrame, score: pd.Series, alpha: float) -> pd.Series:
    """For international rows, blend the model's within-class rank with a deterministic international score: production
    z-scored within his own league-season times how far under 22 he was (iz_young_x_eff), ranked within the class.
    College rows keep the model's rank. alpha = weight on the deterministic score."""
    w = w.assign(_s=score)
    r_model = _rank_within_year(w, "_s")
    intl = w.source.eq("intl") & w.iz_young_x_eff.notna()
    r_intl = w.assign(_i=w.iz_young_x_eff).groupby("draft_year")["_i"].rank(pct=True)
    return np.where(intl, (1 - alpha) * r_model + alpha * r_intl, r_model)


def rule_median(w: pd.DataFrame, configs: list[str]) -> pd.Series:
    """Median of within-year ranks: one wayward model cannot drag a player far."""
    return pd.concat([_rank_within_year(w, c) for c in configs], axis=1).median(axis=1)


def rule_age(w: pd.DataFrame, score: pd.Series, b: float) -> pd.Series:
    """(1 - b) * rank(model) + b * rank(youth). Age on draft night is pre-draft information; unknown ages rank neutral."""
    w = w.assign(_s=score, _a=-w["age_at_draft"].fillna(w["age_at_draft"].median()))
    return (1 - b) * _rank_within_year(w, "_s") + b * _rank_within_year(w, "_a")


# Pre-draft columns a tilt may lean on: sample size, durability, head coach / program pedigree, market disagreement.
TILT_COLUMNS = ["GP", "Min_per", "n_college_seasons", "mock_rank_std", "mock_n_sources", "co_tenure_yrs", "co_car_wl", "co_ncaa_car",
                "co_sw16_car", "co_seas_wl", "co_seas_ap_post", "co_coach_prior_n", "co_coach_prior_rank", "co_coach_prior_vs_pick",
                "co_prog_prior_n", "co_prog_prior_rank", "co_prog_prior_vs_pick"]


def rule_tilt(w: pd.DataFrame, score: pd.Series, col: str, b: float) -> pd.Series:
    """(1 - |b|) * rank(model) + b * rank(col): lean toward (b > 0) or away from (b < 0) a pre-draft column; rows without
    the column keep a neutral rank for it. The games-played tilt is the "did he play enough college?" question, the co_*
    tilts ask whether the head coach or the program should move a prospect."""
    v = pd.to_numeric(w[col], errors="coerce")
    w = w.assign(_s=score, _t=v.fillna(v.groupby(w.draft_year).transform("median")))
    return (1 - abs(b)) * _rank_within_year(w, "_s") + b * _rank_within_year(w, "_t")


# --------------------------------------------------------------------------- stacking fit on the context years

# Pre-draft columns the stacker may lean on, next to the layer-1 model ranks. Fixed list: the fit chooses weights only.
STACK_RAW = ["TS_per", "eFG", "rec_rank", "gl_pred_ws48", "bio_nba_relative_n", "FT_per", "TP_per", "TPA_pg", "stl_per", "usg", "iz_young_x_eff"]
STACK_COVARIATES = ["ts", "efg", "rsci", "gl", "rel", "shoot", "iz", "stl", "usg"]


def _pct(g: pd.DataFrame, v) -> np.ndarray:
    v = pd.to_numeric(v, errors="coerce")
    r = v.rank(pct=True)
    return r.fillna(0.5).values


def stack_features(w: pd.DataFrame, configs: list[str]) -> pd.DataFrame:
    """Within-class percentile ranks of each layer-1 config and of the STACK_COVARIATES, one row per player."""
    parts = []
    for _, g in w.groupby("draft_year"):
        f = pd.DataFrame(index=g.index)
        for c in configs:
            f[c] = _pct(g, g[c])
        f["ts"], f["efg"], f["stl"], f["usg"] = _pct(g, g.TS_per), _pct(g, g.eFG), _pct(g, g.stl_per), _pct(g, g.usg)
        f["rsci"] = _pct(g, -pd.to_numeric(g.rec_rank, errors="coerce"))
        f["gl"], f["rel"] = _pct(g, g.gl_pred_ws48), _pct(g, g.bio_nba_relative_n)
        tpa = pd.to_numeric(g.TPA_pg, errors="coerce").fillna(0)
        tp_shrunk = (pd.to_numeric(g.TP_per, errors="coerce") * tpa + 33 * 2) / (tpa + 2)  # 3P% pulled to 33% by attempts
        f["shoot"] = _pct(g, pd.Series(_pct(g, g.FT_per) + _pct(g, tp_shrunk), index=g.index))
        intl = g.source.eq("intl") & g.iz_young_x_eff.notna()
        f["iz"] = np.where(intl, _pct(g, g.iz_young_x_eff), 0.5)
        parts.append(f)
    return pd.concat(parts).loc[w.index]


def rule_stack(w: pd.DataFrame, configs: list[str], weights: list[float]) -> pd.Series:
    f = stack_features(w, configs)
    return f[configs + STACK_COVARIATES].values @ np.asarray(weights)


def fit_stack(w_ctx: pd.DataFrame, configs: list[str]) -> str:
    """Ridge on the context years only: Gaussian-ranked outcome ~ model ranks + covariate ranks. Returns the rule name
    carrying the fitted weights, so the holdout application is a fixed linear rule, not a fit."""
    from scipy.stats import norm
    from sklearn.linear_model import RidgeCV
    f = stack_features(w_ctx, configs)
    y = np.concatenate([norm.ppf(((g[TARGET].rank() - 0.5) / len(g)).clip(0.01, 0.99)) for _, g in w_ctx.groupby("draft_year")])
    idx = np.concatenate([g.index.values for _, g in w_ctx.groupby("draft_year")])
    m = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(f.loc[idx, configs + STACK_COVARIATES], y)
    return f"stackctx w={','.join(f'{c:.4f}' for c in m.coef_)} on [{' | '.join(configs)}]"


def rule_route(w: pd.DataFrame, base: str) -> pd.Series:
    """Horizon routing: a class whose label can hold at most two NBA seasons is scored by `<base> short3` (trained on
    three-season labels, see tournament.horizon); every other class by `base`. Identical to `base` on context years."""
    twin = f"{base} short3"
    if twin not in w:
        raise KeyError(twin)
    visible = (C.LAST_SEASON - w["draft_year"]).clip(lower=1, upper=C.TARGET_SEASONS)
    return w[base].where(visible > 2, w[twin])


def rule_caruana(w: pd.DataFrame, configs: list[str], steps: int = 12) -> tuple[str, pd.Series] | None:
    """Greedy forward selection with replacement on mean per-year Spearman; weights are the pick counts."""
    if len(configs) < 2:
        return None
    ranks = {c: _rank_within_year(w, c) for c in configs}
    years = {y: (g.index, g[TARGET]) for y, g in w[w.labelled.astype(bool) & w[TARGET].notna()].groupby("draft_year")}

    def fitness(s):
        return float(np.mean([spearmanr(s[idx], y).correlation for idx, y in years.values()]))

    chosen, current = [], None
    for _ in range(steps):
        best = None
        for c in configs:
            cand = ranks[c] if current is None else (current * len(chosen) + ranks[c]) / (len(chosen) + 1)
            f = fitness(cand)
            if best is None or f > best[0]:
                best = (f, c, cand)
        if current is not None and best[0] <= fitness(current) + 1e-6:
            break
        chosen.append(best[1])
        current = best[2]
    counts = pd.Series(chosen).value_counts()
    used = [c for c in configs if c in counts]
    weights = np.array([counts[c] for c in used], dtype=float)
    weights /= weights.sum()
    name = f"caruana w={','.join(f'{v:.4f}' for v in weights)} on [{' | '.join(used)}]"
    return name, rule_blend(w, used, weights.tolist())


def rule_optuna(w: pd.DataFrame, configs: list[str], n_trials: int = 300) -> tuple[str, pd.Series] | None:
    """Tune a sparse rank blend on early context years; the latest two years remain confirmation folds."""
    years = sorted(w.draft_year.unique())
    if len(configs) < 2 or len(years) < 4:
        return None
    import optuna

    dev = years[:-2]
    ranks = {c: _rank_within_year(w, c) for c in configs}

    def objective(trial):
        weights = np.array([trial.suggest_float(f"w{i}", 0.0, 1.0) for i in range(len(configs))])
        weights /= max(weights.sum(), 1e-12)
        score = sum(weights[i] * ranks[c] for i, c in enumerate(configs))
        vals = [spearmanr(score[w.draft_year == y], w.loc[w.draft_year == y, TARGET]).correlation for y in dev]
        return float(np.mean(vals) - 0.1 * np.std(vals))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    weights = np.array([study.best_params[f"w{i}"] for i in range(len(configs))])
    weights /= weights.sum()
    name = f"optuna w={','.join(f'{v:.4f}' for v in weights)} on [{' | '.join(configs)}]"
    return name, rule_blend(w, configs, weights.tolist())


# --------------------------------------------------------------------------- sweep

def sweep(w: pd.DataFrame, configs: list[str]) -> pd.DataFrame:
    rows = []
    def add(name, s, uses_pick=False):
        w["_r"] = s
        rows.append({"rule": name, "uses_pick": uses_pick, **evaluate(w, "_r")})
    bases = [c for c in configs if not c.endswith(" short3")]  # twins only ever enter through `route`
    for c in bases:
        add(f"L1 {c}", w[c])
    for c in bases:
        if f"{c} short3" in w:
            add(f"route [{c}]", rule_route(w, c))
    if len(bases) > 1:
        add("blend all", rule_blend(w, bases))
        add(f"median [{' | '.join(bases)}]", rule_median(w, bases))
        for i, c1 in enumerate(bases):
            for c2 in bases[i + 1:]:
                add(f"blend {c1} | {c2}", rule_blend(w, [c1, c2]))
        # top-k by single-model score, so the blend order is fixed by the layer-1 leaderboard, not by the search
        order = sorted(bases, key=lambda c: -next(r["spearman"] for r in rows if r["rule"] == f"L1 {c}"))
        for k in range(3, min(len(order), 8) + 1):
            add(f"blend {' | '.join(order[:k])}", rule_blend(w, order[:k]))
            add(f"median [{' | '.join(order[:k])}]", rule_median(w, order[:k]))
        for r in (rule_caruana(w, bases), rule_optuna(w, bases)):
            if r:
                add(*r)
    for c in bases:
        if f"{c}::ptop" in w:
            for lam in [0.5, 1.0, 2.0, 4.0]:
                add(f"upside {c} lam={lam}", rule_upside(w, c, lam))
                add(f"floor {c} lam={lam}", rule_floor(w, c, lam))
    # Pre-draft-only overlays sit on top of the three best pure rules.
    top_pure = [r["rule"] for r in sorted(rows, key=lambda r: -r["spearman"])[:3]]
    for pure in top_pure:
        base = _recompute(w, configs, pure)
        if "iz_young_x_eff" in w:
            for al in [0.3, 0.5, 0.7, 1.0]:
                add(f"intl a={al} on [{pure}]", rule_intl(w, base, al))
        for b in [0.1, 0.2, 0.3]:
            add(f"age b={b} on [{pure}]", rule_age(w, base, b))
        for col in TILT_COLUMNS:
            if col in w and w[col].notna().any():
                for b in [-0.15, -0.08, 0.08, 0.15]:
                    add(f"tilt {col} b={b} on [{pure}]", rule_tilt(w, base, col, b))
        if "mock_rank_consensus" in w:  # pre-draft consensus: legitimate pre-draft information, reported as its own variant
            for wc in [0.3, 0.4, 0.5, 0.6, 0.7]:
                add(f"consensus w={wc} on [{pure}]", rule_consensus(w, base, wc))
            for wc in [0.5, 0.6]:
                add(f"consensus-nz w={wc} on [{pure}]", rule_consensus(w, base, wc, skip_unranked=True))
    if "mock_rank_consensus" in w:
        w["_c"] = -w["mock_rank_consensus"].fillna(61)
        add("consensus alone (pre-draft mocks)", _rank_within_year(w, "_c"))
    # second-order: international tilt on top of the best consensus / age overlay
    best_overlay = max((r for r in rows if r["rule"].startswith(("consensus w=", "age b="))), key=lambda r: r["spearman"], default=None)
    if best_overlay and "iz_young_x_eff" in w:
        for al in [0.3, 0.5]:
            add(f"intl a={al} on [{best_overlay['rule']}]", rule_intl(w, _recompute(w, configs, best_overlay["rule"]), al))
    best_l1 = max((r for r in rows if r["rule"].startswith("L1 ")), key=lambda r: r["spearman"])
    for row in rows:
        years = sorted(set(best_l1["_per_year"]) & set(row["_per_year"]))
        diffs = np.array([row["_per_year"][y] - best_l1["_per_year"][y] for y in years])
        row["gain_vs_l1"] = float(diffs.mean()) if len(diffs) else float("-inf")
        row["fold_wins_vs_l1"] = int((diffs > 0).sum())
        needed = len(years) if len(years) <= 5 else int(np.ceil(0.8 * len(years)))
        latest_hold = all(row["_per_year"][y] > best_l1["_per_year"][y] for y in years[-2:])
        row["robust"] = (
            not row["uses_pick"]
            and (row["rule"] == best_l1["rule"]
                 or (row["gain_vs_l1"] >= 0.01 and row["fold_wins_vs_l1"] >= needed and latest_hold))
        )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer1", nargs="+", required=True, help="one or more outputs/layer_1/<tag>.parquet")
    ap.add_argument("--configs", nargs="*", help="layer-1 config names to use (default: all in the files)")
    ap.add_argument("--holdout", action="store_true", help="evaluate on the holdout years (logged to outputs/layer_2/holdout_looks.jsonl)")
    ap.add_argument("--rules", nargs="*", help="with --holdout: only these rule names (from a prior context sweep)")
    ap.add_argument("--years", default=None, help="context window tag in the layer-1 files (default: context; e.g. context7)")
    ap.add_argument("--stack-context", nargs="*", help="layer-1 files scored on context years: fit the stackctx rule on them, then evaluate it")
    ap.add_argument("--stack-years", default="context6", help="years tag of the --stack-context files")
    a = ap.parse_args()
    years = "holdout" if a.holdout else (a.years or "context")
    l1 = pd.concat([pd.read_parquet(p) for p in a.layer1], ignore_index=True)
    if a.holdout:  # the holdout window is whatever config says today, even if the prediction file holds more years
        l1 = l1[l1.draft_year.isin(C.HOLDOUT_YEARS)]
    configs = a.configs or sorted(l1.config.unique())
    w = wide(l1, years)
    rules = list(a.rules or [])
    if a.stack_context:  # weights come from the context years; the holdout only ever sees the finished linear rule
        ctx = pd.concat([pd.read_parquet(p) for p in a.stack_context], ignore_index=True)
        ctx = ctx[ctx.years.astype(str) == a.stack_years]
        w_ctx = wide(ctx, a.stack_years)
        stack_configs = [c for c in configs if c in w_ctx]
        rules.append(fit_stack(w_ctx, stack_configs))
        print("fitted on context:", rules[-1])
    if a.holdout and rules:  # pre-registered rules only, computed exactly as named on the context years
        rows = []
        for name in rules:
            w["_r"] = _recompute(w, configs, name)
            rows.append({"rule": name, "uses_pick": False, "robust": True,
                         "gain_vs_l1": 0.0, "fold_wins_vs_l1": 0, **evaluate(w, "_r")})
        res = pd.DataFrame(rows)
    else:
        res = sweep(w, configs)
    if a.holdout:
        out = C.OUT / "layer_2"
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "holdout_looks.jsonl", "a") as f:
            f.write(json.dumps({"when": datetime.now(timezone.utc).isoformat(), "layer1": a.layer1, "rules": res.rule.tolist(),
                                "results": res.round(4).to_dict("records")}) + "\n")
        n = sum(1 for _ in open(out / "holdout_looks.jsonl"))
        print(f"!! HOLDOUT look #{n} logged -- every extra look leaks information into the choice of rule")
    pd.set_option("display.width", 220)
    res = res.sort_values("spearman", ascending=False)
    print(f"[{years}] {len(w)} players, {w.draft_year.nunique()} years, scouts spearman {res.spearman_nba.iloc[0]:.3f}\n")
    print(res[["rule", "uses_pick", "robust", "spearman", "gap", "gain_vs_l1", "fold_wins_vs_l1", "wc14", "wins", "n_years"]].round(3).to_string(index=False))
    if a.holdout:
        print("\nbootstrap 95% CI of the gap (years and players resampled):")
        for name in res.rule.head(5):
            col = _recompute(w, configs, name)
            lo, hi, p = bootstrap_gap(w.assign(_r=col), "_r")
            print(f"  {name:55s} CI [{lo:+.3f}, {hi:+.3f}]  P(model>scouts)={p:.2f}")


def _recompute(w, configs, name):
    if name.startswith("L1 "):
        return w[name[3:]]
    if name == "blend all":
        return rule_blend(w, [c for c in configs if not c.endswith(" short3")])
    if name.startswith(("optuna w=", "caruana w=")):
        prefix, body = name.split(" on [", 1)
        weights = [float(v) for v in prefix.split("w=", 1)[1].split(",")]
        return rule_blend(w, body[:-1].split(" | "), weights)
    if name.startswith("blend "):
        return rule_blend(w, name[6:].split(" | "))
    if name.startswith("median ["):
        return rule_median(w, name[8:-1].split(" | "))
    if name.startswith("route ["):
        return rule_route(w, name[7:-1])
    if name.startswith("age b="):
        b = float(name.split("b=")[1].split(" ")[0])
        inner = name.split("[", 1)[1][:-1]
        return rule_age(w, _recompute(w, configs, inner), b)
    if name.startswith("stackctx w="):
        prefix, body = name.split(" on [", 1)
        weights = [float(v) for v in prefix.split("w=", 1)[1].split(",")]
        return pd.Series(rule_stack(w, body[:-1].split(" | "), weights), index=w.index)
    if name.startswith("tilt "):
        col, rest = name[5:].split(" b=", 1)
        b = float(rest.split(" ")[0])
        inner = name.split("[", 1)[1][:-1]
        return rule_tilt(w, _recompute(w, configs, inner), col, b)
    if name.startswith("upside ") or name.startswith("floor "):
        kind, rest = name.split(" ", 1)
        cfg, lam = rest.rsplit(" lam=", 1)
        return (rule_upside if kind == "upside" else rule_floor)(w, cfg, float(lam))
    if name.startswith("intl "):
        al = float(name.split("a=")[1].split(" ")[0])
        inner = name.split("[", 1)[1][:-1]
        return rule_intl(w, _recompute(w, configs, inner), al)
    if name.startswith("consensus alone"):
        return _rank_within_year(w.assign(_c=-w["mock_rank_consensus"].fillna(61)), "_c")
    if name.startswith("consensus"):
        wc = float(name.split("w=")[1].split(" ")[0])
        inner = name.split("[", 1)[1][:-1]
        return rule_consensus(w, _recompute(w, configs, inner), wc, skip_unranked=name.startswith("consensus-nz"))
    raise ValueError(name)


if __name__ == "__main__":
    main()
