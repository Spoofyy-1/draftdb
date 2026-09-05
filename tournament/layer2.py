"""Layer 2: deterministic selection rules on top of layer-1 predictions. No GPU, no training.

A rule maps the layer-1 outputs for one draft class (point scores, bin distributions, the actual pick) to a final
order. Rules are kept simple and explainable -- at most two parameters, monotone in their inputs -- and are chosen on
the validation years only. The test years are evaluated behind --test, and every such look is appended to
outputs/layer_2/test_looks.jsonl so the number of peeks stays visible (same idea as the repo's holdout ledger).

Rules
  blend      rank-average of several layer-1 configs (weights)
  market     (1-w) * rank(model) + w * rank(-pick): the model adjusts the scouts' consensus.  Uses the actual pick,
             so any result from it is a "+market" result and must be labelled as such.
  upside     E[bin] + lam * P(top bin): reward a high ceiling on top of the expected outcome (classifier members)
  floor      E[bin] - lam * P(bottom bin): penalise bust risk

Usage: python -m tournament.layer2 --layer1 outputs/layer_1/<tag>.parquet [--test]
"""

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
    sp, nba, wc = [], [], []
    for _, g in df.groupby("draft_year"):
        g = g[g.labelled.astype(bool) & g[TARGET].notna()]
        if len(g) < 3:
            continue
        sp.append(spearmanr(g[col], g[TARGET]).correlation)
        nba.append(spearmanr(-g["pick"], g[TARGET]).correlation)
        ours = g.sort_values(col, ascending=False)[TARGET].head(14).sum()
        act = g.sort_values("pick")[TARGET].head(14).sum()
        best = g[TARGET].nlargest(14).sum()
        wc.append(100 * (ours - act) / (best - act) if best > act else 0.0)
    return {"spearman": float(np.mean(sp)), "spearman_nba": float(np.mean(nba)), "gap": float(np.mean(sp) - np.mean(nba)),
            "wc14": float(np.mean(wc)), "wins": int(np.sum(np.array(sp) > np.array(nba))), "n_years": len(sp)}


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
    out = out.join(tab[[c for c in ("source", "iz_young_x_eff", "iz_eff_36", "mock_rank_consensus", "mock_n_sources") if c in tab.columns]])
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


def rule_market(w: pd.DataFrame, score: pd.Series, wmkt: float) -> pd.Series:
    w = w.assign(_s=score, _p=-w["pick"])
    return (1 - wmkt) * _rank_within_year(w, "_s") + wmkt * _rank_within_year(w, "_p")


def rule_upside(w: pd.DataFrame, cfg: str, lam: float) -> pd.Series:
    return w[f"{cfg}::ebin"] + lam * w[f"{cfg}::ptop"]


def rule_floor(w: pd.DataFrame, cfg: str, lam: float) -> pd.Series:
    return w[f"{cfg}::ebin"] - lam * w[f"{cfg}::pbot"]


def rule_consensus(w: pd.DataFrame, score: pd.Series, wc: float, skip_unranked: bool = False) -> pd.Series:
    """(1 - wc) * rank(model) + wc * rank(pre-draft consensus mock). The mocks were published before draft night, so
    unlike `market` this is a fully pre-draft order; players absent from every mock rank last in the consensus.
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


# --------------------------------------------------------------------------- sweep

def sweep(w: pd.DataFrame, configs: list[str]) -> pd.DataFrame:
    rows = []
    def add(name, s, uses_pick=False):
        w["_r"] = s
        rows.append({"rule": name, "uses_pick": uses_pick, **evaluate(w, "_r")})
    for c in configs:
        add(f"L1 {c}", w[c])
    if len(configs) > 1:
        add("blend all", rule_blend(w, configs))
        for i, c1 in enumerate(configs):
            for c2 in configs[i + 1:]:
                add(f"blend {c1} | {c2}", rule_blend(w, [c1, c2]))
    for c in configs:
        if f"{c}::ptop" in w:
            for lam in [0.5, 1.0, 2.0, 4.0]:
                add(f"upside {c} lam={lam}", rule_upside(w, c, lam))
                add(f"floor {c} lam={lam}", rule_floor(w, c, lam))
    # international override and the market blend sit on top of whichever pure rule ranked best
    best_pure = max(rows, key=lambda r: r["spearman"])["rule"]
    base = _recompute(w, configs, best_pure)
    if "iz_young_x_eff" in w:
        for al in [0.3, 0.5, 0.7, 1.0]:
            add(f"intl a={al} on [{best_pure}]", rule_intl(w, base, al))
    if "mock_rank_consensus" in w:  # pre-draft consensus: legitimate pre-draft information, reported as its own variant
        w["_c"] = -w["mock_rank_consensus"].fillna(61)
        add("consensus alone (pre-draft mocks)", _rank_within_year(w, "_c"))
        for wc in [0.3, 0.4, 0.5, 0.6, 0.7]:
            add(f"consensus w={wc} on [{best_pure}]", rule_consensus(w, base, wc))
        for wc in [0.5, 0.6]:
            add(f"consensus-nz w={wc} on [{best_pure}]", rule_consensus(w, base, wc, skip_unranked=True))
    for wm in [0.3, 0.4, 0.5, 0.6, 0.7]:
        add(f"market w={wm} on [{best_pure}]", rule_market(w, base, wm), uses_pick=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer1", nargs="+", required=True, help="one or more outputs/layer_1/<tag>.parquet")
    ap.add_argument("--configs", nargs="*", help="layer-1 config names to use (default: all in the files)")
    ap.add_argument("--test", action="store_true", help="evaluate on the test years (logged to outputs/layer_2/test_looks.jsonl)")
    ap.add_argument("--rules", nargs="*", help="with --test: only these rule names (from a prior validation sweep)")
    ap.add_argument("--years", default=None, help="validation window tag in the layer-1 files (default: val; e.g. val7)")
    a = ap.parse_args()
    years = "test" if a.test else (a.years or "val")
    l1 = pd.concat([pd.read_parquet(p) for p in a.layer1], ignore_index=True)
    if a.test:  # the test window is whatever config says today, even if the prediction file holds more years
        l1 = l1[l1.draft_year.isin(C.VAL_YEARS)]
    configs = a.configs or sorted(l1.config.unique())
    w = wide(l1, years)
    if a.test and a.rules:  # pre-registered rules only, computed exactly as named on validation
        rows = []
        for name in a.rules:
            w["_r"] = _recompute(w, configs, name)
            rows.append({"rule": name, "uses_pick": name.startswith("market"), **evaluate(w, "_r")})
        res = pd.DataFrame(rows)
    else:
        res = sweep(w, configs)
    if a.test:
        out = C.OUT / "layer_2"
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "test_looks.jsonl", "a") as f:
            f.write(json.dumps({"when": datetime.now(timezone.utc).isoformat(), "layer1": a.layer1, "rules": res.rule.tolist(),
                                "results": res.round(4).to_dict("records")}) + "\n")
        n = sum(1 for _ in open(out / "test_looks.jsonl"))
        print(f"!! TEST look #{n} logged -- every extra look leaks information into the choice of rule")
    pd.set_option("display.width", 220)
    res = res.sort_values("spearman", ascending=False)
    print(f"[{years}] {len(w)} players, {w.draft_year.nunique()} years, scouts spearman {res.spearman_nba.iloc[0]:.3f}\n")
    print(res[["rule", "uses_pick", "spearman", "gap", "wc14", "wins", "n_years"]].round(3).to_string(index=False))
    if a.test:
        print("\nbootstrap 95% CI of the gap (years and players resampled):")
        for name in res.rule.head(5):
            col = _recompute(w, configs, name)
            lo, hi, p = bootstrap_gap(w.assign(_r=col), "_r")
            print(f"  {name:55s} CI [{lo:+.3f}, {hi:+.3f}]  P(model>scouts)={p:.2f}")


def _recompute(w, configs, name):
    if name.startswith("L1 "):
        return w[name[3:]]
    if name == "blend all":
        return rule_blend(w, configs)
    if name.startswith("blend "):
        return rule_blend(w, name[6:].split(" | "))
    if name.startswith("upside ") or name.startswith("floor "):
        kind, rest = name.split(" ", 1)
        cfg, lam = rest.rsplit(" lam=", 1)
        return (rule_upside if kind == "upside" else rule_floor)(w, cfg, float(lam))
    if name.startswith("market "):
        wm = float(name.split("w=")[1].split(" ")[0])
        inner = name.split("[", 1)[1][:-1]
        return rule_market(w, _recompute(w, configs, inner), wm)
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
