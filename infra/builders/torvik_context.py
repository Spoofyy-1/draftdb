"""Season-level, context-adjusted pre-draft Torvik features for every matched draftee (columns prefixed t_).

One row per modelled draftee keyed by (key, draft_year). Y is the draft year, F the matched final Torvik season
(F <= Y; the season ends in April, the draft is in June). Every feature uses Torvik seasons with year <= Y only,
never Torvik's `pick` tag (it is only read by infra.dataset.match_torvik to identify the row), NBA data or later
seasons. Population statistics are causal or same-season:

  team / teammates   the F team-season roster (leave-one-out for teammate quality)
  conference         t_conf_strength_same  = minutes-weighted mean team_bpm_sum of the conference in season F
                     t_conf_strength_prior = the same quantity from season F-1 for the teams that make up the conference in F
  age curves         cubic polyfit stat ~ age on all D1 player-seasons with year < Y and Min_per >= 40 (expanding window
                     2008..Y-1, refit for every Y; the 2008 class has no fit population and gets NaN)
  role z-scores      mean/std within (role, season F) over D1 players with Min_per >= 40
  slope shrinkage    population slope = mean raw slope of draftees with draft_year < Y and >= 2 seasons; own weight n/(n+2)
  shooting priors    beta prior by method of moments on the role's player-seasons with year < Y and >= 20 attempts;
                     all-D1 prior from year < Y when the role is unknown (Torvik has no roles before the 2010 season)

Missing means not observed; nothing is filled with 0.

    from infra.builders.torvik_context import load_torvik_context   # reads the parquet
    python -m infra.builders.torvik_context                         # rebuilds parquet + feature dictionary, prints coverage
"""

import time

import numpy as np
import pandas as pd

from infra import config as C
from infra.dataset import match_torvik, norm_name

OUT = C.PROC / "prospect_torvik_context.parquet"
DICT = C.PROC / "feature_dictionary_t.csv"
KEYS = ["key", "draft_year"]

MIN_PER_POP = 40  # D1 population for age curves and role z-scores
MIN_PER_ROTATION = 20  # a teammate counts for max / top-2 BPM only with >= 20% of minutes (tiny-minute BPM is noise, up to +141)
MIN_ATT_PRIOR = 20  # player-seasons with >= 20 attempts define the shooting priors
AGE_STATS = ["bpm", "obpm", "dbpm", "TS_per", "usg", "AST_per", "stl_per", "blk_per", "ORB_per", "DRB_per"]
ROLE_STATS = ["bpm", "usg", "TS_per", "AST_per", "blk_per", "stl_per", "TP_per", "ftr"]
DEV_STATS = ["bpm", "obpm", "dbpm", "TS_per", "usg", "AST_per", "stl_per", "blk_per", "mpg"]
WEIGHTED = ["bpm", "usg", "adjoe", "adrtg"]

GROUPS = {
    "teammate": ["t_tm_bpm_mw", "t_tm_bpm_max", "t_tm_top2_bpm_mean", "t_tm_recrank_best", "t_tm_n_ranked_recruits", "t_tm_usg_mw",
                 "t_bpm_share", "t_usg_share", "t_pts_share", "t_ast_share", "t_porpag_share", "t_team_talent_conc"],
    "team": ["t_team_bpm_sum", "t_team_adjoe_mean", "t_team_adrtg_mean", "t_team_n_players"],
    "conf": ["t_conf_strength_prior", "t_conf_strength_same"],
    "age": [f"t_{s}_age_{k}" for s in AGE_STATS for k in ("resid", "z")],
    "role": [f"t_{s}_role_z" for s in ROLE_STATS] + ["t_role_rarity"],
    "dev": [f"t_dev_{s}_slope" for s in DEV_STATS] + ["t_dev_n_seasons", "t_dev_bpm_accel", "t_first_season_bpm", "t_first_season_age", "t_bpm_first_to_last"],
    "precocity": ["t_bpm_as_freshman", "t_min_per_as_freshman", "t_age_first_season", "t_freshman_bpm_age_z"],
    "shooting": ["t_ft_shrunk", "t_ft_shrunk_n", "t_3p_shrunk", "t_3p_shrunk_n", "t_3par", "t_ftr", "t_ft_minus_3p"],
}


def load_torvik_context() -> pd.DataFrame:
    return pd.read_parquet(OUT)


# --------------------------------------------------------------------------- inputs

def _prep(raw: pd.DataFrame) -> pd.DataFrame:
    t = raw.drop(columns=["pick"])
    t["minutes"] = t.mpg * t.GP
    t["age"] = (pd.to_datetime(t.year.astype(str) + "-01-15") - t.birthdate).dt.days / 365.25  # season midpoint
    t["bpm_pos"] = t.bpm.clip(lower=0)
    return t


def _draftees(raw: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame:
    """One row per matched draftee: key, draft_year (Y) plus his final Torvik season F (all torvik columns, year == F)."""
    drafts = pd.read_parquet(C.PROC / "drafts.parquet")
    m = match_torvik(drafts, raw)
    m = m[m.torvik_idx.notna()]
    d = pd.DataFrame({"key": m.player.map(norm_name).values, "draft_year": m.draft_year.values})
    f = t.reindex(m.torvik_idx.astype(int).values).reset_index(drop=True)
    assert (f.year.values <= d.draft_year.values).all(), "final season after the draft"
    d = pd.concat([d, f], axis=1)
    assert not d.duplicated(KEYS).any()  # pid is not unique: match_torvik maps Jaquez 2023 to Clayton's Iona season (as draft_table does)
    return d


def _wmean(df, val, w):
    g = df.assign(_p=df[val] * df[w]).groupby(["conf", "year"])
    return g._p.sum() / g[w].sum()


def _top(df, by, k, ascending, cols):
    """Per (team, year): the k best rows by `by`, as wide columns <col>_0..<col>_{k-1}."""
    s = df[df[by].notna()].sort_values(by, ascending=ascending)
    s = s[s.groupby(["team", "year"]).cumcount() < k]
    s["rk"] = s.groupby(["team", "year"]).cumcount()
    wide = s.set_index(["team", "year", "rk"])[cols].unstack("rk")
    wide.columns = [f"{c}_{r}" for c, r in wide.columns]
    return wide.reindex(columns=[f"{c}_{r}" for c in cols for r in range(k)])


# --------------------------------------------------------------------------- 1-3. roster context, team and conference strength

def _team_season(t: pd.DataFrame) -> pd.DataFrame:
    w = t.minutes.fillna(0)
    r = pd.DataFrame({"team": t.team, "year": t.year, "conf": t.conf, "w": w,
                      "wbp": w * t.bpm_pos.fillna(0), "ptsg": t.pts * t.GP, "astg": t.ast * t.GP,
                      "porp": t.porpag.clip(lower=0), "ranked": (t.rec_rank <= 100).astype(int)})
    r["wbp2"] = r.wbp ** 2
    for s in WEIGHTED:  # weights masked where the stat is missing
        r[f"w_{s}"] = w.where(t[s].notna(), 0)
        r[f"ws_{s}"] = r[f"w_{s}"] * t[s].fillna(0)
    g = r.groupby(["team", "year"])
    ts = g.sum(numeric_only=True)
    ts["n"] = g.size()
    ts["conf"] = g.conf.first()
    ts["team_bpm_sum"] = 5 * ts.wbp / ts.w  # positive BPM of the five men on the floor, minutes-weighted
    ts = ts.join(_top(t[t.Min_per >= MIN_PER_ROTATION], "bpm", 3, False, ["bpm", "pid"]))
    ts = ts.join(_top(t, "rec_rank", 2, True, ["rec_rank", "pid"]).rename(columns=lambda c: "r" + c))
    return ts


def _team_features(D: pd.DataFrame, ts: pd.DataFrame) -> pd.DataFrame:
    x = D.merge(ts.drop(columns="conf").reset_index(), on=["team", "year"], how="left", validate="m:1")
    wi = x.minutes.fillna(0)
    o = pd.DataFrame(index=D.index)
    for s, name in (("bpm", "t_tm_bpm_mw"), ("usg", "t_tm_usg_mw")):  # leave-one-out minutes-weighted mean
        wis = wi.where(x[s].notna(), 0)
        o[name] = ((x[f"ws_{s}"] - wis * x[s].fillna(0)) / (x[f"w_{s}"] - wis)).values
    is1, is2 = x.pid == x.pid_0, x.pid == x.pid_1
    o["t_tm_bpm_max"] = np.where(is1, x.bpm_1, x.bpm_0)
    o["t_tm_top2_bpm_mean"] = np.where(is1, (x.bpm_1 + x.bpm_2) / 2, np.where(is2, (x.bpm_0 + x.bpm_2) / 2, (x.bpm_0 + x.bpm_1) / 2))
    o["t_tm_recrank_best"] = np.where(x.pid == x.rpid_0, x.rrec_rank_1, x.rrec_rank_0)
    o["t_tm_n_ranked_recruits"] = (x.ranked - (x.rec_rank <= 100).astype(int)).values
    o["t_bpm_share"] = (wi * x.bpm_pos / x.wbp).values
    o["t_usg_share"] = (wi * x.usg / x.ws_usg).values
    o["t_pts_share"] = (x.pts * x.GP / x.ptsg).values
    o["t_ast_share"] = (x.ast * x.GP / x.astg).values
    o["t_porpag_share"] = (x.porpag.clip(lower=0) / x.porp).values
    o["t_team_talent_conc"] = (x.wbp2 / x.wbp ** 2).values
    o["t_team_bpm_sum"] = x.team_bpm_sum.values
    o["t_team_adjoe_mean"] = (x.ws_adjoe / x.w_adjoe).values
    o["t_team_adrtg_mean"] = (x.ws_adrtg / x.w_adrtg).values
    o["t_team_n_players"] = x.n.values
    return o.replace([np.inf, -np.inf], np.nan)


def _conf_features(D: pd.DataFrame, ts: pd.DataFrame) -> pd.DataFrame:
    c = ts.reset_index()[["team", "year", "conf", "w", "team_bpm_sum"]]
    same = _wmean(c, "team_bpm_sum", "w").rename("t_conf_strength_same")
    prev = c[["team", "year", "team_bpm_sum", "w"]].rename(columns={"team_bpm_sum": "pbs", "w": "pw"}).assign(year=lambda d: d.year + 1)
    prior = _wmean(c[["team", "year", "conf"]].merge(prev, on=["team", "year"]), "pbs", "pw").rename("t_conf_strength_prior")
    x = D[["conf", "year"]].merge(pd.concat([prior, same], axis=1).reset_index(), on=["conf", "year"], how="left")
    return x[["t_conf_strength_prior", "t_conf_strength_same"]].set_index(D.index)


# --------------------------------------------------------------------------- 4. age curves

def _age_fits(t: pd.DataFrame, years) -> dict:
    """(Y, stat) -> (cubic coefficients, residual std, age lo, age hi), fit on seasons < Y with Min_per >= 40."""
    pop = t[(t.Min_per >= MIN_PER_POP) & t.age.notna()]
    fits = {}
    for Y in years:
        p = pop[pop.year < Y]
        if p.empty:
            continue
        lo, hi = p.age.quantile([0.005, 0.995])
        for s in AGE_STATS:
            q = p[p[s].notna()]
            coef = np.polyfit(q.age, q[s], 3)
            fits[(Y, s)] = (coef, (q[s] - np.polyval(coef, q.age)).std(), lo, hi)
    return fits


def _age_resid(fits, Y, s, age, x):
    if (Y, s) not in fits:
        return np.full(len(x), np.nan), np.full(len(x), np.nan)
    coef, sd, lo, hi = fits[(Y, s)]
    r = np.asarray(x, float) - np.polyval(coef, np.clip(np.asarray(age, float), lo, hi))
    return r, r / sd


def _age_features(D: pd.DataFrame, fits: dict) -> pd.DataFrame:
    o = pd.DataFrame(index=D.index)
    for Y, idx in D.groupby("draft_year").groups.items():
        d = D.loc[idx]
        for s in AGE_STATS:
            o.loc[idx, f"t_{s}_age_resid"], o.loc[idx, f"t_{s}_age_z"] = _age_resid(fits, Y, s, d.age, d[s])
    return o[GROUPS["age"]]


# --------------------------------------------------------------------------- 5. role-relative

def _role_features(D: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame:
    pop = t[(t.Min_per >= MIN_PER_POP) & t.role.notna()]
    st = pop.groupby(["role", "year"])[ROLE_STATS].agg(["mean", "std"])
    st.columns = [f"{s}_{a}" for s, a in st.columns]
    roles = t[t.role.notna()]
    st["n_role"] = roles.groupby(["role", "year"]).size()
    x = D[["role", "year"] + ROLE_STATS].merge(st.reset_index(), on=["role", "year"], how="left")
    x = x.merge(roles.groupby("year").size().rename("n_year").reset_index(), on="year", how="left")
    o = pd.DataFrame({f"t_{s}_role_z": (x[s] - x[f"{s}_mean"]) / x[f"{s}_std"] for s in ROLE_STATS})
    o["t_role_rarity"] = x.n_role / x.n_year
    return o.set_index(D.index)


# --------------------------------------------------------------------------- 6-8. career: development, precocity, shooting

def _career(D: pd.DataFrame, t: pd.DataFrame) -> pd.DataFrame:
    """All Torvik seasons of each draftee up to his final season F, oldest first."""
    c = t.merge(D[["pid", "year"] + KEYS].rename(columns={"year": "F"}), on="pid")
    c = c[c.year <= c.F].sort_values(KEYS + ["year"]).reset_index(drop=True)
    g = c.groupby(KEYS)
    c["rk"], c["rk_end"] = g.cumcount(), g.cumcount(ascending=False)
    return c


def _wls_slopes(c: pd.DataFrame) -> pd.DataFrame:
    """Minutes-weighted least-squares slope of each DEV stat on season year, per draftee (NaN with one usable season)."""
    x, w = c.year.astype(float) - 2017.0, c.minutes.fillna(0)  # centred: keeps W*Wxx - Wx^2 well conditioned
    out = {}
    for s in DEV_STATS:
        ws = w.where(c[s].notna(), 0)
        y = c[s].fillna(0)
        acc = pd.DataFrame({"n": (ws > 0).astype(int), "W": ws, "Wx": ws * x, "Wy": ws * y, "Wxy": ws * x * y, "Wxx": ws * x * x,
                            **{k: c[k] for k in KEYS}}).groupby(KEYS).sum()
        den = (acc.W * acc.Wxx - acc.Wx ** 2).where(acc.n >= 2)
        out[s] = (acc.W * acc.Wxy - acc.Wx * acc.Wy) / den
    return pd.DataFrame(out)


def _dev_features(D: pd.DataFrame, c: pd.DataFrame, fits: dict) -> pd.DataFrame:
    key = pd.MultiIndex.from_frame(D[KEYS])
    first = c[c.rk == 0].set_index(KEYS).reindex(key)
    last = c[c.rk_end == 0].set_index(KEYS).reindex(key)
    n = c.groupby(KEYS).size().reindex(key)
    raw = _wls_slopes(c).reindex(key)
    # population slope for class Y = mean raw slope of draftees with draft_year < Y (expanding, causal)
    by_year = raw.groupby(level="draft_year").agg(["sum", "count"])
    by_year = by_year.reindex(range(by_year.index.min(), by_year.index.max() + 1)).fillna(0).cumsum().shift(1)
    pop = pd.DataFrame({s: by_year[(s, "sum")] / by_year[(s, "count")].where(by_year[(s, "count")] > 0) for s in DEV_STATS})
    pop = pop.reindex(D.draft_year.values)
    nn = n.values[:, None].astype(float)
    shrunk = nn / (nn + 2) * raw.values + 2 / (nn + 2) * pop.values

    o = pd.DataFrame(index=D.index)
    for i, s in enumerate(DEV_STATS):
        o[f"t_dev_{s}_slope"] = shrunk[:, i]
    o["t_dev_n_seasons"] = n.values
    b = c[c.rk_end < 3].set_index(KEYS + ["rk_end"]).bpm.unstack("rk_end").reindex(key)
    o["t_dev_bpm_accel"] = ((b[0] - b[1]) - (b[1] - b[2])).values
    multi = n.values >= 2
    o["t_first_season_bpm"] = np.where(multi, first.bpm.values, np.nan)
    o["t_first_season_age"] = np.where(multi, first.age.values, np.nan)
    o["t_bpm_first_to_last"] = np.where(multi, (last.bpm - first.bpm).values, np.nan)
    o["t_bpm_as_freshman"] = first.bpm.values
    o["t_min_per_as_freshman"] = first.Min_per.values
    o["t_age_first_season"] = first.age.values
    z = np.full(len(D), np.nan)
    for Y, idx in D.groupby("draft_year").indices.items():
        z[idx] = _age_resid(fits, Y, "bpm", first.age.values[idx], first.bpm.values[idx])[1]
    o["t_freshman_bpm_age_z"] = z
    return o


def _mom_beta(p: pd.Series, n: pd.Series):
    """Method-of-moments beta prior; the binomial noise of each observed percentage is removed from the variance."""
    m = p.mean()
    v = p.var() - (m * (1 - m) / n).mean()
    k = m * (1 - m) / v - 1 if v > 0 else 500.0
    k = float(np.clip(k, 2, 500))
    return m * k, (1 - m) * k


def _shooting_priors(t: pd.DataFrame, years) -> pd.DataFrame:
    """Rows (draft_year, role, stat, alpha, beta, n_fit); role 'ALL' is the fallback when the draftee's role is unknown."""
    rows = []
    for Y in years:
        p = t[t.year < Y]
        for stat, made, att in (("ft", "FTM", "FTA"), ("3p", "TPM", "TPA")):
            q = p[p[att] >= MIN_ATT_PRIOR]
            for role, grp in [*q[q.role.notna()].groupby("role"), ("ALL", q)]:
                if len(grp) >= 30:
                    a, b = _mom_beta(grp[made] / grp[att], grp[att])
                    rows.append((Y, role, stat, a, b, len(grp)))
    return pd.DataFrame(rows, columns=["draft_year", "role", "stat", "alpha", "beta", "n_fit"])


def _shooting_features(D: pd.DataFrame, c: pd.DataFrame, priors: pd.DataFrame) -> pd.DataFrame:
    tot = c.groupby(KEYS)[["FTM", "FTA", "TPM", "TPA"]].sum().reindex(pd.MultiIndex.from_frame(D[KEYS]))
    o = pd.DataFrame(index=D.index)
    for stat, made, att in (("ft", "FTM", "FTA"), ("3p", "TPM", "TPA")):
        pr = priors[priors.stat == stat]
        x = D[["draft_year", "role"]].merge(pr, on=["draft_year", "role"], how="left")
        fb = D[["draft_year"]].merge(pr[pr.role == "ALL"], on="draft_year", how="left")
        a, b = x.alpha.fillna(fb.alpha).values, x.beta.fillna(fb.beta).values
        made_, att_ = tot[made].values.astype(float), tot[att].values.astype(float)
        o[f"t_{stat}_shrunk"] = np.where(att_ > 0, (made_ + a) / (att_ + a + b), np.nan)
        o[f"t_{stat}_shrunk_n"] = att_
    fga = D.twoPA + D.TPA
    o["t_3par"] = (D.TPA / fga.where(fga > 0)).values
    o["t_ftr"] = D.ftr.values
    o["t_ft_minus_3p"] = o.t_ft_shrunk - o.t_3p_shrunk
    return o


# --------------------------------------------------------------------------- dictionary

_W_F = "final season F team-season (F <= Y), no fitting"
_W_AGE = "cubic age curve fit on all D1 player-seasons with year < Y and Min_per >= 40 (expanding 2008..Y-1); evaluated on season F"
_W_ROLE = "mean/std within (role, season F) over D1 players with Min_per >= 40; season F is complete on draft night"
_W_DEV = "own seasons <= F; shrunk n/(n+2) toward the mean raw slope of draftees with draft_year < Y"
_W_PRIOR = "beta prior (method of moments) on (role, year < Y) player-seasons with >= 20 attempts; all-D1 year < Y fallback when role unknown"
_N_TEAM = "never null for a matched draftee"


def _dictionary() -> pd.DataFrame:
    rows = [
        ("t_tm_bpm_mw", "teammates' BPM, minutes-weighted mean (draftee excluded)", _W_F, _N_TEAM),
        ("t_tm_bpm_max", "best teammate BPM among teammates with Min_per >= 20 (draftee excluded)", _W_F, "no other rotation teammate"),
        ("t_tm_top2_bpm_mean", "mean BPM of the two best rotation teammates (Min_per >= 20, draftee excluded)", _W_F, "fewer than 2 other rotation teammates"),
        ("t_tm_recrank_best", "lowest (best) high-school recruiting rank among teammates", _W_F, "no ranked recruit among teammates"),
        ("t_tm_n_ranked_recruits", "number of teammates with rec_rank <= 100 (0 = none, observed)", _W_F, _N_TEAM),
        ("t_tm_usg_mw", "teammates' usage, minutes-weighted mean (draftee excluded)", _W_F, _N_TEAM),
        ("t_bpm_share", "own max(bpm,0)*minutes / team sum of max(bpm,0)*minutes", _W_F, "team has no positive-BPM minutes"),
        ("t_usg_share", "own usg*minutes / team sum of usg*minutes (share of possessions used)", _W_F, _N_TEAM),
        ("t_pts_share", "own pts*GP / team pts*GP", _W_F, _N_TEAM),
        ("t_ast_share", "own ast*GP / team ast*GP", _W_F, _N_TEAM),
        ("t_porpag_share", "own max(porpag,0) / team sum of max(porpag,0)", _W_F, "team has no positive porpag"),
        ("t_team_talent_conc", "Herfindahl index of the team's max(bpm,0)*minutes shares (draftee included)", _W_F, "team has no positive-BPM minutes"),
        ("t_team_bpm_sum", "5 * sum(max(bpm,0)*minutes) / sum(minutes): positive BPM of the five on the floor", _W_F, _N_TEAM),
        ("t_team_adjoe_mean", "roster adjoe, minutes-weighted mean", _W_F, _N_TEAM),
        ("t_team_adrtg_mean", "roster adrtg, minutes-weighted mean", _W_F, _N_TEAM),
        ("t_team_n_players", "roster size (player-seasons on the team)", _W_F, _N_TEAM),
        ("t_conf_strength_prior", "minutes-weighted mean team_bpm_sum in season F-1 of the teams that form the conference in season F",
         "season F-1 only; production-safe, computable before season F ends", "no member team has a season F-1 (2008 seasons)"),
        ("t_conf_strength_same", "minutes-weighted mean team_bpm_sum of the conference in season F",
         "season F only (complete on draft night); production-safe", _N_TEAM),
    ]
    for s in AGE_STATS:
        rows.append((f"t_{s}_age_resid", f"{s} in season F minus the age-curve expectation at his season-F age (Jan 15)", _W_AGE, "no fit population (2008 class) or stat missing"))
        rows.append((f"t_{s}_age_z", f"t_{s}_age_resid / residual std of the fit population", _W_AGE, "no fit population (2008 class) or stat missing"))
    for s in ROLE_STATS:
        rows.append((f"t_{s}_role_z", f"z-score of {s} within (role, season F)", _W_ROLE, "role unknown (Torvik seasons 2008-09) or < 2 players in the cell"))
    rows.append(("t_role_rarity", "share of D1 players (with a role) in season F that have the draftee's role", _W_ROLE, "role unknown (Torvik seasons 2008-09)"))
    for s in DEV_STATS:
        rows.append((f"t_dev_{s}_slope", f"minutes-weighted LS slope of {s} per season across own D1 seasons, shrunk; n = t_dev_n_seasons", _W_DEV,
                     "fewer than 2 seasons, or no earlier draftee with 2+ seasons to shrink toward (2009 class)"))
    rows += [
        ("t_dev_n_seasons", "number of own D1 seasons <= F (sample size of every t_dev_* slope)", "own seasons <= F", _N_TEAM),
        ("t_dev_bpm_accel", "(bpm_F - bpm_F-1) - (bpm_F-1 - bpm_F-2) over the last three own seasons", "own seasons <= F", "fewer than 3 seasons"),
        ("t_first_season_bpm", "BPM in first D1 season (draftees with 2+ seasons; same value as t_bpm_as_freshman)", "own seasons <= F", "one season only"),
        ("t_first_season_age", "age in first D1 season (draftees with 2+ seasons; same value as t_age_first_season)", "own seasons <= F", "one season only"),
        ("t_bpm_first_to_last", "BPM in season F minus BPM in first D1 season", "own seasons <= F", "one season only"),
        ("t_bpm_as_freshman", "BPM in his first D1 season, whenever it was", "own seasons <= F", _N_TEAM),
        ("t_min_per_as_freshman", "Min_per in his first D1 season", "own seasons <= F", _N_TEAM),
        ("t_age_first_season", "age (Jan 15) in his first D1 season", "own seasons <= F", "birthdate missing"),
        ("t_freshman_bpm_age_z", "age-adjusted z of first-season BPM using the class-Y age curve", _W_AGE, "no fit population (2008 class)"),
        ("t_ft_shrunk", "career FT% (seasons <= F) shrunk to the role's beta prior: (FTM + a) / (FTA + a + b)", _W_PRIOR, "no FT attempts, or no prior seasons (2008 class)"),
        ("t_ft_shrunk_n", "career FTA (sample size of t_ft_shrunk)", "own seasons <= F", _N_TEAM),
        ("t_3p_shrunk", "career 3P% (seasons <= F) shrunk to the role's beta prior", _W_PRIOR, "no 3P attempts, or no prior seasons (2008 class)"),
        ("t_3p_shrunk_n", "career 3PA (sample size of t_3p_shrunk)", "own seasons <= F", _N_TEAM),
        ("t_3par", "season F 3PA / FGA", _W_F, "no field-goal attempts"),
        ("t_ftr", "season F free-throw rate (Torvik ftr)", _W_F, _N_TEAM),
        ("t_ft_minus_3p", "t_ft_shrunk - t_3p_shrunk (latent shooting hint)", _W_PRIOR, "either shrunk percentage missing"),
    ]
    return pd.DataFrame(rows, columns=["feature_name", "description", "fit_window_leakage_note", "null_meaning"])


# --------------------------------------------------------------------------- build

def build() -> pd.DataFrame:
    t0 = time.time()
    raw = pd.read_parquet(C.PROC / "torvik.parquet")
    t = _prep(raw)
    D = _draftees(raw, t)
    years = sorted(D.draft_year.unique())
    ts = _team_season(t)
    fits = _age_fits(t, years)
    c = _career(D, t)
    out = pd.concat([D[KEYS], _team_features(D, ts), _conf_features(D, ts), _age_features(D, fits), _role_features(D, t),
                     _dev_features(D, c, fits), _shooting_features(D, c, _shooting_priors(t, years))], axis=1)
    feats = [f for g in GROUPS.values() for f in g]
    assert set(out.columns) == set(KEYS + feats) and len(feats) == len(set(feats))
    out = out[KEYS + feats]
    out["draft_year"] = out.draft_year.astype(int)
    out.to_parquet(OUT, index=False)
    d = _dictionary()
    assert list(d.feature_name) == feats
    d.to_csv(DICT, index=False)
    print(f"built {out.shape} in {time.time() - t0:.1f}s -> {OUT.name}, {DICT.name}")
    return out


if __name__ == "__main__":
    ctx = build()
    for g, cols in GROUPS.items():
        cov = ctx[cols].notna().mean()
        print(f"{g:10s} {cov.mean():6.1%} mean coverage  min {cov.min():6.1%} ({cov.idxmin()})")
