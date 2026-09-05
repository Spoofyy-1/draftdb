"""Game-level, context-adjusted pre-draft features (every output column is prefixed g_).

Sources
  hoopR / ESPN men's college basketball player box scores, team box scores and schedules, seasons 2007-2026
  (GitHub sportsdataverse/hoopR-mbb-data, mbb/<kind>/parquet/<kind>_<season>.parquet; season = calendar year it ends,
  2018 == 2017-18). player_core_<season>.parquet is only read for birth dates. Torvik player-seasons
  (data/processed/torvik.parquet) supply season BPM / recruiting rank / birth dates for every D1 player, and the
  ayush + jasong draft CSVs under data/external/ supply real birth dates where Torvik only has its Oct-15 placeholder.

Leakage rule (cutoff = one minute before draft night)
  * Only games, prior seasons, recruiting ranks and birth dates are used. Torvik `pick`, ESPN `draft_*` columns and
    anything about NBA careers are never read; nothing here knows who else in the class was drafted.
  * Population fits (age curves, development-slope priors, shooting beta priors) use Torvik seasons < draft_year
    (refit per draft year; the 2008 class, for which no earlier Torvik season exists, uses seasons <= 2008).
  * Per-season population quantities (league-mean game score, the population usage-slope prior, opponent-talent
    quartile cut points) are computed from the games of the prospect's final season, all of which end in early April.
  * Opponent talent is ex ante: the opponent's SRS from the PRIOR season's results, and the prior-season Torvik BPM and
    recruiting ranks of the players who actually dressed in that game. Same-season opponent ratings are never used.
  * Teammate season BPM is the Torvik value of the same (completed) college season, never anything later.

Outputs
  data/processed/player_game.parquet             one row per player-game, all D1 players, 2007-2026
  data/processed/crosswalk_hoopr.parquet         draftee (key, draft_year) -> hoopR athlete_id with match method / confidence
  data/processed/xwalk_hoopr_torvik.parquet      every D1 player-season: hoopR athlete_id <-> Torvik pid
  data/processed/prospect_game_features.parquet  one row per modelled draftee (key, draft_year), columns g_*
  data/processed/feature_dictionary_g.csv        feature_name, description, source, production_safe, null_meaning, minimum_sample

Game BPM-like score: Hollinger game score (PTS + 0.4 FGM - 0.7 FGA - 0.4 (FTA - FTM) + 0.7 ORB + 0.3 DRB + STL + 0.7 AST
+ 0.7 BLK - 0.4 PF - TOV) per 36 minutes, centred on the season's minutes-weighted D1 mean, so 0 = average D1 minute.

Run:  python -m infra.builders.game_features     # idempotent downloads, then rebuilds everything
      from infra.builders.game_features import load_game_features
"""

from __future__ import annotations

import concurrent.futures as cf
import difflib
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

try:  # the package was renamed nbadraft -> infra during the tournament refactor
    from infra import config as C
    from infra.dataset import norm_name
except ImportError:  # pragma: no cover
    from nbadraft import config as C
    from infra.dataset import norm_name

HOOPR = C.ROOT / "data" / "external" / "hoopr"
EXT = C.ROOT / "data" / "external"
PROC = C.PROC
SEASONS = range(2007, 2027)
URL = "https://raw.githubusercontent.com/sportsdataverse/hoopR-mbb-data/main/mbb/{kind}/parquet/{kind}_{season}.parquet"
KINDS = ("player_box", "team_box", "schedules", "player_core")
KEYS = ["key", "draft_year"]

PLAYER_GAME = PROC / "player_game.parquet"
XWALK_POP = PROC / "xwalk_hoopr_torvik.parquet"
XWALK_TEAMS = PROC / "xwalk_hoopr_teams.parquet"
XWALK = PROC / "crosswalk_hoopr.parquet"
FEATURES = PROC / "prospect_game_features.parquet"
DICTIONARY = PROC / "feature_dictionary_g.csv"

AGE_STATS = {"bpm": "bpm", "ts": "TS_per", "usg": "usg", "ast": "AST_per", "blk": "blk_per", "stl": "stl_per"}
DEV_STATS = {"bpm": "bpm", "ts": "TS_per", "usg": "usg", "ast": "AST_per", "stl": "stl_per", "blk": "blk_per", "mpg": "mpg"}
K_SLOPE, K_STAR, K_DEV = 10.0, 5.0, 2.0  # empirical-Bayes shrinkage n / (n + k)


def _log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# --------------------------------------------------------------------------- 0. downloads

def download() -> None:
    """Fetch every hoopR file that is missing (never re-downloads)."""
    HOOPR.mkdir(parents=True, exist_ok=True)
    def fetch(kind: str, season: int) -> None:
        f = HOOPR / f"{kind}_{season}.parquet"
        if f.exists() and f.stat().st_size > 0:
            return
        remote = f"mbb_schedule_{season}" if kind == "schedules" else f"{kind}_{season}"
        url = f"https://raw.githubusercontent.com/sportsdataverse/hoopR-mbb-data/main/mbb/{kind}/parquet/{remote}.parquet"
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        f.write_bytes(r.content)
        _log("downloaded", f.name, len(r.content))

    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(lambda x: fetch(*x), ((kind, s) for kind in KINDS for s in SEASONS)))


# --------------------------------------------------------------------------- 1. player_game

BOX = {"game_id": "game_id", "season": "season", "season_type": "season_type", "game_date": "game_date", "athlete_id": "athlete_id",
       "athlete_display_name": "name", "athlete_position_abbreviation": "pos", "team_id": "team_id", "team_location": "team",
       "opponent_team_id": "opp_team_id", "opponent_team_location": "opp", "home_away": "home_away", "starter": "starter",
       "did_not_play": "dnp", "minutes": "minutes", "points": "pts", "field_goals_attempted": "fga", "field_goals_made": "fgm",
       "three_point_field_goals_attempted": "tpa", "three_point_field_goals_made": "tpm", "free_throws_attempted": "fta",
       "free_throws_made": "ftm", "offensive_rebounds": "oreb", "defensive_rebounds": "dreb", "rebounds": "reb", "assists": "ast",
       "steals": "stl", "blocks": "blk", "turnovers": "tov", "fouls": "pf", "team_score": "team_score", "opponent_team_score": "opp_score"}
COUNTS = ["pts", "fga", "fgm", "tpa", "tpm", "fta", "ftm", "oreb", "dreb", "reb", "ast", "stl", "blk", "tov", "pf"]


def _season_player_game(s: int) -> pd.DataFrame:
    b = pd.read_parquet(HOOPR / f"player_box_{s}.parquet", columns=list(BOX)).rename(columns=BOX)
    b = b[b.season_type.isin([2, 3]) & b.athlete_id.notna() & b.team_id.notna()].drop_duplicates(["game_id", "athlete_id"]).copy()
    b["dnp"] = b.dnp.fillna(False).astype(bool)
    b["starter"] = b.starter.fillna(False).astype(bool)
    b["played"] = ~b.dnp & b.minutes.notna() & (b.minutes > 0)
    for c in COUNTS:
        b[c] = b[c].astype(float)
    # team totals in that game from the players who played (the usage denominator)
    tm = b[b.played].groupby(["game_id", "team_id"])[["minutes", "pts", "fga", "fgm", "fta", "oreb", "ast", "tov"]].sum().add_prefix("tm_")
    b = b.join(tm, on=["game_id", "team_id"])
    b["tm_min_eff"] = np.where((b.tm_minutes >= 190) & (b.tm_minutes <= 300), b.tm_minutes, 200.0)
    # possessions from the team box (has team turnovers); fall back to the player sums
    tb = pd.read_parquet(HOOPR / f"team_box_{s}.parquet", columns=["game_id", "team_id", "field_goals_attempted", "offensive_rebounds", "total_turnovers", "turnovers", "free_throws_attempted"])
    tb = tb.drop_duplicates(["game_id", "team_id"])
    tb["poss"] = tb.field_goals_attempted - tb.offensive_rebounds + tb.total_turnovers.fillna(tb.turnovers) + 0.44 * tb.free_throws_attempted
    poss = tb.set_index(["game_id", "team_id"]).poss
    own = poss.reindex(pd.MultiIndex.from_arrays([b.game_id, b.team_id])).values
    opp = poss.reindex(pd.MultiIndex.from_arrays([b.game_id, b.opp_team_id])).values
    fallback = b.tm_fga - b.tm_oreb + b.tm_tov + 0.44 * b.tm_fta
    with np.errstate(all="ignore"):
        b["tm_poss"] = np.nanmean(np.c_[own, opp], axis=1)
    b["tm_poss"] = b.tm_poss.fillna(fallback)
    # home = +1, away = -1, neutral site = 0
    sched = pd.read_parquet(HOOPR / f"schedules_{s}.parquet", columns=["game_id", "neutral_site"]).drop_duplicates("game_id")
    neutral = b.game_id.map(sched.set_index("game_id").neutral_site).fillna(False).astype(bool)
    b["home"] = np.where(neutral, 0, np.where(b.home_away == "home", 1, -1)).astype(np.int8)
    # per-game rates
    tsa = b.fga + 0.44 * b.fta
    b["ts"] = np.where(tsa > 0, b.pts / (2 * tsa), np.nan)
    b["efg"] = np.where(b.fga > 0, (b.fgm + 0.5 * b.tpm) / b.fga, np.nan)
    load = b.fga + 0.44 * b.fta + b.tov
    tm_load = b.tm_fga + 0.44 * b.tm_fta + b.tm_tov
    b["poss_share"] = np.where(tm_load > 0, load / tm_load, np.nan)
    b["usg"] = np.where(b.played & (tm_load > 0), 100 * load * (b.tm_min_eff / 5) / (b.minutes * tm_load), np.nan)
    ast_den = (b.minutes / (b.tm_min_eff / 5)) * b.tm_fgm - b.fgm
    b["ast_pct"] = np.where(b.played & (ast_den > 0), 100 * b.ast / ast_den, np.nan)
    b["tov_pct"] = np.where(load > 0, 100 * b.tov / load, np.nan)
    b["gs"] = (b.pts + 0.4 * b.fgm - 0.7 * b.fga - 0.4 * (b.fta - b.ftm) + 0.7 * b.oreb + 0.3 * b.dreb + b.stl + 0.7 * b.ast
               + 0.7 * b.blk - 0.4 * b.pf - b.tov)
    b["gs36"] = np.where(b.played, b.gs / b.minutes * 36, np.nan)
    ok = b.played & (b.minutes >= 1)
    league = np.average(b.gs36[ok], weights=b.minutes[ok])
    b["gbpm"] = b.gs36 - league
    b["margin"] = (b.team_score - b.opp_score).astype(float)
    cols = ["athlete_id", "name", "pos", "team_id", "team", "opp_team_id", "opp", "game_id", "game_date", "season", "season_type", "home",
            "starter", "dnp", "played", "minutes", *COUNTS, "ts", "efg", "usg", "poss_share", "ast_pct", "tov_pct", "gs", "gs36", "gbpm",
            "tm_poss", "tm_min_eff", "margin"]
    out = b[cols].copy()
    out["game_date"] = pd.to_datetime(out.game_date)
    for c in ["athlete_id", "team_id", "opp_team_id", "game_id"]:
        out[c] = out[c].astype("int64")
    out["season"] = out.season.astype("int16")
    out["season_type"] = out.season_type.astype("int8")
    return out.sort_values(["game_date", "game_id", "team_id"]).reset_index(drop=True)


def build_player_game() -> None:
    writer = None
    for s in SEASONS:
        df = _season_player_game(s)
        _log(f"player_game {s}: {len(df):,} rows, {df.game_id.nunique():,} games, {df.athlete_id.nunique():,} athletes")
        tbl = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(PLAYER_GAME, tbl.schema, compression="zstd")
        writer.write_table(tbl)
    writer.close()


def load_player_game(season: int | None = None, played_only: bool = False, columns: list[str] | None = None) -> pd.DataFrame:
    filters = []
    if season is not None:
        filters.append(("season", "==", season))
    if played_only:
        filters.append(("played", "==", True))
    return pd.read_parquet(PLAYER_GAME, filters=filters or None, columns=columns)


# --------------------------------------------------------------------------- 2. crosswalks

def _short_key(name: str) -> str:
    """ESPN sometimes lists a full legal name ('Jerome Adolphus Jordan'): first + last token."""
    p = str(name).split()
    return norm_name(p[0] + " " + p[-1]) if len(p) > 2 else norm_name(name)


ATH_COLS = ["athlete_id", "team_id", "name", "team", "game_id", "minutes", "played"]


def _season_athletes(pg: pd.DataFrame) -> pd.DataFrame:
    g = pg[pg.played].groupby(["athlete_id", "team_id"]).agg(name=("name", "first"), team=("team", "first"), games=("game_id", "size"),
                                                             minutes=("minutes", "sum")).reset_index()
    g["k"] = g.name.map(norm_name)
    g["k2"] = g.name.map(_short_key)
    return g


def _team_vote(pairs: pd.DataFrame, min_n: int = 3, min_share: float = 0.6) -> dict:
    """Torvik team name -> ESPN team_id by majority vote over (team, team_id) pairs of exact-name matches."""
    c = pairs.groupby(["team", "team_id"]).size().reset_index(name="n")
    c["share"] = c.n / c.groupby("team").n.transform("sum")
    c = c[(c.n >= min_n) & (c.share >= min_share)].sort_values("n").drop_duplicates("team", keep="last")
    return dict(zip(c.team, c.team_id))


def _team_pairs(H: pd.DataFrame, T: pd.DataFrame) -> pd.DataFrame:
    return H[["k", "team_id"]].merge(T[["k", "team"]], on="k")[["team", "team_id"]]


def _unique_pairs(m: pd.DataFrame) -> pd.DataFrame:
    """Keep matches where both sides appear once (no ambiguity)."""
    return m[~m.duplicated("athlete_id", keep=False) & ~m.duplicated("pid", keep=False)]


def _fuzzy_pairs(H: pd.DataFrame, T: pd.DataFrame, cutoff: float = 0.85, gap: float = 0.08) -> pd.DataFrame:
    """Within one team-season, pair leftover players by string similarity of normalised names; accept only a clear unique best."""
    rows = []
    for team_id, t in T.groupby("team_id"):
        h = H[H.team_id == team_id]
        if h.empty:
            continue
        for pid, kt in zip(t.pid, t.k):
            scores = sorted(((difflib.SequenceMatcher(None, kt, kh).ratio(), aid) for kh, aid in zip(h.k, h.athlete_id)), reverse=True)
            if scores and scores[0][0] >= cutoff and (len(scores) == 1 or scores[0][0] - scores[1][0] >= gap):
                rows.append((scores[0][1], pid, scores[0][0]))
    if not rows:
        return pd.DataFrame(columns=["athlete_id", "pid", "score"])
    return _unique_pairs(pd.DataFrame(rows, columns=["athlete_id", "pid", "score"]))


def _match_season(H: pd.DataFrame, T: pd.DataFrame, tmap: dict) -> pd.DataFrame:
    """One-to-one hoopR athlete <-> Torvik pid for one season, in priority order."""
    T = T.assign(team_id=T.team.map(tmap))
    out = []

    def take(m, method, conf):
        nonlocal H, T
        m = _unique_pairs(m)
        out.append(m[["athlete_id", "pid"]].assign(method=method, confidence=conf))
        H = H[~H.athlete_id.isin(m.athlete_id)]
        T = T[~T.pid.isin(m.pid)]

    take(H.merge(T, on=["k", "team_id"]), "exact_team", 1.0)
    take(H.merge(T.rename(columns={"k": "k2"}), on=["k2", "team_id"]), "short_team", 0.95)
    take(H.merge(T, on="k"), "exact_season", 0.9)  # unique name in the whole season on both sides
    take(H.merge(T.rename(columns={"k": "k2"}), on="k2"), "short_season", 0.85)
    take(_fuzzy_pairs(H, T[T.team_id.notna()]), "fuzzy_team", 0.7)
    return pd.concat(out, ignore_index=True)


def build_xwalk_population(torvik: pd.DataFrame) -> pd.DataFrame:
    """Every D1 player-season 2008-2026: hoopR athlete_id <-> Torvik pid, plus the Torvik team -> ESPN team_id map."""
    T_all = torvik[["pid", "player_name", "team", "year"]].assign(k=torvik.player_name.map(norm_name))
    frames, tmaps, per_season = [], {}, {}
    for s in range(2008, 2027):
        per_season[s] = (_season_athletes(load_player_game(s, played_only=True, columns=ATH_COLS)), T_all[T_all.year == s])
    # per-season majority vote, falling back to the all-seasons vote for team-seasons with few exact-name matches
    gmap = _team_vote(pd.concat([_team_pairs(H, T) for H, T in per_season.values()]), min_n=5)
    for s, (H, T) in per_season.items():
        tmap = _team_vote(_team_pairs(H, T))
        for team in T.team.unique():
            if team not in tmap and team in gmap:
                tmap[team] = gmap[team]
        tmaps[s] = tmap
        m = _match_season(H, T, tmap).assign(season=s)
        m = m.merge(H[["athlete_id", "team_id", "name"]], on="athlete_id").merge(T[["pid", "team", "player_name"]], on="pid")
        frames.append(m)
        _log(f"xwalk {s}: {len(m):,} of {len(T):,} Torvik rows matched ({len(m) / len(T):.1%}); teams mapped {len(tmap)}")
    xw = pd.concat(frames, ignore_index=True)
    tm = pd.DataFrame([(s, t, i) for s, mp in tmaps.items() for t, i in mp.items()], columns=["season", "team", "team_id"])
    xw.to_parquet(XWALK_POP, index=False)
    tm.to_parquet(XWALK_TEAMS, index=False)
    return xw


def _college(draft: pd.DataFrame) -> pd.DataFrame:
    """Draftees with a matched Torvik final season (the draft table's `modelled` flag now also covers international prospects)."""
    return draft[draft.torvik_idx.notna()].drop_duplicates(KEYS).copy()


def build_crosswalk(draft: pd.DataFrame, torvik: pd.DataFrame, xw: pd.DataFrame) -> pd.DataFrame:
    """Draftee -> hoopR athlete_id for his final college season (Torvik year of the matched season).

    Priority: exact name + team + season (1.0), exact name + season (0.9), name + team in an adjacent season (0.8),
    fuzzy within team-season (0.7). Ambiguous candidates are never accepted."""
    tmaps = pd.read_parquet(XWALK_TEAMS)
    tmap = {(s, t): i for s, t, i in zip(tmaps.season, tmaps.team, tmaps.team_id)}
    xw_idx = xw.set_index(["pid", "season"])
    m = _college(draft)
    m["pid"] = torvik.pid.reindex(m.torvik_idx.astype(int).values).values
    m["final_season"] = torvik.year.reindex(m.torvik_idx.astype(int).values).values.astype(int)
    m["torvik_key"] = m.player_name.map(norm_name)
    ath = {s: _season_athletes(load_player_game(s, played_only=True, columns=ATH_COLS)) for s in range(2008, 2027)}
    rows = []
    for r in m.itertuples(index=False):
        s, keys = r.final_season, {r.key, r.torvik_key}
        team_id = tmap.get((s, r.college_team))
        H = ath[s]
        res = dict(key=r.key, draft_year=r.draft_year, player=r.player, college_team=r.college_team, final_season=s, pid=r.pid, team_id_expected=team_id,
                   athlete_id=np.nan, hoopr_name=None, hoopr_team_id=np.nan, hoopr_team=None, match_method="unmatched", confidence=np.nan, n_games=np.nan, note="")
        cand = H[H.k.isin(keys) | H.k2.isin(keys)]
        pick = None
        if (r.pid, s) in xw_idx.index:  # population crosswalk already settled it
            x = xw_idx.loc[(r.pid, s)]
            x = x.iloc[0] if isinstance(x, pd.DataFrame) else x
            hit = H[H.athlete_id == x.athlete_id]
            if len(hit):
                method = {"exact_team": "name_team_season", "short_team": "name_team_season", "exact_season": "name_season", "short_season": "name_season", "fuzzy_team": "fuzzy_team_season"}[x.method]
                pick = (hit.iloc[0], method, float(x.confidence))
        if pick is None and len(cand):
            same = cand[cand.team_id == team_id] if team_id is not None else cand.iloc[0:0]
            if len(same) == 1:
                pick = (same.iloc[0], "name_team_season", 1.0)
            elif len(cand) == 1:
                pick = (cand.iloc[0], "name_season", 0.9)
            else:
                res["note"] = f"ambiguous: {len(cand)} same-name athletes in {s}"
        if pick is None and team_id is not None:  # name + team in an adjacent season (hoopR season tag off by one)
            for s2 in (s - 1, s + 1):
                if s2 in ath:
                    c2 = ath[s2]
                    c2 = c2[(c2.k.isin(keys) | c2.k2.isin(keys)) & (c2.team_id == team_id)]
                    if len(c2) == 1:
                        pick = (c2.iloc[0], "name_team_adjacent_season", 0.8)
                        break
        if pick is None and team_id is not None:  # fuzzy within the team-season
            h = H[H.team_id == team_id]
            scores = sorted(((max(difflib.SequenceMatcher(None, k, kh).ratio() for k in keys), i) for kh, i in zip(h.k, h.index)), reverse=True)
            if scores and scores[0][0] >= 0.8 and (len(scores) == 1 or scores[0][0] - scores[1][0] >= 0.08):
                pick = (h.loc[scores[0][1]], "fuzzy_team_season", round(0.5 + 0.25 * scores[0][0], 3))
            elif scores and scores[0][0] >= 0.8:
                res["note"] = "ambiguous fuzzy candidates"
        if pick is not None:
            a, method, conf = pick
            res.update(athlete_id=int(a.athlete_id), hoopr_name=a["name"], hoopr_team_id=int(a.team_id), hoopr_team=a.team, match_method=method,
                       confidence=conf, n_games=int(a.games))
            if team_id is not None and a.team_id != team_id and method != "name_team_adjacent_season":
                res["note"] = f"team mismatch: expected ESPN team_id {team_id}, got {int(a.team_id)}"
        rows.append(res)
    cw = pd.DataFrame(rows)
    cw.to_parquet(XWALK, index=False)
    return cw


# --------------------------------------------------------------------------- numeric helpers

def _wmean(x, w) -> tuple[float, float, int]:
    """Weighted mean, its standard error (sqrt(sum (w_i (x_i - m))^2) / sum w) and n over the finite pairs."""
    x, w = np.asarray(x, float), np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[ok], w[ok]
    if len(x) == 0:
        return np.nan, np.nan, 0
    m = np.average(x, weights=w)
    se = np.sqrt(np.sum((w * (x - m)) ** 2)) / w.sum() if len(x) > 1 else np.nan
    return float(m), float(se), int(len(x))


def _wls(y, X, w) -> tuple[np.ndarray, np.ndarray]:
    """Weighted least squares (weights normalised to mean 1): coefficients and standard errors. X must contain the intercept."""
    y, X, w = np.asarray(y, float), np.asarray(X, float), np.asarray(w, float)
    n, p = X.shape
    nan = np.full(p, np.nan)
    if n <= p:
        return nan, nan
    w = w / w.mean()
    Xw = X * w[:, None]
    A = X.T @ Xw
    if np.linalg.cond(A) > 1e10:
        return nan, nan
    Ainv = np.linalg.inv(A)
    beta = Ainv @ (Xw.T @ y)
    e = y - X @ beta
    s2 = (w * e ** 2).sum() / (n - p)
    return beta, np.sqrt(np.maximum(np.diag(Ainv) * s2, 0))


def _shrink(x: float, n: float, pop: float, k: float) -> float:
    w = n / (n + k)
    return w * x + (1 - w) * pop


def _ratio_se(a: float, b: float) -> float:
    """Delta-method SE of a/b for two independent Poisson-like counts."""
    return abs(a / b) * np.sqrt(1 / max(a, 1) + 1 / b) if b > 0 else np.nan


def _season_age(dob, year) -> float:
    return (pd.Timestamp(int(year), 2, 1) - pd.Timestamp(dob)).days / 365.25


# --------------------------------------------------------------------------- 3a. birth dates

def _real_dob(s: pd.Series) -> pd.Series:
    """Torvik fills unknown birth dates with Oct 15 of an estimated year."""
    return s.notna() & ~((s.dt.month == 10) & (s.dt.day == 15))


def _external_dobs() -> pd.DataFrame:
    """Birth dates from the ayush / jasong draft CSVs (public biographical fact): key, year, dob, src."""
    a = pd.read_csv(EXT / "ayush_draft_players.csv", usecols=["Name", "Year", "Birthdate"])
    a = pd.DataFrame({"key": a.Name.map(norm_name), "year": pd.to_numeric(a.Year, errors="coerce"),
                      "dob": pd.to_datetime(a.Birthdate, errors="coerce", format="mixed"), "src": "ayush"})
    j = pd.read_csv(EXT / "jasong_draft_db.csv", usecols=["Name", "Season", "Birthday"])
    j = pd.DataFrame({"key": j.Name.map(norm_name), "year": pd.to_numeric(j.Season.astype(str).str[:4], errors="coerce") + 1,
                      "dob": pd.to_datetime(j.Birthday, errors="coerce", format="mixed"), "src": "jasong"})
    return pd.concat([a, j], ignore_index=True).dropna(subset=["dob", "year"])


def _espn_dobs() -> pd.Series:
    """athlete_id -> date of birth from player_core, dropping dates shared by > 3 athletes (ESPN placeholders)."""
    frames = [pd.read_parquet(HOOPR / f"player_core_{s}.parquet", columns=["athlete_id", "date_of_birth"]) for s in SEASONS
              if (HOOPR / f"player_core_{s}.parquet").exists()]
    pc = pd.concat(frames).dropna().drop_duplicates()
    pc["dob"] = pd.to_datetime(pc.date_of_birth.str[:10], errors="coerce")
    pc = pc.dropna(subset=["dob"])
    pc = pc[pc.dob.map(pc.groupby("dob").athlete_id.nunique()) <= 3]
    return pc.drop_duplicates("athlete_id").set_index("athlete_id").dob


def _resolve_dob(key, draft_year, torvik_dob, athlete_id, ext, espn) -> tuple[pd.Timestamp, str]:
    if pd.notna(torvik_dob) and _real_dob(pd.Series([torvik_dob])).iloc[0]:
        return pd.Timestamp(torvik_dob), "torvik"
    c = ext[(ext.key == key) & ((ext.year - draft_year).abs() <= 2)]
    if len(c):
        c = c.assign(d=(c.year - draft_year).abs(), o=(c.src == "jasong").astype(int)).sort_values(["d", "o"])
        return pd.Timestamp(c.dob.iloc[0]), c.src.iloc[0]
    if pd.notna(athlete_id) and int(athlete_id) in espn.index:
        return pd.Timestamp(espn.loc[int(athlete_id)]), "espn"
    if pd.notna(torvik_dob):
        return pd.Timestamp(torvik_dob), "torvik_imputed"
    return pd.NaT, "none"


# --------------------------------------------------------------------------- 3b/8/9. population priors from Torvik seasons < draft_year

def _torvik_prep(torvik: pd.DataFrame) -> pd.DataFrame:
    t = torvik.copy()
    t["real_dob"] = _real_dob(t.birthdate)
    t["age"] = (pd.to_datetime(t.year.astype(str) + "-02-01") - t.birthdate).dt.days / 365.25
    t["minutes"] = t.GP * t.mpg
    band = pd.cut(t.height_in, [0, 75.5, 79.5, 100], labels=["G", "W", "B"]).astype("string")
    t["grp"] = t.role.astype("string").fillna(band)  # Torvik role where it exists (2010+), else a height band
    t["age_band"] = pd.cut(t.age, [0, 20, 22, 99], labels=["u20", "20to22", "o22"]).astype("string")
    return t


def _age_band(age: float):
    return pd.NA if not np.isfinite(age) else "u20" if age <= 20 else "20to22" if age <= 22 else "o22"


def _pop_years(t: pd.DataFrame, draft_year: int) -> pd.DataFrame:
    """Torvik seasons strictly before the draft year; the 2008 class has none, so it uses 2008 (still pre-cutoff)."""
    return t[t.year <= max(draft_year - 1, 2008)]


def _age_curves(t: pd.DataFrame, draft_year: int) -> dict:
    pop = _pop_years(t, draft_year)
    pop = pop[pop.real_dob & (pop.GP >= 10) & (pop.Min_per >= 30) & pop.age.between(17, 27) & (pop.minutes > 0)]
    x = np.clip(pop.age.values, 17.5, 25.5)
    curves = {}
    for name, col in AGE_STATS.items():
        y = pop[col].values
        ok = np.isfinite(y)
        coef = np.polyfit(x[ok], y[ok], 3, w=np.sqrt(pop.minutes.values[ok]))
        resid = y[ok] - np.polyval(coef, x[ok])
        sd = np.sqrt(np.average(resid ** 2, weights=pop.minutes.values[ok]))
        curves[name] = (coef, float(sd), int(ok.sum()))
    return curves


def _group_wslope(d: pd.DataFrame, g: str, x: str, y: str, w: str) -> pd.Series:
    """Per-group weighted least-squares slope of y on x."""
    d = d[[g, x, y, w]].dropna()
    d = d[d[w] > 0]
    sw = d.groupby(g)[w].transform("sum")
    xm = (d[w] * d[x]).groupby(d[g]).transform("sum") / sw
    ym = (d[w] * d[y]).groupby(d[g]).transform("sum") / sw
    num = (d[w] * (d[x] - xm) * (d[y] - ym)).groupby(d[g]).sum()
    den = (d[w] * (d[x] - xm) ** 2).groupby(d[g]).sum()
    return (num / den)[den > 0]


def _dev_priors(t: pd.DataFrame, draft_year: int) -> dict:
    """Population year-over-year slope of each stat (minutes-weighted mean over players with >= 2 seasons, 1-99 pct clipped)."""
    # two earlier seasons are needed for a slope: the 2009 class uses 2008-2009 (its own, completed, pre-cutoff season)
    pop = t[t.year <= max(draft_year - 1, 2009)]
    pop = pop[pop.GP >= 5]
    pop = pop[pop.pid.map(pop.groupby("pid").size()) >= 2]
    tot = pop.groupby("pid").minutes.sum()
    out = {}
    for name, col in DEV_STATS.items():
        sl = _group_wslope(pop, "pid", "year", col, "minutes")
        if len(sl) < 30:
            out[name] = (np.nan, int(len(sl)))
            continue
        lo, hi = sl.quantile([0.01, 0.99])
        out[name] = (float(np.average(sl.clip(lo, hi), weights=tot.reindex(sl.index))), int(len(sl)))
    return out


SHOT = {"ft": ("FTM", "FTA"), "3p": ("TPM", "TPA"), "rim": ("rim_made", "rim_att"), "mid": ("mid_made", "mid_att")}


def _beta_mom(made: np.ndarray, att: np.ndarray) -> tuple[float, float]:
    p = made / att
    m = p.mean()
    v = max(p.var() - (p * (1 - p) / att).mean(), 1e-4)
    k = max(m * (1 - m) / v - 1, 1.0)
    return float(m * k), float((1 - m) * k)


def _shoot_priors(t: pd.DataFrame, draft_year: int) -> dict:
    """Beta priors (alpha, beta) per shot type, keyed by (grp, age_band) with (grp,) and () fallbacks; >= 20 attempts, >= 30 players."""
    pop = _pop_years(t, draft_year)
    priors = {}
    for name, (mcol, acol) in SHOT.items():
        sub = pop[(pop[acol] >= 20) & pop[mcol].notna()]
        if len(sub) < 30:  # no earlier season records this shot type (rim/mid start in 2010): use the completed final season itself
            sub = t[(t.year <= draft_year) & (t[acol] >= 20) & t[mcol].notna()]
        if len(sub) < 30:
            continue
        priors[(name,)] = _beta_mom(sub[mcol].values, sub[acol].values)
        for grp, s1 in sub.groupby("grp"):
            if len(s1) >= 30:
                priors[(name, grp)] = _beta_mom(s1[mcol].values, s1[acol].values)
            for band, s2 in s1.groupby("age_band"):
                if len(s2) >= 30:
                    priors[(name, grp, band)] = _beta_mom(s2[mcol].values, s2[acol].values)
    return priors


def _prior_for(priors: dict, name: str, grp, band) -> tuple[tuple[float, float] | None, str]:
    for k in [(name, grp, band), (name, grp), (name,)]:
        if all(pd.notna(x) for x in k) and k in priors:
            return priors[k], "/".join(map(str, k[1:])) or "all"
    return None, "none"


# --------------------------------------------------------------------------- 5a. prior-season SRS

def _srs(season: int, ridge: float = 1.0) -> pd.Series:
    """Ridge least-squares Simple Rating System (margin = r_home - r_away + hca) from that season's completed results.
    Teams with < 10 games (non-D1 opponents) are dropped; D1 mean = 0."""
    s = pd.read_parquet(HOOPR / f"schedules_{season}.parquet", columns=["home_id", "away_id", "home_score", "away_score", "status_type_completed", "neutral_site", "season_type"])
    s = s[s.status_type_completed.fillna(False).astype(bool) & s.home_score.notna() & s.away_score.notna() & s.season_type.isin([2, 3])]
    teams = np.unique(np.r_[s.home_id.values, s.away_id.values])
    idx = pd.Series(np.arange(len(teams)), index=teams)
    n, m = len(teams), len(s)
    X = np.zeros((m, n + 1))
    X[np.arange(m), idx[s.home_id.values].values] = 1
    X[np.arange(m), idx[s.away_id.values].values] = -1
    X[:, n] = (~s.neutral_site.fillna(False).astype(bool)).astype(float)
    y = (s.home_score - s.away_score).values.astype(float)
    A = X.T @ X + ridge * np.eye(n + 1)
    beta = np.linalg.solve(A, X.T @ y)
    games = pd.Series(np.bincount(np.r_[idx[s.home_id.values].values, idx[s.away_id.values].values], minlength=n), index=teams)
    r = pd.Series(beta[:n], index=teams)[games >= 10]
    return r - r.mean()


# --------------------------------------------------------------------------- 5b/6. season context shared by every prospect of that season

def _season_context(S: int, xw: pd.DataFrame, ath_pid: pd.Series, t: pd.DataFrame) -> dict:
    pg = load_player_game(S)
    xs = xw[xw.season == S].drop_duplicates("athlete_id").set_index("athlete_id").pid
    pg["pid"] = pg.athlete_id.map(xs)
    pg["pid_any"] = pg.pid.fillna(pg.athlete_id.map(ath_pid))
    tS = t[t.year == S].drop_duplicates("pid").set_index("pid")
    tP = t[t.year == S - 1].drop_duplicates("pid").set_index("pid")
    rec = t.dropna(subset=["rec_rank"]).drop_duplicates("pid").set_index("pid").rec_rank
    pg["bpm_season"] = pg.pid.map(tS.bpm)        # completed-season Torvik BPM of a teammate (same season, pre-cutoff)
    pg["rec_rank"] = pg.pid_any.map(rec)          # recruiting rank: known before college
    pg["bpm_prior"] = pg.pid_any.map(tP.bpm)      # Torvik BPM in the PRIOR season: known before this season
    pg["load"] = (pg.minutes * pg.usg).fillna(0.0)

    # ex-ante talent of each team in each game: prior-season SRS, prior-season BPM and recruiting index of the players who dressed
    pl = pg[pg.played]
    gk = [pl.game_id, pl.team_id]
    tot = pl.minutes.groupby(gk).sum()

    def wavg(col, fill=None):
        v = pl[col] if fill is None else pl[col].fillna(fill)
        ok = v.notna()
        num = (pl.minutes * v).where(ok, 0).groupby(gk).sum()
        den = pl.minutes.where(ok, 0).groupby(gk).sum()
        return num / den.replace(0, np.nan), den / tot

    tg = pd.DataFrame(index=tot.index)
    tg["roster_bpm_prior"], tg["roster_bpm_cov"] = wavg("bpm_prior")
    tg.loc[tg.roster_bpm_cov < 0.4, "roster_bpm_prior"] = np.nan
    tg["roster_rec_index"], _ = wavg("rec_rank", fill=0.0)  # unranked recruits sit at 0 on the 0-100 scale
    srs = _srs(S - 1)
    tg["srs_prior"] = srs.reindex(tg.index.get_level_values("team_id")).values

    def z(x):
        return (x - x.mean()) / x.std()

    tg["talent"] = pd.concat([z(tg.srs_prior), z(tg.roster_bpm_prior), z(tg.roster_rec_index)], axis=1).mean(axis=1, skipna=True)
    q75, q50 = tg.talent.quantile([0.75, 0.5])
    okey = pd.MultiIndex.from_arrays([pg.game_id, pg.opp_team_id])
    for c in ["talent", "srs_prior", "roster_bpm_prior"]:
        pg[f"opp_{c}"] = tg[c].reindex(okey).values
    pg["opp_top"] = pg.opp_talent >= q75
    pg["opp_bot"] = pg.opp_talent < q50
    return {"pg": pg, "pop": _population_slopes(pg[pg.played]), "q": (float(q75), float(q50)), "n_teams": int(srs.size)}


ELASTIC = {"ts": "ts", "ast": "ast_pct", "tov": "tov_pct"}


def _usage_slopes(g: pd.DataFrame) -> dict:
    """Within-player WLS of each rate on usage (+ opponent talent, home), minutes as weights."""
    out = {}
    d = g.assign(opp_t=g.opp_talent.fillna(0.0))  # unknown opponent strength = average, control variable only
    for name, col in ELASTIC.items():
        s = d.dropna(subset=[col, "usg"])
        b, se = _wls(s[col], np.c_[np.ones(len(s)), s.usg, s.opp_t, s.home], s.minutes) if len(s) >= 6 else (np.full(4, np.nan),) * 2
        out[name] = (float(b[1]), float(se[1]), int(len(s)))
    return out


def _population_slopes(pl: pd.DataFrame) -> dict:
    """Season prior for the usage slopes: n-weighted mean of the slopes of regulars (>= 15 games, >= 12 mpg), 1-99 pct clipped."""
    reg = pl.groupby("athlete_id").agg(n=("game_id", "size"), mpg=("minutes", "mean"))
    d = pl[pl.athlete_id.isin(reg.index[(reg.n >= 15) & (reg.mpg >= 12)])]
    rows = [_usage_slopes(g) for _, g in d.groupby("athlete_id")]
    out = {}
    for name in ELASTIC:
        s = np.array([(r[name][0], r[name][2]) for r in rows if np.isfinite(r[name][0])])
        lo, hi = np.percentile(s[:, 0], [1, 99])
        out[name] = (float(np.average(np.clip(s[:, 0], lo, hi), weights=s[:, 1])), int(len(s)))
    return out


# --------------------------------------------------------------------------- 7. star-teammate absence

STAR_STATS = {"usg": ("usg", "minutes"), "ts": ("ts", "tsa"), "ast": ("ast_pct", "minutes"), "tov": ("tov_pct", "minutes"),
              "min": ("minutes", None), "bpm": ("gbpm", "minutes")}


def _star_absence(pg: pd.DataFrame, g: pd.DataFrame, aid: int, team_id: int) -> dict:
    team = pg[pg.team_id == team_id]
    games = team[["game_date", "game_id"]].drop_duplicates().sort_values(["game_date", "game_id"]).game_id.values
    load = team.pivot_table(index="athlete_id", columns="game_id", values="load", aggfunc="sum").reindex(columns=games).fillna(0.0)
    mins = team.pivot_table(index="athlete_id", columns="game_id", values="minutes", aggfunc="sum").reindex(columns=games).fillna(0.0)
    L = load.values
    Cum = np.c_[np.zeros(len(L)), np.cumsum(L, axis=1)]
    n_games = len(games)
    prior = np.stack([Cum[:, j] - Cum[:, max(j - 10, 0)] for j in range(n_games)], axis=1)  # rolling sum over the previous 10 team games
    if aid in load.index:
        prior[load.index.get_loc(aid)] = -np.inf
    out_flag = {}
    for j, gid in enumerate(games):
        if j < 5:
            continue
        col = prior[:, j]
        top = np.argsort(col)[-2:]
        top = top[col[top] > 0]
        if len(top) < 2:
            continue
        out_flag[gid] = bool((mins.values[top, j] < 5).any())
    f = {"g_star_anchor_games": 0, "g_star_absence_games": 0}
    s = g[g.game_id.isin(out_flag)].assign(out=lambda d: d.game_id.map(out_flag).astype(bool))
    f["g_star_anchor_games"] = int(len(s))
    f["g_star_absence_games"] = int(s.out.sum())
    if s.out.sum() == 0 or (~s.out).sum() == 0:
        return f
    for name, (col, wcol) in STAR_STATS.items():
        w_out = s.loc[s.out, wcol] if wcol else np.ones(int(s.out.sum()))
        w_in = s.loc[~s.out, wcol] if wcol else np.ones(int((~s.out).sum()))
        m1, se1, n1 = _wmean(s.loc[s.out, col], w_out)
        m0, se0, n0 = _wmean(s.loc[~s.out, col], w_in)
        delta = m1 - m0
        f[f"g_star_out_{name}_delta"] = delta
        f[f"g_star_out_{name}_delta_se"] = float(np.sqrt(se1 ** 2 + se0 ** 2))
        f[f"g_star_out_{name}_delta_n"] = n1
        f[f"g_star_out_{name}_delta_shrunk"] = _shrink(delta, n1, 0.0, K_STAR) if np.isfinite(delta) else np.nan
    return f


# --------------------------------------------------------------------------- 4/5/6. per-prospect game features

SPLIT_STATS = {"ts": ("ts", "tsa"), "bpm": ("gbpm", "minutes"), "usg": ("usg", "minutes")}


def _game_feats(ctx: dict, aid: int, own_bpm: float) -> dict:
    pg = ctx["pg"]
    g = pg[(pg.athlete_id == aid) & pg.played].sort_values("game_date").assign(tsa=lambda d: d.fga + 0.44 * d.fta)
    n = len(g)
    f = {"g_n_games": n, "g_n_starts": int(g.starter.sum()), "g_n_games_post": int((g.season_type == 3).sum()),
         "g_minutes_total": float(g.minutes.sum()), "g_mpg": float(g.minutes.mean()) if n else np.nan}
    if n < 3:
        return f
    team_id = int(g.team_id.mode().iloc[0])
    f["g_usg_mw"], _, _ = _wmean(g.usg, g.minutes)
    f["g_usg_sd"] = float(np.sqrt(np.average((g.usg.fillna(f["g_usg_mw"]) - f["g_usg_mw"]) ** 2, weights=g.minutes)))
    f["g_gbpm_mw"], f["g_gbpm_se"], _ = _wmean(g.gbpm, g.minutes)
    f["g_ts_agg"] = float(g.pts.sum() / (2 * g.tsa.sum())) if g.tsa.sum() > 0 else np.nan

    # ---- 4. teammates, leave-one-out, minutes-weighted over the games he played
    his = pg[pg.game_id.isin(g.game_id) & (pg.team_id == team_id) & pg.played]
    mates = his[his.athlete_id != aid]
    M = mates.groupby("athlete_id").agg(minutes=("minutes", "sum"), bpm=("bpm_season", "first"), rec=("rec_rank", "first"), games=("game_id", "size"))
    f["g_tm_n"] = int(len(M))
    f["g_tm_bpm_mw"], _, f["g_tm_bpm_mw_n"] = _wmean(M.bpm, M.minutes)
    f["g_tm_bpm_cov"] = float(M.minutes[M.bpm.notna()].sum() / M.minutes.sum())
    core = M[M.games >= max(5, 0.25 * n)]
    f["g_tm_bpm_max"] = float(core.bpm.max()) if core.bpm.notna().any() else np.nan
    f["g_tm_recrank_avg"], _, f["g_tm_recrank_n"] = _wmean(M.rec, M.minutes)
    f["g_tm_recrank_share"] = float(M.minutes[M.rec.notna()].sum() / M.minutes.sum())
    if f["g_tm_bpm_cov"] >= 0.6 and pd.notna(own_bpm):
        pos = M.minutes * M.bpm.clip(lower=0)
        own_pos = g.minutes.sum() * max(own_bpm, 0.0)
        tot = pos.sum() + own_pos
        if tot > 0:
            f["g_bpm_share"] = float(own_pos / tot)
            f["g_team_talent_conc"] = float(((pos / tot) ** 2).sum() + (own_pos / tot) ** 2)
    load_own, load_tm = (g.fga + 0.44 * g.fta + g.tov).sum(), (his.fga + 0.44 * his.fta + his.tov).sum()
    f["g_usg_share"] = float(load_own / load_tm) if load_tm > 0 else np.nan
    f["g_scoring_share"] = float(g.pts.sum() / his.pts.sum()) if his.pts.sum() > 0 else np.nan
    f["g_ast_share"] = float(g.ast.sum() / his.ast.sum()) if his.ast.sum() > 0 else np.nan

    # ---- 5. opponent quality (ex ante) and splits
    f["g_opp_talent_mw"], _, f["g_opp_talent_n"] = _wmean(g.opp_talent, g.minutes)
    f["g_opp_srs_prior_mw"], _, _ = _wmean(g.opp_srs_prior, g.minutes)
    f["g_opp_roster_bpm_prior_mw"], _, _ = _wmean(g.opp_roster_bpm_prior, g.minutes)
    known = g[g.opp_talent.notna()]
    top, bot, rest = known[known.opp_top], known[known.opp_bot], known[~known.opp_top]
    f["g_n_games_top_opp"], f["g_n_games_bot_opp"], f["g_n_games_rest_opp"] = len(top), len(bot), len(rest)
    for name, (col, wcol) in SPLIT_STATS.items():
        res = {}
        for lab, sub in [("top", top), ("bot", bot), ("rest", rest)]:
            res[lab] = _wmean(sub[col], sub[wcol])
            f[f"g_{name}_vs_{lab}_opp"], f[f"g_{name}_vs_{lab}_opp_se"], f[f"g_{name}_vs_{lab}_opp_n"] = res[lab]
        f[f"g_{name}_top_minus_rest"] = res["top"][0] - res["rest"][0]
        f[f"g_{name}_top_minus_rest_se"] = float(np.sqrt(res["top"][1] ** 2 + res["rest"][1] ** 2))
    res = {}
    for lab, sub in [("top", top), ("bot", bot), ("rest", rest)]:
        a, b = float(sub.ast.sum()), float(sub.tov.sum())
        res[lab] = (a / b if b > 0 else np.nan, _ratio_se(a, b), len(sub))
        f[f"g_ast_tov_vs_{lab}_opp"], f[f"g_ast_tov_vs_{lab}_opp_se"], f[f"g_ast_tov_vs_{lab}_opp_n"] = res[lab]
    f["g_ast_tov_top_minus_rest"] = res["top"][0] - res["rest"][0]
    f["g_ast_tov_top_minus_rest_se"] = float(np.sqrt(res["top"][1] ** 2 + res["rest"][1] ** 2))

    # ---- 6. role elasticity
    for name, (slope, se, k) in _usage_slopes(g).items():
        pop, _ = ctx["pop"][name]
        f[f"g_{name}_usg_slope"], f[f"g_{name}_usg_slope_se"], f[f"g_{name}_usg_slope_n"] = slope, se, k
        f[f"g_{name}_usg_slope_pop"] = pop
        f[f"g_{name}_usg_slope_shrunk"] = _shrink(slope, k, pop, K_SLOPE) if np.isfinite(slope) else np.nan

    # ---- 7. star-teammate absence
    f.update(_star_absence(pg, g, aid, team_id))
    return f


# --------------------------------------------------------------------------- 3/8/9. Torvik-based features per prospect

def _torvik_feats(t: pd.DataFrame, pid: int, S: int, age_final: float, grp, band, curves: dict, dev_prior: dict, priors: dict) -> dict:
    f = {}
    rows = t[(t.pid == pid) & (t.year <= S)].sort_values("year")
    fin = rows[rows.year == S]
    if fin.empty:
        return f
    fin = fin.iloc[0]
    # 3. age-adjusted production of the final season
    if np.isfinite(age_final):
        xa = np.clip(age_final, 17.5, 25.5)
        for name, col in AGE_STATS.items():
            coef, sd, _ = curves[name]
            if pd.notna(fin[col]):
                f[f"g_{name}_age_z"] = float((fin[col] - np.polyval(coef, xa)) / sd)
        f["g_age_curve_n"] = curves["bpm"][2]
    # 8. development velocity over his college seasons
    rows = rows[(rows.GP >= 5) & (rows.minutes > 0)]
    f["g_dev_n_seasons"] = int(len(rows))
    if len(rows) >= 2:
        X = np.c_[np.ones(len(rows)), rows.year.values - rows.year.values.mean()]  # centred: the slope is unchanged, the design is well conditioned
        for name, col in DEV_STATS.items():
            y = rows[col].values.astype(float)
            ok = np.isfinite(y)
            if ok.sum() >= 2:
                if ok.sum() == 2:
                    slope, se = float(np.diff(y[ok])[0] / np.diff(rows.year.values[ok])[0]), np.nan
                else:
                    b, s = _wls(y[ok], X[ok], rows.minutes.values[ok])
                    slope, se = float(b[1]), float(s[1])
                pop, _ = dev_prior[name]
                f[f"g_dev_{name}_slope"], f[f"g_dev_{name}_slope_se"], f[f"g_dev_{name}_slope_n"] = slope, se, int(ok.sum())
                f[f"g_dev_{name}_slope_pop"] = pop
                f[f"g_dev_{name}_slope_shrunk"] = _shrink(slope, ok.sum(), pop, K_DEV) if np.isfinite(slope) and np.isfinite(pop) else np.nan
        prev = rows[rows.year == S - 1]
        if len(prev) and pd.notna(prev.bpm.iloc[0]) and pd.notna(fin.bpm):
            f["g_latest_year_improvement_bpm"] = float(fin.bpm - prev.bpm.iloc[0])
    # 9. latent shooting
    for name, (mcol, acol) in SHOT.items():
        pr, _ = _prior_for(priors, name, grp, band)
        made, att = fin[mcol], fin[acol]
        f[f"g_{name}_att"] = float(att) if pd.notna(att) else np.nan
        if pr is not None and pd.notna(att) and pd.notna(made):
            a, b = pr
            f[f"g_{name}_shrunk"] = float((made + a) / (att + a + b))
            f[f"g_{name}_prior_mean"] = a / (a + b)
            f[f"g_{name}_prior_n"] = a + b
        if name in ("ft", "3p") and pr is not None:
            cm, ca = rows[mcol].sum(), rows[acol].sum()
            f[f"g_{name}_att_career"] = float(ca)
            f[f"g_{name}_shrunk_career"] = float((cm + pr[0]) / (ca + pr[0] + pr[1])) if ca > 0 else np.nan
    f["g_shoot_prior_group"] = _prior_for(priors, "ft", grp, band)[1]
    fga = fin.twoPA + fin.TPA
    f["g_3par"] = float(fin.TPA / fga) if fga > 0 else np.nan
    f["g_ftr"] = float(fin.FTA / fga) if fga > 0 else np.nan
    return f


# --------------------------------------------------------------------------- build

def build_features(draft: pd.DataFrame, torvik: pd.DataFrame, xw: pd.DataFrame, cw: pd.DataFrame) -> pd.DataFrame:
    t = _torvik_prep(torvik)
    m = _college(draft).merge(cw[["key", "draft_year", "athlete_id", "match_method", "confidence", "final_season", "pid"]], on=KEYS, how="left")
    ext, espn = _external_dobs(), _espn_dobs()
    ath_pid = xw.groupby("athlete_id").pid.agg(lambda s: s.mode().iloc[0])
    # first D1 game of every matched athlete, all seasons in hoopR (2007+)
    ids = m.athlete_id.dropna().astype(int).tolist()
    first = load_player_game(played_only=True, columns=["athlete_id", "game_date", "season", "played"])
    first = first[first.athlete_id.isin(ids)].groupby("athlete_id").agg(first_date=("game_date", "min"), first_season=("season", "min"))
    priors, rows = {}, []
    for S, grp_s in m.groupby("final_season"):
        S = int(S)
        _log(f"features: final season {S}, {len(grp_s)} prospects")
        ctx = _season_context(S, xw, ath_pid, t)
        for r in grp_s.itertuples(index=False):
            Y = int(r.draft_year)
            if Y not in priors:
                priors[Y] = (_age_curves(t, Y), _dev_priors(t, Y), _shoot_priors(t, Y))
            curves, dev_prior, shoot = priors[Y]
            tor = t.loc[int(r.torvik_idx)]
            dob, src = _resolve_dob(r.key, Y, tor.birthdate, r.athlete_id, ext, espn)
            f = {"key": r.key, "draft_year": Y, "g_final_season": S, "g_hoopr_matched": bool(pd.notna(r.athlete_id)), "g_match_method": r.match_method,
                 "g_match_conf": r.confidence, "g_dob_source": src, "g_dob_imputed": src in ("torvik_imputed", "none")}
            age_final = _season_age(dob, S) if pd.notna(dob) else np.nan
            if pd.notna(dob):
                f["g_age_at_draft_exact"] = (pd.Timestamp(f"{Y}-{C.DRAFT_DAY}") - dob).days / 365.25
                f["g_age_final_season"] = age_final
                if pd.notna(r.athlete_id) and int(r.athlete_id) in first.index:
                    fr = first.loc[int(r.athlete_id)]
                    f["g_first_game_season"] = int(fr.first_season)
                    f["g_first_game_censored"] = bool(fr.first_season <= 2007)
                    if fr.first_season > 2007:
                        f["g_age_first_college_game"] = (fr.first_date - dob).days / 365.25
            f.update(_torvik_feats(t, int(r.pid), S, age_final, tor.grp, _age_band(age_final), curves, dev_prior, shoot))
            if pd.notna(r.athlete_id):
                f.update(_game_feats(ctx, int(r.athlete_id), tor.bpm))
            rows.append(f)
    out = pd.DataFrame(rows)
    front = KEYS + ["g_final_season", "g_hoopr_matched", "g_match_method", "g_match_conf"]
    return out[front + sorted(c for c in out.columns if c not in front)]


def coverage_report(feat: pd.DataFrame, cw: pd.DataFrame) -> str:
    probes = {"matched": "g_hoopr_matched", "age_z": "g_bpm_age_z", "first_game": "g_age_first_college_game", "teammates": "g_tm_bpm_mw",
              "opp_split": "g_ts_top_minus_rest", "elastic": "g_ts_usg_slope", "star_abs": "g_star_out_usg_delta",
              "develop": "g_dev_bpm_slope", "shooting": "g_ft_shrunk"}
    tab = feat.groupby("draft_year").agg(n=("key", "size"), **{k: (c, lambda s: int(s.sum()) if s.dtype == bool else int(s.notna().sum())) for k, c in probes.items()})
    tab.loc["all"] = tab.sum()
    un = cw[cw.match_method == "unmatched"]
    return "\n".join([tab.to_string(), "", f"match methods: {cw.match_method.value_counts().to_dict()}",
                      f"unmatched draftees ({len(un)}): " + ", ".join(f"{p} {y}" for p, y in zip(un.player, un.draft_year)),
                      f"dob sources: {feat.g_dob_source.value_counts().to_dict()}"])


def build(rebuild_player_game: bool = True) -> pd.DataFrame:
    """Idempotent downloads, then every table end to end."""
    download()
    if rebuild_player_game or not PLAYER_GAME.exists():
        build_player_game()
    torvik = pd.read_parquet(PROC / "torvik.parquet")
    draft = pd.read_parquet(PROC / "draft_table.parquet")
    xw = build_xwalk_population(torvik)
    cw = build_crosswalk(draft, torvik, xw)
    feat = build_features(draft, torvik, xw, cw)
    feat.to_parquet(FEATURES, index=False)
    write_dictionary(feat)
    print(coverage_report(feat, cw))
    return feat


def load_game_features() -> pd.DataFrame:
    """One row per modelled draftee (key, draft_year); every other column is prefixed g_."""
    return pd.read_parquet(FEATURES)


def player_game_with_age(season: int) -> pd.DataFrame:
    """player_game rows of one season with `age_at_game` (years) from the Torvik birth date via the population crosswalk.
    Null when the athlete has no Torvik match or only Torvik's Oct-15 placeholder date (`dob_imputed` is True then)."""
    pg = load_player_game(season)
    xw = pd.read_parquet(XWALK_POP, columns=["athlete_id", "season", "pid"])
    dob = pd.read_parquet(PROC / "torvik.parquet", columns=["pid", "birthdate"]).dropna().drop_duplicates("pid").set_index("pid").birthdate
    pid = pg.athlete_id.map(xw[xw.season == season].drop_duplicates("athlete_id").set_index("athlete_id").pid)
    d = pid.map(dob)
    pg["dob_imputed"] = ~_real_dob(d) & d.notna()
    pg["age_at_game"] = ((pg.game_date - d).dt.days / 365.25).where(_real_dob(d))
    return pg


# --------------------------------------------------------------------------- feature dictionary

def _dictionary_rows() -> list[dict]:
    TOR, HOOPR_SRC, BOTH = "Torvik", "hoopR/ESPN box scores", "hoopR/ESPN box scores + Torvik"
    rows = []

    def add(name, desc, src, null, minimum):
        rows.append(dict(feature_name=name, description=desc, source=src, production_safe=True, null_meaning=null, minimum_sample=minimum))

    add("g_final_season", "Torvik season of the final college season (calendar year it ends)", TOR, "never null", "")
    add("g_hoopr_matched", "draftee matched to a hoopR athlete_id for his final season", HOOPR_SRC, "never null", "")
    add("g_match_method", "crosswalk method: name_team_season / name_season / name_team_adjacent_season / fuzzy_team_season / unmatched", HOOPR_SRC, "never null", "")
    add("g_match_conf", "crosswalk confidence 1.0 / 0.9 / 0.8 / 0.5+0.25*similarity", HOOPR_SRC, "unmatched", "")
    add("g_dob_source", "birth date source: torvik (real), ayush, jasong, espn, torvik_imputed (Oct-15 placeholder), none", BOTH, "never null", "")
    add("g_dob_imputed", "True when only Torvik's Oct-15 placeholder birth date exists", TOR, "never null", "")
    add("g_age_at_draft_exact", "age in years on draft day (config DRAFT_DAY) from the resolved birth date", BOTH, "no birth date", "")
    add("g_age_final_season", "age on Feb 1 of the final college season", BOTH, "no birth date", "")
    add("g_age_first_college_game", "age at the first D1 game he played in hoopR (2007+)", BOTH, "no birth date, unmatched, or first game before the 2006-07 season (censored)", "")
    add("g_first_game_season", "hoopR season of his first D1 game", HOOPR_SRC, "unmatched", "")
    add("g_first_game_censored", "first observed game is in 2006-07, the first hoopR season, so the true first game may be earlier", HOOPR_SRC, "unmatched", "")
    for k, col in AGE_STATS.items():
        add(f"g_{k}_age_z", f"final-season Torvik {col} minus a cubic age curve, in residual SDs; curve fit on D1 player-seasons < draft_year with real birth dates, GP>=10, Min_per>=30, minutes-weighted", TOR, "no birth date or stat", "fit population in g_age_curve_n")
    add("g_age_curve_n", "player-seasons in the age-curve fit population (seasons < draft_year)", TOR, "no birth date", "")
    add("g_n_games", "final-season games played (minutes > 0), regular season + postseason", HOOPR_SRC, "unmatched", "")
    add("g_n_starts", "final-season starts", HOOPR_SRC, "unmatched", "")
    add("g_n_games_post", "final-season postseason games played (conference + NCAA/NIT tournaments)", HOOPR_SRC, "unmatched", "")
    add("g_minutes_total", "final-season minutes", HOOPR_SRC, "unmatched", "")
    add("g_mpg", "final-season minutes per game played", HOOPR_SRC, "unmatched", "")
    add("g_usg_mw", "minutes-weighted mean game usage rate (100 * (FGA+0.44FTA+TOV) * TmMin/5 / (Min * Tm(FGA+0.44FTA+TOV)))", HOOPR_SRC, "< 3 games", "3 games")
    add("g_usg_sd", "minutes-weighted SD of game usage across his games (how much his role varied)", HOOPR_SRC, "< 3 games", "3 games")
    add("g_gbpm_mw", "minutes-weighted mean game BPM-like score (Hollinger game score per 36 minus the season D1 mean)", HOOPR_SRC, "< 3 games", "3 games")
    add("g_gbpm_se", "standard error of g_gbpm_mw", HOOPR_SRC, "< 3 games", "3 games")
    add("g_ts_agg", "aggregate true shooting over his games (PTS / 2(FGA + 0.44 FTA))", HOOPR_SRC, "< 3 games", "3 games")
    add("g_tm_n", "distinct teammates who played in games he played", HOOPR_SRC, "< 3 games", "3 games")
    add("g_tm_bpm_mw", "teammates' final-season Torvik BPM, weighted by their minutes in the games he played (he is excluded)", BOTH, "no teammate matched to Torvik", "g_tm_bpm_mw_n teammates")
    add("g_tm_bpm_mw_n", "teammates with a Torvik BPM entering g_tm_bpm_mw", BOTH, "< 3 games", "")
    add("g_tm_bpm_cov", "share of teammate minutes covered by a Torvik BPM", BOTH, "< 3 games", "")
    add("g_tm_bpm_max", "best teammate BPM among teammates who played >= max(5, 25% of his games)", BOTH, "no such teammate with BPM", "")
    add("g_tm_recrank_avg", "teammates' Torvik recruiting rank (0-100, 100 = best), minutes-weighted over ranked teammates", BOTH, "no ranked teammate", "g_tm_recrank_n")
    add("g_tm_recrank_n", "ranked teammates entering g_tm_recrank_avg", BOTH, "< 3 games", "")
    add("g_tm_recrank_share", "share of teammate minutes played by ranked recruits", BOTH, "< 3 games", "")
    add("g_bpm_share", "own minutes * max(BPM,0) / team total of minutes * max(BPM,0) over his games (him included in the denominator)", BOTH, "teammate BPM coverage < 60% or team has no positive BPM", "")
    add("g_team_talent_conc", "Herfindahl index of the team's minutes*max(BPM,0) shares over his games (1 = one player has all the talent)", BOTH, "teammate BPM coverage < 60%", "")
    add("g_usg_share", "his FGA+0.44FTA+TOV / team's in the games he played", HOOPR_SRC, "< 3 games", "3 games")
    add("g_scoring_share", "his points / team points in the games he played", HOOPR_SRC, "< 3 games", "3 games")
    add("g_ast_share", "his assists / team assists in the games he played", HOOPR_SRC, "< 3 games", "3 games")
    add("g_opp_talent_mw", "minutes-weighted mean ex-ante opponent talent: season z-score average of prior-season SRS, prior-season Torvik BPM of the opponents who dressed (minutes-weighted, >=40% coverage) and their recruiting index (unranked = 0)", BOTH, "no opponent with any component", "g_opp_talent_n games")
    add("g_opp_talent_n", "games with an opponent talent value", BOTH, "< 3 games", "")
    add("g_opp_srs_prior_mw", "minutes-weighted mean opponent SRS from the PRIOR season (ridge least squares on margins, D1 mean 0)", HOOPR_SRC, "no opponent with a prior-season SRS", "")
    add("g_opp_roster_bpm_prior_mw", "minutes-weighted mean of opponents' prior-season Torvik BPM (players who dressed, minutes-weighted)", BOTH, "no opponent roster with >= 40% prior-season coverage", "")
    add("g_n_games_top_opp", "games vs opponents in the top quartile of the season's team-game talent distribution", BOTH, "< 3 games", "")
    add("g_n_games_bot_opp", "games vs opponents below the season median of talent", BOTH, "< 3 games", "")
    add("g_n_games_rest_opp", "games vs opponents with known talent not in the top quartile", BOTH, "< 3 games", "")
    labels = {"ts": ("game true shooting, weighted by FGA + 0.44 FTA", "TSA"), "bpm": ("game BPM-like score, minutes-weighted", "minutes"), "usg": ("game usage, minutes-weighted", "minutes"),
              "ast_tov": ("aggregate assists / turnovers", "counts")}
    for k, (d, _) in labels.items():
        for lab, what in [("top", "top-quartile"), ("bot", "bottom-half"), ("rest", "non-top-quartile")]:
            add(f"g_{k}_vs_{lab}_opp", f"{d} vs {what} opponents", BOTH, "no such games", f"g_{k}_vs_{lab}_opp_n games")
            add(f"g_{k}_vs_{lab}_opp_se", f"standard error of g_{k}_vs_{lab}_opp" + (" (delta method)" if k == "ast_tov" else ""), BOTH, "< 2 such games", "")
            add(f"g_{k}_vs_{lab}_opp_n", f"games in g_{k}_vs_{lab}_opp", BOTH, "< 3 games", "")
        add(f"g_{k}_top_minus_rest", f"{d}: top-quartile opponents minus the rest", BOTH, "missing either side", "1 game each side")
        add(f"g_{k}_top_minus_rest_se", f"standard error of g_{k}_top_minus_rest", BOTH, "missing either side", "2 games each side")
    for k, col in ELASTIC.items():
        add(f"g_{k}_usg_slope", f"within-player WLS slope of game {col} on game usage, controlling for opponent talent and home/away, minutes as weights", BOTH, "< 6 usable games or no usage variation", "6 games")
        add(f"g_{k}_usg_slope_se", f"standard error of g_{k}_usg_slope", BOTH, "slope missing", "")
        add(f"g_{k}_usg_slope_n", f"games in the g_{k}_usg_slope regression", BOTH, "< 3 games", "")
        add(f"g_{k}_usg_slope_pop", f"season population slope (n-weighted mean over regulars with >= 15 games, >= 12 mpg, 1-99 pct clipped)", BOTH, "< 3 games", "")
        add(f"g_{k}_usg_slope_shrunk", f"g_{k}_usg_slope shrunk to the population slope with weight n/(n+{K_SLOPE:.0f})", BOTH, "slope missing", "")
    add("g_star_anchor_games", "his games where two anchors (top-2 teammates by minutes*usage over the team's previous 10 games) were defined", HOOPR_SRC, "< 3 games", "")
    add("g_star_absence_games", "anchor-defined games in which an anchor did not play or played < 5 minutes", HOOPR_SRC, "< 3 games", "")
    star = {"usg": "game usage (minutes-weighted)", "ts": "game true shooting (TSA-weighted)", "ast": "game assist % (minutes-weighted)", "tov": "game turnover % (minutes-weighted)", "min": "minutes", "bpm": "game BPM-like score (minutes-weighted)"}
    for k, d in star.items():
        add(f"g_star_out_{k}_delta", f"{d} in anchor-out games minus his other anchor-defined games", HOOPR_SRC, "no anchor-out game or no other game", "1 game each side")
        add(f"g_star_out_{k}_delta_se", f"standard error of g_star_out_{k}_delta", HOOPR_SRC, "delta missing", "2 games each side")
        add(f"g_star_out_{k}_delta_n", f"anchor-out games entering g_star_out_{k}_delta", HOOPR_SRC, "delta missing", "")
        add(f"g_star_out_{k}_delta_shrunk", f"g_star_out_{k}_delta shrunk toward 0 with weight n/(n+{K_STAR:.0f})", HOOPR_SRC, "delta missing", "")
    add("g_dev_n_seasons", "Torvik college seasons (GP >= 5) up to the final season", TOR, "never null", "")
    for k, col in DEV_STATS.items():
        add(f"g_dev_{k}_slope", f"minutes-weighted slope per season of Torvik {col} across his college seasons", TOR, "< 2 seasons", "2 seasons")
        add(f"g_dev_{k}_slope_se", f"standard error of g_dev_{k}_slope", TOR, "< 3 seasons", "3 seasons")
        add(f"g_dev_{k}_slope_n", f"seasons in g_dev_{k}_slope", TOR, "< 2 seasons", "")
        add(f"g_dev_{k}_slope_pop", f"population slope of {col} (minutes-weighted mean over D1 players with >= 2 seasons, seasons < draft_year)", TOR, "< 2 seasons", "")
        add(f"g_dev_{k}_slope_shrunk", f"g_dev_{k}_slope shrunk to the population slope with weight n/(n+{K_DEV:.0f})", TOR, "< 2 seasons", "")
    add("g_latest_year_improvement_bpm", "final-season Torvik BPM minus the previous season's", TOR, "no previous season", "2 seasons")
    for k, (mc, ac) in SHOT.items():
        add(f"g_{k}_shrunk", f"final-season {mc}/{ac} shrunk with a beta prior fitted by (Torvik role or height band, age band) on seasons < draft_year", TOR, "no attempts recorded (rim/mid start in 2010)", "attempts in g_{k}_att")
        add(f"g_{k}_att", f"final-season {ac}", TOR, "not recorded", "")
        add(f"g_{k}_prior_mean", f"mean of the beta prior used for g_{k}_shrunk", TOR, "no prior", "")
        add(f"g_{k}_prior_n", f"strength (alpha + beta) of the beta prior used for g_{k}_shrunk", TOR, "no prior", "")
    for k in ("ft", "3p"):
        add(f"g_{k}_shrunk_career", f"career (all college seasons <= final) {k} percentage shrunk with the same prior", TOR, "no attempts", f"attempts in g_{k}_att_career")
        add(f"g_{k}_att_career", f"career {k} attempts through the final season", TOR, "never null", "")
    add("g_shoot_prior_group", "prior cell used: role/age band, role, or all", TOR, "never null", "")
    add("g_3par", "final-season 3PA / FGA", TOR, "no FGA", "")
    add("g_ftr", "final-season FTA / FGA", TOR, "no FGA", "")
    return rows


def write_dictionary(feat: pd.DataFrame) -> None:
    d = pd.DataFrame(_dictionary_rows())
    missing = [c for c in feat.columns if c.startswith("g_") and c not in set(d.feature_name)]
    extra = [c for c in d.feature_name if c not in feat.columns]
    if missing:
        raise ValueError(f"undocumented features: {missing}")
    d = d[~d.feature_name.isin(extra)]
    d.to_csv(DICTIONARY, index=False)


if __name__ == "__main__":
    build()
