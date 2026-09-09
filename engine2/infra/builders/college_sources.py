"""Pre-draft feature loaders for the college-side sources (combine, Kaggle mirror, SCORE, ianstack, hoopR, March Madness).

Every loader returns a frame keyed by (key, draft_year) -- key = infra.dataset.norm_name(player) -- whose other
columns are numeric and carry a source prefix. Only information knowable on draft night: no NBA stats, no mock-draft
ranks, no model outputs. Raw files live under data/external/<source>/; all downloads are idempotent.

  load_combine()        c_   NBA.com stats API draft combine (anthro + drills), combines 2000-2025
  load_kaggle_college() k_   Kaggle "College Basketball Players 2009-2021" via HF mirror -- a Torvik snapshot, nothing new
  load_score()          s_   SCORE network nba_draft 1990-2021 -- pick/college + NBA career only, no pre-draft columns
  load_ianstack()       is_  ianstack/NBA-Draft-Combine-Analysis -- final college season Win Shares (sports-reference)
  load_hoopr()          h_   hoopR/ESPN game-level box scores -> within-season variability, starts, vs top-50, last-10, basic season line
  load_marchmadness()   mm_  Emlembow/march-madness-2026-data -- tournament-team player-seasons with team SOS/SRS/seed

Run `python -m infra.builders.college_sources` for the coverage report against data/processed/draft_table.parquet.
"""

import json

import numpy as np
import pandas as pd

from infra import config as C
from infra.dataset import _same_school, norm_name

EXT = C.ROOT / "data" / "external"
KEYS = ["key", "draft_year"]


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _draftees() -> pd.DataFrame:
    """(key, draft_year, college) of every draftee -- the identity list only, no outcomes."""
    d = pd.read_parquet(C.PROC / "drafts.parquet", columns=["player", "draft_year", "college"])
    return d.assign(key=d.player.map(norm_name)).drop_duplicates(KEYS)


def _resolve_collisions(df: pd.DataFrame, team: str) -> pd.DataFrame:
    """Two players with the same normalised name in one season: keep the row whose team matches the draftee's college,
    else the first row (callers pre-sort by minutes so that is the bigger role). Drops the team column."""
    dup = df.duplicated(KEYS, keep=False)
    if dup.any():
        d = df[dup].merge(_draftees()[KEYS + ["college"]], on=KEYS, how="left")
        d["_ok"] = [_same_school(a, b) for a, b in zip(d.college.fillna(""), d[team].fillna(""))]
        d = d.sort_values("_ok", ascending=False, kind="stable").drop_duplicates(KEYS).drop(columns=["_ok", "college"])
        df = pd.concat([df[~dup], d], ignore_index=True)
    return df.drop(columns=team).reset_index(drop=True)


# --------------------------------------------------------------------------- 1. NBA.com draft combine

# stats.nba.com refuses cloud hosts; this public dump of the same `draftcombinestats` endpoint (identical values on the
# 2022-23 overlap) supplies the combines the local JSON archive lacks, written in the endpoint's own resultSets layout.
COMBINE_MIRROR = "https://raw.githubusercontent.com/MichLitt/nba-draft-oracle-pro/main/data/raw/combine_2000_2026_raw.csv"


def _mirror_recent_combines():
    folder = EXT / "nba_combine"
    have = {f.stem.split("_")[1] for f in folder.glob("draftcombinestats_*.json")}
    want = {f"{y}-{str(y + 1)[-2:]}" for y in range(2000, C.LAST_SEASON + 1)} - have
    if not want:
        return
    csv = folder / "combine_mirror.csv"
    if not csv.exists():
        import urllib.request
        urllib.request.urlretrieve(COMBINE_MIRROR, csv)
    m = pd.read_csv(csv, encoding="utf-8-sig")
    for season, rows in m[m.SEASON.isin(want)].groupby("SEASON"):
        rows = rows.drop(columns=["TEMP_PLAYER_ID", "DRAFT_YEAR"], errors="ignore").astype(object).where(lambda d: d.notna(), None)
        payload = {"resultSets": [{"name": "DraftCombineStats", "headers": rows.columns.tolist(), "rowSet": rows.values.tolist()}]}
        (folder / f"draftcombinestats_{season}.json").write_text(json.dumps(payload))
        print(f"combine {season}: {len(rows)} rows from mirror")

def load_combine() -> pd.DataFrame:
    """data/external/nba_combine/draftcombinestats_<SeasonYear>.json (also *anthro* / *drillresults* files, subsets).
    SeasonYear 'YYYY-YY' is the combine held in May YYYY for the YYYY draft (2019-20 rows are 2019 draftees, not 2020).
    A draftee absent from his own year's combine takes his latest combine up to two years earlier (tested the waters,
    withdrew): measured before draft night, so leakage-free. c_shuttle is the modified lane agility drill (2013+);
    c_shoot_pct pools every spot-up / off-dribble / on-move shooting drill (made / attempted)."""
    cols = {"HEIGHT_WO_SHOES": "c_height_noshoes", "WINGSPAN": "c_wingspan", "STANDING_REACH": "c_standing_reach", "WEIGHT": "c_weight",
            "BODY_FAT_PCT": "c_body_fat", "HAND_LENGTH": "c_hand_length", "HAND_WIDTH": "c_hand_width",
            "STANDING_VERTICAL_LEAP": "c_vert_standing", "MAX_VERTICAL_LEAP": "c_vert_max", "LANE_AGILITY_TIME": "c_lane_agility",
            "MODIFIED_LANE_AGILITY_TIME": "c_shuttle", "THREE_QUARTER_SPRINT": "c_sprint", "BENCH_PRESS": "c_bench"}
    _mirror_recent_combines()
    frames = []
    for f in sorted((EXT / "nba_combine").glob("draftcombinestats_*.json")):
        rs = json.loads(f.read_text())["resultSets"][0]
        frames.append(pd.DataFrame(rs["rowSet"], columns=rs["headers"]).assign(combine_year=int(f.stem[-7:-3])))
    c = pd.concat(frames, ignore_index=True)
    out = pd.DataFrame({"key": c.PLAYER_NAME.map(norm_name), "combine_year": c.combine_year})
    for src, dst in cols.items():
        out[dst] = _num(c[src])
    out["c_wing_minus_height"] = out.c_wingspan - out.c_height_noshoes
    out["c_reach_minus_height"] = out.c_standing_reach - out.c_height_noshoes
    height = out.c_height_noshoes.replace(0, np.nan)
    out["c_wing_height_ratio"] = out.c_wingspan / height
    out["c_reach_height_ratio"] = out.c_standing_reach / height
    out["c_hand_length_height_ratio"] = out.c_hand_length / height
    out["c_hand_width_height_ratio"] = out.c_hand_width / height
    out["c_weight_per_in"] = out.c_weight / height
    out["c_bmi_like"] = 703 * out.c_weight / height.pow(2)
    out["c_lean_mass"] = out.c_weight * (1 - out.c_body_fat / 100)
    out["c_fat_mass"] = out.c_weight * out.c_body_fat / 100
    out["c_lean_mass_height_ratio"] = out.c_lean_mass / height
    out["c_approach_vert_gain"] = out.c_vert_max - out.c_vert_standing
    out["c_max_touch"] = out.c_standing_reach + out.c_vert_max
    out["c_standing_touch"] = out.c_standing_reach + out.c_vert_standing
    anthro = ["c_height_noshoes", "c_wingspan", "c_standing_reach", "c_weight", "c_body_fat", "c_hand_length", "c_hand_width"]
    drills = ["c_vert_standing", "c_vert_max", "c_lane_agility", "c_shuttle", "c_sprint", "c_bench"]
    out["c_anthro_n"] = out[anthro].notna().sum(axis=1)
    out["c_drills_n"] = out[drills].notna().sum(axis=1)
    made = att = 0.0
    for col in [x for x in c.columns if x.startswith(("SPOT_", "OFF_DRIB_", "ON_MOVE_"))]:
        ma = c[col].astype("string").str.extract(r"^(\d+)-(\d+)$").astype(float)
        made, att = made + ma[0].fillna(0), att + ma[1].fillna(0)
    # BRIDGE: the public MichLitt mirror of `draftcombinestats` carries anthro + drills but no shooting-drill
    # BRIDGE: columns, so the loop above finds nothing and `att` is still the scalar 0.0. Emit an empty c_shoot_pct
    # BRIDGE: rather than crashing; with the original stats.nba.com JSON archive present this branch is not taken.
    out["c_shoot_pct"] = (made / att.replace(0, np.nan)) if hasattr(att, "replace") else np.nan

    cand = _draftees()[KEYS].merge(out, on="key")
    cand = cand[(cand.combine_year <= cand.draft_year) & (cand.combine_year >= cand.draft_year - 2)]
    cand = cand.sort_values("combine_year").drop_duplicates(KEYS, keep="last")
    return cand.drop(columns="combine_year").reset_index(drop=True)


# --------------------------------------------------------------------------- 2. Kaggle college players (HF mirror)

def load_kaggle_college() -> pd.DataFrame:
    """data/external/kaggle_college/ (huggingface_hub.snapshot_download of jason1966/adityak2003_college-basketball-players-20092021).
    CollegeBasketballPlayers2009-2021.csv is a 2021 snapshot of barttorvik.com getadvstats: 60,858 of its 61,061 rows match
    data/processed/torvik.parquet on (pid, year) with identical GP / pts / mpg / rec_rank. Column map onto TORVIK_COLS:
    Ortg=ORtg, 'Rec Rank'=rec_rank, 'ast/tov'=ast_tov, rimmade..=rim_made/rim_att/rim_pct, midmade..=mid_made/mid_att/mid_pct,
    dunksmade..=dunk_made/dunk_att/dunk_pct, mp=mpg, 'Unnamed: 64'=role, 'Unnamed: 65'=unk65; 'type' is the constant 'all'.
    No column is new, so this returns only the keys of the player-seasons it holds (season == draft_year), and the
    DraftedPlayers xlsx / RAPTOR files (NBA outcomes) are not used."""
    k = pd.read_csv(EXT / "kaggle_college" / "CollegeBasketballPlayers2009-2021.csv", usecols=["player_name", "year"], low_memory=False)
    out = pd.DataFrame({"key": k.player_name.map(norm_name), "draft_year": k.year.astype(int)})
    return out.drop_duplicates(KEYS).reset_index(drop=True)


# --------------------------------------------------------------------------- 3. SCORE network draft table

def load_score() -> pd.DataFrame:
    """data/external/score/nba_draft.csv (linked from data.scorenetwork.org/basketball/nba_draft_1990-2021.html).
    Columns are pick / team / college / draft_year plus NBA career totals (games, minutes, points, WS, ...). No position,
    height, weight or age, so there is no pre-draft column to expose: returns keys only."""
    s = pd.read_csv(EXT / "score" / "nba_draft.csv", usecols=["player_name", "draft_year"])
    out = pd.DataFrame({"key": s.player_name.map(norm_name), "draft_year": s.draft_year.astype(int)})
    return out.drop_duplicates(KEYS).reset_index(drop=True)


# --------------------------------------------------------------------------- 4. ianstack combine + college WS

def load_ianstack() -> pd.DataFrame:
    """data/external/ianstack/*.csv (GitHub ianstack/NBA-Draft-Combine-Analysis, data/). College_WS.csv = NBA.com combine
    rows 2000-2022 (same source as load_combine, so not repeated here) + 'College WS' = sports-reference Win Shares of the
    player's final college season. Final_dataset_Cleaned.csv is the same after mean-imputing the measurements; not used.
    SEASON is the combine year == draft year (no fallback: WS of an earlier season is not the final one)."""
    w = pd.read_csv(EXT / "ianstack" / "College_WS.csv", index_col=0)
    out = pd.DataFrame({"key": w.PLAYER_NAME.map(norm_name), "draft_year": w.SEASON.astype(int), "is_college_ws": _num(w["College WS"])})
    return out.dropna(subset=["is_college_ws"]).drop_duplicates(KEYS).reset_index(drop=True)


# --------------------------------------------------------------------------- 5. hoopR / ESPN game logs

BOX_COLS = ["game_id", "season_type", "game_date", "athlete_id", "athlete_display_name", "team_id", "team_location", "opponent_team_id",
            "minutes", "field_goals_made", "field_goals_attempted", "three_point_field_goals_made", "three_point_field_goals_attempted",
            "free_throws_made", "free_throws_attempted", "offensive_rebounds", "defensive_rebounds", "rebounds", "assists", "steals", "blocks",
            "turnovers", "fouls", "points", "starter", "did_not_play"]
HOOPR_SEASONS = range(2003, 2027)  # ESPN boxes exist from 2002-03, but 2003 holds 1 game and 2004 11; full coverage starts 2005


def _srs(sched: pd.DataFrame, iters: int = 50) -> tuple[pd.Series, pd.Series]:
    """Simple Rating System from completed results: rating = mean(margin + opponent rating), iterated. Also games per team."""
    g = sched[sched.status_type_completed & sched.home_score.notna() & sched.away_score.notna()]
    a = pd.DataFrame({"team": np.r_[g.home_id, g.away_id], "opp": np.r_[g.away_id, g.home_id],
                      "margin": np.r_[g.home_score - g.away_score, g.away_score - g.home_score].astype(float)})
    rating = pd.Series(0.0, index=np.unique(a.team))
    for _ in range(iters):
        rating = (a.margin + rating.reindex(a.opp).values).groupby(a.team).mean()
        rating -= rating.mean()
    return rating, a.groupby("team").size()


def _espn_keys(names: pd.Series, keys: set) -> pd.Series:
    """ESPN sometimes lists a full legal name ("Jerome Adolphus Jordan"): fall back to first + last token."""
    uniq = names.dropna().drop_duplicates()
    full = uniq.map(norm_name)
    short = uniq.map(lambda s: norm_name(s.split()[0] + " " + s.split()[-1]) if len(s.split()) > 2 else "")
    lut = {n: (k if k in keys else k2 if k2 in keys else None) for n, k, k2 in zip(uniq, full, short)}
    return names.map(lut)


def _hoopr_season(y: int, keys: set) -> pd.DataFrame:
    b = pd.read_parquet(EXT / "hoopr" / f"player_box_{y}.parquet", columns=BOX_COLS)
    b = b[b.season_type.isin([2, 3]) & ~b.did_not_play & b.minutes.notna()]  # regular season + postseason, games actually played
    b = b.assign(key=_espn_keys(b.athlete_display_name, keys)).dropna(subset=["key"])
    if b.empty:
        return pd.DataFrame()
    b["usg"] = b.field_goals_attempted + 0.44 * b.free_throws_attempted + b.turnovers
    b["gsc"] = (b.points + 0.4 * b.field_goals_made - 0.7 * b.field_goals_attempted - 0.4 * (b.free_throws_attempted - b.free_throws_made)
                + 0.7 * b.offensive_rebounds + 0.3 * b.defensive_rebounds + b.steals + 0.7 * b.assists + 0.7 * b.blocks - 0.4 * b.fouls - b.turnovers)
    sched = pd.read_parquet(EXT / "hoopr" / f"schedules_{y}.parquet", columns=["home_id", "away_id", "home_score", "away_score", "status_type_completed"])
    rating, ngames = _srs(sched)
    top50 = set(rating[ngames.reindex(rating.index) >= 20].nlargest(50).index)
    b["top"] = b.opponent_team_id.isin(top50)
    b["opp_srs"] = rating.reindex(b.opponent_team_id).values
    b = b.sort_values(["athlete_id", "game_date"])
    b["from_end"] = b.groupby("athlete_id").cumcount(ascending=False)

    g = b.groupby("athlete_id")
    top, rest = b[b.top].groupby("athlete_id"), b[~b.top].groupby("athlete_id")
    last, prev = b[b.from_end < 10].groupby("athlete_id"), b[b.from_end >= 10].groupby("athlete_id")
    tot = g[["points", "field_goals_made", "field_goals_attempted", "three_point_field_goals_made", "three_point_field_goals_attempted",
             "free_throws_made", "free_throws_attempted"]].sum()
    fga, fg3a, fta = (tot[c].replace(0, np.nan) for c in ("field_goals_attempted", "three_point_field_goals_attempted", "free_throws_attempted"))
    out = pd.DataFrame({
        "key": g.key.first(), "team": g.team_location.first(), "draft_year": y,
        "h_gp": g.size(), "h_gs": g.starter.sum(), "h_post_gp": g.season_type.apply(lambda s: (s == 3).sum()),
        "h_mpg": g.minutes.mean(), "h_min_std": g.minutes.std(), "h_pts_std": g.points.std(), "h_gsc_mean": g.gsc.mean(), "h_gsc_std": g.gsc.std(),
        "h_usg_pg": g.usg.mean(), "h_team_srs": rating.reindex(g.team_id.first()).values, "h_opp_srs_mean": g.opp_srs.mean(),
        "h_gp_top50": g.top.sum(), "h_pts_top50": top.points.mean(), "h_gsc_top50": top.gsc.mean(),
        "h_pts_top50_diff": top.points.mean() - rest.points.mean(), "h_gsc_top50_diff": top.gsc.mean() - rest.gsc.mean(),
        "h_pts_last10_diff": last.points.mean() - prev.points.mean(), "h_gsc_last10_diff": last.gsc.mean() - prev.gsc.mean(),
        "h_min_last10_diff": last.minutes.mean() - prev.minutes.mean(),
        # basic season line (per game / season totals): the only box-score stats for the pre-Torvik 2003-2007 college draftees
        "h_pts_pg": g.points.mean(), "h_reb_pg": g.rebounds.mean(), "h_ast_pg": g.assists.mean(), "h_stl_pg": g.steals.mean(),
        "h_blk_pg": g.blocks.mean(), "h_tov_pg": g.turnovers.mean(), "h_fga_pg": g.field_goals_attempted.mean(),
        "h_fg3a_pg": g.three_point_field_goals_attempted.mean(), "h_fta_pg": g.free_throws_attempted.mean(),
        "h_fg_pct": tot.field_goals_made / fga, "h_fg3_pct": tot.three_point_field_goals_made / fg3a, "h_ft_pct": tot.free_throws_made / fta,
        "h_ts": tot.points / (2 * (tot.field_goals_attempted + 0.44 * tot.free_throws_attempted)).replace(0, np.nan),
        "h_3par": tot.three_point_field_goals_attempted / fga, "h_ftr": tot.free_throws_attempted / fga,
    })
    out["h_start_share"] = out.h_gs / out.h_gp
    return out.sort_values("h_mpg", ascending=False)


def load_hoopr() -> pd.DataFrame:
    """data/external/hoopr/player_box_<season>.parquet + schedules_<season>.parquet (GitHub sportsdataverse/hoopR-mbb-data,
    mbb/player_box/parquet and mbb/schedules/parquet; season = calendar year it ends = draft_year). Per player, from the
    regular-season + postseason games he played in the season ending in draft_year: games / starts / start share,
    minutes and points variability, Hollinger game score, a usage proxy (FGA + 0.44 FTA + TOV per game), and splits vs
    top-50 opponents and over the last 10 games. "Top-50" = the 50 best teams (>= 20 games) by a Simple Rating System
    fitted to that season's results, because AP ranks are only present in the 2025+ schedule files. h_team_srs /
    h_opp_srs_mean are that rating for the player's own team and his average opponent. Also a basic season line
    (h_pts_pg .. h_fta_pg per game; h_fg_pct, h_fg3_pct, h_ft_pct, h_ts, h_3par, h_ftr from season totals) for every
    season, so the 2003-2007 classes, which predate Torvik, still get comparable box-score features. Names are matched
    per season against the college draftees of that year (full name, else first + last token); same-name collisions are
    resolved by school, else by minutes."""
    d = _draftees().dropna(subset=["college"])  # a draftee without a college never played D1 that season: same-name D1 players are not him
    frames = [_hoopr_season(y, set(d.key[d.draft_year == y])) for y in HOOPR_SEASONS if (EXT / "hoopr" / f"player_box_{y}.parquet").exists()]
    out = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    return _resolve_collisions(out, "team")


# --------------------------------------------------------------------------- 6. March Madness player-seasons

def load_marchmadness() -> pd.DataFrame:
    """data/external/marchmadness/training_dataset.parquet (huggingface_hub.snapshot_download of Emlembow/march-madness-2026-data):
    13,299 sports-reference player-seasons on NCAA-tournament teams, 2011-2026. Join keys: (norm_name(player_name),
    season_year); team / team_slug / conference are also available. Box-score and BPM columns duplicate Torvik / ayush, so
    only what is new is exposed: starts, Win Shares, team record / SOS / SRS, tournament seed and tournament wins (the
    tournament ends in early April, before the draft)."""
    mm = pd.read_parquet(EXT / "marchmadness" / "training_dataset.parquet")
    cols = {"games_started": "mm_gs", "ows": "mm_ows", "dws": "mm_dws", "ws": "mm_ws", "team_wins": "mm_team_wins", "team_losses": "mm_team_losses",
            "team_sos": "mm_team_sos", "team_srs": "mm_team_srs", "tournament_seed": "mm_seed", "games_won": "mm_tourney_wins"}
    out = pd.DataFrame({"key": mm.player_name.map(norm_name), "draft_year": mm.season_year.astype(int), "team": mm.team})
    for src, dst in cols.items():
        out[dst] = _num(mm[src])
    out["mm_start_share"] = out.mm_gs / _num(mm.games_played)
    out = out.assign(_mpg=_num(mm.minutes_per_game)).sort_values("_mpg", ascending=False).drop(columns="_mpg")
    return _resolve_collisions(out, "team")


LOADERS = [load_combine, load_kaggle_college, load_score, load_ianstack, load_hoopr, load_marchmadness]


def coverage() -> None:
    """Per loader: feature count and share of the 1193 draftees matched, overall / modelled split / by year."""
    t = pd.read_parquet(C.PROC / "draft_table.parquet", columns=["player", "draft_year", "college", "modelled"])
    t["key"] = t.player.map(norm_name)
    for fn in LOADERS:
        ex = fn()
        feats = [c for c in ex.columns if c not in KEYS]
        assert not ex.duplicated(KEYS).any(), fn.__name__
        assert all(pd.api.types.is_numeric_dtype(ex[c]) for c in feats), fn.__name__
        m = t.merge(ex.assign(_hit=True), on=KEYS, how="left")
        hit = m._hit.fillna(False).astype(bool)
        print(f"{fn.__name__:20s} {len(feats):2d} cols  {len(ex):6d} rows  matched {hit.mean():5.1%}  "
              f"modelled {hit[m.modelled].mean():5.1%}  not-modelled {hit[~m.modelled].mean():5.1%}")
        print("   by year: " + " ".join(f"{y}:{v:.0%}" for y, v in hit.groupby(m.draft_year).mean().items()))
        miss = m[m.modelled & ~hit]
        if len(miss):
            print(f"   unmatched modelled ({len(miss)}), e.g.: " + ", ".join(f"{p} {y}" for p, y in zip(miss.player.head(8), miss.draft_year.head(8))))


if __name__ == "__main__":
    coverage()
