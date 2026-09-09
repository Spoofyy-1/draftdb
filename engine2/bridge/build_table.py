"""Build Colin's draft table (and the per-season WAR table his horizon logic needs) from OUR verified data.

Reads (read-only, never modified):
    <staging>/train_2000_2018.csv          1,458 rows, draft classes 2000-2018, with labels
    <staging>/tests/test_YYYY_inputs.csv   2019-2025, no labels (sealed -- scored through the vault on the box)
    <staging>/input_columns.json           the 1,215 columns that are legal model inputs
    <identity>/tabular_names.csv           pid -> actual_pick (scoring metadata only; NAMES ARE NEVER READ)

Writes:
    data/processed/draft_table.parquet     the table tournament/layer1.py + layer2.py consume
    data/processed/season_war.parquet      (bbref_id, season, war) rows -- what horizon_target / disc85_gaussrank read

Identity rule: the only thing taken from the identity file is `actual_pick`. `player_name` and `nba_id` are never
read into memory, and every row of the output table is keyed by `pid`. `bbref_id`, `key` and `player` are all set to
the pid string, so no name can reach the box.

Usage:
    python -m bridge.build_table [--extras] [--no-picks] [--out data/processed]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from infra import config as C
from infra.dataset import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES
from infra.external import AYUSH_BOX, INTL_FIBA, INTL_PRO, JASONG_NUM, PHYSICAL_FEATURES, TRAJECTORY_FEATURES
from tournament.contract import EXPLICIT, TORVIK

STAGING = Path("/Users/kennakao/nba/datarebuild/v4_build/staging_v419")
IDENTITY = Path("/Users/kennakao/Downloads/nba_redraft_handoff/identity_KEEP_SEPARATE/tabular_names.csv")
HOLDOUT_YEARS = list(range(2019, 2026))
UNRANKED_MOCK = 61.0  # a player no board ranked sits behind the last pick, as in infra/models.py::CONSENSUS_FEATURE

# Every family of ours that the bridge consumes, and the group of his it feeds. Anything not listed here is either
# a fallback for one of his named columns or (with --extras) written as xt_<name>. See MAPPING.md.
CONSUMED_PREFIXES = ("tv_", "col_", "traj_", "bio_", "vmb_", "vcmb_", "intl_", "fy_", "eurs_", "tc_", "gl2_",
                     "sc_", "cp_", "wt_", "misc_", "mock_", "bb_", "vcons_", "cons_", "dis_", "rsci_")


# --------------------------------------------------------------------------- small helpers

class Src:
    """Column accessor over the stacked staging frame: missing columns come back as all-NaN, so a mapping never
    crashes on a family a given staging version does not carry."""

    def __init__(self, df: pd.DataFrame, allowed: set[str]):
        self.df = df
        self.allowed = allowed
        self.used: set[str] = set()

    def __call__(self, name: str) -> pd.Series:
        if name in self.df.columns and name in self.allowed:
            self.used.add(name)
            return pd.to_numeric(self.df[name], errors="coerce")
        return pd.Series(np.nan, index=self.df.index, dtype=float)

    def first(self, *names: str) -> pd.Series:
        out = self(names[0])
        for n in names[1:]:
            out = out.fillna(self(n))
        return out

    def has(self, name: str) -> bool:
        return name in self.df.columns and name in self.allowed

    def prefixed(self, prefix: str) -> list[str]:
        return [c for c in self.df.columns if c.startswith(prefix) and c in self.allowed]


def _pct(s: pd.Series) -> pd.Series:
    """0-1 ratio -> 0-100 percent (Torvik's convention for FT_per / TP_per / twoP_per / ftr)."""
    return s * 100.0


def _within_year_pct(v: pd.Series, year: pd.Series) -> pd.Series:
    return v.groupby(year).rank(pct=True)


# --------------------------------------------------------------------------- the mapping

def build_features(s: Src, year: pd.Series) -> dict[str, pd.Series]:
    """Our columns -> his contract. Every entry is documented in bridge/MAPPING.md."""
    f: dict[str, pd.Series] = {}

    # ---- Torvik final-season line (NUMERIC_FEATURES). tv_ is the player's own Torvik row (2010+ classes only);
    #      col_ is our wider final-college-season line and is the fallback everywhere it lines up on scale.
    f["GP"] = s.first("tv_gp", "col_gp")
    # Min_per is the share of his team's minutes he played, 0-100. col_minutes_share is the same quantity expressed
    # as a share of all 5 x 40 team minutes, so x500 puts it on Torvik's scale (medians 80 vs 76.5 on the overlap).
    f["Min_per"] = s("tv_min_per").fillna(s("col_minutes_share") * 500.0)
    f["ORtg"] = s.first("tv_ortg", "col_ortg")
    f["usg"] = s.first("tv_usg", "col_usg_pct")
    f["eFG"] = s.first("tv_efg", "col_efg_pct")
    f["TS_per"] = s.first("tv_ts", "col_ts_pct")
    f["ORB_per"] = s.first("tv_orb", "col_orb_pct")
    f["DRB_per"] = s.first("tv_drb", "col_drb_pct")
    f["AST_per"] = s.first("tv_ast_pct", "col_ast_pct")
    f["TO_per"] = s.first("tv_to_pct", "col_tov_pct")
    f["FT_per"] = _pct(s("col_ft_pct"))
    f["twoP_per"] = _pct(s("col_fg2_pct"))
    f["TP_per"] = _pct(s("col_fg3_pct"))
    f["blk_per"] = s("col_blk_pct")
    f["stl_per"] = s("col_stl_pct")
    f["ftr"] = s("tv_ftr").fillna(_pct(s("col_ftar")))
    f["porpag"] = s("tv_porpag")
    f["adjoe"] = s("tv_adjoe")
    f["pfr"] = s("tv_pfr")
    # his rec_rank is a recruiting RANK (lower = better; layer2 ranks on -rec_rank). tv_rec_rank is a 0-100 score
    # (higher = better), so it is flipped before it is used as a fallback for rsci_rank.
    f["rec_rank"] = s("rsci_rank").fillna(101.0 - s("tv_rec_rank"))
    f["ast_tov"] = s("tv_ast_tov")
    f["rim_pct"] = s.first("tv_rim_pct", "col_rim_fg_pct")
    f["mid_pct"] = s.first("tv_mid_pct", "col_mid_fg_pct")
    f["drtg"] = s.first("tv_drtg", "col_drtg")
    f["adrtg"] = s.first("tv_adrtg", "col_adj_drtg")
    f["dporpag"] = s("tv_dporpag")
    f["stops"] = s("tv_stops")
    f["bpm"] = s.first("tv_bpm", "col_impact")
    f["obpm"] = s.first("tv_obpm", "col_impact_off")
    f["dbpm"] = s.first("tv_dbpm", "col_impact_def")
    f["gbpm"] = s("tv_gbpm")
    f["ogbpm"] = s("tv_ogbpm")
    f["dgbpm"] = s("tv_dgbpm")
    mpg = s.first("col_mpg").fillna(s("tv_mp"))
    f["mpg"] = mpg
    gp = f["GP"]
    f["dunk_made"] = s("tv_dunks_pg") * gp                      # Torvik counts dunks made; ours is per game
    # per-game counting stats: ours are already per game (col_*_pg); oreb/dreb are split out of total rebounds by
    # the offensive / defensive rebound-percentage share (derived -- Torvik carries them directly, we do not).
    treb = s("col_reb_pg")
    orb_share = s("col_orb_pct") / (s("col_orb_pct") + s("col_drb_pct"))
    f["treb"] = treb
    f["oreb"] = treb * orb_share
    f["dreb"] = treb * (1.0 - orb_share)
    f["ast"] = s("col_ast_pg")
    f["stl"] = s("col_stl_pg")
    f["blk"] = s("col_blk_pg")
    f["pts"] = s.first("tv_pts", "col_pts_pg")
    # TPA_pg / FTA_pg reconstructed from the per-40 three-point volume and the two shot-attempt rates:
    #   FGA/40 = 3PA/40 / 3PAr,  TPA_pg = 3PA/40 * mpg/40,  FTA_pg = FTr * FGA/40 * mpg/40
    per40_to_pg = mpg / 40.0
    tpa40 = s("col_fg3a_per40")
    fga40 = tpa40 / s("col_fg3ar").replace(0, np.nan)
    f["TPA_pg"] = tpa40 * per40_to_pg
    f["FTA_pg"] = s("col_ftar") * fga40 * per40_to_pg
    f["height_in"] = s.first("bio_height_in", "vmb_listed_height_in", "bio_combine_height_in",
                             "vcmb_height_with_shoes_in")
    f["class_year"] = s.first("tv_yr", "col_class_index", "vmb_college_class_year")
    f["n_college_seasons"] = s.first("col_n_seasons", "tv_seasons", "tc_n_seasons")
    f["age_at_draft"] = s.first("bio_age_at_draft", "tv_age_exact", "col_age", "vmb_age_reported_years")

    # ---- categoricals (numeric ids; the winner drops them with "-cat", they exist only to satisfy the contract)
    f["conf"] = s("tv_conf_tier")
    f["role"] = s.first("tv_role", "bio_pos_code")

    # ---- traj: previous college season and the change into the final one
    for his, delta, cur in [("bpm", "tv_d_bpm", "bpm"), ("usg", "tv_d_usg", "usg"), ("TS_per", "tv_d_ts", "TS_per")]:
        d = s(delta)
        f[f"d_{his}"] = d
        f[f"prev_{his}"] = f[cur] - d
    d_mpg = s("traj_d_mpg")     # minutes per game; tv_d_min_per is a minutes-SHARE delta and is not interchangeable
    f["d_mpg"], f["prev_mpg"] = d_mpg, mpg - d_mpg
    for miss in ("obpm", "dbpm", "porpag"):                     # no per-stat previous-season delta in our blocks
        f[f"d_{miss}"] = pd.Series(np.nan, index=s.df.index)
        f[f"prev_{miss}"] = pd.Series(np.nan, index=s.df.index)
    bpm_hist = pd.concat([f["bpm"], f["prev_bpm"], s.first("tc_first_bpm", "tv_first_bpm")], axis=1)
    f["career_bpm_mean"] = bpm_hist.mean(axis=1)                # first / previous / final season, not the full career
    f["career_bpm_max"] = bpm_hist.max(axis=1)

    # ---- phys
    f["wingspan_in"] = s.first("vcmb_wingspan_in", "bio_combine_wingspan_in", "dx_last_wingspan_in")
    f["weight_lb"] = s.first("bio_weight_lb", "vcmb_weight_lb", "vmb_listed_weight_lb")
    f["wing_minus_height"] = s.first("vcmb_wingspan_minus_height_in", "bio_combine_wingspan_minus_height_in").fillna(
        f["wingspan_in"] - f["height_in"])
    f["draft_age_x"] = s.first("bio_age_at_draft", "vmb_age_reported_years").fillna(f["age_at_draft"])

    # ---- intl_pro: last pre-draft professional season abroad / G League
    gpi = s("intl_gp")
    f["i_has_pro"] = (gpi.fillna(0) > 0).astype(float)
    f["i_league_level"] = s("intl_level")
    f["i_top_level"] = s("intl_best_level")
    f["i_n_comps"] = s("intl_n_seasons")
    f["i_age_season"] = s("intl_age")
    f["i_gp"] = gpi
    f["i_mpg"] = s("intl_mpg")
    f["i_pts_36"] = s("intl_pts36")
    f["i_trb_36"] = s("intl_reb36")
    i_orb_share = s("intl_orb_pct") / (s("intl_orb_pct") + s("intl_drb_pct"))
    f["i_orb_36"] = s("intl_reb36") * i_orb_share
    f["i_ast_36"] = s("intl_ast36")
    f["i_stl_36"] = s("intl_stl36")
    f["i_blk_36"] = s("intl_blk36")
    f["i_pir_36"] = s("intl_impact")
    ts = s("intl_ts_pct") / 100.0
    ftr_i = s("intl_ftar")
    # true-shooting attempts back out the shot diet: TSA/36 = pts36 / (2*TS); FGA = TSA / (1 + 0.44*FTr)
    tsa36 = s("intl_pts36") / (2.0 * ts.replace(0, np.nan))
    fga36 = tsa36 / (1.0 + 0.44 * ftr_i)
    f["i_fga_36"] = fga36
    f["i_fg3a_36"] = s("intl_fg3ar") * fga36
    f["i_fta_36"] = ftr_i * fga36
    efg = s("intl_efg_pct") / 100.0
    f["i_fg_pct"] = efg - 0.5 * s("intl_fg3_pct") * s("intl_fg3ar")   # eFG = FG% + 0.5 * 3P% * 3PAr
    f["i_fg3_pct"] = s("intl_fg3_pct")
    f["i_ft_pct"] = s("intl_ft_pct")
    f["i_fg3a_rate"] = s("intl_fg3ar")
    f["i_fta_rate"] = ftr_i
    f["i_ts"] = ts
    f["i_efg"] = efg
    f["i_ast_tov"] = pd.Series(np.nan, index=s.df.index)         # no international turnovers per 36 in our blocks
    top_g = s("eur_el2_ls_games").replace(0, np.nan)
    f["i_top_mpg"] = (s("eur_el2_ls_min") / top_g).fillna(s("intl_mpg").where(s("intl_level") >= s("intl_best_level")))
    f["i_top_pir_36"] = s("eur_el2_ls_pir40") * 36.0 / 40.0

    # ---- intl_fiba: FIBA youth national-team record (ours is age-relative and z-scored inside the event cohort,
    #      his was raw per-36; same information, different scale -- see MAPPING.md)
    f["i_fiba_youth_events"] = s("fy_n_tournaments")
    f["i_fiba_max_level"] = s("fy_best_level")
    f["i_fiba_best_rank"] = s("fy_team_finish_last")
    f["i_fiba_last_age"] = s("fy_age_rel_last")
    f["i_fiba_mpg"] = s("fy_minutes_share_last")
    f["i_fiba_pts_36"] = s("fy_best_pts40_z")
    f["i_fiba_last_pts_36"] = s("fy_pts40_z_last")
    f["i_fiba_ast_36"] = s("fy_ast40_z_last")
    f["i_fiba_stl_36"] = s("fy_stl_blk40_z_last")
    f["i_fiba_pir_36"] = s("fy_best_pir40_z")
    f["i_fiba_last_eff_36"] = s("fy_pir40_z_last")
    f["i_fiba_ts"] = s("fy_ts_z_last")
    f["i_fiba_best_effpg"] = s("fy_z_trend")
    f["i_ngt"] = s("eur_angt_has")
    f["i_ngt_last_age"] = s("eur_angt_age_first")
    f["i_ngt_pir_36"] = s("eur_angt_pir40") * 36.0 / 40.0
    f["i_ngt_pts_36"] = s("eur_angt_pts40") * 36.0 / 40.0

    # ---- combine (prefix c_): NBA combine anthropometrics, athletic tests, shooting drills
    h = s("vcmb_height_without_shoes_in")
    reach, wing = s("vcmb_standing_reach_in"), s("vcmb_wingspan_in")
    vs, vm = s("vcmb_standing_vertical_in"), s("vcmb_max_vertical_in")
    wt, bf = s("vcmb_weight_lb"), s("vcmb_body_fat_pct")
    hl, hw = s("vcmb_hand_length_in"), s("vcmb_hand_width_in")
    f["c_height_noshoes"], f["c_wingspan"], f["c_standing_reach"], f["c_weight"] = h, wing, reach, wt
    f["c_body_fat"], f["c_hand_length"], f["c_hand_width"] = bf, hl, hw
    f["c_vert_standing"], f["c_vert_max"] = vs, vm
    f["c_lane_agility"] = s("vcmb_lane_agility_s")
    f["c_shuttle"] = s("vcmb_modified_lane_agility_s")
    f["c_sprint"] = s("vcmb_three_quarter_sprint_s")
    f["c_bench"] = s("vcmb_bench_press_reps")
    f["c_wing_minus_height"] = s("vcmb_wingspan_minus_height_in").fillna(wing - h)
    f["c_reach_minus_height"] = reach - h
    f["c_wing_height_ratio"] = wing / h
    f["c_reach_height_ratio"] = reach / h
    f["c_hand_length_height_ratio"] = hl / h
    f["c_hand_width_height_ratio"] = hw / h
    f["c_weight_per_in"] = wt / h
    f["c_bmi_like"] = s("vcmb_bmi_kg_m2")
    f["c_lean_mass"] = wt * (1 - bf / 100.0)
    f["c_fat_mass"] = wt * (bf / 100.0)
    f["c_lean_mass_height_ratio"] = f["c_lean_mass"] / h
    f["c_approach_vert_gain"] = vm - vs
    f["c_max_touch"] = reach + vm
    f["c_standing_touch"] = reach + vs
    anthro = [h, s("vcmb_height_with_shoes_in"), wing, reach, wt, bf, hl, hw]
    drills = [vs, vm, s("vcmb_lane_agility_s"), s("vcmb_modified_lane_agility_s"),
              s("vcmb_three_quarter_sprint_s"), s("vcmb_bench_press_reps")]
    f["c_anthro_n"] = pd.concat(anthro, axis=1).notna().sum(axis=1).astype(float)
    f["c_drills_n"] = pd.concat(drills, axis=1).notna().sum(axis=1).astype(float)
    made = ["vcmb_spot_15ft_made", "vcmb_spot_college3_made", "vcmb_spot_nba3_made",
            "vcmb_off_dribble_15ft_made", "vcmb_off_dribble_college3_made",
            "vcmb_on_move_15ft_made", "vcmb_on_move_college3_made"]
    att = [m.replace("_made", "_attempted") for m in made]
    m_sum = pd.concat([s(c) for c in made], axis=1).sum(axis=1, min_count=1)
    a_sum = pd.concat([s(c) for c in att], axis=1).sum(axis=1, min_count=1)
    f["c_shoot_pct"] = m_sum / a_sum.replace(0, np.nan)

    # ---- tctx (prefix t_): our all-D1 Torvik context block, one-for-one
    for c in s.prefixed("tc_"):
        f["t_" + c[3:]] = s(c)

    # ---- intl_z (prefix iz_): production standardised against the player's own international context
    age_i = s("intl_age")
    young = (22.0 - age_i).clip(lower=0)
    impact = s("intl_impact")
    f["iz_pts_36"] = s("intl_lg_adj_pts36")
    f["iz_eff_36"] = impact
    f["iz_mpg"] = s("intl_mpg")
    f["iz_age"] = age_i
    f["iz_ts"] = ts
    f["iz_usg_36"] = s("intl_usg_pct")
    f["iz_n_league"] = s("intl_n_seasons")
    f["iz_trb_36"] = s("intl_reb36")
    f["iz_ast_36"] = s("intl_ast36")
    f["iz_stl_36"] = s("intl_stl36")
    f["iz_blk_36"] = s("intl_blk36")
    f["iz_tov_36"] = s("intl_tov_pct")
    f["iz_fg3a_36"] = s("intl_fg3ar")
    f["iz_fta_36"] = ftr_i
    f["iz_lg_strength"] = s("intl_lg_strength")
    f["iz_season_gap"] = s("intl_season_gap")
    f["iz_prev_pts_36"] = s("intl_prev_pts36")
    f["iz_two_pts_36"] = s("intl_two_pts36")
    f["iz_pct_eff"] = _within_year_pct(impact, year)
    f["iz_young_x_eff"] = young * impact
    f["iz_young_x_mpg"] = young * s("intl_mpg")
    f["iz_young_x_usg"] = young * s("intl_usg_pct")
    for c in s.prefixed("eurs_"):                                # prior-chosen Euroleague/EuroCup/ANGT slim spine
        f["iz_eur_" + c[5:]] = s(c)

    # ---- mock (prefix mock_): pre-draft board consensus
    cons = s.first("vcons_mock_mean_rank", "cons_mock_consensus_rank")
    f["mock_rank_consensus"] = cons.fillna(UNRANKED_MOCK)
    f["mock_rank_consensus_alt"] = s("cons_mock_consensus_rank")
    f["mock_rank_best"] = s.first("vcons_mock_best_rank", "cons_mock_best_rank")
    f["mock_rank_range"] = s.first("vcons_mock_rank_range", "dis_mock_rank_range")
    f["mock_n_sources"] = s.first("vcons_mock_n_sources", "cons_mock_n_sources")
    f["mock_rank_std"] = s("dis_mock_rank_std")
    f["mock_first_round"] = (cons <= 30).astype(float).where(cons.notna())

    # ---- momentum (prefix mo_): how the boards moved before draft night
    for c in s.prefixed("mock_"):                                # our mock_ block is the momentum block
        f["mo_" + c[5:]] = s(c)
    for c in s.prefixed("bb_"):                                  # boards vs mocks, per source, with 30/60-day deltas
        f["mo_" + c[3:]] = s(c)

    # ---- response (prefix rs_): challenge / response from the final-season game log
    for c in s.prefixed("gl2_"):
        f["rs_" + c[4:]] = s(c)

    # ---- person (prefix bio_): biography. NOTE our bio_ block is physicals and feeds phys / all_torvik above;
    #      his bio_ prefix is the "person" group and is fed by our wt_ (Wikipedia rule-based) + misc_ birthplace.
    f["bio_nba_relative_n"] = s.first("wt_n_relatives_pro_any_sport", "wt_nba_relative")
    f["bio_nba_father"] = s("wt_relative_is_parent")
    f["bio_shoots_left"] = s("wt_left_handed")
    f["bio_born_us"] = (1.0 - s("wt_born_outside_usa")).fillna(s("misc_born_usa"))
    f["bio_n_high_schools"] = s("wt_hs_transferred")
    f["bio_prep_academy"] = s.first("wt_prep_year", "misc_prep")
    for c in s.prefixed("wt_"):
        f["bio_" + c[3:]] = s(c)
    for c in ("misc_birth_month", "misc_birth_q", "misc_born_usa", "misc_prep", "misc_birth_us_state",
              "misc_birth_big_metro"):
        f["bio_" + c[5:]] = s(c)

    # ---- scouting (prefix sc_): NBADraft.net grades and write-up flags. Ours already uses his prefix; only four
    #      names differ and the keyword counts are called sc_kw_* instead of sc_flag_*.
    rename = {"sc_nbaready": "sc_nba_ready", "sc_ballhandling": "sc_handle", "sc_postskills": "sc_post"}
    for c in s.prefixed("sc_"):
        f[rename.get(c, "sc_flag_" + c[6:] if c.startswith("sc_kw_") else c)] = s(c)
    ws, ww = s("sc_words_strengths"), s("sc_words_weaknesses")
    f["sc_ratio_strength_weakness"] = ws / ww.replace(0, np.nan)

    # ---- comp (prefix sct_): the NBA comparison's own dated career
    for c in s.prefixed("cp_"):
        f["sct_" + c[3:]] = s(c)
    f["sct_comp_n"] = s("cp_n_comps")
    f["sct_words_total"] = ws.fillna(0) + ww.fillna(0)

    return f


# --------------------------------------------------------------------------- labels

def season_war_rows(train: pd.DataFrame) -> pd.DataFrame:
    """(bbref_id, season, war) from y_s1_war..y_s5_war. Season index i maps to NBA season draft_year + i, which is
    exactly what horizon_target / transform_labels need: `season > draft_year` is always true and `season <= through`
    truncates a label to the seasons completed by draft night `through`.

    Our per-season array is a prefix (no gaps: 1,452 of 1,458 rows have season 1 and nobody has a hole), so the
    calendar mapping and "n-th season played" agree except for a draft-and-stash player whose debut was delayed;
    for him the causal cutoff sees his seasons one or two years early. See MAPPING.md.
    """
    blocks = []
    for i in range(1, C.TARGET_SEASONS + 1):
        col = f"y_s{i}_war"
        if col not in train.columns:
            continue
        b = train.loc[train[col].notna(), ["pid", "draft_year", col]].rename(columns={col: "war"})
        b["season"] = b.draft_year.astype(int) + i
        blocks.append(b[["pid", "season", "war"]])
    out = pd.concat(blocks, ignore_index=True).rename(columns={"pid": "bbref_id"})
    return out.sort_values(["bbref_id", "season"]).reset_index(drop=True)


# --------------------------------------------------------------------------- main

def load_staging(staging: Path) -> tuple[pd.DataFrame, list[str]]:
    inputs = json.load(open(staging / "input_columns.json"))["inputs"]
    frames = [pd.read_csv(staging / "train_2000_2018.csv", low_memory=False)]
    for y in HOLDOUT_YEARS:
        p = staging / "tests" / f"test_{y}_inputs.csv"
        if p.exists():
            frames.append(pd.read_csv(p, low_memory=False))
    df = pd.concat(frames, ignore_index=True, sort=False)
    return df, inputs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging", type=Path, default=STAGING)
    ap.add_argument("--identity", type=Path, default=IDENTITY, help="pid -> actual_pick; player_name is never read")
    ap.add_argument("--no-picks", action="store_true",
                    help="do not read the identity file; fill `pick` with a within-class placeholder. His metrics "
                         "(spearman_nba, WAR captured) then mean nothing, but layer 1 still runs and scores.")
    ap.add_argument("--include-undrafted", action="store_true",
                    help="keep was_drafted=0 rows. Off by default: his pool is the players actually drafted.")
    ap.add_argument("--extras", action="store_true",
                    help="also write every unmapped input column as xt_<name> (contract group `extra`), for building "
                         "on top of the ported winner. Off by default so the winner reproduces on his feature count.")
    ap.add_argument("--out", type=Path, default=C.PROC)
    a = ap.parse_args()

    df, inputs = load_staging(a.staging)
    allowed = set(inputs)
    if not a.include_undrafted:
        df = df[df.was_drafted == 1]
    df = df.reset_index(drop=True)
    year = df.draft_year.astype(int)

    s = Src(df, allowed)
    feats = build_features(s, year)

    table = pd.DataFrame(index=df.index)
    table["bbref_id"] = df.pid.astype(str)      # pid-keyed: no name ever enters this table
    table["key"] = df.pid.astype(str)
    table["player"] = df.pid.astype(str)
    table["draft_year"] = year
    for k, v in feats.items():
        table[k] = pd.to_numeric(v, errors="coerce").astype(float).values

    # ---- pick: scoring metadata only. models.predict() raises if "pick" is passed as a feature; keep it that way.
    if a.no_picks:
        table["pick"] = table.groupby("draft_year").cumcount().add(1).astype(float)
        pick_note = "placeholder (sequential within class) -- his spearman_nba / WAR-captured are meaningless"
    else:
        ident = pd.read_csv(a.identity, usecols=["pid", "draft_year", "actual_pick"])
        ident["pid"] = ident.pid.astype(str)
        pick = table[["bbref_id", "draft_year"]].merge(
            ident.rename(columns={"pid": "bbref_id"}), on=["bbref_id", "draft_year"], how="left")
        table["pick"] = pick.actual_pick.values
        missing = int(table["pick"].isna().sum())
        if missing:                              # layer1 asserts every candidate carries a pick
            table["pick"] = table["pick"].fillna(
                table.groupby("draft_year")["pick"].transform("max").fillna(60) + 1)
        pick_note = f"actual_pick from the identity file ({missing} filled with a per-class placeholder)"

    # ---- who is modelled, and from which population
    has_college = (s("col_gp").notna() | s("tv_gp").notna()).values
    has_pro = (s("intl_gp").fillna(0) > 0).values
    has_combine = s("vcmb_height_without_shoes_in").notna().values
    table["source"] = np.where(has_college, "college", np.where(has_pro, "intl", np.where(has_combine, "other", "none")))
    table["modelled"] = has_college | has_pro | has_combine

    # ---- labels
    train_mask = df.draft_year <= 2018
    war_cols = [f"y_s{i}_war" for i in range(1, C.TARGET_SEASONS + 1) if f"y_s{i}_war" in df.columns]
    table["labelled"] = train_mask.values & df.get("y_early_war", pd.Series(np.nan, index=df.index)).notna().values
    table[C.TARGET] = np.where(table["labelled"], df.get("y_early_war", np.nan), np.nan)
    table["seasons_played"] = np.where(table["labelled"], df[war_cols].notna().sum(axis=1), 0).astype(int)

    # ---- every column his contract lists must exist; families we cannot fill are present and empty (see MAPPING.md)
    required = (set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES) | set(FEATURES)
                | {c for v in TORVIK.values() for c in v} | {c for v in EXPLICIT.values() for c in v})
    unfillable = sorted(required - set(table.columns))
    if unfillable:
        table = pd.concat([table, pd.DataFrame(np.nan, index=table.index, columns=unfillable)], axis=1)

    # ---- optional: everything we did not consume, for building past the ported winner
    if a.extras:
        consumed = set(s.used)
        extra = [c for c in inputs if c in df.columns and c not in consumed]
        for c in extra:
            table["xt_" + c] = pd.to_numeric(df[c], errors="coerce").astype(float).values
        print(f"extras: {len(extra)} unmapped input columns written as xt_*")

    a.out.mkdir(parents=True, exist_ok=True)
    swar = season_war_rows(df[train_mask]) if train_mask.any() else pd.DataFrame(columns=["bbref_id", "season", "war"])
    swar.to_parquet(a.out / "season_war.parquet", index=False)
    table.to_parquet(a.out / "draft_table.parquet", index=False)

    # ---- report
    fam = {
        "all_torvik": NUMERIC_FEATURES + CATEGORICAL_FEATURES, "traj": TRAJECTORY_FEATURES,
        "phys": PHYSICAL_FEATURES, "intl_pro": INTL_PRO, "intl_fiba": INTL_FIBA,
        "a_box": list(AYUSH_BOX.values()), "jasong": list(JASONG_NUM.values()),
        "combine (c_)": [c for c in table if c.startswith("c_")], "tctx (t_)": [c for c in table if c.startswith("t_")],
        "intl_z (iz_)": [c for c in table if c.startswith("iz_")], "mock (mock_)": [c for c in table if c.startswith("mock_")],
        "momentum (mo_)": [c for c in table if c.startswith("mo_")], "response (rs_)": [c for c in table if c.startswith("rs_")],
        "person (bio_)": [c for c in table if c.startswith("bio_")], "scouting (sc_)": [c for c in table if c.startswith("sc_")],
        "comp (sct_)": [c for c in table if c.startswith("sct_")], "extra (xt_)": [c for c in table if c.startswith("xt_")],
    }
    ctx, hold = table[table.draft_year <= 2018], table[table.draft_year >= 2019]
    rows = []
    for name, cols in fam.items():
        cols = [c for c in cols if c in table.columns]
        live = [c for c in cols if table[c].notna().any()]
        rows.append({"family": name, "cols": len(cols), "filled": len(live),
                     "cov_2003_18": round(float(ctx[live].notna().mean().mean()), 3) if live else 0.0,
                     "cov_2019_25": round(float(hold[live].notna().mean().mean()), 3) if live else 0.0})
    pd.set_option("display.width", 160)
    print("\ncolumns filled per family")
    print(pd.DataFrame(rows).to_string(index=False))
    per_class = table.groupby("draft_year").agg(
        rows=("bbref_id", "size"), modelled=("modelled", "sum"),
        college=("source", lambda x: int((x == "college").sum())), intl=("source", lambda x: int((x == "intl").sum())),
        labelled=("labelled", "sum"))
    print("\nrows per draft class")
    print(per_class.T.to_string())
    print(f"\npick: {pick_note}")
    print(f"season_war.parquet: {len(swar)} player-seasons, {swar.bbref_id.nunique()} players, "
          f"seasons {int(swar.season.min())}-{int(swar.season.max())}" if len(swar) else "season_war.parquet: empty")
    print(f"draft_table.parquet: {table.shape[0]} rows x {table.shape[1]} columns -> {a.out / 'draft_table.parquet'}")


if __name__ == "__main__":
    main()
