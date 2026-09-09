"""Build Colin's draft table (and the per-season WAR table his horizon logic needs) from OUR verified data.

Reads (read-only, never modified):
    <staging>/train_2000_2018.csv          1,458 rows, draft classes 2000-2018, with labels
    <staging>/tests/test_YYYY_inputs.csv   2019-2025, no labels (sealed -- scored through the vault on the box)
    <staging>/input_columns.json           the 1,215 columns that are legal model inputs
    <identity>/tabular_names.csv           pid -> actual_pick (scoring metadata only; NAMES ARE NEVER READ)
    <screen>                               v4_build/screen_blocks.csv, for --response-max (recomputed if it
                                           carries no gl2_ rows; the result is cached in bridge/response_screen.csv)

Writes:
    data/processed/draft_table.parquet     the table tournament/layer1.py + layer2.py consume
    data/processed/season_war.parquet      (bbref_id, season, war) rows -- what horizon_target / disc85_gaussrank read

Identity rule: the only thing taken from the identity file is `actual_pick`. `player_name` and `nba_id` are never
read into memory, and every row of the output table is keyed by `pid`. `bbref_id`, `key` and `player` are all set to
the pid string, so no name can reach the box.

Every named feature is filled by an ordered *coalesce chain*: `tv_` (the player's own Torvik final-season row, our
2010+ classes only) first, then our own final-college line `col_`, then the `ctx_base_` / `cgd_` restatements of the
same season, each converted onto Torvik's scale first so a column carries ONE scale across rows. `src_torvik`
records which side of that a row came from. See MAPPING.md for the pair list and the scale audit behind it.

Usage:
    python -m bridge.build_table [--extras] [--families] [--intl-mask] [--response-max N]
                                 [--no-fallback] [--no-picks] [--out data/processed]
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

HERE = Path(__file__).resolve().parent
STAGING = Path("/Users/kennakao/nba/datarebuild/v4_build/staging_v419")
IDENTITY = Path("/Users/kennakao/Downloads/nba_redraft_handoff/identity_KEEP_SEPARATE/tabular_names.csv")
SCREEN = Path("/Users/kennakao/nba/datarebuild/v4_build/screen_blocks.csv")
SCREEN_CACHE = HERE / "response_screen.csv"          # our own gl2_ recompute, if screen_blocks.csv has no gl2_ rows
BEFORE_COV = HERE / "coverage_before.csv"            # frozen coverage of the pre-fallback (v1) mapping, for the report
HOLDOUT_YEARS = list(range(2019, 2026))
BANDS = [("2003_09", 2003, 2009), ("2010_18", 2010, 2018), ("2019_25", 2019, 2025)]

UNRANKED_MOCK = 61.0   # a player no board ranked sits behind the last pick, as in infra/models.py::CONSENSUS_FEATURE
UNRANKED_RSCI = 101.0  # a college player on no RSCI top-100 list sits behind the last ranked recruit
RSCI_PER_POINT = 5.0   # Torvik's rec_rank is a 0-100 score over the top ~500 recruits: rank = (100 - score) * 5
MIN_PER_PER_SHARE = 500.0  # our col_minutes_share is a share of all 5x40 team minutes; x500 is Torvik's 0-100 min%

# bio_pos_code (1=PG..5=C, 6=other) -> the modal Torvik `role` id on the 667-row overlap. `role` is categorical and
# the winner drops it with "-cat"; the map exists so the column carries one code set instead of two.
POS_TO_ROLE = {1.0: 2.0, 2.0: 4.0, 3.0: 5.0, 4.0: 7.0, 5.0: 8.0, 6.0: 7.0}

# The named features that describe the final COLLEGE season. Filled only on rows that have one -- see
# Map.gate_college. rec_rank (recruiting), height_in, age_at_draft and role (listed position) are not college-season
# quantities and are filled for everyone.
# Hard plausibility ceilings for the per-game / per-season counting columns, set well above the NCAA single-season
# records (~30 pts, 16 reb, 13 ast, 5 stl, 6 blk per game). They exist to stop a corrupt staging cell propagating
# through a unit conversion: col_stl36 carries one value of 67.7 steals per 36 minutes (a 2016 row), and
# `col_stl36 x mpg/36` turned that into 69.0 steals per game in `stl`. Anything above the ceiling becomes NaN.
COUNT_CEILING = {"GP": 45.0, "Min_per": 100.0, "mpg": 40.0, "pts": 45.0, "treb": 25.0, "oreb": 15.0, "dreb": 20.0,
                 "ast": 15.0, "stl": 6.0, "blk": 10.0, "TPA_pg": 20.0, "FTA_pg": 20.0, "dunk_made": 250.0}

COLLEGE_LINE = [
    "GP", "Min_per", "ORtg", "usg", "eFG", "TS_per", "ORB_per", "DRB_per", "AST_per", "TO_per", "FT_per", "twoP_per",
    "TP_per", "blk_per", "stl_per", "ftr", "porpag", "adjoe", "pfr", "ast_tov", "rim_pct", "mid_pct", "dunk_made",
    "drtg", "adrtg", "dporpag", "stops", "bpm", "obpm", "dbpm", "gbpm", "ogbpm", "dgbpm", "mpg", "oreb", "dreb",
    "treb", "ast", "stl", "blk", "pts", "TPA_pg", "FTA_pg", "class_year", "n_college_seasons", "conf",
]

# Our extra dated blocks, promoted to first-class contract groups by --families (see tournament/contract.py PREFIX).
# wt_ -> person (bio_), bb_ -> momentum (mo_) and cp_ -> comp (sct_) are already first-class and are not repeated.
FAMILY_GROUPS = {"fy_": "fiba_youth", "dx_": "growth", "eur_": "euroleague", "dp_": "draftpage"}

# Every family of ours that the bridge consumes, and the group of his it feeds. Anything not listed here is either
# a fallback for one of his named columns or (with --extras) written as xt_<name>. See MAPPING.md.
CONSUMED_PREFIXES = ("tv_", "col_", "ctx_", "cgd_", "traj_", "bio_", "vmb_", "vcmb_", "intl_", "fy_", "eurs_", "tc_",
                     "gl2_", "sc_", "cp_", "wt_", "misc_", "mock_", "bb_", "vcons_", "cons_", "dis_", "rsci_", "hs_")

# --intl-mask: the columns the old engine's IMASK gene blanks for a college-path row. patch_opt_v34.py does
#     ic = [c for c in B.columns if c.startswith(("intl_", "adj_intl_")) and c != "intl_youth_n"]
#     B.loc[meta.path.str.startswith("college"), ic] = np.nan
# because on a college-path row the intl_ block is a 6-8 game FIBA youth line, not a pro season (117 of the 125
# college rows that carry one are at intl_level == 1). These are the bridge columns fed from an intl_* column.
# i_top_pir_36 and every iz_eur_* come from eur_ / eurs_ only and are NOT blanked; i_top_mpg is, because it falls
# back to intl_mpg. Raw intl_* columns written by --families / --extras are blanked too, except intl_youth_n.
INTL_MASK_COLUMNS = [
    "i_has_pro", "i_league_level", "i_top_level", "i_n_comps", "i_age_season", "i_gp", "i_mpg", "i_pts_36",
    "i_trb_36", "i_orb_36", "i_ast_36", "i_stl_36", "i_blk_36", "i_pir_36", "i_fga_36", "i_fg3a_36", "i_fta_36",
    "i_fg_pct", "i_fg3_pct", "i_ft_pct", "i_fg3a_rate", "i_fta_rate", "i_ts", "i_efg", "i_top_mpg",
    "iz_pts_36", "iz_eff_36", "iz_mpg", "iz_age", "iz_ts", "iz_usg_36", "iz_n_league", "iz_trb_36", "iz_ast_36",
    "iz_stl_36", "iz_blk_36", "iz_tov_36", "iz_fg3a_36", "iz_fta_36", "iz_lg_strength", "iz_season_gap",
    "iz_prev_pts_36", "iz_two_pts_36", "iz_pct_eff", "iz_young_x_eff", "iz_young_x_mpg", "iz_young_x_usg",
]
INTL_MASK_KEEP = "intl_youth_n"   # the engine keeps the youth-event counter: "did he play youth ball at all"


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


class Map:
    """Src plus a recorded coalesce chain per output column.

    `coal(his, ("tv_bpm", s("tv_bpm")), ("col_impact", s("col_impact")))` fills `his` from the first non-null step.
    Every step must already be converted onto the target (Torvik) scale -- that is the whole point of the exercise,
    and the conversions are audited column by column in MAPPING.md. The chain is kept for the build report, and the
    first step alone is kept as the "primary only" coverage the report prints as *before*.
    """

    def __init__(self, s: Src, fallback: bool = True):
        self.s = s
        self.fallback = fallback
        self.chain: dict[str, list[str]] = {}
        self.n_fallback: dict[str, int] = {}
        self.primary: dict[str, pd.Series] = {}

    def coal(self, his: str, *steps: tuple[str, pd.Series]) -> pd.Series:
        out = steps[0][1].astype(float).copy()
        self.primary[his] = out.notna()
        labels = [steps[0][0]]
        if self.fallback:
            for label, ser in steps[1:]:
                out = out.fillna(ser)
                labels.append(label)
        self.chain[his] = labels
        self.n_fallback[his] = len(labels) - 1
        return out

    def gate_college(self, f: dict[str, pd.Series], names, college: pd.Series) -> None:
        """A column of the Torvik COLLEGE line may only be non-null on a row that has a college season.

        Most fallback sources are college-only by construction, but `traj_*` is a pre-draft trajectory across any
        competition: 101 international rows carry traj_last_gp / traj_last_mpg (median 23.7 games, 21.3 mpg -- a pro
        season, not a college one). Without this gate those would land in `GP` / `mpg` / `pts` and silently give the
        column two populations, which is exactly what the fallbacks exist to avoid.
        """
        for k in names:
            f[k] = f[k].where(college)
            self.primary[k] = self.primary[k] & college
            self.chain[k].append("college rows only")


def _pct(s: pd.Series) -> pd.Series:
    """0-1 ratio -> 0-100 percent (Torvik's convention for FT_per / TP_per / twoP_per / ftr)."""
    return s * 100.0


def _within_year_pct(v: pd.Series, year: pd.Series) -> pd.Series:
    return v.groupby(year).rank(pct=True)


def _nz(s: pd.Series) -> pd.Series:
    """Zero -> NaN, so a ratio never divides by an exact zero."""
    return s.replace(0, np.nan)


def _has_college(s: Src) -> pd.Series:
    """A final college season *line* -- what `source == "college"` and `modelled` have always meant here."""
    return s("col_gp").notna() | s("tv_gp").notna()


def _college_path(s: Src) -> pd.Series:
    """A college CAREER, whether or not we hold his box line: the staging `path` meta column (college_2008plus /
    college_pre2008_or_unlinked), which is exactly the row set the old engine's IMASK gene keys on
    (`meta.path.astype(str).str.startswith("college")`, genes/patch_opt_v34.py).

    Wider than `_has_college` by the 122 pre-2008 / unlinked college players whose box line we do not carry: every
    tv_gp row and all but one col_gp row is inside it. Used to decide (a) which rows may carry a college-line
    column at all, (b) who counts as "unranked" rather than "not a recruit", and (c) what --intl-mask blanks.
    """
    df = s.df
    if "path" in df.columns:
        return df.path.astype(str).str.startswith("college") | _has_college(s)
    return _has_college(s)


# --------------------------------------------------------------------------- the mapping

def build_features(m: Map, year: pd.Series) -> dict[str, pd.Series]:
    """Our columns -> his contract. Every entry is documented in bridge/MAPPING.md.

    Scale rules applied here (audited on the tv_/col_ overlap, medians of the ratio in MAPPING.md §2):
      * 0-100 percentages stay 0-100      (col_*_pct already are; col_ft/fg3/fg2_pct and col_ftar are 0-1 -> x100)
      * per-36 -> per game                x mpg/36     (col_*36)
      * per-40 -> per game                x mpg/40     (col_fg3a_per40, col_dunks_per40)
      * season total -> per game          / GP         (ctx_base_fg3a, ctx_base_fg2a, ...)
      * minutes share -> Torvik min%      x 500        (col_minutes_share; ctx_base_minutes_share already is 0-100)
      * recruiting score -> recruiting rank  (100 - score) * 5, clipped to 1..101 (lower = better, his convention)
    """
    s, f = m.s, {}
    coal = m.coal
    college = _college_path(s)

    # ---- Torvik final-season line (NUMERIC_FEATURES). tv_ is the player's own Torvik row (2010+ classes only);
    #      col_ is our wider final-college-season line (2002+ for the box line, 2008+ for the ratings) and
    #      ctx_base_ / cgd_ are the same season restated (2008+). All three are the same college season.
    f["GP"] = coal("GP", ("tv_gp", s("tv_gp")), ("col_gp", s("col_gp")), ("ctx_base_gp", s("ctx_base_gp")),
                   ("cgd_observed_gp", s("cgd_observed_gp")), ("traj_last_gp", s("traj_last_gp")))
    gp = f["GP"]
    # Min_per is the share of his team's minutes he played, 0-100. ctx_base_minutes_share already is that;
    # col_minutes_share is a share of all 5 x 40 team minutes, so x500 puts it on Torvik's scale (ratio 1.007).
    f["Min_per"] = coal("Min_per", ("tv_min_per", s("tv_min_per")),
                        ("ctx_base_minutes_share", s("ctx_base_minutes_share")),
                        ("col_minutes_share x500", s("col_minutes_share") * MIN_PER_PER_SHARE))
    f["mpg"] = coal("mpg", ("col_mpg", s("col_mpg")), ("tv_mp", s("tv_mp")), ("ctx_base_mpg", s("ctx_base_mpg")),
                    ("traj_last_mpg", s("traj_last_mpg")))
    mpg = f["mpg"]
    per36_to_pg, per40_to_pg = mpg / 36.0, mpg / 40.0

    f["ORtg"] = coal("ORtg", ("tv_ortg", s("tv_ortg")), ("col_ortg", s("col_ortg")))
    f["adjoe"] = coal("adjoe", ("tv_adjoe", s("tv_adjoe")), ("col_adj_ortg", s("col_adj_ortg")))
    f["drtg"] = coal("drtg", ("tv_drtg", s("tv_drtg")), ("col_drtg", s("col_drtg")))
    f["adrtg"] = coal("adrtg", ("tv_adrtg", s("tv_adrtg")), ("col_adj_drtg", s("col_adj_drtg")))
    f["usg"] = coal("usg", ("tv_usg", s("tv_usg")), ("col_usg_pct", s("col_usg_pct")),
                    ("ctx_base_usage", s("ctx_base_usage")))
    f["eFG"] = coal("eFG", ("tv_efg", s("tv_efg")), ("col_efg_pct", s("col_efg_pct")),
                    ("ctx_base_efg", s("ctx_base_efg")))
    f["TS_per"] = coal("TS_per", ("tv_ts", s("tv_ts")), ("col_ts_pct", s("col_ts_pct")),
                       ("ctx_base_ts", s("ctx_base_ts")))
    f["ORB_per"] = coal("ORB_per", ("tv_orb", s("tv_orb")), ("col_orb_pct", s("col_orb_pct")),
                        ("ctx_base_orb", s("ctx_base_orb")))
    f["DRB_per"] = coal("DRB_per", ("tv_drb", s("tv_drb")), ("col_drb_pct", s("col_drb_pct")),
                        ("ctx_base_drb", s("ctx_base_drb")))
    f["AST_per"] = coal("AST_per", ("tv_ast_pct", s("tv_ast_pct")), ("col_ast_pct", s("col_ast_pct")),
                        ("ctx_base_ast_pct", s("ctx_base_ast_pct")))
    f["TO_per"] = coal("TO_per", ("tv_to_pct", s("tv_to_pct")), ("col_tov_pct", s("col_tov_pct")),
                       ("ctx_base_tov_pct", s("ctx_base_tov_pct")))
    f["blk_per"] = coal("blk_per", ("col_blk_pct", s("col_blk_pct")), ("ctx_base_blk_pct", s("ctx_base_blk_pct")))
    f["stl_per"] = coal("stl_per", ("col_stl_pct", s("col_stl_pct")), ("ctx_base_stl_pct", s("ctx_base_stl_pct")))
    # ours are 0-1 ratios, Torvik's are 0-100 percentages
    f["FT_per"] = coal("FT_per", ("col_ft_pct x100", _pct(s("col_ft_pct"))),
                       ("ctx_base_ft_pct x100", _pct(s("ctx_base_ft_pct"))))
    f["twoP_per"] = coal("twoP_per", ("col_fg2_pct x100", _pct(s("col_fg2_pct"))),
                         ("ctx_base_fg2_pct x100", _pct(s("ctx_base_fg2_pct"))))
    f["TP_per"] = coal("TP_per", ("col_fg3_pct x100", _pct(s("col_fg3_pct"))),
                       ("ctx_base_fg3_pct x100", _pct(s("ctx_base_fg3_pct"))),
                       ("ctx fg3m/fg3a x100", _pct(s("ctx_base_fg3m") / _nz(s("ctx_base_fg3a")))))
    f["ftr"] = coal("ftr", ("tv_ftr", s("tv_ftr")), ("ctx_base_ftr", s("ctx_base_ftr")),
                    ("col_ftar x100", _pct(s("col_ftar"))))
    # col_value is our points-over-replacement-per-game figure and lands on tv_porpag exactly (ratio 1.000)
    f["porpag"] = coal("porpag", ("tv_porpag", s("tv_porpag")), ("col_value", s("col_value")))
    # personal fouls per 40 minutes
    f["pfr"] = coal("pfr", ("tv_pfr", s("tv_pfr")), ("ctx_base_fouls40", s("ctx_base_fouls40")),
                    ("cgd_pf_pg x40/mpg", s("cgd_pf_pg") * 40.0 / _nz(mpg)))
    f["ast_tov"] = coal("ast_tov", ("tv_ast_tov", s("tv_ast_tov")), ("ctx_base_ast_tov", s("ctx_base_ast_tov")),
                        ("cgd_ast_pg/cgd_tov_pg", s("cgd_ast_pg") / _nz(s("cgd_tov_pg"))))
    # his rec_rank is a recruiting RANK, lower = better (layer2 ranks on -rec_rank). rsci_rank / hs_* already are.
    # tv_rec_rank and col_recruit_score are 0-100 SCORES over the top ~500 recruits: rank = (100 - score) * 5
    # (validated decile by decile -- score 95.0 -> rank 25, 90.8 -> 46, 98.4 -> 8). The old `101 - tv_rec_rank`
    # put a second, incompatible scale into the same column (median 5 against rsci_rank's 22 on the overlap).
    score_to_rank = lambda v: ((100.0 - v) * RSCI_PER_POINT).clip(lower=1.0, upper=UNRANKED_RSCI)
    f["rec_rank"] = coal("rec_rank", ("rsci_rank", s("rsci_rank")), ("hs_rsci_rank", s("hs_rsci_rank")),
                         ("hs_recruit_rank_final", s("hs_recruit_rank_final")),
                         ("col_recruit_score ->rank", score_to_rank(s("col_recruit_score"))),
                         ("tv_rec_rank ->rank", score_to_rank(s("tv_rec_rank"))))
    if m.fallback:
        # a college player on no RSCI top-100 list is not missing, he is unranked: put him behind the last ranked
        # recruit, exactly as mock_rank_consensus puts an un-mocked player behind the last pick. Restricted to
        # college rows in classes the RSCI source actually covers -- an international player is out of universe.
        rsci_class = s("rsci_rank").notna().groupby(year).transform("any")
        unranked = college & rsci_class & f["rec_rank"].isna()
        f["rec_rank"] = f["rec_rank"].mask(unranked, UNRANKED_RSCI)
        m.chain["rec_rank"].append(f"unranked college -> {UNRANKED_RSCI:.0f}")
    f["rim_pct"] = coal("rim_pct", ("tv_rim_pct", s("tv_rim_pct")), ("col_rim_fg_pct", s("col_rim_fg_pct")),
                        ("ctx rim_made/rim_att", s("ctx_base_rim_made") / _nz(s("ctx_base_rim_attempts"))))
    f["mid_pct"] = coal("mid_pct", ("tv_mid_pct", s("tv_mid_pct")), ("col_mid_fg_pct", s("col_mid_fg_pct")),
                        ("ctx (fg2m-rim)/(fg2a-rim)",
                         (s("ctx_base_fg2m") - s("ctx_base_rim_made")) / _nz(s("ctx_base_fg2a") - s("ctx_base_rim_attempts"))))
    f["bpm"] = coal("bpm", ("tv_bpm", s("tv_bpm")), ("col_impact", s("col_impact")),
                    ("tc_first_bpm+tc_bpm_first_to_last", s("tc_first_bpm") + s("tc_bpm_first_to_last")))
    f["obpm"] = coal("obpm", ("tv_obpm", s("tv_obpm")), ("col_impact_off", s("col_impact_off")))
    f["dbpm"] = coal("dbpm", ("tv_dbpm", s("tv_dbpm")), ("col_impact_def", s("col_impact_def")))
    # Torvik-only: no column of ours is on these scales (tc_porpag_share is a team share, tc_slope_bpm a per-season
    # slope, col_impact_def a BPM component -- all checked, none within a factor of the target).
    for only in ("dporpag", "stops", "gbpm", "ogbpm", "dgbpm"):
        f[only] = coal(only, (f"tv_{only}", s(f"tv_{only}")))

    # ---- per-game counting stats. ctx_base_*_pg / cgd_*_pg are per game already; col_*36 needs x mpg/36.
    f["oreb"] = coal("oreb", ("ctx_base_oreb_pg", s("ctx_base_oreb_pg")), ("cgd_orb_pg", s("cgd_orb_pg")),
                     ("col_reb_pg x orb share", s("col_reb_pg") * (s("col_orb_pct") / _nz(s("col_orb_pct") + s("col_drb_pct")))))
    f["dreb"] = coal("dreb", ("ctx_base_dreb_pg", s("ctx_base_dreb_pg")), ("cgd_drb_pg", s("cgd_drb_pg")),
                     ("col_reb_pg x drb share", s("col_reb_pg") * (s("col_drb_pct") / _nz(s("col_orb_pct") + s("col_drb_pct")))))
    f["treb"] = coal("treb", ("col_reb_pg", s("col_reb_pg")), ("oreb+dreb", f["oreb"] + f["dreb"]),
                     ("col_reb36 x mpg/36", s("col_reb36") * per36_to_pg))
    f["ast"] = coal("ast", ("col_ast_pg", s("col_ast_pg")), ("ctx_base_ast_pg", s("ctx_base_ast_pg")),
                    ("cgd_ast_pg", s("cgd_ast_pg")), ("col_ast36 x mpg/36", s("col_ast36") * per36_to_pg))
    f["stl"] = coal("stl", ("col_stl_pg", s("col_stl_pg")), ("ctx_base_stl_pg", s("ctx_base_stl_pg")),
                    ("cgd_stl_pg", s("cgd_stl_pg")), ("col_stl36 x mpg/36", s("col_stl36") * per36_to_pg))
    f["blk"] = coal("blk", ("col_blk_pg", s("col_blk_pg")), ("ctx_base_blk_pg", s("ctx_base_blk_pg")),
                    ("cgd_blk_pg", s("cgd_blk_pg")), ("col_blk36 x mpg/36", s("col_blk36") * per36_to_pg))
    f["pts"] = coal("pts", ("tv_pts", s("tv_pts")), ("col_pts_pg", s("col_pts_pg")),
                    ("ctx_base_pts_pg", s("ctx_base_pts_pg")), ("cgd_points_pg", s("cgd_points_pg")),
                    ("col_pts36 x mpg/36", s("col_pts36") * per36_to_pg),
                    ("traj_last_pts36 x mpg/36", s("traj_last_pts36") * per36_to_pg))
    # shot volume: cgd_*_pg is the per-game attempt line; otherwise reconstruct from the per-40 three-point volume
    # and the two attempt rates (FGA/40 = 3PA/40 / 3PAr); ctx_base_fg2a / fg3a are SEASON TOTALS, hence / GP.
    tpa40 = s("col_fg3a_per40")
    fga40 = tpa40 / _nz(s("col_fg3ar"))
    f["TPA_pg"] = coal("TPA_pg", ("cgd_three_a_pg", s("cgd_three_a_pg")),
                       ("col_fg3a_per40 x mpg/40", tpa40 * per40_to_pg),
                       ("ctx_base_fg3a / GP", s("ctx_base_fg3a") / _nz(gp)))
    f["FTA_pg"] = coal("FTA_pg", ("cgd_ft_a_pg", s("cgd_ft_a_pg")),
                       ("col_ftar x FGA/40 x mpg/40", s("col_ftar") * fga40 * per40_to_pg),
                       ("ctx ftr x (fg2a+fg3a) / GP",
                        s("ctx_base_ftr") / 100.0 * (s("ctx_base_fg2a") + s("ctx_base_fg3a")) / _nz(gp)))
    # his dunk_made is a SEASON COUNT; tv_dunks_pg and col_dunks_per40 are rates, ctx_base_dunk_made is a count
    f["dunk_made"] = coal("dunk_made", ("tv_dunks_pg x GP", s("tv_dunks_pg") * gp),
                          ("ctx_base_dunk_made", s("ctx_base_dunk_made")),
                          ("col_dunks_per40 x mpg/40 x GP", s("col_dunks_per40") * per40_to_pg * gp))

    # ---- bio columns of the Torvik line
    f["height_in"] = coal("height_in", ("bio_height_in", s("bio_height_in")),
                          ("vmb_listed_height_in", s("vmb_listed_height_in")),
                          ("bio_combine_height_in", s("bio_combine_height_in")),
                          ("vcmb_height_with_shoes_in", s("vcmb_height_with_shoes_in")),
                          ("dx_last_height_in", s("dx_last_height_in")),
                          ("rsci_hs_height_in", s("rsci_hs_height_in")))
    # Torvik's yr is Fr/So/Jr/Sr = 1-4; col_class_index runs to 6 for a super-senior, so it is clipped to Torvik's
    f["class_year"] = coal("class_year", ("tv_yr", s("tv_yr")),
                           ("col_class_index clip 1-4", s("col_class_index").clip(lower=1, upper=4)),
                           ("vmb_college_class_year", s("vmb_college_class_year")))
    f["n_college_seasons"] = coal("n_college_seasons", ("col_n_seasons", s("col_n_seasons")),
                                  ("tv_seasons", s("tv_seasons")), ("tc_n_seasons", s("tc_n_seasons")),
                                  ("traj_n_seasons", s("traj_n_seasons")))
    # col_age is his age during the final COLLEGE SEASON, ~0.27 yr below draft age; last in the chain and in
    # practice never reached, because bio_age_at_draft covers every row.
    f["age_at_draft"] = coal("age_at_draft", ("bio_age_at_draft", s("bio_age_at_draft")),
                             ("tv_age_exact", s("tv_age_exact")),
                             ("vmb_age_reported_years", s("vmb_age_reported_years")), ("col_age", s("col_age")))

    # ---- categoricals (numeric ids; the winner drops them with "-cat", they exist only to satisfy the contract)
    f["conf"] = coal("conf", ("tv_conf_tier", s("tv_conf_tier")))   # tc_conf_strength_prior is a rating, not a tier
    f["role"] = coal("role", ("tv_role", s("tv_role")),
                     ("bio_pos_code ->role", s("bio_pos_code").map(POS_TO_ROLE)))

    # ---- src_torvik: 1 = the row's college line is the player's own Torvik final season, 0 = a fallback filled it.
    #      Lets a model separate the two populations instead of reading one column as if it had one provenance.
    f["src_torvik"] = s("tv_gp").notna().astype(float)

    # ---- a college-line column may only be non-null where there IS a college season (see Map.gate_college)
    m.gate_college(f, COLLEGE_LINE, college)
    for k, hi in COUNT_CEILING.items():                     # a corrupt staging cell must not survive a conversion
        f[k] = f[k].where((f[k] >= 0) & (f[k] <= hi))

    # ---- traj: previous college season and the change into the final one
    for his, delta, cur in [("bpm", "tv_d_bpm", "bpm"), ("usg", "tv_d_usg", "usg"), ("TS_per", "tv_d_ts", "TS_per")]:
        d = s(delta)
        f[f"d_{his}"] = d
        f[f"prev_{his}"] = f[cur] - d
    # minutes per game. tv_d_min_per is a minutes-SHARE delta, col_usg_delta / traj_mpg_growth / tc_slope_bpm are
    # first-to-last (not previous-to-last) changes -- none is interchangeable with a one-season delta, so none is
    # used as a fallback here (checked: rho 0.44-0.68 with a systematic offset).
    d_mpg = s("traj_d_mpg")
    f["d_mpg"], f["prev_mpg"] = d_mpg, f["mpg"] - d_mpg
    for miss in ("obpm", "dbpm", "porpag"):                     # no per-stat previous-season delta in our blocks
        f[f"d_{miss}"] = pd.Series(np.nan, index=s.df.index)
        f[f"prev_{miss}"] = pd.Series(np.nan, index=s.df.index)
    bpm_hist = pd.concat([f["bpm"], f["prev_bpm"], s.first("tc_first_bpm", "tv_first_bpm")], axis=1)
    f["career_bpm_mean"] = bpm_hist.mean(axis=1)                # first / previous / final season, not the full career
    f["career_bpm_max"] = bpm_hist.max(axis=1)

    # ---- phys
    f["wingspan_in"] = coal("wingspan_in", ("vcmb_wingspan_in", s("vcmb_wingspan_in")),
                            ("bio_combine_wingspan_in", s("bio_combine_wingspan_in")),
                            ("dx_last_wingspan_in", s("dx_last_wingspan_in")))
    f["weight_lb"] = coal("weight_lb", ("bio_weight_lb", s("bio_weight_lb")), ("vcmb_weight_lb", s("vcmb_weight_lb")),
                          ("vmb_listed_weight_lb", s("vmb_listed_weight_lb")),
                          ("dx_last_weight_lb", s("dx_last_weight_lb")))
    f["wing_minus_height"] = coal(
        "wing_minus_height", ("vcmb_wingspan_minus_height_in", s("vcmb_wingspan_minus_height_in")),
        ("bio_combine_wingspan_minus_height_in", s("bio_combine_wingspan_minus_height_in")),
        ("wingspan - height", f["wingspan_in"] - f["height_in"]))
    f["draft_age_x"] = coal("draft_age_x", ("bio_age_at_draft", s("bio_age_at_draft")),
                            ("vmb_age_reported_years", s("vmb_age_reported_years")),
                            ("age_at_draft", f["age_at_draft"]))

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
    i_orb_share = s("intl_orb_pct") / _nz(s("intl_orb_pct") + s("intl_drb_pct"))
    f["i_orb_36"] = s("intl_reb36") * i_orb_share
    f["i_ast_36"] = s("intl_ast36")
    f["i_stl_36"] = s("intl_stl36")
    f["i_blk_36"] = s("intl_blk36")
    f["i_pir_36"] = s("intl_impact")
    ts = s("intl_ts_pct") / 100.0
    ftr_i = s("intl_ftar")
    # true-shooting attempts back out the shot diet: TSA/36 = pts36 / (2*TS); FGA = TSA / (1 + 0.44*FTr)
    tsa36 = s("intl_pts36") / (2.0 * _nz(ts))
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
    top_g = _nz(s("eur_el2_ls_games"))
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

    # ---- combine (prefix c_): NBA combine anthropometrics, athletic tests, shooting drills. bio_combine_* is the
    #      same tape from the biography feed and backs every measurement vcmb_ does not carry.
    h = s.first("vcmb_height_without_shoes_in", "bio_combine_height_in")
    reach = s.first("vcmb_standing_reach_in", "bio_combine_reach_in")
    wing = s.first("vcmb_wingspan_in", "bio_combine_wingspan_in")
    vs = s.first("vcmb_standing_vertical_in", "bio_combine_vertical_standing_in")
    vm = s.first("vcmb_max_vertical_in", "bio_combine_vertical_max_in")
    wt = s.first("vcmb_weight_lb", "bio_combine_weight_lb")
    bf = s.first("vcmb_body_fat_pct", "bio_combine_body_fat_pct")
    hl = s.first("vcmb_hand_length_in", "bio_combine_hand_length_in")
    hw = s.first("vcmb_hand_width_in", "bio_combine_hand_width_in")
    lane = s.first("vcmb_lane_agility_s", "bio_combine_lane_agility_s")
    sprint = s.first("vcmb_three_quarter_sprint_s", "bio_combine_sprint_s")
    bench = s.first("vcmb_bench_press_reps", "bio_combine_bench_reps")
    f["c_height_noshoes"], f["c_wingspan"], f["c_standing_reach"], f["c_weight"] = h, wing, reach, wt
    f["c_body_fat"], f["c_hand_length"], f["c_hand_width"] = bf, hl, hw
    f["c_vert_standing"], f["c_vert_max"] = vs, vm
    f["c_lane_agility"] = lane
    f["c_shuttle"] = s("vcmb_modified_lane_agility_s")
    f["c_sprint"] = sprint
    f["c_bench"] = bench
    f["c_wing_minus_height"] = s.first("vcmb_wingspan_minus_height_in",
                                       "bio_combine_wingspan_minus_height_in").fillna(wing - h)
    f["c_reach_minus_height"] = reach - h
    f["c_wing_height_ratio"] = wing / _nz(h)
    f["c_reach_height_ratio"] = reach / _nz(h)
    f["c_hand_length_height_ratio"] = hl / _nz(h)
    f["c_hand_width_height_ratio"] = hw / _nz(h)
    f["c_weight_per_in"] = wt / _nz(h)
    f["c_bmi_like"] = s.first("vcmb_bmi_kg_m2", "bio_bmi_proxy")
    f["c_lean_mass"] = wt * (1 - bf / 100.0)
    f["c_fat_mass"] = wt * (bf / 100.0)
    f["c_lean_mass_height_ratio"] = f["c_lean_mass"] / _nz(h)
    f["c_approach_vert_gain"] = vm - vs
    f["c_max_touch"] = reach + vm
    f["c_standing_touch"] = reach + vs
    anthro = [h, s("vcmb_height_with_shoes_in"), wing, reach, wt, bf, hl, hw]
    drills = [vs, vm, lane, s("vcmb_modified_lane_agility_s"), sprint, bench]
    f["c_anthro_n"] = pd.concat(anthro, axis=1).notna().sum(axis=1).astype(float)
    f["c_drills_n"] = pd.concat(drills, axis=1).notna().sum(axis=1).astype(float)
    made = ["vcmb_spot_15ft_made", "vcmb_spot_college3_made", "vcmb_spot_nba3_made",
            "vcmb_off_dribble_15ft_made", "vcmb_off_dribble_college3_made",
            "vcmb_on_move_15ft_made", "vcmb_on_move_college3_made"]
    att = [c.replace("_made", "_attempted") for c in made]
    m_sum = pd.concat([s(c) for c in made], axis=1).sum(axis=1, min_count=1)
    a_sum = pd.concat([s(c) for c in att], axis=1).sum(axis=1, min_count=1)
    f["c_shoot_pct"] = m_sum / _nz(a_sum)

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
    f["bio_born_us"] = (1.0 - s("wt_born_outside_usa")).fillna(s("misc_born_usa")).fillna(s("bio_country_usa"))
    f["bio_n_high_schools"] = s.first("wt_hs_transferred", "pre_n_schools")
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
    f["sc_ratio_strength_weakness"] = ws / _nz(ww)

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


# --------------------------------------------------------------------------- the gl2_ response screen

def response_screen(df: pd.DataFrame, identity: Path | None, screen: Path) -> pd.DataFrame:
    """`col, n, rho_raw, rho_res, abs_res` for every gl2_ column, strongest first.

    Identical definition to datarebuild/v4_build/screen_blocks.py: draft classes 2000-2011 only (the walk-forward
    folds start at 2012, so this never touches a fold), drafted rows with a label; `rho_raw` is the n-weighted mean
    within-class Spearman against y_early_war_log and `rho_res` the same against the residual of that label on
    log(actual_pick) -- information beyond the draft slot. Verified to reproduce the frozen `gls_` slim block: its
    ten members come out ranked 1, 2, 4, 6, 7, 8, 9, 10, 12, 13 (the gaps are the near-duplicates that script drops).

    Preference order: gl2_ rows already in screen_blocks.csv -> our cached recompute -> recompute and cache.
    """
    if screen.exists():
        S = pd.read_csv(screen)
        S = S[S.col.astype(str).str.startswith("gl2_") & S.get("abs_res", pd.Series(dtype=float)).notna()]
        if len(S):
            return S.sort_values("abs_res", ascending=False).reset_index(drop=True)
    if SCREEN_CACHE.exists():
        return pd.read_csv(SCREEN_CACHE).sort_values("abs_res", ascending=False).reset_index(drop=True)

    cols = [c for c in df.columns if c.startswith("gl2_")]
    d = df[(df.draft_year <= 2011) & (df.was_drafted == 1) & df.get("y_early_war_log", np.nan).notna()].copy()
    y = "y_early_war_log"
    if identity is not None and identity.exists():
        ident = pd.read_csv(identity, usecols=["pid", "actual_pick"])
        d = d.merge(ident, on="pid", how="left")
        d = d[d.actual_pick.notna()]
        A = np.c_[np.ones(len(d)), np.log(d.actual_pick.clip(lower=1))]
        beta = np.linalg.lstsq(A, d[y].values, rcond=None)[0]
        d["y_res"] = d[y] - A @ beta
    else:
        d["y_res"] = d[y]                       # --no-picks: fall back to the raw label screen, and say so
    print(f"response screen: recomputed on {len(d)} rows, classes {int(d.draft_year.min())}-{int(d.draft_year.max())}"
          f"{'' if 'actual_pick' in d else ' (raw label, no pick residual -- --no-picks)'}")

    def wcorr(col: str, tgt: str) -> float:
        rs, ws = [], []
        for _, g in d.groupby("draft_year"):
            mk = g[col].notna() & g[tgt].notna()
            if mk.sum() >= 12 and g.loc[mk, col].nunique() > 1:
                r = g.loc[mk, col].corr(g.loc[mk, tgt], method="spearman")
                if pd.notna(r):
                    rs.append(r), ws.append(int(mk.sum()))
        return float(np.average(rs, weights=ws)) if rs else np.nan

    rows = []
    for c in cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
        nn = int(d[c].notna().sum())
        rows.append((c, nn, np.nan, np.nan) if nn < 60 else (c, nn, wcorr(c, y), wcorr(c, "y_res")))
    S = pd.DataFrame(rows, columns=["col", "n", "rho_raw", "rho_res"])
    S["abs_res"] = S.rho_res.abs()
    S = S.sort_values("abs_res", ascending=False).reset_index(drop=True)
    S.to_csv(SCREEN_CACHE, index=False)
    print(f"response screen: cached to {SCREEN_CACHE}")
    return S


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


def _band_cov(mask: pd.Series, year: pd.Series) -> dict[str, float]:
    return {b: round(float(mask[(year >= a) & (year <= z)].mean()), 3) for b, a, z in BANDS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging", type=Path, default=STAGING)
    ap.add_argument("--identity", type=Path, default=IDENTITY, help="pid -> actual_pick; player_name is never read")
    ap.add_argument("--no-picks", action="store_true",
                    help="do not read the identity file; fill `pick` with a within-class placeholder. His metrics "
                         "(spearman_nba, WAR captured) then mean nothing, but layer 1 still runs and scores.")
    ap.add_argument("--include-undrafted", action="store_true",
                    help="keep was_drafted=0 rows. Off by default: his pool is the players actually drafted.")
    ap.add_argument("--no-fallback", action="store_true",
                    help="primary source only: every named feature keeps just the first step of its coalesce chain "
                         "(tv_ where there is one). The A/B floor for what the fallbacks are worth.")
    ap.add_argument("--intl-mask", action="store_true",
                    help="blank the intl_-fed columns on rows with a college line, as the old engine's IMASK gene "
                         "does: there the intl_ block is a 6-8 game FIBA youth line, not a pro season.")
    ap.add_argument("--families", action="store_true",
                    help="write our extra dated blocks as first-class contract groups with their own prefixes "
                         "(fy_ fiba_youth, dx_ growth, eur_ euroleague, dp_ draftpage) instead of the xt_ catch-all.")
    ap.add_argument("--response-max", type=int, default=None, metavar="N",
                    help="keep only the N strongest rs_ (gl2_) columns by the 2000-2011 pre-fold label screen "
                         "(abs_res in screen_blocks.csv). Default: all 147.")
    ap.add_argument("--screen", type=Path, default=SCREEN, help="the pre-fold screen used by --response-max")
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
    m = Map(s, fallback=not a.no_fallback)
    feats = build_features(m, year)

    table = pd.DataFrame(index=df.index)
    table["bbref_id"] = df.pid.astype(str)      # pid-keyed: no name ever enters this table
    table["key"] = df.pid.astype(str)
    table["player"] = df.pid.astype(str)
    table["draft_year"] = year
    block = pd.DataFrame({k: pd.to_numeric(v, errors="coerce").astype(float).values for k, v in feats.items()},
                         index=table.index)
    table = pd.concat([table, block], axis=1)

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
    has_college = _has_college(s).values
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

    # ---- optional: our extra dated blocks as first-class contract groups, under their own prefixes
    fam_written: dict[str, list[str]] = {}
    if a.families:
        cols = {}
        for prefix, group in FAMILY_GROUPS.items():
            names = s.prefixed(prefix)
            fam_written[group] = names
            for c in names:
                cols[c] = pd.to_numeric(df[c], errors="coerce").astype(float).values
        table = pd.concat([table, pd.DataFrame(cols, index=table.index)], axis=1)

    # ---- optional: trim the response group to the N strongest columns by the pre-fold screen
    response_note = f"all {len([c for c in table.columns if c.startswith('rs_')])} gl2_ columns"
    if a.response_max is not None:
        S = response_screen(df, None if a.no_picks else a.identity, a.screen)
        keep = ["rs_" + c[4:] for c in S.col.head(a.response_max)]
        drop = [c for c in table.columns if c.startswith("rs_") and c not in keep]
        table = table.drop(columns=drop)
        response_note = (f"top {a.response_max} of {a.response_max + len(drop)} by |rho beyond pick|, "
                         f"2000-2011 screen; dropped {len(drop)}")

    # ---- optional: the old engine's IMASK -- on a college row the intl_ block is a youth-tournament line
    masked: list[str] = []
    college_rows = _college_path(s).values          # the engine's IMASK row set, not just "has a box line"
    if a.intl_mask:
        masked = [c for c in INTL_MASK_COLUMNS if c in table.columns]
        masked += [c for c in table.columns
                   if c.startswith(("intl_", "xt_intl_")) and c not in (INTL_MASK_KEEP, "xt_" + INTL_MASK_KEEP)]
        table.loc[college_rows, masked] = np.nan

    # ---- every column his contract lists must exist; families we cannot fill are present and empty (see MAPPING.md)
    required = (set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES) | set(FEATURES)
                | {c for v in TORVIK.values() for c in v} | {c for v in EXPLICIT.values() for c in v})
    unfillable = sorted(required - set(table.columns))
    if unfillable:
        table = pd.concat([table, pd.DataFrame(np.nan, index=table.index, columns=unfillable)], axis=1)

    # ---- optional: everything we did not consume, for building past the ported winner
    if a.extras:
        consumed = set(s.used) | {c for v in fam_written.values() for c in v}
        extra = [c for c in inputs if c in df.columns and c not in consumed]
        table = pd.concat([table, pd.DataFrame({"xt_" + c: pd.to_numeric(df[c], errors="coerce").astype(float).values
                                                for c in extra}, index=table.index)], axis=1)
        print(f"extras: {len(extra)} unmapped input columns written as xt_*")
        if a.intl_mask:                                 # the raw intl_ copies must be masked too
            extra_intl = [c for c in table.columns if c.startswith("xt_intl_") and c != "xt_" + INTL_MASK_KEEP]
            table.loc[college_rows, extra_intl] = np.nan
            masked += extra_intl

    table = table.copy()                                # de-fragment before writing
    a.out.mkdir(parents=True, exist_ok=True)
    swar = season_war_rows(df[train_mask]) if train_mask.any() else pd.DataFrame(columns=["bbref_id", "season", "war"])
    swar.to_parquet(a.out / "season_war.parquet", index=False)
    table.to_parquet(a.out / "draft_table.parquet", index=False)

    # ---- report ------------------------------------------------------------------------------------------------
    pd.set_option("display.width", 200)
    before = pd.read_csv(BEFORE_COV).set_index("feature") if BEFORE_COV.exists() else None
    rows = []
    for k in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        pri, fin = _band_cov(m.primary[k], year), _band_cov(table[k].notna(), year)
        r = {"feature": k, "chain": " <- ".join(m.chain[k])}
        for b, _, _ in BANDS:
            if before is not None and k in before.index:
                r["v1_" + b] = round(float(before.loc[k, b]), 3)
            r["pri_" + b], r["now_" + b] = pri[b], fin[b]
        rows.append(r)
    cov = pd.DataFrame(rows)
    print("\nnamed-feature coverage -- pri_ = primary source only (before), now_ = with fallbacks (after)"
          + (", v1_ = the pre-fallback mapping" if before is not None else ""))
    print(cov.to_string(index=False))
    mean_cols = [c for c in cov.columns if c.split("_", 1)[0] in ("v1", "pri", "now")]
    print("mean over the 50 named features:", {c: round(float(cov[c].mean()), 3) for c in mean_cols})

    print("\nfallback chains actually used (first non-null wins; every step converted onto Torvik's scale)")
    for k in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        if m.n_fallback[k]:
            print(f"  {k:20s} {' <- '.join(m.chain[k])}")
    solo = [k for k in NUMERIC_FEATURES + CATEGORICAL_FEATURES if not m.n_fallback[k]]
    print(f"  no scale-compatible fallback: {', '.join(solo)}")

    fam = {
        "all_torvik": NUMERIC_FEATURES + CATEGORICAL_FEATURES, "traj": TRAJECTORY_FEATURES,
        "phys": PHYSICAL_FEATURES, "intl_pro": INTL_PRO, "intl_fiba": INTL_FIBA,
        "a_box": list(AYUSH_BOX.values()), "jasong": list(JASONG_NUM.values()),
        "combine (c_)": [c for c in table if c.startswith("c_")], "tctx (t_)": [c for c in table if c.startswith("t_")],
        "intl_z (iz_)": [c for c in table if c.startswith("iz_")], "mock (mock_)": [c for c in table if c.startswith("mock_")],
        "momentum (mo_)": [c for c in table if c.startswith("mo_")], "response (rs_)": [c for c in table if c.startswith("rs_")],
        "person (bio_)": [c for c in table if c.startswith("bio_")], "scouting (sc_)": [c for c in table if c.startswith("sc_")],
        "comp (sct_)": [c for c in table if c.startswith("sct_")], "srcflag (src_)": [c for c in table if c.startswith("src_")],
        **{f"{g} ({p})": [c for c in table if c.startswith(p)] for p, g in FAMILY_GROUPS.items() if a.families},
        "extra (xt_)": [c for c in table if c.startswith("xt_")],
    }
    ctx, hold = table[table.draft_year <= 2018], table[table.draft_year >= 2019]
    rows = []
    for name, cols in fam.items():
        cols = [c for c in cols if c in table.columns]
        live = [c for c in cols if table[c].notna().any()]
        rows.append({"family": name, "cols": len(cols), "filled": len(live),
                     "cov_2003_18": round(float(ctx[live].notna().mean().mean()), 3) if live else 0.0,
                     "cov_2019_25": round(float(hold[live].notna().mean().mean()), 3) if live else 0.0})
    print("\ncolumns filled per family")
    print(pd.DataFrame(rows).to_string(index=False))

    per_class = table.groupby("draft_year").agg(
        rows=("bbref_id", "size"), modelled=("modelled", "sum"),
        college=("source", lambda x: int((x == "college").sum())), intl=("source", lambda x: int((x == "intl").sum())),
        labelled=("labelled", "sum"), src_torvik=("src_torvik", "sum"))
    print("\nrows per draft class")
    print(per_class.T.to_string())

    print(f"\nsrc_torvik: {int(table.src_torvik.sum())} of {len(table)} rows carry the player's own Torvik line; "
          f"the rest are filled from col_ / ctx_base_ / cgd_ and marked 0")
    if a.intl_mask:
        print(f"intl-mask: {len(masked)} columns blanked on {int(college_rows.sum())} rows with a college career "
              f"(keeping {INTL_MASK_KEEP}):")
        print("  " + ", ".join(masked))
    else:
        would = [c for c in INTL_MASK_COLUMNS if c in table.columns]
        print(f"intl-mask: off (--intl-mask would blank {len(would)} intl_-fed columns on "
              f"{int(college_rows.sum())} rows with a college career, keeping {INTL_MASK_KEEP}; MAPPING.md section 5)")
    if a.families:
        print("families: " + ", ".join(f"{g} ({p}) {len(fam_written[g])} cols" for p, g in FAMILY_GROUPS.items()))
    else:
        print("families: off (fy_ / dx_ / eur_ / dp_ reachable only through --extras, as xt_*)")
    print(f"response: {response_note}")
    print(f"fallbacks: {'OFF (--no-fallback: primary source only)' if a.no_fallback else 'on'}")
    print(f"pick: {pick_note}")
    print(f"season_war.parquet: {len(swar)} player-seasons, {swar.bbref_id.nunique()} players, "
          f"seasons {int(swar.season.min())}-{int(swar.season.max())}" if len(swar) else "season_war.parquet: empty")
    print(f"draft_table.parquet: {table.shape[0]} rows x {table.shape[1]} columns -> {a.out / 'draft_table.parquet'}")


if __name__ == "__main__":
    main()
