"""The contract: every feature group, model and label framing the tournament can use, in one place.

  FEATURE GROUPS  named lists of columns in data/processed/draft_table.parquet. Torvik groups are explicit; every
                  external source is a group defined by its column prefix, so a source becomes sweepable the moment
                  its parquet is materialised (tournament/builders/materialize.py) and the table rebuilt.
  MODELS          the estimators layer 1 knows how to run, with the framing each one uses.
  LABELS          how the context label is rewritten before the model sees it.

`validate(table)` fails if any listed column is missing and reports coverage per group for college and international
draftees. `assert_covered(configs)` fails if a sweep grid leaves any group or model untouched -- the guarantee that
"all the features and both models" really are in the sweep.
"""

import pandas as pd

from infra.dataset import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES
from infra.external import INTL_FIBA, INTL_PRO, PHYSICAL_FEATURES, TRAJECTORY_FEATURES, AYUSH_BOX, JASONG_NUM

# --------------------------------------------------------------------------- feature groups

TORVIK = {
    "rate": ["usg", "eFG", "TS_per", "ORB_per", "DRB_per", "AST_per", "TO_per", "FT_per", "twoP_per", "TP_per", "blk_per", "stl_per", "ftr", "ast_tov"],
    "adv": ["ORtg", "porpag", "adjoe", "drtg", "adrtg", "dporpag", "stops", "bpm", "obpm", "dbpm", "gbpm", "ogbpm", "dgbpm", "pfr"],
    "count": ["GP", "Min_per", "mpg", "oreb", "dreb", "treb", "ast", "stl", "blk", "pts", "TPA_pg", "FTA_pg"],
    "shotloc": ["rim_pct", "mid_pct", "dunk_made"],
    "bio": ["height_in", "class_year", "n_college_seasons", "age_at_draft", "rec_rank"],
    "cat": list(CATEGORICAL_FEATURES),
    "core": ["bpm", "obpm", "dbpm", "porpag", "dporpag", "adjoe", "adrtg", "usg", "TS_per", "age_at_draft", "height_in", "rec_rank", "class_year"],
}
EXPLICIT = {
    "traj": TRAJECTORY_FEATURES,                        # previous college season + change into the final one (Torvik)
    "phys": PHYSICAL_FEATURES,                          # wingspan, weight, wingspan-height, draft age (AyushBatra)
    "a_box": list(AYUSH_BOX.values()),                  # second, independent college stat line (sports-reference via AyushBatra)
    "j_bio": ["rsci", "j_height", "j_weight", "j_age"],  # JasonG: recruiting rank, measurements, age
    "j_team": ["sos", "j_wins", "j_losses"],
    "j_box": [v for v in JASONG_NUM.values() if v.startswith("j_") and v not in ("j_height", "j_weight", "j_age", "j_wins", "j_losses")
              and not any(v.startswith(p) for p in ("j_dunk", "j_rim", "j_fg_", "j_astd", "j_pct"))],
    "j_shot": ["dunks_per_min", "j_dunks", "j_dunk_vs_rim", "j_dunks_unast", "j_rim_unast_100", "pct_rim", "j_fg_rim", "j_astd_rim", "j_pct_mid", "j_fg_mid", "j_astd_mid", "j_pct_3", "j_astd_3", "pct_astd"],
    "j_aau": [v for v in JASONG_NUM.values() if v.startswith("aau_")],
    "j_event": [v for v in JASONG_NUM.values() if v.startswith("ev_")],
    "intl_pro": INTL_PRO,                               # last pre-draft pro season abroad / G League, per 36, league level
    "intl_fiba": INTL_FIBA,                             # FIBA U16-U20 national-team tournaments + EuroLeague NGT
}
# Prefix-defined groups from materialised sources (present only if the table has the columns).
PREFIX = {
    "combine": "c_",        # NBA draft combine anthro + athletic testing (NBA.com)
    "kaggle": "k_",         # Kaggle college players 2009-21 (HF mirror), columns not already in Torvik
    "score": "s_",          # SCORE draft table, pre-draft columns only
    "ianstack": "is_",      # combine + final-college win shares
    "hoopr": "h_",          # hoopR season player boxes: game-level variability, starts, trend
    "marchmadness": "mm_",  # tournament-team player seasons with team SOS/SRS/seed
    "tctx": "t_",           # Torvik context: teammate quality, shares, age/role residuals, conference strength, development
    "game": "g_",           # game-level context: opponent quality, role elasticity, star absence, trajectory
    "intl_z": "iz_",        # international production z-scored within the player's own league-season, young-x-production
    "shrunk": "sh_",        # rate stats shrunk toward the same-season D1 mean by minutes played (3-game seasons stop looking elite)
    "eurocamp": "ec_",      # adidas Eurocamp attendance + measurements
    "bwb": "bwb_",          # Basketball Without Borders selection
    "academy": "acad_",     # NBA Academy membership / stats
    "transfers": "tr_",     # college transfers inferred from Torvik: deltas across the move, level change
    "mock": "mock_",        # pre-draft public mock consensus and source ranks; explicitly excluded from pure pipeline FEATURES
    "momentum": "mo_",      # pre-draft mock-rank movement across 90/60/30/7-day snapshots
    "odds": "odds_",        # pre-draft sportsbook position markets, vig removed
    "scouting": "sc_",      # archived pre-draft NBADraft.net grades and text-derived flags
    "dev": "dv_",           # population development priors: expected next-season improvement for the profile, projected level, surprise
    "gleague": "gl_",       # G League translation priors: expected first-pro-season WS/48, PER, TS, usage, minutes for the college profile
    "population": "pop_",   # every-D1-player NBA-outcome priors: P(NBA minutes), expected 3-season WAR and minutes for the profile
    "response": "rs_",      # challenge -> response from the final-season game log: rematches, bounce-back, error persistence, close games
    "comp": "sct_",         # scouting write-up length and the NBA-comparison player's realised value through draft night
    "impact": "he_",        # Hoop Explorer lineup impact (2019+ classes): RAPM, on/off, play-type profile, vs-top-100 versions
    "person": "bio_",       # basketball-reference biography: NBA relatives, shooting hand, birthplace, high-school path
    "coach": "co_",         # head coach tenure / record / tournament history, team record, causal coach + program NBA track record
    # BRIDGE: our own dated blocks, promoted to first-class groups by `bridge/build_table.py --families`. Each is
    # BRIDGE: absent (and the group therefore absent) unless that flag is used; see bridge/MAPPING.md section 5.
    "fiba_youth": "fy_",    # BRIDGE: FIBA U16-U20 record, age-relative and z-scored inside the event cohort
    "growth": "dx_",        # BRIDGE: measurement history across youth events: height/wingspan/weight growth per year
    "euroleague": "eur_",   # BRIDGE: full EuroLeague / EuroCup / ANGT spine (the iz_eur_ slim variant is intl_z)
    "draftpage": "dp_",     # BRIDGE: draft-page facts: green room, early entrant, auto-eligible, projected
    # BRIDGE: `src_torvik`, 1 when the row's college line is the player's own Torvik final season and 0 when a
    # BRIDGE: fallback (col_ / ctx_base_ / cgd_) filled it. Always written; name a config's features "srcflag".
    "srcflag": "src_",
    # BRIDGE: every verified pre-draft column of ours that his groups have no home for, written by
    # BRIDGE: `bridge/build_table.py --extras`. Absent (and the group therefore absent) unless that flag is used.
    "extra": "xt_",
}

# --------------------------------------------------------------------------- models and labels

MODELS = {
    "tabfm": "Google TabFM 1.0 regression head (1.65B, in-context) on the transformed WAR target",
    "tabfm_cls": "TabFM classification head on within-class quantile bins, read out as the expected bin",
    "exaone": "LG EXAONE-Tabular regressor (21M, in-context); wants narrow feature sets",
    "exaone_cls": "EXAONE-Tabular classifier on within-class bins, expected-bin readout; best with ~30-50 features",
    "xgb": "shallow XGBoost regressor with a monotone age constraint",
    "xgbrank": "pairwise XGBoost ranker grouped by draft class",
    "catboost": "CatBoost regressor",
    "catrank": "pairwise CatBoost ranker grouped by draft class",
    "extratrees": "randomised-tree regression baseline",
    "tabicl": "TabICL v2 in-context regressor; complementary rank-stack member",
    "tabldm": "Xiaomi-TabLDM in-context regressor",
    "tabpfn26": "TabPFN 2.6 regressor (internal evaluation only; restricted model license)",
    "tabpfn3": "TabPFN 3 regressor (internal evaluation only; restricted model license)",
    "ridge": "RidgeCV baseline (median-impute + standardise); cheap diverse stack member",
}
LABELS = {
    "raw": "the WAR target as is",
    "zscore": "z-scored within each draft class",
    "rank": "percentile within class",
    "gaussrank": "Gaussianised percentile within class",
    "disc85_gaussrank": "first-five-season WAR discounted 0.85 per season, clipped, then Gaussian-ranked within class",
}

# --------------------------------------------------------------------------- api

def groups(table: pd.DataFrame) -> dict[str, list[str]]:
    g = {**TORVIK, **EXPLICIT}
    for name, p in PREFIX.items():
        cols = [c for c in table.columns if c.startswith(p) and pd.api.types.is_numeric_dtype(table[c])]
        if cols:
            g[name] = cols
    g["pipeline"] = list(FEATURES)                       # exactly what pipeline.run publishes today
    g["all_torvik"] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    return g


def resolve(spec, g: dict[str, list[str]], table_cols) -> list[str]:
    """spec: list of group names or column names; '-name' removes a group; None -> pipeline FEATURES."""
    if spec is None:
        return [f for f in FEATURES if f in table_cols]
    feats: list[str] = []
    for s in spec:
        if s.startswith("-"):
            feats = [f for f in feats if f not in g[s[1:]]]
        else:
            feats += g.get(s, [s])
    return [f for f in dict.fromkeys(feats) if f in table_cols]


def validate(table: pd.DataFrame, verbose=True) -> dict[str, list[str]]:
    g = groups(table)
    missing = {k: [c for c in v if c not in table.columns] for k, v in g.items()}
    missing = {k: v for k, v in missing.items() if v}
    assert not missing, f"contract columns missing from draft_table: {missing}"
    if verbose:
        m = table[table.modelled]
        rows = []
        for k, v in g.items():
            if k in ("pipeline", "all_torvik"):
                continue
            cov = lambda d: float(d[v].notna().mean().mean()) if len(d) else float("nan")
            rows.append({"group": k, "n_cols": len(v), "college": cov(m[m.source == "college"]), "intl": cov(m[m.source == "intl"]),
                         "test_2018_25": cov(m[m.draft_year.between(2018, 2025)])})
        print(pd.DataFrame(rows).round(2).to_string(index=False))
    return g


def assert_covered(configs: list[dict], g: dict[str, list[str]]):
    """Every contract group and model must appear in at least one config of the grid."""
    used_groups, used_models = set(), set()
    for c in configs:
        for m in c.get("stack") or [c]:
            used_models.add(m.get("model", c.get("model", "tabfm")))
            for s in m.get("features", c.get("features")) or ["pipeline"]:
                used_groups.add(s.lstrip("-"))
    if "all_torvik" in used_groups or "pipeline" in used_groups:  # both contain every Torvik group
        used_groups |= set(TORVIK)
    groups_needed = {k for k in g if k not in ("pipeline", "all_torvik")}
    missing_g = groups_needed - used_groups
    # TabPFN checkpoints are license-gated and therefore optional unless the machine is authenticated.
    missing_m = {"tabfm", "tabfm_cls", "exaone_cls", "xgb", "xgbrank", "catboost", "catrank",
                 "extratrees", "tabicl", "tabldm"} - used_models
    assert not missing_g, f"grid never uses feature groups: {sorted(missing_g)}"
    assert not missing_m, f"grid never uses models: {sorted(missing_m)}"
    print(f"contract check: {len(used_groups)} groups, models {sorted(used_models)} -- all covered")
