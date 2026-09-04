"""Paths, years, split definitions and WAR constants. Single source of truth."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
MODEL_DIR = ROOT / "models" / "tabfm-1.0.0-pytorch"

# The 20 drafts we redo. Torvik college data starts with the 2007-08 season, so
# 2007 draftees have no pre-draft features and 2007 is reported but not modelled.
DRAFT_YEARS = range(2007, 2027)
TORVIK_YEARS = range(2008, 2027)
# bbref season label = calendar year the season ends (2016 == 2015-16).
NBA_SEASONS = range(2008, 2027)
LAST_SEASON = 2026

# Target: peak WAR = mean WAR over a player's PEAK_SEASONS best NBA seasons known at the time of judging
# (all of them if he has played fewer). Judges how good a player became; does not punish a career cut
# short by injury, and puts a rookie's one season and a veteran's fifteen on one scale.
# A player who never played is not "0 wins" -- he is in a bucket of his own, below every real value
# (the worst real per-season WAR is around -3).
PEAK_SEASONS = 3
NEVER_PLAYED_WAR = -5.0
FIRST_LABELED_DRAFT = 2008
LAST_LABELED_DRAFT = LAST_SEASON - 1  # 2025: last draft class that has played at least one season

# Splits (draft years). Holdout is locked: touched once, at the end, via `holdout`.
VAL_YEARS = (2014, 2015, 2016, 2017, 2018)
HOLDOUT_YEARS = (2019, 2020, 2021)

# WAR equation: war_season = WAR_PER_MIN * (rating + WAR_REPLACEMENT) * minutes
# Constants fit to FiveThirtyEight's published RAPTOR WAR, seasons 2008-2022 (R^2 = 0.9999).
WAR_REPLACEMENT = 2.75  # replacement level, points per 100 possessions below average
WAR_PER_MIN = 0.000514  # wins per (point/100 poss) per minute (~1/1946)
RAPTOR_LAST_SEASON = 2022  # 538 stopped publishing RAPTOR after 2021-22

DRAFT_DAY = "06-25"  # approximate draft date, used for age-at-draft
