"""Paths, years, split definitions and WAR constants. Single source of truth."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
MODEL_DIR = ROOT / "models" / "tabfm-1.0.0-pytorch"

# The 24 drafts we redo. Torvik college data starts with the 2007-08 season, so the 2003-2007 classes have no Torvik
# features; their college features come from hoopR/ESPN season box scores (h_*) and the AyushBatra college line (a_*).
DRAFT_YEARS = range(2003, 2027)
TORVIK_YEARS = range(2008, 2027)
# bbref season label = calendar year the season ends (2016 == 2015-16). The 2003 class's first season is 2004.
NBA_SEASONS = range(2004, 2027)
LAST_SEASON = 2026

# Target. Two definitions are implemented; TARGET_KIND picks the one every model, rule and page is judged on.
#   "war5": WAR summed over the player's FIRST TARGET_SEASONS NBA seasons (all of them if he has played fewer) --
#           what the pick produced early in his career. Switched to this on 2026-09-05 at the user's request;
#           the window was 3 seasons until then.
#   "peak": mean WAR over his TARGET_SEASONS BEST seasons known at the time of judging -- how good he became;
#           does not punish a career cut short by injury.
# Either way a class observed for one season is judged on that season. A drafted player who never plays has 0 WAR,
# matching the frozen public benchmark.
TARGET_KIND = "war5"
TARGET_SEASONS = 5
TARGET = {"war5": "war5", "peak": "peak_war"}[TARGET_KIND]  # column name of the score everywhere downstream
NEVER_PLAYED_WAR = {"war5": 0.0, "peak": 0.0}[TARGET_KIND]
PEAK_SEASONS = TARGET_SEASONS  # legacy name
# Context labels are rewritten within each draft class before the model sees them, so a class observed for one rookie
# season and one observed for ten sit on one scale (the metric is a within-class rank). Percentile rank won on the full
# feature set in the tournament (2013-2017: rank 0.532 vs z-score 0.504 vs raw 0.47).
LABEL_TRANSFORM = "rank"
FIRST_LABELED_DRAFT = 2003  # first class with pre-draft features (hoopR / AyushBatra) and an NBA outcome
LAST_LABELED_DRAFT = LAST_SEASON - 1  # 2025: last draft class that has played at least one season

# Two kinds of draft class and nothing else: context (what a model may learn from) and holdout (what it is scored on).
# Holdout is walk-forward: when scoring holdout class Y the model sees only classes before Y, and each label contains
# only NBA seasons completed by draft night Y.
CONTEXT_YEARS = tuple(range(2003, 2019))
HOLDOUT_YEARS = tuple(range(2019, 2026))

# FiveThirtyEight regular-season formula:
# WAR = (rating + replacement) * minutes * ((league pace + individual pace impact) / league pace) * multiplier.
WAR_REPLACEMENT = 2.75  # replacement level, points per 100 possessions below average
WAR_PER_MIN = 0.0005102  # official regular-season WAR multiplier
RAPTOR_LAST_SEASON = 2022  # 538 stopped publishing RAPTOR after 2021-22

DRAFT_DAY = "06-25"  # approximate draft date, used for age-at-draft
