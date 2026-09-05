"""Materialise every optional feature source as data/external/feat_<name>.parquet so the pipeline table can merge it.

Builders live in tournament/builders/ (college_sources.py, intl.py, torvik_context.py, game_features.py); the published
pipeline never imports them -- it only reads the parquets this script writes. Re-run after any builder changes.

Usage: python -m infra.builders.materialize
"""

import importlib
import signal

import pandas as pd

from infra.config import PROC
from infra.external import EXT

# A loader must read its cache and return in seconds. Anything slower is downloading (or still being built by a worker)
# and is skipped this round rather than stalling the table build.
TIMEOUT_S = 90


class _Timeout(Exception):
    pass


def _alarm(*_):
    raise _Timeout()

# name -> (module, function) returning a DataFrame keyed by key, draft_year; or a parquet already keyed that way
SOURCES = {
    "combine": ("infra.builders.college_sources", "load_combine"),
    "kaggle": ("infra.builders.college_sources", "load_kaggle_college"),
    "score": ("infra.builders.college_sources", "load_score"),
    "ianstack": ("infra.builders.college_sources", "load_ianstack"),
    "hoopr": ("infra.builders.college_sources", "load_hoopr"),
    "marchmadness": ("infra.builders.college_sources", "load_marchmadness"),
    "intl_zscore": ("infra.builders.intl_zscore", "load_intl_zscore"),
    "mock": ("infra.builders.mocks", "load_mocks"),
    "momentum": ("infra.builders.momentum", "load_momentum"),
    "shrunk": ("infra.builders.shrink", "load_shrunk"),
    "eurocamp": ("infra.builders.intl_extras", "load_eurocamp"),
    "bwb": ("infra.builders.intl_extras", "load_bwb"),
    "academy": ("infra.builders.intl_extras", "load_academy"),
    "transfers": ("infra.builders.transfers", "load_transfers"),
    "odds": ("infra.builders.odds", "load_odds"),
    "scouting": ("infra.builders.scouting", "load_scouting"),
    "coach": ("infra.builders.coaches", "load_coaches"),
    "development": ("infra.builders.development", "load_development"),
    "gleague": ("infra.builders.gleague", "load_gleague"),
    "population": ("infra.builders.population", "load_population"),
    "response": ("infra.builders.response", "load_response"),
    "bio": ("infra.builders.bio", "load_bio"),
    "hoopexplorer": ("infra.builders.hoopexplorer", "load_hoopexplorer"),
    "scouting_text": ("infra.builders.scouting_text", "load_scouting_text"),
    "torvik_context": PROC / "prospect_torvik_context.parquet",
    "game": PROC / "prospect_game_features.parquet",
}


def main():
    signal.signal(signal.SIGALRM, _alarm)
    for name, src in SOURCES.items():
        signal.alarm(TIMEOUT_S)
        try:
            df = pd.read_parquet(src) if not isinstance(src, tuple) else getattr(importlib.import_module(src[0]), src[1])()
        except _Timeout:
            print(f"{name:15s} skipped: loader exceeded {TIMEOUT_S}s (downloading inside load_*? move it to build())")
            continue
        except Exception as e:  # a builder that is not finished yet is simply skipped
            print(f"{name:15s} skipped: {type(e).__name__}: {str(e)[:80]}")
            continue
        finally:
            signal.alarm(0)
        if not {"key", "draft_year"}.issubset(df.columns):
            print(f"{name:15s} skipped: missing key or draft_year")
            continue
        df = df.drop_duplicates(["key", "draft_year"])
        df["draft_year"] = df.draft_year.astype(int)
        df.to_parquet(EXT / f"feat_{name}.parquet", index=False)
        print(f"{name:15s} {len(df):5d} rows  {df.shape[1] - 2:3d} cols")


if __name__ == "__main__":
    main()
