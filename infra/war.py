"""Basketball WAR and the per-season draft target.

    war_season = WAR_PER_MIN * (rating + WAR_REPLACEMENT) * minutes

* rating  = FiveThirtyEight RAPTOR (points/100 poss above average) for seasons <= 2022.
* rating  = BPM mapped onto the RAPTOR scale (linear fit on the 2008-2022 overlap) for
            2023-2026, since 538 stopped publishing RAPTOR.
* minutes = basketball-reference regular-season minutes for every season.
* target  = config.TARGET_KIND: "war5" = WAR summed over the player's first TARGET_SEASONS NBA seasons (all of
            them if fewer) -- what the pick produced early; or "peak" = mean over his TARGET_SEASONS best seasons
            -- how good he became. Never played = NEVER_PLAYED_WAR, below every real value.
"""

import numpy as np
import pandas as pd

from infra.config import LAST_SEASON, NEVER_PLAYED_WAR, PROC, RAPTOR_LAST_SEASON, TARGET, TARGET_KIND, TARGET_SEASONS, WAR_PER_MIN, WAR_REPLACEMENT


def war_from_rating(rating, minutes):
    return WAR_PER_MIN * (rating + WAR_REPLACEMENT) * minutes


def season_war(nba: pd.DataFrame, raptor: pd.DataFrame) -> pd.DataFrame:
    """One row per (bbref_id, season) with `war`, `rating`, `rating_source`."""
    d = nba.merge(raptor[["bbref_id", "season", "raptor_total", "war_reg_season"]], on=["bbref_id", "season"], how="left")
    d = d[d.mp > 0].copy()

    # BPM -> RAPTOR scale, minutes-weighted least squares on seasons where both exist
    ov = d[(d.season <= RAPTOR_LAST_SEASON) & d.raptor_total.notna() & d.bpm.notna()]
    w = np.sqrt(ov.mp.values)
    A = np.c_[np.ones(len(ov)), ov.bpm.values] * w[:, None]
    alpha, beta = np.linalg.lstsq(A, ov.raptor_total.values * w, rcond=None)[0]

    use_raptor = (d.season <= RAPTOR_LAST_SEASON) & d.raptor_total.notna()
    d["rating"] = np.where(use_raptor, d.raptor_total, alpha + beta * d.bpm)
    d["rating_source"] = np.where(use_raptor, "raptor", "bpm_calibrated")
    d["war"] = war_from_rating(d.rating, d.mp)

    # report: how well the equation reproduces 538's own WAR, and how noisy the fallback is
    r = d[use_raptor]
    fb = war_from_rating(alpha + beta * r.bpm, r.mp)
    print(f"rating calibration: raptor ~= {alpha:.3f} + {beta:.3f} * bpm   (n={len(ov)})")
    print(f"equation vs 538 war_reg_season: r={np.corrcoef(r.war, r.war_reg_season)[0, 1]:.4f}")
    print(f"bpm-fallback war vs raptor war (overlap): r={np.corrcoef(fb, r.war)[0, 1]:.4f}, rmse={np.sqrt(np.mean((fb - r.war) ** 2)):.3f} wins")
    return d[["bbref_id", "season", "mp", "rating", "rating_source", "war"]]


def war_target(drafts: pd.DataFrame, swar: pd.DataFrame, through: int = LAST_SEASON) -> pd.DataFrame:
    """The target (column TARGET) over seasons draft_year+1 .. `through`, one row per draftee in input order.

    `through` is the last season known to whoever is judging: today that is LAST_SEASON; for the labelled
    context a model sees on draft night Y it is Y, so nothing from the future leaks into training labels.
    `labelled` is False for classes that have not had a season to play yet.
    """
    m = drafts[["bbref_id", "draft_year"]].merge(swar[["bbref_id", "season", "war"]], on="bbref_id", how="left")
    m = m[(m.season > m.draft_year) & (m.season <= through)]
    if TARGET_KIND == "war5":  # first seasons played, summed
        sel = m.sort_values("season").groupby("bbref_id").head(TARGET_SEASONS)
        score = sel.groupby("bbref_id").war.sum()
    else:  # best seasons, averaged
        sel = m.sort_values("war", ascending=False).groupby("bbref_id").head(TARGET_SEASONS)
        score = sel.groupby("bbref_id").war.mean()
    agg = pd.DataFrame({TARGET: score, "seasons_played": m.groupby("bbref_id").season.count()})
    out = drafts[["bbref_id", "draft_year"]].merge(agg, left_on="bbref_id", right_index=True, how="left")
    out["seasons_played"] = out.seasons_played.fillna(0).astype(int)
    out[TARGET] = out[TARGET].fillna(NEVER_PLAYED_WAR)
    out["labelled"] = out.draft_year < through
    return out


if __name__ == "__main__":
    nba = pd.read_parquet(PROC / "nba_seasons.parquet")
    raptor = pd.read_parquet(PROC / "raptor.parquet")
    drafts = pd.read_parquet(PROC / "drafts.parquet")
    swar = season_war(nba, raptor)
    swar.to_parquet(PROC / "season_war.parquet", index=False)
    tgt = war_target(drafts, swar)
    tgt.to_parquet(PROC / "target.parquet", index=False)
    top = tgt.merge(drafts[["bbref_id", "player", "pick"]], on="bbref_id").sort_values(TARGET, ascending=False)
    print(f"target {TARGET_KIND} ({TARGET}); worst real value {tgt.loc[tgt.seasons_played > 0, TARGET].min():.2f}, never played = {NEVER_PLAYED_WAR}")
    print(top.head(10)[["draft_year", "pick", "player", TARGET, "seasons_played"]].to_string(index=False))
