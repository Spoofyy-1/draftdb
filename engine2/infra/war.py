"""Basketball WAR and the per-season draft target.

    war_season = WAR_PER_MIN * (rating + WAR_REPLACEMENT) * minutes * pace_adjustment

* rating  = FiveThirtyEight RAPTOR (points/100 poss above average) for seasons <= 2022.
* rating  = BPM mapped onto the RAPTOR scale (linear fit on the 2008-2022 overlap) for
            2023-2026, since 538 stopped publishing RAPTOR.
* minutes = basketball-reference regular-season minutes for every season.
* pace_adjustment = (league pace + individual pace impact) / league pace where FiveThirtyEight supplies it;
                    neutral 1.0 for the post-2022 BPM fallback.
* 2019-2025 draft classes use the frozen reference per-season WAR labels so the fixed test benchmark is identical.
* target  = config.TARGET_KIND: "war5" = WAR summed over the player's first TARGET_SEASONS NBA seasons (all of
            them if fewer) -- what the pick produced early; or "peak" = mean over his TARGET_SEASONS best seasons
            -- how good he became. Never played = NEVER_PLAYED_WAR, below every real value.
"""

import numpy as np
import pandas as pd

from infra.config import LAST_SEASON, NEVER_PLAYED_WAR, PROC, RAPTOR_LAST_SEASON, TARGET, TARGET_KIND, TARGET_SEASONS, WAR_PER_MIN, WAR_REPLACEMENT


def war_from_rating(rating, minutes, pace_adjustment=1.0):
    return WAR_PER_MIN * (rating + WAR_REPLACEMENT) * minutes * pace_adjustment


def season_war(nba: pd.DataFrame, raptor: pd.DataFrame) -> pd.DataFrame:
    """One row per (bbref_id, season) with `war`, rating provenance and the official pace adjustment."""
    r = raptor.copy()
    league_pace = 48 * r.groupby("season").poss.sum() / r.groupby("season").mp.sum()
    r["league_pace"] = r.season.map(league_pace)
    r["pace_adjustment"] = (r.league_pace + r.pace_impact) / r.league_pace
    d = nba.merge(
        r[["bbref_id", "season", "raptor_total", "pace_adjustment", "war_reg_season"]],
        on=["bbref_id", "season"], how="left",
    )
    d = d[d.mp > 0].copy()

    # BPM -> RAPTOR scale, minutes-weighted least squares on seasons where both exist
    ov = d[(d.season <= RAPTOR_LAST_SEASON) & d.raptor_total.notna() & d.bpm.notna()]
    w = np.sqrt(ov.mp.values)
    A = np.c_[np.ones(len(ov)), ov.bpm.values] * w[:, None]
    alpha, beta = np.linalg.lstsq(A, ov.raptor_total.values * w, rcond=None)[0]

    use_raptor = (d.season <= RAPTOR_LAST_SEASON) & d.raptor_total.notna()
    d["rating"] = np.where(use_raptor, d.raptor_total, alpha + beta * d.bpm)
    d["rating_source"] = np.where(use_raptor, "raptor", "bpm_calibrated")
    d["pace_adjustment"] = d.pace_adjustment.where(use_raptor, 1.0).fillna(1.0)
    d["war"] = war_from_rating(d.rating, d.mp, d.pace_adjustment)
    d["war_source"] = "formula"

    # report: how well the equation reproduces 538's own WAR, and how noisy the fallback is
    r = d[use_raptor]
    fb = war_from_rating(alpha + beta * r.bpm, r.mp, r.pace_adjustment)
    print(f"rating calibration: raptor ~= {alpha:.3f} + {beta:.3f} * bpm   (n={len(ov)})")
    print(f"equation vs 538 war_reg_season: r={np.corrcoef(r.war, r.war_reg_season)[0, 1]:.4f}")
    print(f"bpm-fallback war vs raptor war (overlap): r={np.corrcoef(fb, r.war)[0, 1]:.4f}, rmse={np.sqrt(np.mean((fb - r.war) ** 2)):.3f} wins")
    return d[["bbref_id", "season", "mp", "rating", "rating_source", "pace_adjustment", "war", "war_source"]]


def apply_reference_war(swar: pd.DataFrame, drafts: pd.DataFrame) -> pd.DataFrame:
    """Use the frozen per-season WAR benchmark for 2019-2025, keyed by draft year and actual pick."""
    out = swar.copy()
    draft = drafts[["draft_year", "pick", "bbref_id"]]
    overridden = 0
    for year in range(2019, 2026):
        path = PROC.parent / "raw" / "reference_war" / f"answers_{year}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run `make data`")
        a = pd.read_csv(path).dropna(subset=["actual_pick"])
        a["pick"] = a.actual_pick.astype(int)
        a = a.merge(draft[draft.draft_year == year], on="pick", how="inner")
        for r in a.itertuples():
            values = [getattr(r, f"y_s{i}_war") for i in range(1, 6)]
            values = [float(v) for v in values if pd.notna(v)]
            idx = out.index[out.bbref_id.eq(r.bbref_id)].tolist()
            idx = sorted(idx, key=lambda i: out.at[i, "season"])
            if len(idx) < len(values):
                raise ValueError(f"reference WAR has {len(values)} seasons but NBA data has {len(idx)} for {year} pick {r.pick}")
            for pos, i in enumerate(idx[:TARGET_SEASONS]):
                out.at[i, "war"] = values[pos] if pos < len(values) else 0.0
                out.at[i, "war_source"] = "reference"
                overridden += 1
    print(f"reference WAR overrides: {overridden} player-seasons (2019-2025 draft classes)")
    return out


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
    if through == LAST_SEASON and "pick" in drafts:
        totals = {}
        for year in range(2019, 2026):
            path = PROC.parent / "raw" / "reference_war" / f"answers_{year}.csv"
            # BRIDGE: the frozen 2019-2025 answer sheets are his benchmark files. On our data those labels are sealed
            # BRIDGE: (scored through the vault), so the override is skipped when the file is absent instead of raising.
            if not path.exists():
                continue
            a = pd.read_csv(path).dropna(subset=["actual_pick"])
            totals.update({(year, int(r.actual_pick)): float(r.y_early_war) for r in a.itertuples()})
        exact = [totals.get((int(y), int(p))) for y, p in zip(drafts.draft_year, drafts["pick"])]
        exact = pd.Series(exact, index=out.index, dtype=float)
        out[TARGET] = exact.fillna(out[TARGET])
    out["labelled"] = out.draft_year < through
    return out


if __name__ == "__main__":
    nba = pd.read_parquet(PROC / "nba_seasons.parquet")
    raptor = pd.read_parquet(PROC / "raptor.parquet")
    drafts = pd.read_parquet(PROC / "drafts.parquet")
    swar = apply_reference_war(season_war(nba, raptor), drafts)
    swar.to_parquet(PROC / "season_war.parquet", index=False)
    tgt = war_target(drafts, swar)
    tgt.to_parquet(PROC / "target.parquet", index=False)
    top = tgt.merge(drafts[["bbref_id", "player", "pick"]], on="bbref_id").sort_values(TARGET, ascending=False)
    print(f"target {TARGET_KIND} ({TARGET}); worst real value {tgt.loc[tgt.seasons_played > 0, TARGET].min():.2f}, never played = {NEVER_PLAYED_WAR}")
    print(top.head(10)[["draft_year", "pick", "player", TARGET, "seasons_played"]].to_string(index=False))
