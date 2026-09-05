"""Integration test: is the data we claim to have actually in the table the models read? Fails loudly.

Checks, on data/processed/draft_table.parquet as it is on disk right now:
  pool        every draft year has its real draftees, every row has an actual pick; no undrafted player in the pool
  population  international and G League draftees are modelled, by name (Jokic, Doncic, Wembanyama, Sengun, Scoot
              Henderson via G League, Giddey via NBL ...), with a real pro line; FIBA youth reaches US college players too
  sources     every contract feature group is present and covers the population it should (thresholds per group)
  named       stars have the features we say they have (trajectory, Torvik context, hoopR, combine ...)
  leakage     no target/market/consensus column in the published FEATURES or LEAN_FEATURES

Usage: python -m tournament.integration_test
"""

import sys

import pandas as pd

from infra import config as C
from infra.dataset import FEATURES, LEAN_FEATURES, MARKET_FEATURE
from tournament import contract as F

RESULTS: list[tuple[str, bool, str]] = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))


def main():
    t = pd.read_parquet(C.PROC / "draft_table.parquet")
    m = t[t.modelled]
    G = F.groups(t)

    # -------------------------------------------------------------- pool
    per_year = t.groupby("draft_year").size()
    check("pool: every draft year 2007-2026 present", set(range(2007, 2027)) <= set(per_year.index), f"years {per_year.index.min()}-{per_year.index.max()}")
    check("pool: 58-60 draftees per year", per_year.between(58, 60).all(), per_year[~per_year.between(58, 60)].to_dict() or "ok")
    check("pool: every row has an actual pick", t[MARKET_FEATURE].notna().all())
    check("pool: no duplicate (bbref_id, draft_year)", not t.duplicated(["bbref_id", "draft_year"]).any())
    check("pool: only drafted players anywhere in the table (no undrafted rows)", "undrafted" not in t.columns or not t.undrafted.fillna(False).any())

    # -------------------------------------------------------------- population
    since = t[t.draft_year >= 2008]
    check("population: >= 90% of 2008+ draftees modelled", since.modelled.mean() >= 0.90, f"{since.modelled.mean():.1%}")
    intl = m[m.source == "intl"]
    check("population: >= 150 international / G League rows modelled", len(intl) >= 150, f"{len(intl)}")
    named = {"Nikola Jokić": 2014, "Luka Dončić": 2018, "Victor Wembanyama": 2023, "Alperen Şengün": 2021, "Rudy Gobert": 2013,
             "Kristaps Porziņģis": 2015, "Josh Giddey": 2021, "Scoot Henderson": 2023, "Jalen Green": 2021, "Jonathan Kuminga": 2021,
             "LaMelo Ball": 2020, "Zaccharie Risacher": 2024, "Alex Sarr": 2024}
    for p, y in named.items():
        r = t[(t.player == p) & (t.draft_year == y)]
        ok = len(r) == 1 and bool(r.modelled.iloc[0]) and r.i_gp.notna().iloc[0]
        check(f"  intl/GLG modelled with a pro line: {p} {y}", ok, f"source={r.source.iloc[0] if len(r) else '?'} league_level={r.i_league_level.iloc[0] if len(r) else '?'} gp={r.i_gp.iloc[0] if len(r) else '?'}")
    glg = m[(m.source == "intl") & (m.i_league_level.notna())]
    check("population: G League rows present", (t.i_gp.notna() & (t.source == "intl")).sum() >= 15, f"{len(glg)} intl rows with league level")
    fiba_college = (m[m.source == "college"].i_fiba_youth_events.fillna(0) > 0).sum()
    check("population: FIBA youth history reaches US college players", fiba_college >= 150, f"{fiba_college} college players with FIBA youth events")

    # -------------------------------------------------------------- sources: coverage thresholds
    # (group, population, minimum share of non-null cells among modelled rows of that population, 2010+)
    m10 = m[m.draft_year >= 2010]
    college, international = m10[m10.source == "college"], m10[m10.source == "intl"]
    cov = lambda d, cols: float(d[cols].notna().mean().mean()) if len(d) else 0.0
    thresholds = [("core", college, 0.95), ("rate", college, 0.95), ("adv", college, 0.95), ("traj", college, 0.55), ("phys", college, 0.60),
                  ("a_box", college, 0.60), ("j_box", college, 0.50), ("j_shot", college, 0.50), ("hoopr", college, 0.80), ("combine", college, 0.50),
                  ("marchmadness", college, 0.40), ("tctx", college, 0.70), ("intl_pro", international, 0.80), ("intl_z", international, 0.50),
                  ("intl_fiba", international, 0.50)]
    for g, pop, thr in thresholds:
        if g not in G:
            check(f"source {g}: present in contract", False, "group missing from table")
            continue
        c = cov(pop, G[g])
        check(f"source {g}: coverage >= {thr:.0%} on its population", c >= thr, f"{c:.0%} over {len(G[g])} cols")
    for g in ("game", "eurocamp", "bwb", "academy", "transfers", "shrunk"):
        check(f"source {g}: present", g in G, "ok" if g in G else "not materialised yet")
    mock_cov = t.loc[t.draft_year >= 2008, "mock_rank_consensus"].notna().mean() if "mock_rank_consensus" in t.columns else 0.0
    check("consensus: pre-draft mock ranks present for the '+consensus' variant (kept out of pure FEATURES)", mock_cov >= 0.95, f"{mock_cov:.0%} of 2008+ draftees")

    # -------------------------------------------------------------- named college stars have the features
    stars = {"Zion Williamson": 2019, "Anthony Edwards": 2020, "Cade Cunningham": 2021, "Paolo Banchero": 2022, "Stephen Curry": 2009, "Kawhi Leonard": 2011}
    for p, y in stars.items():
        r = t[(t.player == p) & (t.draft_year == y)]
        have = {g: r[G[g]].notna().mean(axis=1).iloc[0] if len(r) and g in G else 0 for g in ("core", "traj", "tctx", "hoopr", "phys")}
        check(f"  college star has core/traj/tctx/hoopr/phys: {p} {y}", len(r) == 1 and have["core"] > 0.9 and have["tctx"] > 0.5 and have["hoopr"] > 0.5,
              " ".join(f"{k}={v:.0%}" for k, v in have.items()))

    # -------------------------------------------------------------- leakage
    banned = {MARKET_FEATURE, "peak_war", "war5", "war", "seasons_played", "labelled", "modelled"}
    check("leakage: no target/market column in FEATURES", not (banned & set(FEATURES)), sorted(banned & set(FEATURES)) or "ok")
    check("leakage: no target/market column in LEAN_FEATURES", not (banned & set(LEAN_FEATURES)), sorted(banned & set(LEAN_FEATURES)) or "ok")
    check("leakage: no consensus (mock_*) column in published FEATURES", not any(f.startswith("mock_") for f in FEATURES))
    check("leakage: FEATURES all present in table", all(f in t.columns for f in FEATURES), f"{len(FEATURES)} features")

    # -------------------------------------------------------------- report
    width = max(len(n) for n, _, _ in RESULTS)
    fails = 0
    for n, ok, d in RESULTS:
        fails += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {n:<{width}}  {d}")
    print(f"\n{len(RESULTS) - fails} passed, {fails} failed  |  table {t.shape}, modelled {t.modelled.sum()} ({(t.source == 'college').sum()} college, {(t.source == 'intl').sum()} intl/GLG)")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
