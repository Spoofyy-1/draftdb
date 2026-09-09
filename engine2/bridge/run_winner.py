"""Run the frozen winner (`winners/catboost_tabicl_market_bio_scout_comp_m3`) on OUR draft table.

The winner is an equal rank-average of four layer-1 configs -- CatBoost (context from 2003, 800 trees, depth 5,
lr 0.03, two batches of 5 seeds) and TabICL v2 (context from 2010, outlier_threshold 2.0, two batches of 4 seeds),
both on the `disc85_gaussrank` label with the `match3` horizon. This script builds those configs, shells out to his
`tournament.layer1` unchanged, then applies his `tournament.layer2.rule_blend` and writes per-player scores.

    outputs/bridge/predictions_holdout.csv      pid, draft_year, cutoff, score, rank_in_class, <per-config score>
    outputs/bridge/predictions_walkforward.csv  same, for the context classes scored walk-forward
    outputs/bridge/run_manifest.json            argv, tags, year keys and the exact config dicts that produced them

(`--stub` redirects all three to outputs/bridge_stub/, so a plumbing run can never be mistaken for a real one.)

"Walk-forward" is what layer 1 already does: scoring class Y it learns only from classes < Y (`context_years(y,
"causal", ...)`), so running it on 2012-2018 gives an out-of-sample prediction for every context class.

Holdout classes carry no labels in our staging files, so his own scoring is skipped -- these CSVs are scored through
the sealed vault on the box.

Usage (full winner, one GPU):
    python -m bridge.run_winner --window both --device cuda:0

Usage (CPU smoke test on this Mac; --stub swaps in the numpy stand-ins under bridge/smoke_stubs):
    python -m bridge.run_winner --window holdout --years 2019,2020 --members catboost \
        --seeds 1 --batches 1 --n-estimators 50 --cutoff full --device cpu --stub
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from infra import config as C

ROOT = C.ROOT
OUT = C.OUT / "bridge"          # --stub redirects to outputs/bridge_stub/ so the two can never be confused

# Exactly the feature groups of the two winning members (handoff_colin.md section 3): the CatBoost member also gets
# the game-log challenge-response group, the TabICL member does not.
FEATURES_CATBOOST = ["all_torvik", "-cat", "traj", "phys", "intl_pro", "intl_fiba", "combine", "tctx", "intl_z",
                     "mock", "momentum", "response", "person", "scouting", "comp"]
FEATURES_TABICL = [g for g in FEATURES_CATBOOST if g != "response"]

WINDOWS = {                        # window -> (layer1 --years tag, default year list)
    "holdout": ("holdout", list(C.HOLDOUT_YEARS)),
    "walkforward": ("context8", list(range(2012, 2019))),
}
DEFAULT_SEEDS = {"catboost": 5, "tabicl": 4}
WINDOWS_ALL = {"holdout": list(C.HOLDOUT_YEARS), "context8": list(range(2011, 2019))}


def build_configs(members: list[str], seeds: int | None, batches: int, n_estimators: int, tabicl_estimators: int,
                  cutoff: str, ctx_start_catboost: int, ctx_start_tabicl: int, seed_offset: int = 0, extra_groups: list[str] | None = None) -> list[dict]:  # BRIDGE: seed offset + extra feature groups
    """The winner's four layer-1 configs. At the defaults the names are byte-identical to winner.json's
    `layer2_rule`, so the blend below is the archived rule."""
    cfgs = []
    for m in members:
        k = seeds or DEFAULT_SEEDS[m]
        for b in range(batches):
            start = b * k + seed_offset  # BRIDGE: shifted seed sets for confirmation
            if m == "catboost":
                name = f"C03 m3 +mo+rs+person+sc+comp2 x{k}" + (f" s{start}" if start else "")
                cfg = {"model": "catboost", "ctx_start": ctx_start_catboost, "features": FEATURES_CATBOOST + list(extra_groups or []),
                       "n_estimators": n_estimators}
            else:
                name = f"T10 o2 m3 +mo+person+sc+comp2 x{k}" + (f" s{start}" if start else "")
                cfg = {"model": "tabicl", "ctx_start": ctx_start_tabicl, "features": FEATURES_TABICL + list(extra_groups or []),
                       "n_estimators": tabicl_estimators, "model_options": {"outlier_threshold": 2.0}}
            if extra_groups: name += " +" + "+".join(extra_groups)
            if seed_offset: name += f" o{seed_offset}"
            cfg.update({"label": "disc85_gaussrank", "label_cutoff": cutoff, "label_horizon": "match3",
                        "seeds": k, "seed_start": start, "name": name})
            cfgs.append(cfg)
    return cfgs


def run_layer1(configs: list[dict], years_tag: str, years: list[int], tag: str, device: str, stub: bool,
               dry_run: bool) -> tuple[Path, str]:
    """Shell out to his entrypoint, unmodified. Returns the prediction parquet and the `years` key inside it."""
    full = years == WINDOWS_ALL[years_tag]
    cmd = [sys.executable, "-m", "tournament.layer1", "--device", device, "--years", years_tag,
           "--tag", tag, "--configs", json.dumps(configs)]
    if not full:
        cmd += ["--year-list", ",".join(str(y) for y in years)]
    years_key = years_tag if full else f"{years_tag}:{','.join(str(y) for y in years)}"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        ([str(ROOT / "bridge" / "smoke_stubs")] if stub else []) + [str(ROOT), env.get("PYTHONPATH", "")]).strip(os.pathsep)
    print("$ " + " ".join(f"'{c}'" if " " in c else c for c in cmd), flush=True)
    if dry_run:
        return C.OUT / "layer_1" / f"{tag}.parquet", years_key
    t0 = time.time()
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)
    print(f"  layer1 [{tag}] {time.time() - t0:.1f}s", flush=True)
    return C.OUT / "layer_1" / f"{tag}.parquet", years_key


def blend(parquet: Path, years_key: str, configs: list[dict]) -> pd.DataFrame:
    """His layer-2 `blend A | B | C | D`: equal rank-average of the stacked score of each config, within class."""
    from tournament.layer2 import rule_blend, wide

    l1 = pd.read_parquet(parquet)
    l1 = l1[l1.years.astype(str) == years_key]
    names = [c["name"] for c in configs]
    w = wide(l1, years_key)
    missing = [n for n in names if n not in w.columns]
    if missing:
        raise SystemExit(f"layer-1 output is missing configs {missing}; have {sorted(set(l1.config))}")
    score = rule_blend(w, names)
    out = pd.DataFrame({"pid": w.bbref_id.values, "draft_year": w.draft_year.astype(int).values,
                        "score": np.asarray(score, dtype=float)})
    out["rank_in_class"] = out.groupby("draft_year")["score"].rank(ascending=False, method="first").astype(int)
    for n in names:
        out[n] = w[n].values
    return out.sort_values(["draft_year", "rank_in_class"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", default="catboost,tabicl", help="comma list: catboost,tabicl")
    ap.add_argument("--seeds", type=int, help="seeds per batch (default: the winner's 5 CatBoost / 4 TabICL)")
    ap.add_argument("--batches", type=int, default=2, help="seed batches per member (winner: 2)")
    ap.add_argument("--n-estimators", type=int, default=800, help="CatBoost iterations (winner: 800)")
    ap.add_argument("--tabicl-estimators", type=int, default=8, help="TabICL ensemble members (winner: 8)")
    ap.add_argument("--cutoff", default="full,causal", help="label cutoff(s): full, causal, or both")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--window", default="both", choices=["holdout", "walkforward", "both"])
    ap.add_argument("--years", help="comma list restricting the window (e.g. 2019,2020 for a smoke test)")
    ap.add_argument("--ctx-start-catboost", type=int, default=2003)
    ap.add_argument("--ctx-start-tabicl", type=int, default=2010)
    ap.add_argument("--tag", default="bridge")
    ap.add_argument("--seed-offset", type=int, default=0, help="BRIDGE: shift every member's seed_start (confirmation runs)")
    ap.add_argument("--extra-groups", default="", help="BRIDGE: comma list of extra contract groups appended to both members' feature lists")
    ap.add_argument("--stub", action="store_true",
                    help="put bridge/smoke_stubs first on PYTHONPATH: numpy stand-ins for catboost / tabicl so the "
                         "whole layer-1 + layer-2 path can be exercised on a machine without them. NEVER on the box.")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    members = [m.strip() for m in a.members.split(",") if m.strip()]
    assert set(members) <= {"catboost", "tabicl"}, members
    cutoffs = [c.strip() for c in a.cutoff.split(",") if c.strip()]
    assert set(cutoffs) <= {"full", "causal"}, cutoffs
    windows = ["holdout", "walkforward"] if a.window == "both" else [a.window]
    out = OUT
    if a.stub:
        out = C.OUT / "bridge_stub"
        print("!! --stub: CatBoost / TabICL are numpy stand-ins. Plumbing test only, not the winner's numbers.")
        print(f"!! writing to {out} -- never mistake these for model output.\n")

    out.mkdir(parents=True, exist_ok=True)
    manifest = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "argv": sys.argv[1:], "stub": a.stub, "runs": []}
    for window in windows:
        years_tag, default_years = WINDOWS[window]
        years = [int(y) for y in a.years.split(",")] if a.years else default_years
        years = [y for y in years if y in WINDOWS_ALL[years_tag]]
        if not years:
            print(f"skip {window}: no requested year falls inside {years_tag}")
            continue
        frames = []
        for cutoff in cutoffs:
            cfgs = build_configs(members, a.seeds, a.batches, a.n_estimators, a.tabicl_estimators, cutoff,
                                 a.ctx_start_catboost, a.ctx_start_tabicl, seed_offset=a.seed_offset, extra_groups=[g for g in a.extra_groups.split(',') if g])
            tag = f"{a.tag}_{window}_{cutoff}"
            parquet, years_key = run_layer1(cfgs, years_tag, years, tag, a.device, a.stub, a.dry_run)
            manifest["runs"].append({"window": window, "cutoff": cutoff, "tag": tag, "years": years,
                                     "years_key": years_key, "configs": cfgs})
            if a.dry_run:
                continue
            frames.append(blend(parquet, years_key, cfgs).assign(cutoff=cutoff))
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        cols = ["pid", "draft_year", "cutoff", "score", "rank_in_class"] + \
               [c for c in df.columns if c not in ("pid", "draft_year", "cutoff", "score", "rank_in_class")]
        path = out / f"predictions_{window}.csv"
        df[cols].to_csv(path, index=False)
        print(f"\n{window}: {len(df)} rows -> {path}")
        print(df.groupby(["cutoff", "draft_year"]).score.agg(["size", "min", "mean", "max"]).round(3).to_string())
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=1))
    print("\nmanifest:", out / "run_manifest.json")


if __name__ == "__main__":
    main()
