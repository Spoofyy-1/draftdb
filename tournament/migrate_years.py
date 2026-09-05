"""One-off: re-key the layer-1 cache and prediction files after val/test were renamed to context/holdout.

The cache filename is config_sig(cfg, years_tag, ...), so renaming the window tags orphaned every cached GPU
prediction. This recomputes each cached config's signature under the old tag and the new one and renames the file,
then rewrites the `years` column inside outputs/layer_1/*.parquet.

Usage: python -m tournament.migrate_years [--apply]
"""

import argparse
import json

import pandas as pd

from infra import config as C
from tournament import contract as F
from tournament.layer1 import config_sig

RENAME = {"val": "context", "val6": "context6", "val7": "context7", "val8": "context8", "test": "holdout"}


def new_tag(old: str) -> str | None:
    base, _, shard = old.partition(":")
    if base not in RENAME:
        return None
    return RENAME[base] + (f":{shard}" if shard else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    table = pd.read_parquet(C.PROC / "draft_table.parquet")
    G = F.validate(table, verbose=False)
    cache = C.OUT / "layer_1" / "cache"

    # Every window tag any prediction file was written under, so sharded tags are covered too.
    tags = set()
    for p in sorted((C.OUT / "layer_1").glob("*.parquet")):
        tags |= set(pd.read_parquet(p, columns=["years"]).years.astype(str).unique())
    tags = {t: new_tag(t) for t in tags}
    print("window tags found:", {k: v for k, v in tags.items()})

    configs = []
    for p in sorted((C.OUT / "sweeps").glob("*/configs_gpu*.json")):
        configs += json.load(open(p))
    seen, uniq = set(), []
    for c in configs:
        k = json.dumps(c, sort_keys=True)
        if k not in seen:
            seen.add(k)
            uniq.append(c)
    print(f"{len(uniq)} distinct configs from {len(list((C.OUT / 'sweeps').glob('*/configs_gpu*.json')))} shard files")

    moved, missing = 0, 0
    for cfg in uniq:
        for old, new in tags.items():
            if not new:
                continue
            src = cache / f"{config_sig(cfg, old, 0, table, G)}.parquet"
            dst = cache / f"{config_sig(cfg, new, 0, table, G)}.parquet"
            if not src.exists() or dst.exists():
                continue
            print(f"  {src.name} -> {dst.name}  [{old} -> {new}] {cfg['name']}")
            if a.apply:
                src.rename(dst)
            moved += 1
    print(f"{moved} cache files re-keyed" + ("" if a.apply else " (dry run)"))

    for p in sorted((C.OUT / "layer_1").glob("*.parquet")):
        df = pd.read_parquet(p)
        if "years" not in df.columns:
            continue
        mapped = df.years.astype(str).map(lambda t: new_tag(t) or t)
        if mapped.equals(df.years.astype(str)):
            continue
        print(f"  {p.name}: years {sorted(df.years.unique())} -> {sorted(mapped.unique())}")
        if a.apply:
            df.assign(years=mapped).to_parquet(p, index=False)

    orphans = sorted(q.name for q in cache.glob("*.parquet"))
    print(f"{len(orphans)} cache files present, {missing} configs without a cached prediction")


if __name__ == "__main__":
    main()
