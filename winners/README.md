# winners

Frozen, scored systems. Everything else the tournament writes lives under `outputs/`, which is regenerated on every
sweep and deleted by `make clean`; `outputs/sweeps/<tag>/best.json` is also overwritten by the next sweep that reuses
the tag. A system worth keeping gets exported here, where nothing overwrites it.

One folder per winner:

```
winners/<name>/winner.json         config, rule, metrics, features, protocol, provenance
winners/<name>/predictions.parquet the per-player layer-1 predictions it was scored from
```

`winner.json` records the exact layer-1 config dict, the layer-2 rule (if any), per-year and mean Spearman against
the real draft order, the resolved feature list, the protocol constants in force (target, label transform, context
and holdout years), a fingerprint of the `draft_table.parquet` it was built on, and the command to re-run it. Export
re-scores from the prediction file rather than trusting a number passed on the command line, so the metrics in the
folder always match the predictions sitting next to them.

```bash
make winners                                   # leaderboard of everything archived
python -m tournament.winner show --name <name> # full record for one

python -m tournament.winner export --name <name> \
    --layer1 outputs/layer_1/<tag>.parquet \
    --config "<layer-1 config name>" \
    [--rule "<layer-2 rule>"] [--window holdout|context] [--note "..."]
```

A number is only clean if nothing about the system was chosen by looking at the holdout years. A blend whose weights
were fit on holdout (anything named `optuna w=...` from a `--holdout` sweep) is an optimistic number, not a result;
note that in `--note` if you archive one anyway.
