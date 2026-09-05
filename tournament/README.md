# tournament

The sweep. Run it, read the report, improve a builder or a rule, run it again.

```bash
.venv/bin/python -m tournament.sweep --tag s1 --grid full      # audit -> layer 1 on all GPUs -> layer 2 -> report
```

The sweep selects on context years (2013-2018, inside the 2003-2018 context block). The holdout years 2019-2025 are
never touched by the sweep; confirm a short, pre-registered list once, and every look is logged:

```bash
.venv/bin/python -m tournament.layer2 --layer1 outputs/layer_1/s1_gpu*.parquet --holdout --rules "blend all"
```

```
contract.py     every feature group, model and label framing; validate() + assert_covered() guarantee the grid uses all of them
audit.py        who is modelled (college / international / nobody), coverage per group, leakage tests
layer1.py       raw AI predictions per config (point + bin distribution) -> outputs/layer_1/<tag>.parquet
layer2.py       pre-draft-only rules on layer-1 outputs (blends, upside/floor, consensus); holdout looks -> outputs/layer_2/holdout_looks.jsonl
sweep.py        the entrypoint: grid over the contract, sharded across GPUs, then layer 2 and a report in outputs/sweeps/<tag>/
builders/       feature builders -> data/external/feat_<name>.parquet, merged into the table by infra.dataset
```

Leakage rule for every feature: knowable one minute before the draft starts. No NBA data, no draft results of the
same year, no population statistic fitted on later seasons.

When a configuration wins on the context years and survives its holdout look, it is written into `infra/` + `pipeline/` (features in
`dataset.py`/`external.py`, model in `models.py`, label in `config.py`) and republished with `make run`.
