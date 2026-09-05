# tournament

The sweep. Run it, read the report, improve a builder or a rule, run it again.

```bash
.venv/bin/python -m tournament.sweep --tag s1 --grid full      # audit -> layer 1 on all GPUs -> layer 2 -> report
```

Validation years are 2013-2017. The test years 2018-2025 are never touched by the sweep; confirm a short,
pre-registered list once, and every look is logged:

```bash
.venv/bin/python -m tournament.layer2 --layer1 outputs/layer_1/s1_gpu*.parquet --test --rules "blend all" "market w=0.6 on [blend all]"
```

```
contract.py     every feature group, model and label framing; validate() + assert_covered() guarantee the grid uses all of them
audit.py        who is modelled (college / international / nobody), coverage per group, leakage tests
layer1.py       raw AI predictions per config (point + bin distribution) -> outputs/layer_1/<tag>.parquet
layer2.py       deterministic rules on layer-1 outputs (blends, upside/floor, market); test looks -> outputs/layer_2/test_looks.jsonl
sweep.py        the entrypoint: grid over the contract, sharded across GPUs, then layer 2 and a report in outputs/sweeps/<tag>/
builders/       feature builders -> data/external/feat_<name>.parquet, merged into the table by infra.dataset
```

Leakage rule for every feature: knowable one minute before the draft starts. No NBA data, no draft results of the
same year, no population statistic fitted on later seasons.

When a configuration wins on validation and survives its test look, it is written into `infra/` + `validation/` (features in
`dataset.py`/`external.py`, model in `models.py`, label in `config.py`) and republished with `make run`.
