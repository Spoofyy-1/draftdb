# One command from a clean machine to a running site:  make all
# Needs: python3.11+, uv (https://docs.astral.sh/uv), bun (https://bun.sh), NVIDIA GPU(s) for the in-context models.

PY := .venv/bin/python

.PHONY: all setup data model features tables run sweep web winners clean

all: setup data model features tables run web

setup:            ## python venv + deps, web deps
	uv venv .venv --python 3.12 --allow-existing
	uv pip install --python $(PY) -e .
	cd web && bun install

data:             ## download Torvik, Basketball-Reference and RAPTOR sources, parse to parquet (~15 min, bbref is rate-limited)
	$(PY) -m infra.download_data

model:            ## download the TabFM checkpoints (~12 GB); EXAONE-Tabular fetches its own on first use
	$(PY) -m infra.download_model

features:         ## build every extra pre-draft source (international, combine, hoopR, Torvik context, ...) -> data/external/feat_*.parquet
	$(PY) -m infra.builders.materialize

tables:           ## WAR per season, configured WAR target, draft table with every feature
	$(PY) -m infra.war
	$(PY) -m infra.dataset

run:              ## score every draft with the selected model and all pre-draft signals
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $(PY) -m pipeline.run --models stack+momentum --protocols causal --tag published

run-all-models:   ## same, plus the stack's members and the ridge / LightGBM baselines and the pooled diagnostic (~10 min)
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $(PY) -m pipeline.run --models stack,stack+market,tabfm,tabfm_cls,ridge,lgbm --tag published_full

sweep:            ## the tournament: audit, every feature group x model on context years across all GPUs, layer-2 rules, report
	$(PY) -m tournament.sweep --tag $(shell date +%m%d_%H%M) --grid full

sweep-status:     ## progress + best so far for TAG=<tag> (also live in ./logs.txt)
	$(PY) -m tournament.sweep --tag $(TAG) --status

sweep-cut:        ## stop the running sweep TAG=<tag>, finish with what is done, export outputs/sweeps/<tag>/best.json
	$(PY) -m tournament.sweep --tag $(TAG) --cut

test:             ## read-only integration test: is the data we claim to have actually in the table? (0.5 s)
	$(PY) -m tournament.integration_test

winners:          ## leaderboard of every frozen system in winners/
	$(PY) -m tournament.winner list

web:              ## serve the site on http://localhost:3010 (must run under bun: the app uses bun:sqlite)
	cd web && bun run dev

clean:            ## remove everything the pipeline generated (keeps raw downloads)
	rm -rf data/processed outputs
