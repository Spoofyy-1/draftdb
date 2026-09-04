# One command from a clean machine to a running site:  make all
# Needs: python3.11+, uv (https://docs.astral.sh/uv), bun (https://bun.sh). GPU optional; runs on CPU.

PY := .venv/bin/python

.PHONY: all setup data model tables run web clean

all: setup data model tables run web

setup:            ## python venv + deps, web deps
	uv venv .venv --python 3.12 --allow-existing
	uv pip install --python $(PY) -e .
	cd web && bun install

data:             ## download Torvik, Basketball-Reference and RAPTOR sources, parse to parquet (~15 min, bbref is rate-limited)
	$(PY) -m nbadraft.data

model:            ## download the TabFM checkpoint (~6 GB)
	$(PY) -m nbadraft.download_model

tables:           ## WAR per season, peak-WAR target, draft table with college features
	$(PY) -m nbadraft.war
	$(PY) -m nbadraft.dataset

run:              ## score every draft; writes outputs/runs/<id>/run.json and rebuilds outputs/results.sqlite
	$(PY) -m nbadraft.run --models tabfm,lgbm,ridge --tag peak_war

web:              ## serve the site on http://localhost:3010 (must run under bun: the app uses bun:sqlite)
	cd web && bun run dev

clean:            ## remove everything the pipeline generated (keeps raw downloads)
	rm -rf data/processed outputs
