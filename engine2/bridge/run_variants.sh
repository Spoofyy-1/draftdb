#!/bin/bash
# Sequential engine2 walk-forward runs (selection window only): bash run_variants.sh v0 v1 v2 v3 v5
cd ~/nba/engine2 && source ../.venv/bin/activate && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
for V in "$@"; do
  cp data/variants/draft_table_$V.parquet data/processed/draft_table.parquet
  echo "[$(date -u +%H:%M)] variant $V walkforward" >> outputs/bridge/variants.log
  python -m bridge.run_winner --window walkforward --cutoff causal --device cuda --ctx-start-catboost 2008 --tag wf_$V > outputs/bridge/run_wf_$V.log 2>&1
  cp outputs/bridge/predictions_walkforward.csv outputs/bridge/predictions_walkforward_$V.csv
  grep "^\[context8\]" outputs/bridge/run_wf_$V.log | sed "s/^/  $V /" | cut -c1-150 >> outputs/bridge/variants.log
done
echo "[$(date -u +%H:%M)] variants done" >> outputs/bridge/variants.log
