#!/bin/bash
# engine2: seed-shifted confirmation for v0 and v3 (walk-forward), then the pick's strict holdout (once) and expanding holdout.
cd ~/nba/engine2 && source ../.venv/bin/activate && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
for V in v0 v3; do
  cp data/variants/draft_table_$V.parquet data/processed/draft_table.parquet
  python -m bridge.run_winner --window walkforward --cutoff causal --device cuda --ctx-start-catboost 2008 --seed-offset 10 --tag wfo10_$V > outputs/bridge/run_wfo10_$V.log 2>&1
  cp outputs/bridge/predictions_walkforward.csv outputs/bridge/predictions_walkforward_${V}_o10.csv; echo "[$(date -u +%H:%M)] confirm $V done" >> outputs/bridge/confirm.log
done
cp data/variants/draft_table_v3.parquet data/processed/draft_table.parquet
python -m bridge.run_winner --window holdout --cutoff causal --device cuda --ctx-start-catboost 2008 --tag hold_v3 > outputs/bridge/run_hold_v3.log 2>&1
cp outputs/bridge/predictions_holdout.csv outputs/bridge/predictions_holdout_v3.csv; echo "[$(date -u +%H:%M)] holdout v3 done" >> outputs/bridge/confirm.log
python bridge/expanding_holdout.py data/variants/draft_table_v3.parquet v3 --ctx-start-catboost 2008 > outputs/bridge/run_exp_v3.log 2>&1; echo "[$(date -u +%H:%M)] expanding v3 done" >> outputs/bridge/confirm.log
echo "[$(date -u +%H:%M)] confirm chain done" >> outputs/bridge/confirm.log
