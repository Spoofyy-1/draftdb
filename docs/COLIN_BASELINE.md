# Baseline: the `colin` branch

The `colin` branch of this repository is a separate pipeline (Torvik base, Wayback-dated mock consensus and momentum, NBADraft.net pre-draft grades and comparison-player value, basketball-reference biography, G League translation priors, `match3` labels, CatBoost from 2003 + TabICL from 2010, equal rank-average). It is kept separate on purpose and serves as the baseline for the main branch.

Independent cross-check (2026-09-08): its archived clean predictions, mapped to this repository's player ids (393 of 413 names) and scored through the sealed vault at this repository's horizons, strict protocol (trained on classes through 2018 only):

| class | Colin clean blend | real draft |
|---|---|---|
| 2019 | 0.458 | 0.408 |
| 2020 | 0.535 | 0.350 |
| 2021 | 0.580 | 0.408 |
| 2022 | 0.523 | 0.262 |
| 2023 | 0.522 | 0.111 |
| 2024 | 0.547 | 0.119 |
| 2025 | 0.402 | 0.176 |
| **mean** | **0.510** | 0.262 |

His own report: 0.504 clean, 0.516 for a ridge stacker whose covariate list was chosen after holdout looks (about a point optimistic by his account). His bound: training even on future classes with the original features gives 0.43-0.45, so the gain over that comes from new dated pre-draft information, not from modelling.

The main branch's verified model (same metric, same protocol, verified inputs only) is at 0.434 strict / 0.452 expanding. The gap is the dated information above, which the main branch does not yet ingest.
