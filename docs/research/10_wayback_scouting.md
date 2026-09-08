# 10 — Wayback / archived scouting-media sources (sub-scan of 08), 2026-09-08. All items verified by direct CDX/timemap probes.

| Rank | Source | Years | Coverage of draftees | Hours | EV |
|---|---|---|---|---|---|
| 1 | DraftExpress measurements DB: ONE archived page holds the whole DB: https://web.archive.org/web/20161112235654/http://www.draftexpress.com/nba-pre-draft-measurements/ (11.87 MB, 3,683 players, 1987-2016, non-combine events incl. Hoop Summit, Nike camps, Portsmouth, EuroCamp, USA Basketball, college listings). Drop the `Drafted` and `Rank` columns (post-hoc). | 1987-2016 | 74.5% of 2000-16 draftees (exact-name lower bound) | 3-5 | HIGH |
| 2 | NBADraft.net grades (2001-2025 profiles; 12 numeric 1-10 grades + Overall + comparison; 2008-15 Drupal layout has labels in a position banner image, scores in text) + big board https://www.nbadraft.net/ranking/bigboard/ (873 mementos 2008-2025, 100 entries, Change column) | 2001-2025 | 88-98% (2006+) | 12-20 | HIGH |
| 3 | DraftExpress profiles (DOB, RSCI, agent, hometown, multi-source measurements with Year/Source) — safe fields only; 2017 shutdown crawl captures carry post-draft data | 2000-2017 | >=17,471 player ids archived | 10-15 | HIGH |
| 4 | DraftExpress Top-100 board (355 mementos 2008-2026; on-page "last updated" from ~2012; post-draft captures preserve the final pre-draft board) + DE mock (2007-2017) -> fit gap = mock pick - board rank, momentum, volatility | 2008-2017 | top 100 / top 60 | 18-27 | HIGH (2012+) |
| 5 | The Stepien boards (publication date in URL; several named analysts per class -> dispersion) | 2018-2022 | top 30-100 | 6-10 | MEDIUM |
| 6 | ESPN best-available (top 25 only per capture; Givony mock is paywalled and not in the archive) | 2017-2026 | top 25 | 4-6 | MEDIUM |
| 7 | Tankathon big board (robots.txt bans ClaudeBot/Anthropic-ai explicitly; only rank + TIER are non-redundant) | 2018-2026 | top ~75 | 4-6 | MEDIUM (ToS) |
| 8 | The Ringer guide (2020 capture parses; 2021 is a JS shell; historical guides return 500 live) | 2019-2025 patchy | top 30-50 | 8-12 | LOW-MED |
| 9 | No Ceilings versioned boards (2022+; top 30; some NC+ paywalled) | 2022-2026 | top 30 | 3-5 | LOW-MED |
| 10 | Reddit r/NBA_Draft (robots Disallow: /; archive essentially one 2023 crawl) | — | — | — | DO NOT PURSUE |

Cross-cutting: accept a capture iff capture_ts < draft_date OR on-page last_updated < draft_date (boards freeze after the draft and reset in late August). Scouting prose is copyrighted: rule-based features only, never store/republish text. Situational (Synergy-style) DE stats: no working archived endpoint found.
