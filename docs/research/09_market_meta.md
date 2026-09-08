# 09 — Market and meta sources (sub-scan of 08), 2026-09-08

Test applied: does the feature exist in BOTH the 2000-2018 train window and the 2019-2025 test window, datable pre-draft?

| Source | Verdict |
|---|---|
| Wikipedia pre-draft revisions of "YYYY NBA draft" pages | HIGH: green-room invitees (NBA's own pre-draft signal, dated in prose), combine invitees, early-entrant declarations, pick ownership/provenance; usable 2006/2007-2025 (pages for 2000-2004 created after the fact) |
| Agents / representation (RealGM, HoopsHype, NBPA) | LOW: client lists are NBA-roster-derived (post-draft by construction), window mismatch (HoopsHype 2007-15 only; RealGM sub-pages unarchived), selection bias; robots.txt of both sites bans Anthropic agents |
| Spotrac | DEAD: 403 to non-browser clients; the non-redundant items (two-way, Exhibit 10, rights trades) are post-draft anyway |
| Sportsbook draft props (VegasInsider/BetMGM) | LOW: 2020+ only, ~15-20 players per class, hole in 2023; usable only as a held-out sanity check of the mock-consensus feature |
| Wikidata / DBpedia | LOW: cannot enumerate draft classes (P1836 on 537 players); truthy statements carry the whole career (leakage) |
| Google Books Ngrams | DEAD: indexed by publication year, i.e. the label with extra steps |
| GDELT DOC 2.0 | LOW: starts 2017-01-01, redundant with Google Trends, aggressive rate limit |

Leakage note for the draft-page parser: cut strictly by revision timestamp with rvdir=older from the draft date; pre-draft revisions have empty player cells and a placeholder "Draft-day trades" section, so a correct cut is clean by construction. Markup changed from <small>(from X)</small> to {{small|(from X)}}.
