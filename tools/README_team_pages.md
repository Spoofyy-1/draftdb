# Team pages generator

`tools/build_team_pages.py` turns `data/team_cards.json` into the 30 team pages
and the team index of the static site. Python 3, standard library only, no build
step — the files it writes are the files Vercel serves.

## Rebuild the real pages

```sh
cd /Users/kennakao/nba/site
python3 tools/build_team_pages.py
```

That reads `data/team_cards.json` and rewrites `teams/<slug>.html` for every team
in the dataset plus `teams/index.html`. It prints the file list and a validation
summary, and exits non-zero if any check fails.

## Preview against the mock fixture

`data/team_cards.mock.json` is a two-team fixture with invented players and
invented numbers (it exercises escaping, a year with no pick, an unranked player,
a slot where the model picked the same player, and a slot with no board entry):

```sh
cd /Users/kennakao/nba/site
python3 tools/build_team_pages.py \
    --data data/team_cards.mock.json \
    --out  tools/_preview \
    --asset-prefix ../../
```

`--asset-prefix` is the path from a written page back to the site root, so the
preview pages still load `css/base.css`, `css/team.css` and `js/base.js`.
`tools/_preview/` is git-ignored so a preview never ships. View it with the local
server: `python3 serve.py 5174` from the repo root, then
<http://127.0.0.1:5174/tools/_preview/index.html>.

## What it writes

Per team page, in order: hero (verdict sentence + three stat tiles + year jump
strip), one section per draft year 2019–2025 with a comparison card per pick,
"Read this honestly", "How DraftDB works", footer with the generation date.

Each page keeps its own accent colour. The accent is re-read from the page being
replaced, so a hand-tuned colour survives a rebuild; `TEAMS` in the script is only
the fallback table.

Styles live in `css/team.css`, in the marked block at the bottom
("NEW (2026-09) — draft-review team pages"). There is no separate team-cards.css.

## Validation

Every run checks each written page for: 7 draft sections and 7 year anchors,
no `None` / `nan` / `null` / `undefined` / `{{ }}` / `%`-placeholder tokens, no
unescaped `&` inside `<main>`, balanced tags, and a well-formed document. It also
audits the dataset and reports teams missing from it, picks with no model board,
and any pick whose `verdict` disagrees with the sign of `delta_war` (reported,
never silently corrected).
