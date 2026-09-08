# 07 — International & Non-College Pathway Signals

Ranked by expected value. Every item is pre-draft-knowable and free. Verified live where marked ✅ (probed 2026-09-08).

---

## 1. EuroLeague official API as the international data spine (2000–2025) — ✅ verified live

**The single biggest win in this slice.** `api-live.euroleague.net` is an open, unauthenticated, undocumented-but-stable API. Probed directly:

- `GET /v2/competitions` → **43 competitions**: Euroleague (`E`), Eurocup (`U`), Adriatic (`AL`), **U18 Tournament (`J`)**, plus legacy Spanish/Italian/Turkish/Greek/French/German/Israeli/Serbian/Lithuanian/Russian codes.
- `GET /v1/games?seasonCode=E2000&gameCode=1` → **full player box scores back to E2000** (E1999 → `NotFoundProblemDetails`, so 2000-01 is the floor). Each `<stat>` carries TimePlayedSeconds, all counting stats, StartFive.
- `GET /v1/results?seasonCode=U2002` → EuroCup back to **2002-03** (its first season).
- `GET /v1/players?playerCode=ADK&seasonCode=E2001` → **height, birthdate, country, position** — age without any scraping.
- `GET /v2/.../games/{n}/stats` → richer JSON with `person.birthDate`, height, weight, birthCountry, and per-registration `startDate`/`endDate`.

**Dating:** every game carries `cetdate` (e.g. `2001-10-25T20:30:00`). Filter `cetdate < draft_date` — exact, no inference.

**Coverage:** Euroleague 2000-01→2025-26 (26 seasons); EuroCup 2002-03→present. Essentially every European first-rounder and most second-rounders 2001–2025. Gap: players whose only senior minutes were domestic-only (Şengün's BSL year, Jokić's ABA year) — see §3.

**Hours:** ~0.5 s/call measured. ~18k games across E+U+J ≈ 2.5 h serial, <30 min concurrent; +6–8 h parsing, name/ID mapping and QA. **10–12 h total.**

**Features (new block, `el_*`):** per-40 and per-100 rates for pts/reb(off,def)/ast/stl/blk/TO/PF; TS%, 3PAr, FTr; `el_min_total`, `el_games`, `el_age_at_first_game`, `el_minutes_share`. Compute separately for `E` and `U`; do **not** pool.

**Leakage risk: low**, if you filter on `cetdate` and not `seasonCode` — an `E2020` season extends past June draft night, so season-level filtering leaks post-draft playoff games.

**ToS:** `api-live.euroleague.net/robots.txt` is empty. The main site sits behind a Vercel checkpoint (403), so I could **not** read the written terms — unverified; keep rates polite, re-check before redistribution.

Sources: probed endpoints; wrappers [giasemidis/euroleague_api](https://github.com/giasemidis/euroleague_api), [euroleaguer](https://cran.r-project.org/package=euroleaguer).

---

## 2. ANGT / NIJT U18 club tournament as a *separate* feature block — ✅ verified live

Your miss analysis found that **mixing youth-tournament lines into pro season lines hurt college players**. This fixes the cause rather than patching it.

`GET /v2/competitions/J/seasons` → **66 season-tournaments spanning 2002–2025**. `GET /v2/competitions/J/seasons/JT12/games/79/stats` → full player box scores with `birthDate` embedded (verified: Alberto Abalde, b. 1995-12-15, 2012-13 NIJT). `v1` also works for `JT12` with `cetdate`.

ANGT is **club** U18 competition, annual, against age-peers, run by the same academies that produce Euro draftees (past MVPs: Dončić, Šarić, Motiejūnas). That makes it a genuinely different signal from FIBA national-team youth events — 5-game samples with severe roster-quality confounds — and from senior pro lines.

**Features:** namespace `angt_*` — per-40 rates, `angt_min`, `angt_games`, `angt_age_at_tournament`, `angt_rel_age = age − mean age of tournament field`, plus `has_angt`. **Never** average ANGT into `el_*` or pro season lines. Give college players `has_angt = 0` and let the model learn a block-missing offset rather than imputing.

**Coverage:** 2002 onward for European academy players; ~0% for Americans, Australians, Africans without European clubs. Expect ~40–60% of drafted Europeans. **Hours: 4–6 h.** **Leakage risk: low** — tournaments end in spring; still filter on `cetdate`.

---

## 3. League-tier ladder (competition-difficulty scalar) — strongest published effect sizes

Layne Vashro published a **186-venue** difficulty table (standardized units, 0 = mean, 1 = 1 SD), built by projecting every player-season into NBA terms with a fixed-effects model, then iterating between-league calibration off shared players:

| Venue | Rating | | Venue | Rating |
|---|---|---|---|---|
| NBA | 3.26 | | EuroCup | 0.92 |
| Olympics | 2.56 | | ABA/Adriatic | 0.82 |
| FIBA World Cup | 1.85 | | Australian NBL | 0.80 |
| **Euroleague** | **1.61** | | Turkish BSL | 0.77 |
| Spanish ACB | 1.52 | | French LNB Pro A | 0.69 |
| NBA D-League | 1.21 | | German BBL | 0.65 |
| Italian Serie A | 1.12 | | Israeli BSL | 0.52 |
| Russian PBL | 1.13 | | Greek A1 | 0.43 |
| VTB United | 0.87 | | Spanish LEB Oro | 0.30 |

Youth: U20 Euro final round **0.00**, U19 World Cup **−0.23**, U18 Euro final round **−0.49**, U20 Euro B **−0.99**, Nike Hoop Summit **−0.93**. NCAA: Pac-10 0.96, ACC 0.76, SEC 0.68, Big East 0.56, Big 12 0.50, Big Ten 0.48; Patriot −0.55.

Two things fall out: (a) **EuroCup ≈ Pac-10 and Euroleague sits well above any NCAA conference** — raw per-minute Euroleague production is systematically under-credited if leagues are treated as interchangeable; (b) **U18/U19 events sit ~2 full SD below Euroleague**, quantifying exactly why pooling youth into pro lines corrupts the scale.

A 2026 replication by Ignacio Rissotto (No Ceilings), SRS-style over ~250k RealGM season lines, reaches the same shape and adds that **lower-tier D-I now ranks near U16 EuroBasket** after transfer-portal talent drain.

**Feature:** `league_tier_score` — published table as a **prior**, then refit as a random intercept on your 2000–2018 data — plus `tier × production` interactions. Don't hard-code the 2015-vintage numbers as final; BSL/VTB/ABA have moved since.

**Coverage:** all 2000–2025 draftees (everyone has a highest-level venue). **Hours: 2–3 h.** **Leakage risk: low** — league identity is known on draft night, but refit the random intercepts **inside** your CV folds or you leak outcomes into the tier scores.

Sources: [Vashro, "Measuring Level of Competition Around the World"](https://fansided.com/2015/11/06/deep-dives-measuring-level-of-competition-around-the-world/); [Rissotto, "Ranking the Top Basketball Leagues in the World"](https://www.noceilingsnba.com/p/ranking-the-top-basketball-leagues).

---

## 4. Recalibrate sample-size shrinkage for internationals — cheapest high-EV change

Vashro reports international projection scores built on **<100-minute samples correlate 0.60** with observed NBA performance versus **0.65 for >1,000-minute samples** — a gap of only 0.05, which he calls "pretty shocking."

Your v4 config applies sample-size shrinkage globally. If calibrated on college volume it is almost certainly **over-shrinking** teenagers with 200 Euroleague minutes toward the prior — precisely the population your miss analysis flags as the steal pool (Jokic: 25 ABA games; Sengun: one BSL season).

**Change:** make the shrinkage coefficient a fitted parameter *interacted with pathway* (`college` vs `intl_pro`), or add `min_bucket × intl` terms. **Hours: 2 h.** **Leakage risk: none** — it is a hyperparameter. A config change, not a data pull; run it before anything else here.

Source: [Vashro, "Projecting International Prospects, Part 3"](https://fansided.com/2014/08/21/projecting-international-prospects-part-3/).

---

## 5. Young-senior-minutes feature (senior pro minutes at age ≤ 19)

Pelton's system weights seasons **in the opposite direction for internationals vs. college players** — for college, older seasons predict better (recent 2×, prior 3×, two-prior 5×); for internationals the reverse, because they already face older opponents. That asymmetry is real signal a pooled treatment washes out.

Weak support: a 47-player, 4-draft descriptive study finds only 6 European draftees averaged <18 mpg pre-draft, and only 2 of those went in round one. Directional, not an effect size.

**Feature definition:** `senior_min_by_19` = total minutes in venues rated ≥ 0.4 on the tier scale, accrued before the player's 20th birthday; `age_at_first_senior_game`; `minutes_share_at_18`. All derivable from §1 with birthdates already in the API. **Hours: 1–2 h** on top of §1. **Coverage:** high for Europeans, zero for Americans (encode as block-missing). **Leakage risk: low.**

Sources: [Pelton, "How my NBA draft projections work"](https://www.espn.com/nba/story/_/id/16235135/explaining-kevin-pelton-nba-draft-projection-system); [Basketball Analytics Lab](https://basketballanalyticslab.substack.com/p/playing-time-matters-the-path-to).

---

## 6. Draft-and-stash — **use only as an outcome-side base rate, never a feature**

The Center Hub's audit of **54 stashed players, 2012–2024**: hit rate **24.1% (7/29) for 2013–2016**, collapsing to **3.4% (1/29) since 2015** and **0% (0/25) since 2016**. Picks 44–60: **2.5% (1/39)**; picks 48–60: **0% (0/33)**. One star in 13 years (Jokić, 1.8%).

**Leakage risk: HIGH — do not build a `was_stashed` feature.** Stashing is decided *after* selection and is downstream of the drafting team's own valuation; it leaks both the pick and the team's private information.

What it *is* good for: a sanity prior. The "second-round international steals" pool your miss analysis identified is largely a **2012–2015 phenomenon**. Verify your 2019–2025 gains aren't an artifact of a regime that has since closed, and consider an era interaction on `intl × round2`.

Source: [Examining the State of Draft and Stash Picks](https://thecenterhub.substack.com/p/examining-the-state-of-draft-and-stash-picks).

---

## 7. Alternative-pathway categorical (Ignite / OTE / NBL Next Stars)

Not a predictive feature — a **coverage-correctness** fix. Without it, OTE and Ignite lines get silently treated as college or as generic pro lines.

Counts are small: **Ignite 13 drafted in 4 years** (Green 2, Kuminga 7, Todd 31; Daniels 8, Beauchamp 24, Hardy 37; Henderson 3, Miller 33, Cissoko 44, King 47…), **NBL Next Stars 11 since 2020** (LaMelo 3, Hampton 24, Giddey 6, Dieng 11, Rupert 43, Sarr 2, Johnson 23, Klintman 37, Hukporti 58, Zikarsky, Toohey), OTE a handful (Thompson twins 4 and 5).

Ignite's context is extreme and needs a control: **6-31 and 2-32 records, 102.6 ORtg, −12.7 net rating**, worst in the G League. Those per-game lines are not comparable to NBL Next Stars lines on competitive teams (NBL = 0.80 on the Vashro scale).

**Feature:** one-hot `pathway ∈ {ncaa, intl_pro, ignite, ote, nbl_next_star, hs_prep, other}` + `team_net_rating` where available. **Coverage:** the 2000–2018 training set has ~zero of these (Ignite began 2020), so they can only help 2019–2025 scoring, and at n≈30 they will not move Spearman much. **Hours: 1–2 h** (hand-coded). **Leakage risk: low.**

Sources: [NBL Next Stars](https://www.nbl.com.au/news/next-stars-programs-proven-path-to-nba); [Why Ignite hasn't lived up to its promise](https://sports.yahoo.com/why-the-nbas-g-league-ignite-has-not-lived-up-to-its-promise-153128872.html).

---

## 8. Senior national-team debut before the draft — weak evidence, cheap to test

No basketball study with an effect size found. The best adjacent evidence is negative: in football, **youth international experience is a limited predictor of senior elite status** (U21 > U19 > U17), and across Olympic sports **junior performance explains only ~2.2% of reliable variance in senior performance**. FIBA youth events produce volume (U18 EuroBasket: **66 first-rounders in 15 years**; U19 World Cup: **~9 per tournament**) but that is selection, not prediction.

**Features:** `senior_nt_debut_before_draft`, `age_at_senior_nt_debut`. **Source:** FIBA event pages (robots.txt disallows only auth routes) or Wikipedia squad lists. **Coverage:** good Europe/Oceania, poor Africa pre-2010. **Hours: 6–10 h** (no clean API). **Leakage risk: moderate** — draft-year summer call-ups can post-date draft night, so date every cap.

Sources: [Youth International Experience Is a Limited Predictor](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9044071/); [Frontiers, athletic profiles & relative age](https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2025.1616800/full).

---

## 9. Loan spells / club movement — speculative, but nearly free

The v2 API's player-registration records carry `startDate`/`endDate` per club per season (verified), so **club changes and mid-season moves are derivable at zero extra collection cost** once §1 is done. Loaning young players to lower-division clubs for minutes is standard practice (Avdalas → Peristeri; Penda's two years in the French second division).

**Features:** `n_clubs_pre_draft`, `n_midseason_moves`, `max_tier_drop_on_loan`. **Hours: +1 h** on §1. **Leakage risk: low.** **Prior: weak**, and the sign is ambiguous — a loan can mean "the club is investing in his minutes" or "the club doesn't rate him." 

---

## 10. Data-source ToS audit (checked 2026-09-08) — read before spending collection hours

| Source | robots.txt / access | Verdict |
|---|---|---|
| `api-live.euroleague.net` | empty robots.txt; open API | ✅ **Use.** Site ToS unreadable (Vercel checkpoint) — unverified, be polite |
| `aba-liga.com` | `User-agent: *` with no rules; HTTP 200 | ✅ Permissive; fills the Jokić-shaped ABA gap post-2006 |
| `vtb-league.com` | empty robots.txt | ✅ Permissive |
| `nbl.com.au` | disallows only `/nbl-tipping`, `/angel-`, `/test-` | ✅ Permissive |
| `lnb.fr` | `User-agent: *`, no disallow | ✅ Permissive |
| `fiba.basketball` | disallows only auth routes | ✅ Permissive |
| `legabasket.it` | **disallows `*/statistiche`, `*/dettaglio-giocatori`, `*/dettaglio-gare`** | ❌ The exact pages you want are blocked |
| `tblstat.net` | **disallows `/pd.asp`, `/td.asp`, `/gd.asp`** (player/team/game detail) | ❌ Blocked for the useful pages |
| `acb.com` | Lists `ClaudeBot`, `Claude-SearchBot`, `Claude-User`, `GPTBot`, `CCBot`… then `User-agent: *` / `Allow: /` | ⚠️ **Ambiguous/malformed.** Read literally it's one group allowing all; the intent is plainly to name AI agents. Treat as **do not automate** — get ACB via the Euroleague API's `SP` code or skip |
| `proballers.com` | Cloudflare interstitial ("Just a moment…"), `noindex,nofollow` | ❌ Bot-challenged; don't automate |
| `realgm.com` | robots permissive (`crawl-delay: 2`) **but HTTP 403 to non-browser clients** — verified on two URLs | ⚠️ Access controls contradict robots; don't automate |
| `eurobasket.com` | Cloudflare Content-Signals block (Art. 4 EU DSM reservation of rights) | ⚠️ Read signals before use |

---

## Recommended order of work

1. **§4 shrinkage recalibration (2 h, no data pull)** — test first; it may capture part of the international-steal gap for free.
2. **§1 EuroLeague API spine (10–12 h)** — replaces the brittle international pipeline and fixes the Wembanyama-style stale-season bug at the root, since every row is dated by `cetdate`.
3. **§2 ANGT block (4–6 h)** — the structural fix for the youth/pro contamination finding.
4. **§3 tier ladder (2–3 h)** — refit inside CV folds.
5. **§5 young-senior-minutes (1–2 h)** and **§9 club movement (1 h)** — near-free once §1 lands.
6. **§7 pathway categorical (1–2 h)** — correctness, not lift.
7. Use **§6** only to sanity-check era effects. Skip **§8** unless the rest underdelivers.

**Biggest risk to flag:** §6's base rates say the second-round-international steal window largely **closed after 2015**. If your model's 0.45 comes disproportionately from 2019–2021 classes via an `intl × round2` effect learned on 2000–2018, check it per-class before trusting it to reach 0.55 on 2022–2025.
