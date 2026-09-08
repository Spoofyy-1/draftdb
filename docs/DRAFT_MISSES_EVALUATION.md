# Why good prospects were drafted low: an evaluation

Data: 2007-2018 draft classes with complete five-season outcomes (634 drafted players; outcomes = WAR over the first five NBA seasons, verified pre-draft inputs from data v4), plus the 2019-2025 classes with the seasons played so far. "Steal" = a player whose outcome rank in his class is top-10 but who was drafted 15th or later. "Bust" = a top-10 pick whose outcome rank is 25th or worse.

## 1. What steals look like before the draft (2007-2018)

| pre-draft profile (medians) | steals (n=55) | hits: top-10 outcome drafted top-10 (n=53) | all drafted (n=634) |
|---|---|---|---|
| age at draft | 21.0 | 19.6 | 21.0 |
| mock consensus rank | 26 | 5 | 25 |
| RSCI recruiting rank | 42 (67% unranked) | 4.5 (40% unranked) | 19 (51% unranked) |
| height / wingspan (in) | 79.5 / 81.4 | 79.5 / 82.8 | 79.5 / 82.8 |
| college impact (BPM-style) | 8.3 | 8.9 | 7.2 |
| true shooting | 58.3 | 59.1 | 57.5 |
| assist rate | 18.9 | 15.2 | 13.2 |
| 3PA per 40 | 4.1 | 3.6 | 3.9 |
| steal rate | 2.5 | 2.5 | 2.0 |
| international share | 27% | 8% | 14% |

The steals were nearly as productive in college as the hits (impact 8.3 vs 8.9, same efficiency), passed the ball more, and shot more threes. What they lacked was pedigree and youth: two thirds were unranked recruits, they were a year and a half older, and the mock consensus had them around 26th. A quarter were second-round internationals (Jokic, Gasol, Gobert, Bertans, Bjelica).

## 2. The draft's two discounts, measured

Among players who became top-15 outcomes (n=180), the share drafted in the top 15:

| age at draft | n | drafted top-15 | median pick |
|---|---|---|---|
| 19.5 or younger | 51 | 69% | 8 |
| 19.5-21 | 85 | 54% | 14 |
| 21-22 | 29 | 10% | 35 |
| over 22 | 15 | 13% | 27 |

| recruiting pedigree | n | drafted top-15 | median pick |
|---|---|---|---|
| RSCI top-30 | 67 | 75% | 7 |
| RSCI 31-100 | 30 | 30% | 24.5 |
| unranked | 83 | 33% | 23 |

Within the same college production quartile, the median pick moves from 4 (age 19.5 or younger) to 13 (19.5-21) to 26 (21-22) to 31 (over 22) for the most productive quartile. The outcomes in those cells (median five-season WAR 12.8, 10.6, 2.3, 3.4) show the age discount is right on average; the misses are the exceptions inside the older, unranked, productive group, which the draft treats as a block.

A least-squares model of "drafted later than his outcome rank" on standardized pre-draft factors (college players, n=444, R2 0.26) ranks the drivers: consensus rank (+9.1 slots per standard deviation: the draft follows the mocks, and the mocks carry the miss), college impact (+4.0: productive players are the ones who get under-drafted), steal rate (+1.1); age (-0.7), true shooting (-1.4) and usage (-0.6) reduce it. In words: productive, disruptive defenders with a middling mock rank are the classic steal; the draft under-weights production relative to consensus.

## 3. What busts look like (2007-2018)

Top-10 picks with outcome rank 25 or worse (n=35) versus top-10 picks that became top-10 outcomes (n=53): busts were slightly older (20.0 vs 19.6), equally pedigreed (RSCI 6 vs 4), less productive (impact 7.3 vs 8.9), less efficient (TS 57.0 vs 59.1), far weaker passers (assist rate 11.6 vs 15.1), and three times as often international (23% vs 8%). Bennett, Fultz, Turner, Parker, Derrick Williams, Thabeet, Beasley, Bagley, Kanter, Okafor, Josh Jackson, Bender, Wesley Johnson, Len, Hezonja, Thomas Robinson.

## 4. The 2019-2025 steals, and who saw them


31 players drafted 15th or later who are top-10 outcomes of their class so far. Median age 22.0; 24 of 31 were unranked recruits; median mock consensus 29. Same profile as the historical steals: older, unheralded, productive college players (Thybulle, Bane, Herb Jones, Kessler, Podziemski, Kalkbrenner) plus a few pedigreed teenagers the market cooled on (Maxey, Jaden McDaniels, Jalen Johnson).

| class | player | pick | outcome rank so far | our verified model | Colin's model | mock consensus | age | RSCI |
|---|---|---|---|---|---|---|---|---|
| 2019 | Terance Mann | 48 | 10 | 28 | 31 | 53 | 23 | unranked |
| 2019 | Daniel Gafford | 38 | 7 | 45 | 29 | 33 | 21 | 36 |
| 2019 | Nic Claxton | 31 | 4 | - | 19 | - | - | unranked |
| 2019 | Matisse Thybulle | 20 | 5 | 5 | 11 | 26 | 22 | unranked |
| 2019 | Brandon Clarke | 21 | 9 | 2 | 3 | 14 | 23 | unranked |
| 2020 | Desmond Bane | 30 | 3 | 11 | 10 | 23 | 22 | unranked |
| 2020 | Immanuel Quickley | 25 | 5 | 43 | 52 | 42 | 21 | unranked |
| 2020 | Payton Pritchard | 26 | 7 | 16 | 13 | 48 | 23 | 45 |
| 2020 | Jaden McDaniels | 28 | 10 | 52 | 54 | 24 | 20 | 7 |
| 2020 | Tyrese Maxey | 21 | 4 | 46 | 17 | 19 | 20 | 10 |
| 2021 | Herbert Jones | 35 | 9 | 19 | 31 | 44 | 23 | unranked |
| 2021 | Alperen Şengün | 16 | 2 | 5 | 16 | 16 | 19 | unranked |
| 2021 | Jalen Johnson | 20 | 8 | 15 | 10 | 12 | 20 | 11 |
| 2021 | Trey Murphy III | 17 | 6 | 41 | 25 | 20 | 21 | unranked |
| 2022 | Jaylin Williams | 34 | 7 | 15 | 24 | 34 | 20 | 79 |
| 2022 | Walker Kessler | 22 | 6 | 8 | 4 | 22 | 21 | 18 |
| 2022 | Tari Eason | 17 | 8 | 6 | 6 | 16 | 21 | 93 |
| 2022 | Mark Williams | 15 | 9 | 5 | 7 | 13 | 21 | 25 |
| 2023 | Trayce Jackson-Davis | 57 | 10 | 14 | 9 | 31 | 23 | unranked |
| 2023 | Toumani Camara | 52 | 6 | 41 | 33 | 53 | 23 | unranked |
| 2023 | Brandin Podziemski | 19 | 5 | 1 | 7 | 24 | 20 | 79 |
| 2023 | Jaime Jaquez Jr. | 18 | 9 | 5 | 7 | 22 | 22 | unranked |
| 2024 | Cam Spencer | 53 | 5 | 16 | 11 | 44 | 24 | unranked |
| 2024 | Oso Ighodaro | 40 | 7 | 31 | 15 | 48 | 22 | unranked |
| 2024 | Ajay Mitchell | 38 | 6 | 50 | 39 | 38 | 22 | unranked |
| 2024 | Baylor Scheierman | 30 | 8 | 26 | 19 | 29 | 24 | unranked |
| 2024 | Kel'el Ware | 15 | 2 | 8 | 20 | 22 | 20 | 6 |
| 2024 | Jaylon Tyson | 20 | 10 | 40 | 23 | 27 | 22 | 31 |
| 2025 | Javon Small | 48 | 9 | 23 | 35 | 46 | 23 | unranked |
| 2025 | Ryan Kalkbrenner | 34 | 5 | 31 | 7 | 29 | 23 | unranked |
| 2025 | Kasparas Jakučionis | 20 | 10 | 20 | 11 | 9 | 19 | unranked |

Where the two models put them: median rank 18 for our verified model and 16 for Colin's, against a median pick of 26 and a median consensus of 26. Our model had 42% of the steals in its top 15, Colin's 48%. Both models see the productive-older-unranked group well (Clarke 2/3, Thybulle 5/11, Kessler 8/4, Eason 6/6, Mark Williams 5/7, Podziemski 1/7, Jaquez 5/7) and both miss the teenage risers whose college numbers were ordinary (Maxey 46/17, Quickley 43/52, McDaniels 52/54, Trey Murphy 41/25) and the second-round bigs who developed physically (Gafford 45/29, Camara 41/33, Ajay Mitchell 50/39).

## 5. Answer in one paragraph

Good prospects fall in the draft for two reasons the data can measure: the draft follows the mock consensus, and the consensus discounts age and lack of recruiting pedigree far more than five-year outcomes justify for players who were already producing in college. The typical steal is a 21-23-year-old unranked recruit with top-quartile college production, above-average passing and defensive playmaking, a middling mock rank, or a second-round international big. Busts are the mirror image: pedigreed teenagers with ordinary production, weak passing, and a higher international share. The draft's age discount is right on average (older producers do turn out worse), which is why the misses are individual, not systematic, and why both models recover roughly half of them: the half whose production was already visible. The other half, late-blooming athletes and shooters whose college numbers were ordinary, is not in the pre-draft box score at all.

## 6. Why OUR model missed them (attribution) and what is being fixed

The ridge member carries 75% of the stack, so its standardized contributions say what pushed each missed steal down. Across the 30 steals of 2019-2025:

- **A column I added was hurting them.** `cons_src2_n` (how many of my scraped mock boards listed the player) was among the top-4 negative contributions for 19 of 30 steals. It varies by class (0.9 to 3.0 boards per year) and punishes low-profile players twice. Removed (data v4.7).
- **Youth-tournament lines read as pro seasons.** For college players the international block only holds FIBA youth lines (6-8 games): Maxey, Quickley, Camara and Kalkbrenner were pushed down by `intl_impact`, `adj_intl_reb36`, `intl_season_gap`, features the model learned from real pro seasons. New option: mask those columns for college-path players (`intl_mask`).
- **The linear member reads every column.** TabICL uses the 100 most important inputs; the ridge uses all 540, including sparse ones. New option: ridge on the same top-100 (`ridge_top`).
- **Pedigreed teenagers with modest numbers.** McDaniels (RSCI 7, impact 2.2) and Maxey (RSCI 10, impact 4.1) were ranked 52 and 46 because production dominates pedigree in the features. New engineered features: recruiting pedigree x youth and pedigree x production (`ped`).
- **The honest part of the misses.** Camara, Mann, Small and Ighodaro were ranked low because their mock consensus was 45-53 and their production, while good, sat in the second tier; their rise came after the draft. Kalkbrenner and Scheierman were pushed down by seniority (`col_n_seasons`, `col_class_index`), the age discount that is right on average.

These four changes are being tested together and separately on the verified data with the confirmed gate (panel `opt`); results are appended below when they land.
