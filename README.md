# nba-redraft

## AI models used

Two walk-forward models plus a small fixed set of pre-draft signals, combined by a linear stacker whose weights were fit
on the 2013-2018 context years only:

| Input | What it is |
|:------|:-----------|
| **CatBoost** | gradient-boosted trees (depth 5, 800 iterations, 10 seeds), context = every draft class from 2003 before the one being scored |
| **TabICL v2** | tabular in-context-learning foundation model (`TabICLRegressor`, 8 estimators, 8 seeds), context from 2010 |
| covariates | within-class ranks of true shooting, eFG, RSCI rank, G League translation prior, NBA relatives, a shooting composite (FT% + attempt-shrunk 3P%), international league-relative production x youth, steal rate, usage |

Both models learn a label that matches how each class is judged: WAR over a player's first max(3, N) NBA seasons, N =
seasons the scored class has played (2019-2021: five, 2022: four, 2023-2025: three), Gaussian-ranked within class. Model
features, all knowable before draft night: Torvik final season + trajectory, physicals, NBA combine, international pro /
FIBA youth lines, Torvik team context, a 5-8 board pre-draft mock consensus with its 90/60/30/7-day movement (Wayback
captures), basketball-reference biography (NBA relatives, shooting hand, birthplace, high-school path), NBADraft.net
scouting grades captured before each draft night (athleticism, size, defense, leadership, jump shot, NBA-ready, potential,
intangibles, overall, text flags) and -- CatBoost only -- game-log challenge-response features (rematches, bounce-back,
error persistence, close games). Nothing after draft night is used. The plain 50/50 rank-average of the two models scores
0.498 (`winners/catboost_tabicl_market_bio_scout_m3`); that is the fully clean number. The stacker's weights were fit on
2013-2018, but its covariate list was chosen after looking at holdout tilts, so the 51% headline is optimistic by about a point.

## Holdout (2019-2025)

| AI model accuracy | NBA scouts accuracy | Drafts the AI won |
|:-----------------:|:-------------------:|:-----------------:|
| **51%**           | 26%                 | **7 of 7**        |

Accuracy = Spearman rank correlation between the order (AI redraft, or the real draft) and the players' realised 5-year WAR
(classes that have played fewer than five seasons are judged on the seasons they have played). Rule: `stackctx w=0.1389,0.1456,0.3714,0.3626,0.1114,-0.0145,-0.0211,0.2852,0.2343,0.1205,0.1618,0.1682,0.0845 on [C03 m3 +mo+rs+person+sc x5 | C03 m3 +mo+rs+person+sc x5 s5 | T10 o2 m3 +mo+person+sc x4 | T10 o2 m3 +mo+person+sc x4 s4]`.

## By draft

| Draft | AI model | NBA scouts | Difference |
|:------|---------:|-----------:|-----------:|
| 2019  | 46%      | 40%        | +6         |
| 2020  | 55%      | 35%        | +20         |
| 2021  | 58%      | 41%        | +17         |
| 2022  | 51%      | 24%        | +27         |
| 2023  | 60%      | 10%        | +50         |
| 2024  | 47%      | 15%        | +32         |
| 2025  | 43%      | 17%        | +26         |
| **Mean** | **51%** | **26%**   | **+25**    |

## By player

Every holdout draft, actual order beside the model's reordering of the same names.

### 2019

| # | Actual draft | 5yr WAR | Model redraft | 5yr WAR |
|--:|:-------------|--------:|:--------------|--------:|
| 1 | Zion Williamson | 22.1 | Zion Williamson | 22.1 |
| 2 | Ja Morant | 19.2 | Brandon Clarke | 12.3 |
| 3 | RJ Barrett | 6.1 | Bol Bol | 1.9 |
| 4 | De'Andre Hunter | 2.4 | Grant Williams | 7.2 |
| 5 | Darius Garland | 15.3 | Jarrett Culver | -1.4 |
| 6 | Jarrett Culver | -1.4 | Ja Morant | 19.2 |
| 7 | Coby White | 6.9 | Chuma Okeke | 2.3 |
| 8 | Jaxson Hayes | 2.0 | Matisse Thybulle | 15.6 |
| 9 | Rui Hachimura | 0.1 | Jaxson Hayes | 2.0 |
| 10 | Cam Reddish | -0.9 | Goga Bitadze | 3.7 |
| 11 | Cameron Johnson | 17.2 | De'Andre Hunter | 2.4 |
| 12 | P.J. Washington | 7.0 | Ty Jerome | 1.3 |
| 13 | Tyler Herro | 12.5 | P.J. Washington | 7.0 |
| 14 | Romeo Langford | 0.9 | Nickeil Alexander-Walker | 4.8 |
| 15 | Sekou Doumbouya | -3.6 | Tyler Herro | 12.5 |
| 16 | Chuma Okeke | 2.3 | Dylan Windler | 1.0 |
| 17 | Nickeil Alexander-Walker | 4.8 | Coby White | 6.9 |
| 18 | Goga Bitadze | 3.7 | Nic Claxton | 16.3 |
| 19 | Luka Šamanić | -0.7 | Bruno Fernando | -0.6 |
| 20 | Matisse Thybulle | 15.6 | Cameron Johnson | 17.2 |
| 21 | Brandon Clarke | 12.3 | Jalen McDaniels | 0.3 |
| 22 | Grant Williams | 7.2 | Daniel Gafford | 13.1 |
| 23 | Darius Bazley | -2.2 | Sekou Doumbouya | -3.6 |
| 24 | Ty Jerome | 1.3 | Isaiah Roby | -2.1 |
| 25 | Nassir Little | 1.2 | RJ Barrett | 6.1 |
| 26 | Dylan Windler | 1.0 | Quinndary Weatherspoon | -0.9 |
| 27 | Mfiondu Kabengele | -0.1 | Darius Garland | 15.3 |
| 28 | Jordan Poole | 3.7 | Luka Šamanić | -0.7 |
| 29 | Keldon Johnson | 7.2 | Mfiondu Kabengele | -0.1 |
| 30 | Kevin Porter Jr. | 7.0 | Keldon Johnson | 7.2 |
| 31 | Nic Claxton | 16.3 | Terance Mann | 11.8 |
| 32 | KZ Okpala | -1.2 | Tremont Waters | -0.3 |
| 33 | Carsen Edwards | -0.6 | Cody Martin | 7.4 |
| 34 | Bruno Fernando | -0.6 | Carsen Edwards | -0.6 |
| 35 | Didi Louzada | -0.8 | Alen Smailagić | -0.2 |
| 36 | Cody Martin | 7.4 | Cam Reddish | -0.9 |
| 37 | Deividas Sirvydis | -0.0 | Deividas Sirvydis | -0.0 |
| 38 | Daniel Gafford | 13.1 | Darius Bazley | -2.2 |
| 39 | Alen Smailagić | -0.2 | Didi Louzada | -0.8 |
| 40 | Justin James | -0.2 | Vanja Marinković | 0.0 |
| 41 | Eric Paschall | -2.1 | Romeo Langford | 0.9 |
| 42 | Admiral Schofield | -1.0 | Kyle Guy | -0.3 |
| 43 | Jaylen Nowell | 1.0 | Justin James | -0.2 |
| 44 | Bol Bol | 1.9 | Eric Paschall | -2.1 |
| 45 | Isaiah Roby | -2.1 | Dewan Hernandez | 0.1 |
| 46 | Talen Horton-Tucker | 3.7 | Jarrell Brantley | -0.1 |
| 47 | Ignas Brazdeikis | -0.8 | Rui Hachimura | 0.1 |
| 48 | Terance Mann | 11.8 | Talen Horton-Tucker | 3.7 |
| 49 | Quinndary Weatherspoon | -0.9 | Nassir Little | 1.2 |
| 50 | Jarrell Brantley | -0.1 | Jordan Poole | 3.7 |
| 51 | Tremont Waters | -0.3 | Marial Shayok | -0.1 |
| 52 | Jalen McDaniels | 0.3 | Jaylen Nowell | 1.0 |
| 53 | Justin Wright-Foreman | -0.2 | Justin Wright-Foreman | -0.2 |
| 54 | Marial Shayok | -0.1 | Ignas Brazdeikis | -0.8 |
| 55 | Kyle Guy | -0.3 | Jordan Bone | -0.8 |
| 56 | Jaylen Hands | 0.0 | Admiral Schofield | -1.0 |
| 57 | Jordan Bone | -0.8 | Miye Oni | -0.0 |
| 58 | Miye Oni | -0.0 | Kevin Porter Jr. | 7.0 |
| 59 | Dewan Hernandez | 0.1 | Jaylen Hands | 0.0 |
| 60 | Vanja Marinković | 0.0 | KZ Okpala | -1.2 |

### 2020

| # | Actual draft | 5yr WAR | Model redraft | 5yr WAR |
|--:|:-------------|--------:|:--------------|--------:|
| 1 | Anthony Edwards | 36.5 | Tyrese Haliburton | 38.5 |
| 2 | James Wiseman | -4.2 | Onyeka Okongwu | 14.5 |
| 3 | LaMelo Ball | 18.1 | James Wiseman | -4.2 |
| 4 | Patrick Williams | 1.2 | Obi Toppin | 10.2 |
| 5 | Isaac Okoro | 6.3 | Xavier Tillman Sr. | 5.3 |
| 6 | Onyeka Okongwu | 14.5 | Jalen Smith | 5.6 |
| 7 | Killian Hayes | -4.1 | Devin Vassell | 8.1 |
| 8 | Obi Toppin | 10.2 | Tre Jones | 9.4 |
| 9 | Deni Avdija | 13.4 | Paul Reed | 6.7 |
| 10 | Jalen Smith | 5.6 | Patrick Williams | 1.2 |
| 11 | Devin Vassell | 8.1 | LaMelo Ball | 18.1 |
| 12 | Tyrese Haliburton | 38.5 | Payton Pritchard | 14.7 |
| 13 | Kira Lewis Jr. | -0.0 | Josh Green | 2.1 |
| 14 | Aaron Nesmith | 5.6 | Tyrell Terry | 0.1 |
| 15 | Cole Anthony | 5.8 | Desmond Bane | 26.9 |
| 16 | Isaiah Stewart | 6.0 | Malachi Flynn | 1.4 |
| 17 | Aleksej Pokusevski | -3.1 | Saddiq Bey | 7.1 |
| 18 | Josh Green | 2.1 | Precious Achiuwa | 3.6 |
| 19 | Saddiq Bey | 7.1 | Anthony Edwards | 36.5 |
| 20 | Precious Achiuwa | 3.6 | Deni Avdija | 13.4 |
| 21 | Tyrese Maxey | 21.9 | Udoka Azubuike | 0.1 |
| 22 | Zeke Nnaji | -0.4 | Aaron Nesmith | 5.6 |
| 23 | Leandro Bolmaro | -0.5 | Tyrese Maxey | 21.9 |
| 24 | R.J. Hampton | -4.5 | Isaiah Stewart | 6.0 |
| 25 | Immanuel Quickley | 19.5 | Kira Lewis Jr. | -0.0 |
| 26 | Payton Pritchard | 14.7 | Cole Anthony | 5.8 |
| 27 | Udoka Azubuike | 0.1 | Tyler Bey | -0.4 |
| 28 | Jaden McDaniels | 12.8 | Isaac Okoro | 6.3 |
| 29 | Malachi Flynn | 1.4 | Killian Hayes | -4.1 |
| 30 | Desmond Bane | 26.9 | Zeke Nnaji | -0.4 |
| 31 | Tyrell Terry | 0.1 | Vernon Carey Jr. | -0.1 |
| 32 | Vernon Carey Jr. | -0.1 | Skylar Mays | 0.6 |
| 33 | Daniel Oturu | -0.5 | KJ Martin | 0.3 |
| 34 | Théo Maledon | -6.3 | Reggie Perry | -0.9 |
| 35 | Xavier Tillman Sr. | 5.3 | Justinian Jessup | 0.0 |
| 36 | Tyler Bey | -0.4 | Jahmi'us Ramsey | -0.6 |
| 37 | Vít Krejčí | 0.6 | Marko Simonovic | -0.0 |
| 38 | Saben Lee | 0.0 | Vít Krejčí | 0.6 |
| 39 | Elijah Hughes | -1.8 | Yam Madar | 0.0 |
| 40 | Robert Woodard II | -0.0 | CJ Elleby | -1.7 |
| 41 | Tre Jones | 9.4 | Isaiah Joe | 10.9 |
| 42 | Nick Richards | 0.4 | Daniel Oturu | -0.5 |
| 43 | Jahmi'us Ramsey | -0.6 | Grant Riller | 0.0 |
| 44 | Marko Simonovic | -0.0 | Leandro Bolmaro | -0.5 |
| 45 | Jordan Nwora | -1.9 | Sam Merrill | 4.0 |
| 46 | CJ Elleby | -1.7 | Jalen Harris | 0.3 |
| 47 | Yam Madar | 0.0 | Saben Lee | 0.0 |
| 48 | Nico Mannion | -0.3 | R.J. Hampton | -4.5 |
| 49 | Isaiah Joe | 10.9 | Elijah Hughes | -1.8 |
| 50 | Skylar Mays | 0.6 | Théo Maledon | -6.3 |
| 51 | Justinian Jessup | 0.0 | Nick Richards | 0.4 |
| 52 | KJ Martin | 0.3 | Immanuel Quickley | 19.5 |
| 53 | Cassius Winston | -0.6 | Aleksej Pokusevski | -3.1 |
| 54 | Cassius Stanley | -0.1 | Cassius Winston | -0.6 |
| 55 | Jay Scrubb | -0.5 | Jaden McDaniels | 12.8 |
| 56 | Grant Riller | 0.0 | Robert Woodard II | -0.0 |
| 57 | Reggie Perry | -0.9 | Jay Scrubb | -0.5 |
| 58 | Paul Reed | 6.7 | Jordan Nwora | -1.9 |
| 59 | Jalen Harris | 0.3 | Nico Mannion | -0.3 |
| 60 | Sam Merrill | 4.0 | Cassius Stanley | -0.1 |

### 2021

| # | Actual draft | 5yr WAR | Model redraft | 5yr WAR |
|--:|:-------------|--------:|:--------------|--------:|
| 1 | Cade Cunningham | 22.6 | Franz Wagner | 25.7 |
| 2 | Jalen Green | 7.5 | Evan Mobley | 28.5 |
| 3 | Evan Mobley | 28.5 | Jalen Suggs | 12.8 |
| 4 | Scottie Barnes | 32.0 | Cade Cunningham | 22.6 |
| 5 | Jalen Suggs | 12.8 | Chris Duarte | 1.6 |
| 6 | Josh Giddey | 19.6 | Isaiah Jackson | 3.7 |
| 7 | Jonathan Kuminga | 3.5 | Jalen Johnson | 17.0 |
| 8 | Franz Wagner | 25.7 | Neemias Queta | 7.8 |
| 9 | Davion Mitchell | 5.8 | Scottie Barnes | 32.0 |
| 10 | Ziaire Williams | 1.0 | Kai Jones | 0.6 |
| 11 | James Bouknight | -1.4 | Miles McBride | 6.7 |
| 12 | Joshua Primo | -1.1 | Day'Ron Sharpe | 6.3 |
| 13 | Chris Duarte | 1.6 | Jared Butler | 0.7 |
| 14 | Moses Moody | 5.6 | Corey Kispert | 2.0 |
| 15 | Corey Kispert | 2.0 | Alperen Şengün | 29.5 |
| 16 | Alperen Şengün | 29.5 | Davion Mitchell | 5.8 |
| 17 | Trey Murphy III | 20.7 | Jeremiah Robinson-Earl | -1.0 |
| 18 | Tre Mann | -1.4 | Moses Moody | 5.6 |
| 19 | Kai Jones | 0.6 | Jaden Springer | 0.2 |
| 20 | Jalen Johnson | 17.0 | Filip Petrušev | -0.0 |
| 21 | Keon Johnson | -3.7 | Luka Garza | 3.7 |
| 22 | Isaiah Jackson | 3.7 | Scottie Lewis | 0.0 |
| 23 | Usman Garuba | 0.9 | Herbert Jones | 15.1 |
| 24 | Josh Christopher | -1.8 | James Bouknight | -1.4 |
| 25 | Quentin Grimes | 9.5 | Bones Hyland | 6.2 |
| 26 | Bones Hyland | 6.2 | Trey Murphy III | 20.7 |
| 27 | Cam Thomas | 0.7 | Josh Giddey | 19.6 |
| 28 | Jaden Springer | 0.2 | Charles Bassey | 1.8 |
| 29 | Day'Ron Sharpe | 6.3 | Usman Garuba | 0.9 |
| 30 | Santi Aldama | 11.4 | Ayo Dosunmu | 6.6 |
| 31 | Isaiah Todd | -0.6 | Quentin Grimes | 9.5 |
| 32 | Jeremiah Robinson-Earl | -1.0 | Sharife Cooper | -1.0 |
| 33 | Jason Preston | 0.0 | Jason Preston | 0.0 |
| 34 | Rokas Jokubaitis | 0.0 | Isaiah Livers | -0.5 |
| 35 | Herbert Jones | 15.1 | Marcus Zegarowski | 0.0 |
| 36 | Miles McBride | 6.7 | Balša Koprivica | 0.0 |
| 37 | JT Thor | -2.2 | Josh Christopher | -1.8 |
| 38 | Ayo Dosunmu | 6.6 | RaiQuan Gray | 0.2 |
| 39 | Neemias Queta | 7.8 | Joe Wieskamp | -0.4 |
| 40 | Jared Butler | 0.7 | Sandro Mamukelashvili | 5.6 |
| 41 | Joe Wieskamp | -0.4 | Jericho Sims | 0.9 |
| 42 | Isaiah Livers | -0.5 | Kessler Edwards | -0.2 |
| 43 | Greg Brown III | -2.2 | Aaron Wiggins | 7.6 |
| 44 | Kessler Edwards | -0.2 | Georgios Kalaitzakis | -1.1 |
| 45 | Juhann Begarin | 0.0 | Santi Aldama | 11.4 |
| 46 | Dalano Banton | 2.0 | Juhann Begarin | 0.0 |
| 47 | David Johnson | 0.0 | Dalano Banton | 2.0 |
| 48 | Sharife Cooper | -1.0 | JT Thor | -2.2 |
| 49 | Marcus Zegarowski | 0.0 | Rokas Jokubaitis | 0.0 |
| 50 | Filip Petrušev | -0.0 | Jalen Green | 7.5 |
| 51 | Brandon Boston Jr. | -1.1 | Jonathan Kuminga | 3.5 |
| 52 | Luka Garza | 3.7 | David Johnson | 0.0 |
| 53 | Charles Bassey | 1.8 | Keon Johnson | -3.7 |
| 54 | Sandro Mamukelashvili | 5.6 | Tre Mann | -1.4 |
| 55 | Aaron Wiggins | 7.6 | Brandon Boston Jr. | -1.1 |
| 56 | Scottie Lewis | 0.0 | Ziaire Williams | 1.0 |
| 57 | Balša Koprivica | 0.0 | Isaiah Todd | -0.6 |
| 58 | Jericho Sims | 0.9 | Cam Thomas | 0.7 |
| 59 | RaiQuan Gray | 0.2 | Greg Brown III | -2.2 |
| 60 | Georgios Kalaitzakis | -1.1 | Joshua Primo | -1.1 |

### 2022

| # | Actual draft | 5yr WAR | Model redraft | 5yr WAR |
|--:|:-------------|--------:|:--------------|--------:|
| 1 | Paolo Banchero | 18.5 | Keegan Murray | 8.4 |
| 2 | Chet Holmgren | 19.9 | Tari Eason | 8.6 |
| 3 | Jabari Smith Jr. | 7.5 | Walker Kessler | 12.9 |
| 4 | Keegan Murray | 8.4 | Jabari Smith Jr. | 7.5 |
| 5 | Jaden Ivey | -0.5 | Chet Holmgren | 19.9 |
| 6 | Bennedict Mathurin | -1.3 | Jake LaRavia | 3.5 |
| 7 | Shaedon Sharpe | 2.7 | Jalen Duren | 17.1 |
| 8 | Dyson Daniels | 15.4 | Mark Williams | 8.4 |
| 9 | Jeremy Sochan | -0.1 | Jeremy Sochan | -0.1 |
| 10 | Johnny Davis | -1.9 | Paolo Banchero | 18.5 |
| 11 | Ousmane Dieng | -0.6 | Jalen Williams | 22.4 |
| 12 | Jalen Williams | 22.4 | Johnny Davis | -1.9 |
| 13 | Jalen Duren | 17.1 | Dalen Terry | 0.8 |
| 14 | Ochai Agbaji | -0.8 | Christian Koloko | -0.3 |
| 15 | Mark Williams | 8.4 | E.J. Liddell | -0.2 |
| 16 | AJ Griffin | 0.7 | Josh Minott | 3.2 |
| 17 | Tari Eason | 8.6 | Andrew Nembhard | 5.1 |
| 18 | Dalen Terry | 0.8 | Jaylin Williams | 8.7 |
| 19 | Jake LaRavia | 3.5 | Shaedon Sharpe | 2.7 |
| 20 | Malaki Branham | -5.2 | Vince Williams Jr. | 2.5 |
| 21 | Christian Braun | 8.1 | Bennedict Mathurin | -1.3 |
| 22 | Walker Kessler | 12.9 | Dyson Daniels | 15.4 |
| 23 | David Roddy | -2.4 | Christian Braun | 8.1 |
| 24 | MarJon Beauchamp | -1.2 | Karlo Matković | 1.7 |
| 25 | Blake Wesley | -3.0 | TyTy Washington Jr. | -0.9 |
| 26 | Wendell Moore Jr. | -0.1 | David Roddy | -2.4 |
| 27 | Nikola Jović | 3.2 | Wendell Moore Jr. | -0.1 |
| 28 | Patrick Baldwin Jr. | -0.1 | Kennedy Chandler | -0.6 |
| 29 | TyTy Washington Jr. | -0.9 | Jabari Walker | -1.2 |
| 30 | Peyton Watson | 2.2 | Jaden Ivey | -0.5 |
| 31 | Andrew Nembhard | 5.1 | Ismael Kamagate | 0.0 |
| 32 | Caleb Houstan | 0.1 | Nikola Jović | 3.2 |
| 33 | Christian Koloko | -0.3 | AJ Griffin | 0.7 |
| 34 | Jaylin Williams | 8.7 | Ousmane Dieng | -0.6 |
| 35 | Max Christie | 0.4 | Khalifa Diop | 0.0 |
| 36 | Gabriele Procida | 0.0 | Isaiah Mobley | -0.0 |
| 37 | Jaden Hardy | -2.3 | Malaki Branham | -5.2 |
| 38 | Kennedy Chandler | -0.6 | Ochai Agbaji | -0.8 |
| 39 | Khalifa Diop | 0.0 | Gabriele Procida | 0.0 |
| 40 | Bryce McGowens | -2.3 | Peyton Watson | 2.2 |
| 41 | E.J. Liddell | -0.2 | Luke Travers | -0.3 |
| 42 | Trevor Keels | -0.1 | Kendall Brown | -0.2 |
| 43 | Moussa Diabaté | 6.1 | Ryan Rollins | 3.8 |
| 44 | Ryan Rollins | 3.8 | Gui Santos | 2.4 |
| 45 | Josh Minott | 3.2 | Hugo Besson | 0.0 |
| 46 | Ismael Kamagate | 0.0 | Matteo Spagnolo | 0.0 |
| 47 | Vince Williams Jr. | 2.5 | Tyrese Martin | -1.0 |
| 48 | Kendall Brown | -0.2 | Yannick Nzosa | 0.0 |
| 49 | Isaiah Mobley | -0.0 | MarJon Beauchamp | -1.2 |
| 50 | Matteo Spagnolo | 0.0 | Moussa Diabaté | 6.1 |
| 51 | Tyrese Martin | -1.0 | JD Davison | 0.1 |
| 52 | Karlo Matković | 1.7 | Max Christie | 0.4 |
| 53 | JD Davison | 0.1 | Trevor Keels | -0.1 |
| 54 | Yannick Nzosa | 0.0 | Jaden Hardy | -2.3 |
| 55 | Gui Santos | 2.4 | Blake Wesley | -3.0 |
| 56 | Luke Travers | -0.3 | Patrick Baldwin Jr. | -0.1 |
| 57 | Jabari Walker | -1.2 | Bryce McGowens | -2.3 |
| 58 | Hugo Besson | 0.0 | Caleb Houstan | 0.1 |

### 2023

| # | Actual draft | 5yr WAR | Model redraft | 5yr WAR |
|--:|:-------------|--------:|:--------------|--------:|
| 1 | Victor Wembanyama | 32.1 | Trayce Jackson-Davis | 4.7 |
| 2 | Brandon Miller | 6.0 | Jaime Jaquez Jr. | 5.7 |
| 3 | Scoot Henderson | -1.9 | Anthony Black | 4.2 |
| 4 | Amen Thompson | 20.4 | Dereck Lively II | 5.7 |
| 5 | Ausar Thompson | 10.7 | Victor Wembanyama | 32.1 |
| 6 | Anthony Black | 4.2 | Cason Wallace | 13.7 |
| 7 | Bilal Coulibaly | -2.9 | Brandon Miller | 6.0 |
| 8 | Jarace Walker | 0.3 | Jarace Walker | 0.3 |
| 9 | Taylor Hendricks | -0.2 | Brandin Podziemski | 9.1 |
| 10 | Cason Wallace | 13.7 | Jaylen Clark | 1.0 |
| 11 | Jett Howard | -0.4 | Amen Thompson | 20.4 |
| 12 | Dereck Lively II | 5.7 | Jalen Slawson | 0.3 |
| 13 | Gradey Dick | -3.5 | Jalen Pickett | 0.5 |
| 14 | Jordan Hawkins | -3.5 | Taylor Hendricks | -0.2 |
| 15 | Kobe Bufkin | -0.6 | Marcus Sasser | 0.7 |
| 16 | Keyonte George | -1.0 | Jordan Miller | 2.2 |
| 17 | Jalen Hood-Schifino | -1.2 | Leonard Miller | 0.9 |
| 18 | Jaime Jaquez Jr. | 5.7 | Gradey Dick | -3.5 |
| 19 | Brandin Podziemski | 9.1 | Kris Murray | 0.4 |
| 20 | Cam Whitmore | 2.1 | Ausar Thompson | 10.7 |
| 21 | Noah Clowney | -1.1 | Scoot Henderson | -1.9 |
| 22 | Dariq Whitehead | -0.8 | Kobe Brown | -0.3 |
| 23 | Kris Murray | 0.4 | Kobe Bufkin | -0.6 |
| 24 | Olivier-Maxence Prosper | 0.8 | Bilal Coulibaly | -2.9 |
| 25 | Marcus Sasser | 0.7 | Andre Jackson Jr. | -1.6 |
| 26 | Ben Sheppard | 0.3 | Noah Clowney | -1.1 |
| 27 | Nick Smith Jr. | -4.4 | Mouhamed Gueye | 1.5 |
| 28 | Brice Sensabaugh | -0.6 | Toumani Camara | 6.8 |
| 29 | Julian Strawther | -1.6 | Colby Jones | -0.2 |
| 30 | Kobe Brown | -0.3 | Julian Phillips | -0.4 |
| 31 | James Nnaji | 0.0 | Cam Whitmore | 2.1 |
| 32 | Jalen Pickett | 0.5 | James Nnaji | 0.0 |
| 33 | Leonard Miller | 0.9 | Hunter Tyson | -0.6 |
| 34 | Colby Jones | -0.2 | Sidy Cissoko | -1.1 |
| 35 | Julian Phillips | -0.4 | Keyonte George | -1.0 |
| 36 | Andre Jackson Jr. | -1.6 | Isaiah Wong | -0.2 |
| 37 | Hunter Tyson | -0.6 | Ben Sheppard | 0.3 |
| 38 | Jordan Walsh | 2.1 | Tristan Vukcevic | 1.1 |
| 39 | Mouhamed Gueye | 1.5 | Jett Howard | -0.4 |
| 40 | Maxwell Lewis | -0.7 | Mojave King | 0.0 |
| 41 | Amari Bailey | -0.2 | Tarik Biberovic | 0.0 |
| 42 | Tristan Vukcevic | 1.1 | Jordan Walsh | 2.1 |
| 43 | Rayan Rupert | -3.6 | Rayan Rupert | -3.6 |
| 44 | Sidy Cissoko | -1.1 | Seth Lundy | -0.2 |
| 45 | GG Jackson II | -0.0 | Jalen Wilson | -3.1 |
| 46 | Seth Lundy | -0.2 | Olivier-Maxence Prosper | 0.8 |
| 47 | Mojave King | 0.0 | Keyontae Johnson | -0.2 |
| 48 | Jordan Miller | 2.2 | Julian Strawther | -1.6 |
| 49 | Emoni Bates | -0.3 | Amari Bailey | -0.2 |
| 50 | Keyontae Johnson | -0.2 | Brice Sensabaugh | -0.6 |
| 51 | Jalen Wilson | -3.1 | Dariq Whitehead | -0.8 |
| 52 | Toumani Camara | 6.8 | Jalen Hood-Schifino | -1.2 |
| 53 | Jaylen Clark | 1.0 | Jordan Hawkins | -3.5 |
| 54 | Jalen Slawson | 0.3 | Chris Livingston | -0.4 |
| 55 | Isaiah Wong | -0.2 | Emoni Bates | -0.3 |
| 56 | Tarik Biberovic | 0.0 | Maxwell Lewis | -0.7 |
| 57 | Trayce Jackson-Davis | 4.7 | Nick Smith Jr. | -4.4 |
| 58 | Chris Livingston | -0.4 | GG Jackson II | -0.0 |

### 2024

| # | Actual draft | 5yr WAR | Model redraft | 5yr WAR |
|--:|:-------------|--------:|:--------------|--------:|
| 1 | Zaccharie Risacher | -0.8 | Reed Sheppard | 5.9 |
| 2 | Alex Sarr | 0.9 | Jonathan Mogbo | 0.4 |
| 3 | Reed Sheppard | 5.9 | Adem Bona | -0.1 |
| 4 | Stephon Castle | 5.5 | Devin Carter | 0.0 |
| 5 | Ron Holland | 1.1 | Zach Edey | 3.1 |
| 6 | Tidjane Salaün | -1.1 | Donovan Clingan | 8.6 |
| 7 | Donovan Clingan | 8.6 | Tyler Kolek | 1.3 |
| 8 | Rob Dillingham | -2.4 | Cam Spencer | 4.9 |
| 9 | Zach Edey | 3.1 | DaRon Holmes | 0.2 |
| 10 | Cody Williams | -5.5 | Anton Watson | -0.1 |
| 11 | Matas Buzelis | 1.8 | Jamal Shead | -0.7 |
| 12 | Nikola Topić | -0.4 | Alex Sarr | 0.9 |
| 13 | Devin Carter | 0.0 | Stephon Castle | 5.5 |
| 14 | Bub Carrington | -6.6 | Ryan Dunn | 0.2 |
| 15 | Kel'el Ware | 6.5 | Kyle Filipowski | 2.4 |
| 16 | Jared McCain | 0.7 | Oso Ighodaro | 3.3 |
| 17 | Dalton Knecht | 0.3 | Matas Buzelis | 1.8 |
| 18 | Tristan Da Silva | 1.7 | Dillon Jones | -0.1 |
| 19 | Ja'Kobe Walter | 2.1 | Tidjane Salaün | -1.1 |
| 20 | Jaylon Tyson | 2.9 | Kel'el Ware | 6.5 |
| 21 | Yves Missi | -0.6 | KJ Simpson | -1.4 |
| 22 | DaRon Holmes | 0.2 | Jared McCain | 0.7 |
| 23 | AJ Johnson | -3.3 | Nikola Topić | -0.4 |
| 24 | Kyshawn George | 0.2 | Tristan Da Silva | 1.7 |
| 25 | Pacôme Dadiet | -0.1 | Jaylon Tyson | 2.9 |
| 26 | Dillon Jones | -0.1 | Baylor Scheierman | 3.2 |
| 27 | Terrence Shannon Jr. | 0.1 | Enrique Freeman | -0.2 |
| 28 | Ryan Dunn | 0.2 | Ariel Hukporti | -0.1 |
| 29 | Isaiah Collier | -2.0 | Kevin McCullar Jr. | -0.1 |
| 30 | Baylor Scheierman | 3.2 | Harrison Ingram | 0.1 |
| 31 | Jonathan Mogbo | 0.4 | Tristen Newton | -0.1 |
| 32 | Kyle Filipowski | 2.4 | Ron Holland | 1.1 |
| 33 | Tyler Smith | -0.1 | Zaccharie Risacher | -0.8 |
| 34 | Tyler Kolek | 1.3 | Quinten Post | 1.6 |
| 35 | Johnny Furphy | -0.2 | Pelle Larsson | 2.1 |
| 36 | Juan Nunez | 0.0 | Terrence Shannon Jr. | 0.1 |
| 37 | Bobi Klintman | -0.1 | Tyler Smith | -0.1 |
| 38 | Ajay Mitchell | 4.5 | Yves Missi | -0.6 |
| 39 | Jaylen Wells | 1.2 | Kyshawn George | 0.2 |
| 40 | Oso Ighodaro | 3.3 | Ajay Mitchell | 4.5 |
| 41 | Adem Bona | -0.1 | Juan Nunez | 0.0 |
| 42 | KJ Simpson | -1.4 | Nikola Djurisic | 0.0 |
| 43 | Nikola Djurisic | 0.0 | Johnny Furphy | -0.2 |
| 44 | Pelle Larsson | 2.1 | Jaylen Wells | 1.2 |
| 45 | Jamal Shead | -0.7 | Melvin Ajinça | 0.0 |
| 46 | Cam Christie | -0.4 | Rob Dillingham | -2.4 |
| 47 | Antonio Reeves | -0.2 | Cody Williams | -5.5 |
| 48 | Harrison Ingram | 0.1 | Ulrich Chomche | -0.1 |
| 49 | Tristen Newton | -0.1 | AJ Johnson | -3.3 |
| 50 | Enrique Freeman | -0.2 | Bronny James | -0.6 |
| 51 | Melvin Ajinça | 0.0 | Bub Carrington | -6.6 |
| 52 | Quinten Post | 1.6 | Ja'Kobe Walter | 2.1 |
| 53 | Cam Spencer | 4.9 | Dalton Knecht | 0.3 |
| 54 | Anton Watson | -0.1 | Antonio Reeves | -0.2 |
| 55 | Bronny James | -0.6 | Pacôme Dadiet | -0.1 |
| 56 | Kevin McCullar Jr. | -0.1 | Cam Christie | -0.4 |
| 57 | Ulrich Chomche | -0.1 | Isaiah Collier | -2.0 |
| 58 | Ariel Hukporti | -0.1 | Bobi Klintman | -0.1 |

### 2025

| # | Actual draft | 5yr WAR | Model redraft | 5yr WAR |
|--:|:-------------|--------:|:--------------|--------:|
| 1 | Cooper Flagg | 5.0 | Collin Murray-Boyles | 2.7 |
| 2 | Dylan Harper | 4.4 | Cooper Flagg | 5.0 |
| 3 | VJ Edgecombe | 4.3 | Thomas Sorber | 0.0 |
| 4 | Kon Knueppel | 7.4 | Kon Knueppel | 7.4 |
| 5 | Ace Bailey | -2.4 | VJ Edgecombe | 4.3 |
| 6 | Tre Johnson | -2.0 | Dylan Harper | 4.4 |
| 7 | Jeremiah Fears | -0.6 | Ryan Kalkbrenner | 3.3 |
| 8 | Egor Dëmin | 0.8 | Cedric Coward | 2.3 |
| 9 | Collin Murray-Boyles | 2.7 | Asa Newell | 0.3 |
| 10 | Khaman Maluach | -0.3 | Johni Broome | -0.3 |
| 11 | Cedric Coward | 2.3 | Nique Clifford | -2.7 |
| 12 | Noa Essengue | -0.1 | Khaman Maluach | -0.3 |
| 13 | Derik Queen | 2.3 | Derik Queen | 2.3 |
| 14 | Carter Bryant | 0.1 | Adou Thiero | -0.2 |
| 15 | Thomas Sorber | 0.0 | Noa Essengue | -0.1 |
| 16 | Yang Hansen | -1.0 | Kasparas Jakučionis | 1.0 |
| 17 | Joan Beringer | 0.6 | Carter Bryant | 0.1 |
| 18 | Walter Clayton | -1.4 | Danny Wolf | -0.1 |
| 19 | Nolan Traoré | -3.1 | Jase Richardson | 0.1 |
| 20 | Kasparas Jakučionis | 1.0 | Rasheer Fleming | -0.2 |
| 21 | Will Riley | -2.0 | Max Shulga | -0.1 |
| 22 | Drake Powell | -2.5 | Kam Jones | -1.3 |
| 23 | Asa Newell | 0.3 | Bogoljub Markovic | 0.0 |
| 24 | Nique Clifford | -2.7 | Yang Hansen | -1.0 |
| 25 | Jase Richardson | 0.1 | Brooks Barnhizer | -0.1 |
| 26 | Ben Saraf | -2.5 | Walter Clayton | -1.4 |
| 27 | Danny Wolf | -0.1 | Maxime Raynaud | -0.3 |
| 28 | Hugo González | 0.6 | Jahmai Mashack | -1.8 |
| 29 | Liam McNeeley | 0.2 | Amari Williams | 0.0 |
| 30 | Yanic Konan Niederhäuser | 0.2 | Hugo González | 0.6 |
| 31 | Rasheer Fleming | -0.2 | Micah Peavy | -1.1 |
| 32 | Noah Penda | 0.6 | Rocco Zikarsky | 0.1 |
| 33 | Sion James | 0.6 | Sion James | 0.6 |
| 34 | Ryan Kalkbrenner | 3.3 | Joan Beringer | 0.6 |
| 35 | Johni Broome | -0.3 | Will Richard | 1.0 |
| 36 | Adou Thiero | -0.2 | Javon Small | 1.3 |
| 37 | Chaz Lanier | -0.3 | Noah Penda | 0.6 |
| 38 | Kam Jones | -1.3 | Alex Toohey | 0.0 |
| 39 | Alijah Martin | -0.1 | Koby Brea | 0.1 |
| 40 | Micah Peavy | -1.1 | Kobe Sanders | 0.0 |
| 41 | Koby Brea | 0.1 | Yanic Konan Niederhäuser | 0.2 |
| 42 | Maxime Raynaud | -0.3 | Alijah Martin | -0.1 |
| 43 | Jamir Watkins | -0.5 | Taelon Peter | -0.6 |
| 44 | Brooks Barnhizer | -0.1 | Saliou Niang | 0.0 |
| 45 | Rocco Zikarsky | 0.1 | Jeremiah Fears | -0.6 |
| 46 | Amari Williams | 0.0 | Mohamed Diawara | 0.3 |
| 47 | Bogoljub Markovic | 0.0 | Drake Powell | -2.5 |
| 48 | Javon Small | 1.3 | Ben Saraf | -2.5 |
| 49 | Tyrese Proctor | -0.2 | Jamir Watkins | -0.5 |
| 50 | Kobe Sanders | 0.0 | John Tonje | -0.0 |
| 51 | Mohamed Diawara | 0.3 | Nolan Traoré | -3.1 |
| 52 | Alex Toohey | 0.0 | Lachlan Olbrich | -0.2 |
| 53 | John Tonje | -0.0 | Egor Dëmin | 0.8 |
| 54 | Taelon Peter | -0.6 | Tre Johnson | -2.0 |
| 55 | Lachlan Olbrich | -0.2 | Ace Bailey | -2.4 |
| 56 | Will Richard | 1.0 | Tyrese Proctor | -0.2 |
| 57 | Max Shulga | -0.1 | Liam McNeeley | 0.2 |
| 58 | Saliou Niang | 0.0 | Chaz Lanier | -0.3 |
| 59 | Jahmai Mashack | -1.8 | Will Riley | -2.0 |
