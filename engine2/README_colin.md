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
profiles captured before each draft night (1-10 scouting grades, write-up length, and the value of the NBA player the
scout compared him to) and -- CatBoost only -- game-log challenge-response features (rematches, bounce-back, error
persistence, close games). Nothing after draft night is used. The plain 50/50 rank-average of the two models scores
0.504 (`winners/catboost_tabicl_market_bio_scout_comp_m3`); that is the fully clean number. The stacker's weights were fit
on 2013-2018, but its covariate list was chosen after looking at holdout tilts, so the 52% headline is optimistic by about a point.

## Holdout (2019-2025)

| AI model accuracy | NBA scouts accuracy | Drafts the AI won |
|:-----------------:|:-------------------:|:-----------------:|
| **52%**           | 26%                 | **7 of 7**        |

Accuracy = Spearman rank correlation between the order (AI redraft, or the real draft) and the players' realised 5-year WAR
(classes that have played fewer than five seasons are judged on the seasons they have played). Rule: `stackctx w=0.1113,0.1429,0.3846,0.3837,0.1094,-0.0149,-0.0207,0.2859,0.2301,0.1226,0.1570,0.1687,0.0875 on [C03 m3 +mo+rs+person+sc+comp2 x5 | C03 m3 +mo+rs+person+sc+comp2 x5 s5 | T10 o2 m3 +mo+person+sc+comp2 x4 | T10 o2 m3 +mo+person+sc+comp2 x4 s4]`.

## By draft

| Draft | AI model | NBA scouts | Difference |
|:------|---------:|-----------:|-----------:|
| 2019  | 47%      | 40%        | +7         |
| 2020  | 55%      | 35%        | +20         |
| 2021  | 58%      | 41%        | +17         |
| 2022  | 50%      | 24%        | +26         |
| 2023  | 60%      | 10%        | +50         |
| 2024  | 48%      | 15%        | +33         |
| 2025  | 43%      | 17%        | +27         |
| **Mean** | **52%** | **26%**   | **+26**    |

## By player

Every holdout draft, actual order beside the model's reordering of the same names.

### 2019

| # | Actual draft | 5yr WAR | Model redraft | 5yr WAR |
|--:|:-------------|--------:|:--------------|--------:|
| 1 | Zion Williamson | 22.1 | Zion Williamson | 22.1 |
| 2 | Ja Morant | 19.2 | Brandon Clarke | 12.3 |
| 3 | RJ Barrett | 6.1 | Bol Bol | 1.9 |
| 4 | De'Andre Hunter | 2.4 | Jarrett Culver | -1.4 |
| 5 | Darius Garland | 15.3 | Grant Williams | 7.2 |
| 6 | Jarrett Culver | -1.4 | Ja Morant | 19.2 |
| 7 | Coby White | 6.9 | Chuma Okeke | 2.3 |
| 8 | Jaxson Hayes | 2.0 | Matisse Thybulle | 15.6 |
| 9 | Rui Hachimura | 0.1 | Jaxson Hayes | 2.0 |
| 10 | Cam Reddish | -0.9 | Goga Bitadze | 3.7 |
| 11 | Cameron Johnson | 17.2 | De'Andre Hunter | 2.4 |
| 12 | P.J. Washington | 7.0 | Ty Jerome | 1.3 |
| 13 | Tyler Herro | 12.5 | P.J. Washington | 7.0 |
| 14 | Romeo Langford | 0.9 | Nickeil Alexander-Walker | 4.8 |
| 15 | Sekou Doumbouya | -3.6 | Dylan Windler | 1.0 |
| 16 | Chuma Okeke | 2.3 | Bruno Fernando | -0.6 |
| 17 | Nickeil Alexander-Walker | 4.8 | Coby White | 6.9 |
| 18 | Goga Bitadze | 3.7 | Tyler Herro | 12.5 |
| 19 | Luka Šamanić | -0.7 | Nic Claxton | 16.3 |
| 20 | Matisse Thybulle | 15.6 | Cameron Johnson | 17.2 |
| 21 | Brandon Clarke | 12.3 | Daniel Gafford | 13.1 |
| 22 | Grant Williams | 7.2 | Jalen McDaniels | 0.3 |
| 23 | Darius Bazley | -2.2 | Sekou Doumbouya | -3.6 |
| 24 | Ty Jerome | 1.3 | Darius Garland | 15.3 |
| 25 | Nassir Little | 1.2 | Isaiah Roby | -2.1 |
| 26 | Dylan Windler | 1.0 | RJ Barrett | 6.1 |
| 27 | Mfiondu Kabengele | -0.1 | Quinndary Weatherspoon | -0.9 |
| 28 | Jordan Poole | 3.7 | Luka Šamanić | -0.7 |
| 29 | Keldon Johnson | 7.2 | Terance Mann | 11.8 |
| 30 | Kevin Porter Jr. | 7.0 | Keldon Johnson | 7.2 |
| 31 | Nic Claxton | 16.3 | Mfiondu Kabengele | -0.1 |
| 32 | KZ Okpala | -1.2 | Cody Martin | 7.4 |
| 33 | Carsen Edwards | -0.6 | Tremont Waters | -0.3 |
| 34 | Bruno Fernando | -0.6 | Alen Smailagić | -0.2 |
| 35 | Didi Louzada | -0.8 | Carsen Edwards | -0.6 |
| 36 | Cody Martin | 7.4 | Darius Bazley | -2.2 |
| 37 | Deividas Sirvydis | -0.0 | Cam Reddish | -0.9 |
| 38 | Daniel Gafford | 13.1 | Deividas Sirvydis | -0.0 |
| 39 | Alen Smailagić | -0.2 | Didi Louzada | -0.8 |
| 40 | Justin James | -0.2 | Vanja Marinković | 0.0 |
| 41 | Eric Paschall | -2.1 | Dewan Hernandez | 0.1 |
| 42 | Admiral Schofield | -1.0 | Romeo Langford | 0.9 |
| 43 | Jaylen Nowell | 1.0 | Kyle Guy | -0.3 |
| 44 | Bol Bol | 1.9 | Justin James | -0.2 |
| 45 | Isaiah Roby | -2.1 | Eric Paschall | -2.1 |
| 46 | Talen Horton-Tucker | 3.7 | Rui Hachimura | 0.1 |
| 47 | Ignas Brazdeikis | -0.8 | Nassir Little | 1.2 |
| 48 | Terance Mann | 11.8 | Jarrell Brantley | -0.1 |
| 49 | Quinndary Weatherspoon | -0.9 | Talen Horton-Tucker | 3.7 |
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
| 8 | Obi Toppin | 10.2 | Paul Reed | 6.7 |
| 9 | Deni Avdija | 13.4 | Tre Jones | 9.4 |
| 10 | Jalen Smith | 5.6 | Patrick Williams | 1.2 |
| 11 | Devin Vassell | 8.1 | Tyrell Terry | 0.1 |
| 12 | Tyrese Haliburton | 38.5 | LaMelo Ball | 18.1 |
| 13 | Kira Lewis Jr. | -0.0 | Payton Pritchard | 14.7 |
| 14 | Aaron Nesmith | 5.6 | Josh Green | 2.1 |
| 15 | Cole Anthony | 5.8 | Desmond Bane | 26.9 |
| 16 | Isaiah Stewart | 6.0 | Malachi Flynn | 1.4 |
| 17 | Aleksej Pokusevski | -3.1 | Precious Achiuwa | 3.6 |
| 18 | Josh Green | 2.1 | Anthony Edwards | 36.5 |
| 19 | Saddiq Bey | 7.1 | Saddiq Bey | 7.1 |
| 20 | Precious Achiuwa | 3.6 | Tyrese Maxey | 21.9 |
| 21 | Tyrese Maxey | 21.9 | Deni Avdija | 13.4 |
| 22 | Zeke Nnaji | -0.4 | Aaron Nesmith | 5.6 |
| 23 | Leandro Bolmaro | -0.5 | Isaiah Stewart | 6.0 |
| 24 | R.J. Hampton | -4.5 | Udoka Azubuike | 0.1 |
| 25 | Immanuel Quickley | 19.5 | Kira Lewis Jr. | -0.0 |
| 26 | Payton Pritchard | 14.7 | Cole Anthony | 5.8 |
| 27 | Udoka Azubuike | 0.1 | Tyler Bey | -0.4 |
| 28 | Jaden McDaniels | 12.8 | Isaac Okoro | 6.3 |
| 29 | Malachi Flynn | 1.4 | Killian Hayes | -4.1 |
| 30 | Desmond Bane | 26.9 | Zeke Nnaji | -0.4 |
| 31 | Tyrell Terry | 0.1 | Skylar Mays | 0.6 |
| 32 | Vernon Carey Jr. | -0.1 | Reggie Perry | -0.9 |
| 33 | Daniel Oturu | -0.5 | KJ Martin | 0.3 |
| 34 | Théo Maledon | -6.3 | Justinian Jessup | 0.0 |
| 35 | Xavier Tillman Sr. | 5.3 | Vernon Carey Jr. | -0.1 |
| 36 | Tyler Bey | -0.4 | Jahmi'us Ramsey | -0.6 |
| 37 | Vít Krejčí | 0.6 | Marko Simonovic | -0.0 |
| 38 | Saben Lee | 0.0 | Vít Krejčí | 0.6 |
| 39 | Elijah Hughes | -1.8 | CJ Elleby | -1.7 |
| 40 | Robert Woodard II | -0.0 | Yam Madar | 0.0 |
| 41 | Tre Jones | 9.4 | Isaiah Joe | 10.9 |
| 42 | Nick Richards | 0.4 | Grant Riller | 0.0 |
| 43 | Jahmi'us Ramsey | -0.6 | Daniel Oturu | -0.5 |
| 44 | Marko Simonovic | -0.0 | Leandro Bolmaro | -0.5 |
| 45 | Jordan Nwora | -1.9 | Sam Merrill | 4.0 |
| 46 | CJ Elleby | -1.7 | Jalen Harris | 0.3 |
| 47 | Yam Madar | 0.0 | R.J. Hampton | -4.5 |
| 48 | Nico Mannion | -0.3 | Elijah Hughes | -1.8 |
| 49 | Isaiah Joe | 10.9 | Saben Lee | 0.0 |
| 50 | Skylar Mays | 0.6 | Nick Richards | 0.4 |
| 51 | Justinian Jessup | 0.0 | Immanuel Quickley | 19.5 |
| 52 | KJ Martin | 0.3 | Cassius Winston | -0.6 |
| 53 | Cassius Winston | -0.6 | Aleksej Pokusevski | -3.1 |
| 54 | Cassius Stanley | -0.1 | Théo Maledon | -6.3 |
| 55 | Jay Scrubb | -0.5 | Robert Woodard II | -0.0 |
| 56 | Grant Riller | 0.0 | Jordan Nwora | -1.9 |
| 57 | Reggie Perry | -0.9 | Jaden McDaniels | 12.8 |
| 58 | Paul Reed | 6.7 | Jay Scrubb | -0.5 |
| 59 | Jalen Harris | 0.3 | Nico Mannion | -0.3 |
| 60 | Sam Merrill | 4.0 | Cassius Stanley | -0.1 |

### 2021

| # | Actual draft | 5yr WAR | Model redraft | 5yr WAR |
|--:|:-------------|--------:|:--------------|--------:|
| 1 | Cade Cunningham | 22.6 | Evan Mobley | 28.5 |
| 2 | Jalen Green | 7.5 | Franz Wagner | 25.7 |
| 3 | Evan Mobley | 28.5 | Jalen Suggs | 12.8 |
| 4 | Scottie Barnes | 32.0 | Cade Cunningham | 22.6 |
| 5 | Jalen Suggs | 12.8 | Isaiah Jackson | 3.7 |
| 6 | Josh Giddey | 19.6 | Chris Duarte | 1.6 |
| 7 | Jonathan Kuminga | 3.5 | Jalen Johnson | 17.0 |
| 8 | Franz Wagner | 25.7 | Neemias Queta | 7.8 |
| 9 | Davion Mitchell | 5.8 | Scottie Barnes | 32.0 |
| 10 | Ziaire Williams | 1.0 | Kai Jones | 0.6 |
| 11 | James Bouknight | -1.4 | Miles McBride | 6.7 |
| 12 | Joshua Primo | -1.1 | Day'Ron Sharpe | 6.3 |
| 13 | Chris Duarte | 1.6 | Jared Butler | 0.7 |
| 14 | Moses Moody | 5.6 | Corey Kispert | 2.0 |
| 15 | Corey Kispert | 2.0 | Davion Mitchell | 5.8 |
| 16 | Alperen Şengün | 29.5 | Alperen Şengün | 29.5 |
| 17 | Trey Murphy III | 20.7 | Moses Moody | 5.6 |
| 18 | Tre Mann | -1.4 | Jeremiah Robinson-Earl | -1.0 |
| 19 | Kai Jones | 0.6 | Jaden Springer | 0.2 |
| 20 | Jalen Johnson | 17.0 | Luka Garza | 3.7 |
| 21 | Keon Johnson | -3.7 | Scottie Lewis | 0.0 |
| 22 | Isaiah Jackson | 3.7 | Filip Petrušev | -0.0 |
| 23 | Usman Garuba | 0.9 | Herbert Jones | 15.1 |
| 24 | Josh Christopher | -1.8 | Josh Giddey | 19.6 |
| 25 | Quentin Grimes | 9.5 | Bones Hyland | 6.2 |
| 26 | Bones Hyland | 6.2 | James Bouknight | -1.4 |
| 27 | Cam Thomas | 0.7 | Usman Garuba | 0.9 |
| 28 | Jaden Springer | 0.2 | Trey Murphy III | 20.7 |
| 29 | Day'Ron Sharpe | 6.3 | Charles Bassey | 1.8 |
| 30 | Santi Aldama | 11.4 | Sharife Cooper | -1.0 |
| 31 | Isaiah Todd | -0.6 | Quentin Grimes | 9.5 |
| 32 | Jeremiah Robinson-Earl | -1.0 | Jason Preston | 0.0 |
| 33 | Jason Preston | 0.0 | Ayo Dosunmu | 6.6 |
| 34 | Rokas Jokubaitis | 0.0 | Isaiah Livers | -0.5 |
| 35 | Herbert Jones | 15.1 | Marcus Zegarowski | 0.0 |
| 36 | Miles McBride | 6.7 | RaiQuan Gray | 0.2 |
| 37 | JT Thor | -2.2 | Balša Koprivica | 0.0 |
| 38 | Ayo Dosunmu | 6.6 | Joe Wieskamp | -0.4 |
| 39 | Neemias Queta | 7.8 | Josh Christopher | -1.8 |
| 40 | Jared Butler | 0.7 | Sandro Mamukelashvili | 5.6 |
| 41 | Joe Wieskamp | -0.4 | Jericho Sims | 0.9 |
| 42 | Isaiah Livers | -0.5 | Georgios Kalaitzakis | -1.1 |
| 43 | Greg Brown III | -2.2 | Kessler Edwards | -0.2 |
| 44 | Kessler Edwards | -0.2 | Aaron Wiggins | 7.6 |
| 45 | Juhann Begarin | 0.0 | Santi Aldama | 11.4 |
| 46 | Dalano Banton | 2.0 | Juhann Begarin | 0.0 |
| 47 | David Johnson | 0.0 | Dalano Banton | 2.0 |
| 48 | Sharife Cooper | -1.0 | Rokas Jokubaitis | 0.0 |
| 49 | Marcus Zegarowski | 0.0 | Jonathan Kuminga | 3.5 |
| 50 | Filip Petrušev | -0.0 | JT Thor | -2.2 |
| 51 | Brandon Boston Jr. | -1.1 | Jalen Green | 7.5 |
| 52 | Luka Garza | 3.7 | Keon Johnson | -3.7 |
| 53 | Charles Bassey | 1.8 | David Johnson | 0.0 |
| 54 | Sandro Mamukelashvili | 5.6 | Brandon Boston Jr. | -1.1 |
| 55 | Aaron Wiggins | 7.6 | Tre Mann | -1.4 |
| 56 | Scottie Lewis | 0.0 | Ziaire Williams | 1.0 |
| 57 | Balša Koprivica | 0.0 | Cam Thomas | 0.7 |
| 58 | Jericho Sims | 0.9 | Isaiah Todd | -0.6 |
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
| 14 | Ochai Agbaji | -0.8 | Josh Minott | 3.2 |
| 15 | Mark Williams | 8.4 | E.J. Liddell | -0.2 |
| 16 | AJ Griffin | 0.7 | Christian Koloko | -0.3 |
| 17 | Tari Eason | 8.6 | Vince Williams Jr. | 2.5 |
| 18 | Dalen Terry | 0.8 | Andrew Nembhard | 5.1 |
| 19 | Jake LaRavia | 3.5 | Shaedon Sharpe | 2.7 |
| 20 | Malaki Branham | -5.2 | Jaylin Williams | 8.7 |
| 21 | Christian Braun | 8.1 | Bennedict Mathurin | -1.3 |
| 22 | Walker Kessler | 12.9 | TyTy Washington Jr. | -0.9 |
| 23 | David Roddy | -2.4 | Christian Braun | 8.1 |
| 24 | MarJon Beauchamp | -1.2 | Dyson Daniels | 15.4 |
| 25 | Blake Wesley | -3.0 | David Roddy | -2.4 |
| 26 | Wendell Moore Jr. | -0.1 | Kennedy Chandler | -0.6 |
| 27 | Nikola Jović | 3.2 | Jabari Walker | -1.2 |
| 28 | Patrick Baldwin Jr. | -0.1 | Wendell Moore Jr. | -0.1 |
| 29 | TyTy Washington Jr. | -0.9 | Karlo Matković | 1.7 |
| 30 | Peyton Watson | 2.2 | AJ Griffin | 0.7 |
| 31 | Andrew Nembhard | 5.1 | Jaden Ivey | -0.5 |
| 32 | Caleb Houstan | 0.1 | Nikola Jović | 3.2 |
| 33 | Christian Koloko | -0.3 | Ismael Kamagate | 0.0 |
| 34 | Jaylin Williams | 8.7 | Ousmane Dieng | -0.6 |
| 35 | Max Christie | 0.4 | Khalifa Diop | 0.0 |
| 36 | Gabriele Procida | 0.0 | Isaiah Mobley | -0.0 |
| 37 | Jaden Hardy | -2.3 | Malaki Branham | -5.2 |
| 38 | Kennedy Chandler | -0.6 | Gabriele Procida | 0.0 |
| 39 | Khalifa Diop | 0.0 | Peyton Watson | 2.2 |
| 40 | Bryce McGowens | -2.3 | Luke Travers | -0.3 |
| 41 | E.J. Liddell | -0.2 | Ochai Agbaji | -0.8 |
| 42 | Trevor Keels | -0.1 | Kendall Brown | -0.2 |
| 43 | Moussa Diabaté | 6.1 | Ryan Rollins | 3.8 |
| 44 | Ryan Rollins | 3.8 | Gui Santos | 2.4 |
| 45 | Josh Minott | 3.2 | Matteo Spagnolo | 0.0 |
| 46 | Ismael Kamagate | 0.0 | Hugo Besson | 0.0 |
| 47 | Vince Williams Jr. | 2.5 | Tyrese Martin | -1.0 |
| 48 | Kendall Brown | -0.2 | Yannick Nzosa | 0.0 |
| 49 | Isaiah Mobley | -0.0 | Moussa Diabaté | 6.1 |
| 50 | Matteo Spagnolo | 0.0 | MarJon Beauchamp | -1.2 |
| 51 | Tyrese Martin | -1.0 | JD Davison | 0.1 |
| 52 | Karlo Matković | 1.7 | Max Christie | 0.4 |
| 53 | JD Davison | 0.1 | Trevor Keels | -0.1 |
| 54 | Yannick Nzosa | 0.0 | Jaden Hardy | -2.3 |
| 55 | Gui Santos | 2.4 | Blake Wesley | -3.0 |
| 56 | Luke Travers | -0.3 | Bryce McGowens | -2.3 |
| 57 | Jabari Walker | -1.2 | Patrick Baldwin Jr. | -0.1 |
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
| 11 | Jett Howard | -0.4 | Jalen Slawson | 0.3 |
| 12 | Dereck Lively II | 5.7 | Amen Thompson | 20.4 |
| 13 | Gradey Dick | -3.5 | Taylor Hendricks | -0.2 |
| 14 | Jordan Hawkins | -3.5 | Jalen Pickett | 0.5 |
| 15 | Kobe Bufkin | -0.6 | Marcus Sasser | 0.7 |
| 16 | Keyonte George | -1.0 | Jordan Miller | 2.2 |
| 17 | Jalen Hood-Schifino | -1.2 | Kris Murray | 0.4 |
| 18 | Jaime Jaquez Jr. | 5.7 | Ausar Thompson | 10.7 |
| 19 | Brandin Podziemski | 9.1 | Gradey Dick | -3.5 |
| 20 | Cam Whitmore | 2.1 | Scoot Henderson | -1.9 |
| 21 | Noah Clowney | -1.1 | Leonard Miller | 0.9 |
| 22 | Dariq Whitehead | -0.8 | Kobe Brown | -0.3 |
| 23 | Kris Murray | 0.4 | Kobe Bufkin | -0.6 |
| 24 | Olivier-Maxence Prosper | 0.8 | Andre Jackson Jr. | -1.6 |
| 25 | Marcus Sasser | 0.7 | Bilal Coulibaly | -2.9 |
| 26 | Ben Sheppard | 0.3 | Noah Clowney | -1.1 |
| 27 | Nick Smith Jr. | -4.4 | Mouhamed Gueye | 1.5 |
| 28 | Brice Sensabaugh | -0.6 | Toumani Camara | 6.8 |
| 29 | Julian Strawther | -1.6 | Colby Jones | -0.2 |
| 30 | Kobe Brown | -0.3 | Julian Phillips | -0.4 |
| 31 | James Nnaji | 0.0 | Cam Whitmore | 2.1 |
| 32 | Jalen Pickett | 0.5 | James Nnaji | 0.0 |
| 33 | Leonard Miller | 0.9 | Sidy Cissoko | -1.1 |
| 34 | Colby Jones | -0.2 | Hunter Tyson | -0.6 |
| 35 | Julian Phillips | -0.4 | Isaiah Wong | -0.2 |
| 36 | Andre Jackson Jr. | -1.6 | Ben Sheppard | 0.3 |
| 37 | Hunter Tyson | -0.6 | Tristan Vukcevic | 1.1 |
| 38 | Jordan Walsh | 2.1 | Jett Howard | -0.4 |
| 39 | Mouhamed Gueye | 1.5 | Keyonte George | -1.0 |
| 40 | Maxwell Lewis | -0.7 | Mojave King | 0.0 |
| 41 | Amari Bailey | -0.2 | Tarik Biberovic | 0.0 |
| 42 | Tristan Vukcevic | 1.1 | Rayan Rupert | -3.6 |
| 43 | Rayan Rupert | -3.6 | Jordan Walsh | 2.1 |
| 44 | Sidy Cissoko | -1.1 | Seth Lundy | -0.2 |
| 45 | GG Jackson II | -0.0 | Jalen Wilson | -3.1 |
| 46 | Seth Lundy | -0.2 | Olivier-Maxence Prosper | 0.8 |
| 47 | Mojave King | 0.0 | Julian Strawther | -1.6 |
| 48 | Jordan Miller | 2.2 | Keyontae Johnson | -0.2 |
| 49 | Emoni Bates | -0.3 | Amari Bailey | -0.2 |
| 50 | Keyontae Johnson | -0.2 | Brice Sensabaugh | -0.6 |
| 51 | Jalen Wilson | -3.1 | Dariq Whitehead | -0.8 |
| 52 | Toumani Camara | 6.8 | Jalen Hood-Schifino | -1.2 |
| 53 | Jaylen Clark | 1.0 | Jordan Hawkins | -3.5 |
| 54 | Jalen Slawson | 0.3 | Chris Livingston | -0.4 |
| 55 | Isaiah Wong | -0.2 | Maxwell Lewis | -0.7 |
| 56 | Tarik Biberovic | 0.0 | Emoni Bates | -0.3 |
| 57 | Trayce Jackson-Davis | 4.7 | Nick Smith Jr. | -4.4 |
| 58 | Chris Livingston | -0.4 | GG Jackson II | -0.0 |

### 2024

| # | Actual draft | 5yr WAR | Model redraft | 5yr WAR |
|--:|:-------------|--------:|:--------------|--------:|
| 1 | Zaccharie Risacher | -0.8 | Reed Sheppard | 5.9 |
| 2 | Alex Sarr | 0.9 | Jonathan Mogbo | 0.4 |
| 3 | Reed Sheppard | 5.9 | Zach Edey | 3.1 |
| 4 | Stephon Castle | 5.5 | Devin Carter | 0.0 |
| 5 | Ron Holland | 1.1 | Adem Bona | -0.1 |
| 6 | Tidjane Salaün | -1.1 | Donovan Clingan | 8.6 |
| 7 | Donovan Clingan | 8.6 | Tyler Kolek | 1.3 |
| 8 | Rob Dillingham | -2.4 | Cam Spencer | 4.9 |
| 9 | Zach Edey | 3.1 | DaRon Holmes | 0.2 |
| 10 | Cody Williams | -5.5 | Jamal Shead | -0.7 |
| 11 | Matas Buzelis | 1.8 | Anton Watson | -0.1 |
| 12 | Nikola Topić | -0.4 | Alex Sarr | 0.9 |
| 13 | Devin Carter | 0.0 | Stephon Castle | 5.5 |
| 14 | Bub Carrington | -6.6 | Kyle Filipowski | 2.4 |
| 15 | Kel'el Ware | 6.5 | Ryan Dunn | 0.2 |
| 16 | Jared McCain | 0.7 | Matas Buzelis | 1.8 |
| 17 | Dalton Knecht | 0.3 | Dillon Jones | -0.1 |
| 18 | Tristan Da Silva | 1.7 | Oso Ighodaro | 3.3 |
| 19 | Ja'Kobe Walter | 2.1 | Tristan Da Silva | 1.7 |
| 20 | Jaylon Tyson | 2.9 | Kel'el Ware | 6.5 |
| 21 | Yves Missi | -0.6 | Tidjane Salaün | -1.1 |
| 22 | DaRon Holmes | 0.2 | Jaylon Tyson | 2.9 |
| 23 | AJ Johnson | -3.3 | Jared McCain | 0.7 |
| 24 | Kyshawn George | 0.2 | KJ Simpson | -1.4 |
| 25 | Pacôme Dadiet | -0.1 | Nikola Topić | -0.4 |
| 26 | Dillon Jones | -0.1 | Baylor Scheierman | 3.2 |
| 27 | Terrence Shannon Jr. | 0.1 | Ariel Hukporti | -0.1 |
| 28 | Ryan Dunn | 0.2 | Enrique Freeman | -0.2 |
| 29 | Isaiah Collier | -2.0 | Kevin McCullar Jr. | -0.1 |
| 30 | Baylor Scheierman | 3.2 | Harrison Ingram | 0.1 |
| 31 | Jonathan Mogbo | 0.4 | Zaccharie Risacher | -0.8 |
| 32 | Kyle Filipowski | 2.4 | Tristen Newton | -0.1 |
| 33 | Tyler Smith | -0.1 | Quinten Post | 1.6 |
| 34 | Tyler Kolek | 1.3 | Ron Holland | 1.1 |
| 35 | Johnny Furphy | -0.2 | Pelle Larsson | 2.1 |
| 36 | Juan Nunez | 0.0 | Terrence Shannon Jr. | 0.1 |
| 37 | Bobi Klintman | -0.1 | Tyler Smith | -0.1 |
| 38 | Ajay Mitchell | 4.5 | Yves Missi | -0.6 |
| 39 | Jaylen Wells | 1.2 | Kyshawn George | 0.2 |
| 40 | Oso Ighodaro | 3.3 | Ajay Mitchell | 4.5 |
| 41 | Adem Bona | -0.1 | Juan Nunez | 0.0 |
| 42 | KJ Simpson | -1.4 | Johnny Furphy | -0.2 |
| 43 | Nikola Djurisic | 0.0 | Nikola Djurisic | 0.0 |
| 44 | Pelle Larsson | 2.1 | Jaylen Wells | 1.2 |
| 45 | Jamal Shead | -0.7 | Melvin Ajinça | 0.0 |
| 46 | Cam Christie | -0.4 | Rob Dillingham | -2.4 |
| 47 | Antonio Reeves | -0.2 | Ulrich Chomche | -0.1 |
| 48 | Harrison Ingram | 0.1 | Cody Williams | -5.5 |
| 49 | Tristen Newton | -0.1 | AJ Johnson | -3.3 |
| 50 | Enrique Freeman | -0.2 | Dalton Knecht | 0.3 |
| 51 | Melvin Ajinça | 0.0 | Antonio Reeves | -0.2 |
| 52 | Quinten Post | 1.6 | Bub Carrington | -6.6 |
| 53 | Cam Spencer | 4.9 | Bronny James | -0.6 |
| 54 | Anton Watson | -0.1 | Ja'Kobe Walter | 2.1 |
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
| 4 | Kon Knueppel | 7.4 | VJ Edgecombe | 4.3 |
| 5 | Ace Bailey | -2.4 | Kon Knueppel | 7.4 |
| 6 | Tre Johnson | -2.0 | Ryan Kalkbrenner | 3.3 |
| 7 | Jeremiah Fears | -0.6 | Dylan Harper | 4.4 |
| 8 | Egor Dëmin | 0.8 | Cedric Coward | 2.3 |
| 9 | Collin Murray-Boyles | 2.7 | Johni Broome | -0.3 |
| 10 | Khaman Maluach | -0.3 | Asa Newell | 0.3 |
| 11 | Cedric Coward | 2.3 | Nique Clifford | -2.7 |
| 12 | Noa Essengue | -0.1 | Derik Queen | 2.3 |
| 13 | Derik Queen | 2.3 | Khaman Maluach | -0.3 |
| 14 | Carter Bryant | 0.1 | Adou Thiero | -0.2 |
| 15 | Thomas Sorber | 0.0 | Noa Essengue | -0.1 |
| 16 | Yang Hansen | -1.0 | Carter Bryant | 0.1 |
| 17 | Joan Beringer | 0.6 | Kasparas Jakučionis | 1.0 |
| 18 | Walter Clayton | -1.4 | Danny Wolf | -0.1 |
| 19 | Nolan Traoré | -3.1 | Rasheer Fleming | -0.2 |
| 20 | Kasparas Jakučionis | 1.0 | Jase Richardson | 0.1 |
| 21 | Will Riley | -2.0 | Bogoljub Markovic | 0.0 |
| 22 | Drake Powell | -2.5 | Max Shulga | -0.1 |
| 23 | Asa Newell | 0.3 | Kam Jones | -1.3 |
| 24 | Nique Clifford | -2.7 | Yang Hansen | -1.0 |
| 25 | Jase Richardson | 0.1 | Brooks Barnhizer | -0.1 |
| 26 | Ben Saraf | -2.5 | Amari Williams | 0.0 |
| 27 | Danny Wolf | -0.1 | Walter Clayton | -1.4 |
| 28 | Hugo González | 0.6 | Jahmai Mashack | -1.8 |
| 29 | Liam McNeeley | 0.2 | Maxime Raynaud | -0.3 |
| 30 | Yanic Konan Niederhäuser | 0.2 | Hugo González | 0.6 |
| 31 | Rasheer Fleming | -0.2 | Sion James | 0.6 |
| 32 | Noah Penda | 0.6 | Micah Peavy | -1.1 |
| 33 | Sion James | 0.6 | Rocco Zikarsky | 0.1 |
| 34 | Ryan Kalkbrenner | 3.3 | Javon Small | 1.3 |
| 35 | Johni Broome | -0.3 | Will Richard | 1.0 |
| 36 | Adou Thiero | -0.2 | Noah Penda | 0.6 |
| 37 | Chaz Lanier | -0.3 | Alex Toohey | 0.0 |
| 38 | Kam Jones | -1.3 | Koby Brea | 0.1 |
| 39 | Alijah Martin | -0.1 | Joan Beringer | 0.6 |
| 40 | Micah Peavy | -1.1 | Kobe Sanders | 0.0 |
| 41 | Koby Brea | 0.1 | Yanic Konan Niederhäuser | 0.2 |
| 42 | Maxime Raynaud | -0.3 | Alijah Martin | -0.1 |
| 43 | Jamir Watkins | -0.5 | Saliou Niang | 0.0 |
| 44 | Brooks Barnhizer | -0.1 | Jeremiah Fears | -0.6 |
| 45 | Rocco Zikarsky | 0.1 | Taelon Peter | -0.6 |
| 46 | Amari Williams | 0.0 | Mohamed Diawara | 0.3 |
| 47 | Bogoljub Markovic | 0.0 | Jamir Watkins | -0.5 |
| 48 | Javon Small | 1.3 | Drake Powell | -2.5 |
| 49 | Tyrese Proctor | -0.2 | Ben Saraf | -2.5 |
| 50 | Kobe Sanders | 0.0 | Nolan Traoré | -3.1 |
| 51 | Mohamed Diawara | 0.3 | John Tonje | -0.0 |
| 52 | Alex Toohey | 0.0 | Lachlan Olbrich | -0.2 |
| 53 | John Tonje | -0.0 | Egor Dëmin | 0.8 |
| 54 | Taelon Peter | -0.6 | Tyrese Proctor | -0.2 |
| 55 | Lachlan Olbrich | -0.2 | Ace Bailey | -2.4 |
| 56 | Will Richard | 1.0 | Tre Johnson | -2.0 |
| 57 | Max Shulga | -0.1 | Liam McNeeley | 0.2 |
| 58 | Saliou Niang | 0.0 | Chaz Lanier | -0.3 |
| 59 | Jahmai Mashack | -1.8 | Will Riley | -2.0 |
