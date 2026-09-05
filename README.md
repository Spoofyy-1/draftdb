# nba-redraft

## AI models used

Two walk-forward models plus a small fixed set of pre-draft signals, combined by a linear stacker whose weights were fit
on the 2013-2018 context years only (nothing about the holdout years chose them):

| Input | What it is |
|:------|:-----------|
| **CatBoost** | gradient-boosted trees (depth 5, 800 iterations, 10 seeds), context = every draft class from 2003 before the one being scored |
| **TabICL v2** | tabular in-context-learning foundation model (`TabICLRegressor`, 8 estimators, 8 seeds), context from 2010 |
| covariates | within-class ranks of true shooting, eFG, RSCI rank, G League translation prior, NBA relatives, a shooting composite (FT% + attempt-shrunk 3P%), international league-relative production x youth, steal rate, usage |

Both models learn a label that matches how each class is judged: WAR over a player's first max(3, N) NBA seasons, N =
seasons the scored class has played (2019-2021: five, 2022: four, 2023-2025: three), Gaussian-ranked within class. Model
features, all knowable before draft night: Torvik final season + trajectory, physicals, NBA combine, international pro /
FIBA youth lines, Torvik team context, a 5-8 board pre-draft mock consensus with its 90/60/30/7-day movement (Wayback
captures), basketball-reference biography (NBA relatives, shooting hand, birthplace, high-school path) and -- CatBoost
only -- game-log challenge-response features (rematches, bounce-back, error persistence, close games). Nothing after
draft night is used. The plain 50/50 rank-average of the two models scores 0.495 (`winners/catboost_tabicl_market_bio_m3`).

## Holdout (2019-2025)

| AI model accuracy | NBA scouts accuracy | Drafts the AI won |
|:-----------------:|:-------------------:|:-----------------:|
| **51%**           | 26%                 | **7 of 7**        |

Accuracy = Spearman rank correlation between the order (AI redraft, or the real draft) and the players' realised 5-year WAR
(classes that have played fewer than five seasons are judged on the seasons they have played). Rule: `stackctx w=0.1394,0.0413,0.3975,0.4651,0.1493,-0.0404,-0.0229,0.3325,0.3141,0.1395,0.2790,0.1974,0.1067 on [C03 m3 +mo+rs+person x5 | C03 m3 +mo+rs+person x5 s5 | T10 o2 m3 +mo+person x4 | T10 o2 m3 +mo+person x4 s4]`.

## By draft

| Draft | AI model | NBA scouts | Difference |
|:------|---------:|-----------:|-----------:|
| 2019  | 48%      | 40%        | +8         |
| 2020  | 54%      | 35%        | +19         |
| 2021  | 54%      | 41%        | +13         |
| 2022  | 50%      | 24%        | +26         |
| 2023  | 60%      | 10%        | +49         |
| 2024  | 46%      | 15%        | +31         |
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
| 5 | Darius Garland | 15.3 | Ja Morant | 19.2 |
| 6 | Jarrett Culver | -1.4 | Jarrett Culver | -1.4 |
| 7 | Coby White | 6.9 | Matisse Thybulle | 15.6 |
| 8 | Jaxson Hayes | 2.0 | Chuma Okeke | 2.3 |
| 9 | Rui Hachimura | 0.1 | Nickeil Alexander-Walker | 4.8 |
| 10 | Cam Reddish | -0.9 | Jaxson Hayes | 2.0 |
| 11 | Cameron Johnson | 17.2 | Goga Bitadze | 3.7 |
| 12 | P.J. Washington | 7.0 | Tyler Herro | 12.5 |
| 13 | Tyler Herro | 12.5 | Dylan Windler | 1.0 |
| 14 | Romeo Langford | 0.9 | P.J. Washington | 7.0 |
| 15 | Sekou Doumbouya | -3.6 | Ty Jerome | 1.3 |
| 16 | Chuma Okeke | 2.3 | De'Andre Hunter | 2.4 |
| 17 | Nickeil Alexander-Walker | 4.8 | Nic Claxton | 16.3 |
| 18 | Goga Bitadze | 3.7 | Bruno Fernando | -0.6 |
| 19 | Luka Šamanić | -0.7 | Coby White | 6.9 |
| 20 | Matisse Thybulle | 15.6 | Cameron Johnson | 17.2 |
| 21 | Brandon Clarke | 12.3 | Sekou Doumbouya | -3.6 |
| 22 | Grant Williams | 7.2 | Daniel Gafford | 13.1 |
| 23 | Darius Bazley | -2.2 | Darius Garland | 15.3 |
| 24 | Ty Jerome | 1.3 | Jalen McDaniels | 0.3 |
| 25 | Nassir Little | 1.2 | Quinndary Weatherspoon | -0.9 |
| 26 | Dylan Windler | 1.0 | Isaiah Roby | -2.1 |
| 27 | Mfiondu Kabengele | -0.1 | RJ Barrett | 6.1 |
| 28 | Jordan Poole | 3.7 | Luka Šamanić | -0.7 |
| 29 | Keldon Johnson | 7.2 | Cody Martin | 7.4 |
| 30 | Kevin Porter Jr. | 7.0 | Terance Mann | 11.8 |
| 31 | Nic Claxton | 16.3 | Keldon Johnson | 7.2 |
| 32 | KZ Okpala | -1.2 | Mfiondu Kabengele | -0.1 |
| 33 | Carsen Edwards | -0.6 | Alen Smailagić | -0.2 |
| 34 | Bruno Fernando | -0.6 | Tremont Waters | -0.3 |
| 35 | Didi Louzada | -0.8 | Carsen Edwards | -0.6 |
| 36 | Cody Martin | 7.4 | Cam Reddish | -0.9 |
| 37 | Deividas Sirvydis | -0.0 | Vanja Marinković | 0.0 |
| 38 | Daniel Gafford | 13.1 | Darius Bazley | -2.2 |
| 39 | Alen Smailagić | -0.2 | Deividas Sirvydis | -0.0 |
| 40 | Justin James | -0.2 | Didi Louzada | -0.8 |
| 41 | Eric Paschall | -2.1 | Kyle Guy | -0.3 |
| 42 | Admiral Schofield | -1.0 | Dewan Hernandez | 0.1 |
| 43 | Jaylen Nowell | 1.0 | Romeo Langford | 0.9 |
| 44 | Bol Bol | 1.9 | Eric Paschall | -2.1 |
| 45 | Isaiah Roby | -2.1 | Justin James | -0.2 |
| 46 | Talen Horton-Tucker | 3.7 | Jarrell Brantley | -0.1 |
| 47 | Ignas Brazdeikis | -0.8 | Talen Horton-Tucker | 3.7 |
| 48 | Terance Mann | 11.8 | Nassir Little | 1.2 |
| 49 | Quinndary Weatherspoon | -0.9 | Rui Hachimura | 0.1 |
| 50 | Jarrell Brantley | -0.1 | Jordan Poole | 3.7 |
| 51 | Tremont Waters | -0.3 | Marial Shayok | -0.1 |
| 52 | Jalen McDaniels | 0.3 | Jaylen Nowell | 1.0 |
| 53 | Justin Wright-Foreman | -0.2 | Ignas Brazdeikis | -0.8 |
| 54 | Marial Shayok | -0.1 | Justin Wright-Foreman | -0.2 |
| 55 | Kyle Guy | -0.3 | Jordan Bone | -0.8 |
| 56 | Jaylen Hands | 0.0 | Admiral Schofield | -1.0 |
| 57 | Jordan Bone | -0.8 | Kevin Porter Jr. | 7.0 |
| 58 | Miye Oni | -0.0 | Miye Oni | -0.0 |
| 59 | Dewan Hernandez | 0.1 | Jaylen Hands | 0.0 |
| 60 | Vanja Marinković | 0.0 | KZ Okpala | -1.2 |

### 2020

| # | Actual draft | 5yr WAR | Model redraft | 5yr WAR |
|--:|:-------------|--------:|:--------------|--------:|
| 1 | Anthony Edwards | 36.5 | Tyrese Haliburton | 38.5 |
| 2 | James Wiseman | -4.2 | Onyeka Okongwu | 14.5 |
| 3 | LaMelo Ball | 18.1 | James Wiseman | -4.2 |
| 4 | Patrick Williams | 1.2 | Obi Toppin | 10.2 |
| 5 | Isaac Okoro | 6.3 | LaMelo Ball | 18.1 |
| 6 | Onyeka Okongwu | 14.5 | Xavier Tillman Sr. | 5.3 |
| 7 | Killian Hayes | -4.1 | Jalen Smith | 5.6 |
| 8 | Obi Toppin | 10.2 | Devin Vassell | 8.1 |
| 9 | Deni Avdija | 13.4 | Paul Reed | 6.7 |
| 10 | Jalen Smith | 5.6 | Tre Jones | 9.4 |
| 11 | Devin Vassell | 8.1 | Patrick Williams | 1.2 |
| 12 | Tyrese Haliburton | 38.5 | Josh Green | 2.1 |
| 13 | Kira Lewis Jr. | -0.0 | Malachi Flynn | 1.4 |
| 14 | Aaron Nesmith | 5.6 | Desmond Bane | 26.9 |
| 15 | Cole Anthony | 5.8 | Tyrell Terry | 0.1 |
| 16 | Isaiah Stewart | 6.0 | Payton Pritchard | 14.7 |
| 17 | Aleksej Pokusevski | -3.1 | Saddiq Bey | 7.1 |
| 18 | Josh Green | 2.1 | Precious Achiuwa | 3.6 |
| 19 | Saddiq Bey | 7.1 | Anthony Edwards | 36.5 |
| 20 | Precious Achiuwa | 3.6 | Kira Lewis Jr. | -0.0 |
| 21 | Tyrese Maxey | 21.9 | Udoka Azubuike | 0.1 |
| 22 | Zeke Nnaji | -0.4 | Cole Anthony | 5.8 |
| 23 | Leandro Bolmaro | -0.5 | Isaiah Stewart | 6.0 |
| 24 | R.J. Hampton | -4.5 | Aaron Nesmith | 5.6 |
| 25 | Immanuel Quickley | 19.5 | Killian Hayes | -4.1 |
| 26 | Payton Pritchard | 14.7 | Deni Avdija | 13.4 |
| 27 | Udoka Azubuike | 0.1 | Tyler Bey | -0.4 |
| 28 | Jaden McDaniels | 12.8 | Tyrese Maxey | 21.9 |
| 29 | Malachi Flynn | 1.4 | Reggie Perry | -0.9 |
| 30 | Desmond Bane | 26.9 | Skylar Mays | 0.6 |
| 31 | Tyrell Terry | 0.1 | KJ Martin | 0.3 |
| 32 | Vernon Carey Jr. | -0.1 | Zeke Nnaji | -0.4 |
| 33 | Daniel Oturu | -0.5 | Marko Simonovic | -0.0 |
| 34 | Théo Maledon | -6.3 | Justinian Jessup | 0.0 |
| 35 | Xavier Tillman Sr. | 5.3 | Isaac Okoro | 6.3 |
| 36 | Tyler Bey | -0.4 | Vernon Carey Jr. | -0.1 |
| 37 | Vít Krejčí | 0.6 | Vít Krejčí | 0.6 |
| 38 | Saben Lee | 0.0 | Yam Madar | 0.0 |
| 39 | Elijah Hughes | -1.8 | CJ Elleby | -1.7 |
| 40 | Robert Woodard II | -0.0 | Jahmi'us Ramsey | -0.6 |
| 41 | Tre Jones | 9.4 | Daniel Oturu | -0.5 |
| 42 | Nick Richards | 0.4 | Grant Riller | 0.0 |
| 43 | Jahmi'us Ramsey | -0.6 | Isaiah Joe | 10.9 |
| 44 | Marko Simonovic | -0.0 | Nick Richards | 0.4 |
| 45 | Jordan Nwora | -1.9 | Sam Merrill | 4.0 |
| 46 | CJ Elleby | -1.7 | Saben Lee | 0.0 |
| 47 | Yam Madar | 0.0 | Jalen Harris | 0.3 |
| 48 | Nico Mannion | -0.3 | Leandro Bolmaro | -0.5 |
| 49 | Isaiah Joe | 10.9 | Jaden McDaniels | 12.8 |
| 50 | Skylar Mays | 0.6 | Elijah Hughes | -1.8 |
| 51 | Justinian Jessup | 0.0 | Jay Scrubb | -0.5 |
| 52 | KJ Martin | 0.3 | R.J. Hampton | -4.5 |
| 53 | Cassius Winston | -0.6 | Immanuel Quickley | 19.5 |
| 54 | Cassius Stanley | -0.1 | Théo Maledon | -6.3 |
| 55 | Jay Scrubb | -0.5 | Aleksej Pokusevski | -3.1 |
| 56 | Grant Riller | 0.0 | Nico Mannion | -0.3 |
| 57 | Reggie Perry | -0.9 | Robert Woodard II | -0.0 |
| 58 | Paul Reed | 6.7 | Jordan Nwora | -1.9 |
| 59 | Jalen Harris | 0.3 | Cassius Winston | -0.6 |
| 60 | Sam Merrill | 4.0 | Cassius Stanley | -0.1 |

### 2021

| # | Actual draft | 5yr WAR | Model redraft | 5yr WAR |
|--:|:-------------|--------:|:--------------|--------:|
| 1 | Cade Cunningham | 22.6 | Evan Mobley | 28.5 |
| 2 | Jalen Green | 7.5 | Franz Wagner | 25.7 |
| 3 | Evan Mobley | 28.5 | Jalen Suggs | 12.8 |
| 4 | Scottie Barnes | 32.0 | Cade Cunningham | 22.6 |
| 5 | Jalen Suggs | 12.8 | Isaiah Jackson | 3.7 |
| 6 | Josh Giddey | 19.6 | Jalen Johnson | 17.0 |
| 7 | Jonathan Kuminga | 3.5 | Chris Duarte | 1.6 |
| 8 | Franz Wagner | 25.7 | Neemias Queta | 7.8 |
| 9 | Davion Mitchell | 5.8 | Scottie Barnes | 32.0 |
| 10 | Ziaire Williams | 1.0 | Kai Jones | 0.6 |
| 11 | James Bouknight | -1.4 | Miles McBride | 6.7 |
| 12 | Joshua Primo | -1.1 | Day'Ron Sharpe | 6.3 |
| 13 | Chris Duarte | 1.6 | Alperen Şengün | 29.5 |
| 14 | Moses Moody | 5.6 | Jared Butler | 0.7 |
| 15 | Corey Kispert | 2.0 | Davion Mitchell | 5.8 |
| 16 | Alperen Şengün | 29.5 | Corey Kispert | 2.0 |
| 17 | Trey Murphy III | 20.7 | Jeremiah Robinson-Earl | -1.0 |
| 18 | Tre Mann | -1.4 | Filip Petrušev | -0.0 |
| 19 | Kai Jones | 0.6 | Luka Garza | 3.7 |
| 20 | Jalen Johnson | 17.0 | Moses Moody | 5.6 |
| 21 | Keon Johnson | -3.7 | Jaden Springer | 0.2 |
| 22 | Isaiah Jackson | 3.7 | Charles Bassey | 1.8 |
| 23 | Usman Garuba | 0.9 | Herbert Jones | 15.1 |
| 24 | Josh Christopher | -1.8 | James Bouknight | -1.4 |
| 25 | Quentin Grimes | 9.5 | Scottie Lewis | 0.0 |
| 26 | Bones Hyland | 6.2 | Usman Garuba | 0.9 |
| 27 | Cam Thomas | 0.7 | Ayo Dosunmu | 6.6 |
| 28 | Jaden Springer | 0.2 | Bones Hyland | 6.2 |
| 29 | Day'Ron Sharpe | 6.3 | Trey Murphy III | 20.7 |
| 30 | Santi Aldama | 11.4 | Josh Christopher | -1.8 |
| 31 | Isaiah Todd | -0.6 | Isaiah Livers | -0.5 |
| 32 | Jeremiah Robinson-Earl | -1.0 | Quentin Grimes | 9.5 |
| 33 | Jason Preston | 0.0 | Marcus Zegarowski | 0.0 |
| 34 | Rokas Jokubaitis | 0.0 | Joe Wieskamp | -0.4 |
| 35 | Herbert Jones | 15.1 | Jason Preston | 0.0 |
| 36 | Miles McBride | 6.7 | Sharife Cooper | -1.0 |
| 37 | JT Thor | -2.2 | RaiQuan Gray | 0.2 |
| 38 | Ayo Dosunmu | 6.6 | Balša Koprivica | 0.0 |
| 39 | Neemias Queta | 7.8 | Josh Giddey | 19.6 |
| 40 | Jared Butler | 0.7 | Sandro Mamukelashvili | 5.6 |
| 41 | Joe Wieskamp | -0.4 | Kessler Edwards | -0.2 |
| 42 | Isaiah Livers | -0.5 | Jericho Sims | 0.9 |
| 43 | Greg Brown III | -2.2 | Aaron Wiggins | 7.6 |
| 44 | Kessler Edwards | -0.2 | Georgios Kalaitzakis | -1.1 |
| 45 | Juhann Begarin | 0.0 | JT Thor | -2.2 |
| 46 | Dalano Banton | 2.0 | Santi Aldama | 11.4 |
| 47 | David Johnson | 0.0 | Juhann Begarin | 0.0 |
| 48 | Sharife Cooper | -1.0 | Jalen Green | 7.5 |
| 49 | Marcus Zegarowski | 0.0 | Dalano Banton | 2.0 |
| 50 | Filip Petrušev | -0.0 | Jonathan Kuminga | 3.5 |
| 51 | Brandon Boston Jr. | -1.1 | Brandon Boston Jr. | -1.1 |
| 52 | Luka Garza | 3.7 | Keon Johnson | -3.7 |
| 53 | Charles Bassey | 1.8 | Rokas Jokubaitis | 0.0 |
| 54 | Sandro Mamukelashvili | 5.6 | Tre Mann | -1.4 |
| 55 | Aaron Wiggins | 7.6 | David Johnson | 0.0 |
| 56 | Scottie Lewis | 0.0 | Ziaire Williams | 1.0 |
| 57 | Balša Koprivica | 0.0 | Isaiah Todd | -0.6 |
| 58 | Jericho Sims | 0.9 | Greg Brown III | -2.2 |
| 59 | RaiQuan Gray | 0.2 | Cam Thomas | 0.7 |
| 60 | Georgios Kalaitzakis | -1.1 | Joshua Primo | -1.1 |

### 2022

| # | Actual draft | 5yr WAR | Model redraft | 5yr WAR |
|--:|:-------------|--------:|:--------------|--------:|
| 1 | Paolo Banchero | 18.5 | Keegan Murray | 8.4 |
| 2 | Chet Holmgren | 19.9 | Tari Eason | 8.6 |
| 3 | Jabari Smith Jr. | 7.5 | Jabari Smith Jr. | 7.5 |
| 4 | Keegan Murray | 8.4 | Walker Kessler | 12.9 |
| 5 | Jaden Ivey | -0.5 | Jalen Duren | 17.1 |
| 6 | Bennedict Mathurin | -1.3 | Chet Holmgren | 19.9 |
| 7 | Shaedon Sharpe | 2.7 | Jake LaRavia | 3.5 |
| 8 | Dyson Daniels | 15.4 | Mark Williams | 8.4 |
| 9 | Jeremy Sochan | -0.1 | Jeremy Sochan | -0.1 |
| 10 | Johnny Davis | -1.9 | Johnny Davis | -1.9 |
| 11 | Ousmane Dieng | -0.6 | Paolo Banchero | 18.5 |
| 12 | Jalen Williams | 22.4 | Christian Koloko | -0.3 |
| 13 | Jalen Duren | 17.1 | Jalen Williams | 22.4 |
| 14 | Ochai Agbaji | -0.8 | E.J. Liddell | -0.2 |
| 15 | Mark Williams | 8.4 | Josh Minott | 3.2 |
| 16 | AJ Griffin | 0.7 | Dalen Terry | 0.8 |
| 17 | Tari Eason | 8.6 | Shaedon Sharpe | 2.7 |
| 18 | Dalen Terry | 0.8 | Vince Williams Jr. | 2.5 |
| 19 | Jake LaRavia | 3.5 | AJ Griffin | 0.7 |
| 20 | Malaki Branham | -5.2 | Andrew Nembhard | 5.1 |
| 21 | Christian Braun | 8.1 | Jaylin Williams | 8.7 |
| 22 | Walker Kessler | 12.9 | Dyson Daniels | 15.4 |
| 23 | David Roddy | -2.4 | Bennedict Mathurin | -1.3 |
| 24 | MarJon Beauchamp | -1.2 | Karlo Matković | 1.7 |
| 25 | Blake Wesley | -3.0 | Kennedy Chandler | -0.6 |
| 26 | Wendell Moore Jr. | -0.1 | Christian Braun | 8.1 |
| 27 | Nikola Jović | 3.2 | David Roddy | -2.4 |
| 28 | Patrick Baldwin Jr. | -0.1 | Wendell Moore Jr. | -0.1 |
| 29 | TyTy Washington Jr. | -0.9 | Jabari Walker | -1.2 |
| 30 | Peyton Watson | 2.2 | TyTy Washington Jr. | -0.9 |
| 31 | Andrew Nembhard | 5.1 | Jaden Ivey | -0.5 |
| 32 | Caleb Houstan | 0.1 | Ismael Kamagate | 0.0 |
| 33 | Christian Koloko | -0.3 | Isaiah Mobley | -0.0 |
| 34 | Jaylin Williams | 8.7 | Khalifa Diop | 0.0 |
| 35 | Max Christie | 0.4 | Nikola Jović | 3.2 |
| 36 | Gabriele Procida | 0.0 | Ousmane Dieng | -0.6 |
| 37 | Jaden Hardy | -2.3 | Kendall Brown | -0.2 |
| 38 | Kennedy Chandler | -0.6 | Ochai Agbaji | -0.8 |
| 39 | Khalifa Diop | 0.0 | Gabriele Procida | 0.0 |
| 40 | Bryce McGowens | -2.3 | Malaki Branham | -5.2 |
| 41 | E.J. Liddell | -0.2 | Luke Travers | -0.3 |
| 42 | Trevor Keels | -0.1 | Peyton Watson | 2.2 |
| 43 | Moussa Diabaté | 6.1 | Gui Santos | 2.4 |
| 44 | Ryan Rollins | 3.8 | Hugo Besson | 0.0 |
| 45 | Josh Minott | 3.2 | Ryan Rollins | 3.8 |
| 46 | Ismael Kamagate | 0.0 | Tyrese Martin | -1.0 |
| 47 | Vince Williams Jr. | 2.5 | MarJon Beauchamp | -1.2 |
| 48 | Kendall Brown | -0.2 | Matteo Spagnolo | 0.0 |
| 49 | Isaiah Mobley | -0.0 | JD Davison | 0.1 |
| 50 | Matteo Spagnolo | 0.0 | Moussa Diabaté | 6.1 |
| 51 | Tyrese Martin | -1.0 | Max Christie | 0.4 |
| 52 | Karlo Matković | 1.7 | Trevor Keels | -0.1 |
| 53 | JD Davison | 0.1 | Yannick Nzosa | 0.0 |
| 54 | Yannick Nzosa | 0.0 | Jaden Hardy | -2.3 |
| 55 | Gui Santos | 2.4 | Blake Wesley | -3.0 |
| 56 | Luke Travers | -0.3 | Patrick Baldwin Jr. | -0.1 |
| 57 | Jabari Walker | -1.2 | Bryce McGowens | -2.3 |
| 58 | Hugo Besson | 0.0 | Caleb Houstan | 0.1 |

### 2023

| # | Actual draft | 5yr WAR | Model redraft | 5yr WAR |
|--:|:-------------|--------:|:--------------|--------:|
| 1 | Victor Wembanyama | 32.1 | Trayce Jackson-Davis | 4.7 |
| 2 | Brandon Miller | 6.0 | Victor Wembanyama | 32.1 |
| 3 | Scoot Henderson | -1.9 | Anthony Black | 4.2 |
| 4 | Amen Thompson | 20.4 | Jaime Jaquez Jr. | 5.7 |
| 5 | Ausar Thompson | 10.7 | Cason Wallace | 13.7 |
| 6 | Anthony Black | 4.2 | Dereck Lively II | 5.7 |
| 7 | Bilal Coulibaly | -2.9 | Brandon Miller | 6.0 |
| 8 | Jarace Walker | 0.3 | Brandin Podziemski | 9.1 |
| 9 | Taylor Hendricks | -0.2 | Jarace Walker | 0.3 |
| 10 | Cason Wallace | 13.7 | Amen Thompson | 20.4 |
| 11 | Jett Howard | -0.4 | Jaylen Clark | 1.0 |
| 12 | Dereck Lively II | 5.7 | Jalen Slawson | 0.3 |
| 13 | Gradey Dick | -3.5 | Kris Murray | 0.4 |
| 14 | Jordan Hawkins | -3.5 | Scoot Henderson | -1.9 |
| 15 | Kobe Bufkin | -0.6 | Marcus Sasser | 0.7 |
| 16 | Keyonte George | -1.0 | Jalen Pickett | 0.5 |
| 17 | Jalen Hood-Schifino | -1.2 | Taylor Hendricks | -0.2 |
| 18 | Jaime Jaquez Jr. | 5.7 | Jordan Miller | 2.2 |
| 19 | Brandin Podziemski | 9.1 | Leonard Miller | 0.9 |
| 20 | Cam Whitmore | 2.1 | Kobe Brown | -0.3 |
| 21 | Noah Clowney | -1.1 | Ausar Thompson | 10.7 |
| 22 | Dariq Whitehead | -0.8 | Gradey Dick | -3.5 |
| 23 | Kris Murray | 0.4 | Andre Jackson Jr. | -1.6 |
| 24 | Olivier-Maxence Prosper | 0.8 | Bilal Coulibaly | -2.9 |
| 25 | Marcus Sasser | 0.7 | Kobe Bufkin | -0.6 |
| 26 | Ben Sheppard | 0.3 | Toumani Camara | 6.8 |
| 27 | Nick Smith Jr. | -4.4 | Colby Jones | -0.2 |
| 28 | Brice Sensabaugh | -0.6 | Mouhamed Gueye | 1.5 |
| 29 | Julian Strawther | -1.6 | James Nnaji | 0.0 |
| 30 | Kobe Brown | -0.3 | Cam Whitmore | 2.1 |
| 31 | James Nnaji | 0.0 | Noah Clowney | -1.1 |
| 32 | Jalen Pickett | 0.5 | Julian Phillips | -0.4 |
| 33 | Leonard Miller | 0.9 | Hunter Tyson | -0.6 |
| 34 | Colby Jones | -0.2 | Isaiah Wong | -0.2 |
| 35 | Julian Phillips | -0.4 | Sidy Cissoko | -1.1 |
| 36 | Andre Jackson Jr. | -1.6 | Keyonte George | -1.0 |
| 37 | Hunter Tyson | -0.6 | Mojave King | 0.0 |
| 38 | Jordan Walsh | 2.1 | Tarik Biberovic | 0.0 |
| 39 | Mouhamed Gueye | 1.5 | Ben Sheppard | 0.3 |
| 40 | Maxwell Lewis | -0.7 | Rayan Rupert | -3.6 |
| 41 | Amari Bailey | -0.2 | Jett Howard | -0.4 |
| 42 | Tristan Vukcevic | 1.1 | Jordan Walsh | 2.1 |
| 43 | Rayan Rupert | -3.6 | Tristan Vukcevic | 1.1 |
| 44 | Sidy Cissoko | -1.1 | Seth Lundy | -0.2 |
| 45 | GG Jackson II | -0.0 | Jalen Wilson | -3.1 |
| 46 | Seth Lundy | -0.2 | Olivier-Maxence Prosper | 0.8 |
| 47 | Mojave King | 0.0 | Amari Bailey | -0.2 |
| 48 | Jordan Miller | 2.2 | Julian Strawther | -1.6 |
| 49 | Emoni Bates | -0.3 | Jordan Hawkins | -3.5 |
| 50 | Keyontae Johnson | -0.2 | Keyontae Johnson | -0.2 |
| 51 | Jalen Wilson | -3.1 | Dariq Whitehead | -0.8 |
| 52 | Toumani Camara | 6.8 | Brice Sensabaugh | -0.6 |
| 53 | Jaylen Clark | 1.0 | Chris Livingston | -0.4 |
| 54 | Jalen Slawson | 0.3 | Jalen Hood-Schifino | -1.2 |
| 55 | Isaiah Wong | -0.2 | Maxwell Lewis | -0.7 |
| 56 | Tarik Biberovic | 0.0 | Emoni Bates | -0.3 |
| 57 | Trayce Jackson-Davis | 4.7 | Nick Smith Jr. | -4.4 |
| 58 | Chris Livingston | -0.4 | GG Jackson II | -0.0 |

### 2024

| # | Actual draft | 5yr WAR | Model redraft | 5yr WAR |
|--:|:-------------|--------:|:--------------|--------:|
| 1 | Zaccharie Risacher | -0.8 | Reed Sheppard | 5.9 |
| 2 | Alex Sarr | 0.9 | Jonathan Mogbo | 0.4 |
| 3 | Reed Sheppard | 5.9 | Devin Carter | 0.0 |
| 4 | Stephon Castle | 5.5 | Adem Bona | -0.1 |
| 5 | Ron Holland | 1.1 | Zach Edey | 3.1 |
| 6 | Tidjane Salaün | -1.1 | Donovan Clingan | 8.6 |
| 7 | Donovan Clingan | 8.6 | Cam Spencer | 4.9 |
| 8 | Rob Dillingham | -2.4 | Alex Sarr | 0.9 |
| 9 | Zach Edey | 3.1 | DaRon Holmes | 0.2 |
| 10 | Cody Williams | -5.5 | Tyler Kolek | 1.3 |
| 11 | Matas Buzelis | 1.8 | Jamal Shead | -0.7 |
| 12 | Nikola Topić | -0.4 | Stephon Castle | 5.5 |
| 13 | Devin Carter | 0.0 | Anton Watson | -0.1 |
| 14 | Bub Carrington | -6.6 | Ryan Dunn | 0.2 |
| 15 | Kel'el Ware | 6.5 | Oso Ighodaro | 3.3 |
| 16 | Jared McCain | 0.7 | Dillon Jones | -0.1 |
| 17 | Dalton Knecht | 0.3 | Kyle Filipowski | 2.4 |
| 18 | Tristan Da Silva | 1.7 | Tidjane Salaün | -1.1 |
| 19 | Ja'Kobe Walter | 2.1 | Matas Buzelis | 1.8 |
| 20 | Jaylon Tyson | 2.9 | Kel'el Ware | 6.5 |
| 21 | Yves Missi | -0.6 | KJ Simpson | -1.4 |
| 22 | DaRon Holmes | 0.2 | Jared McCain | 0.7 |
| 23 | AJ Johnson | -3.3 | Nikola Topić | -0.4 |
| 24 | Kyshawn George | 0.2 | Tristan Da Silva | 1.7 |
| 25 | Pacôme Dadiet | -0.1 | Jaylon Tyson | 2.9 |
| 26 | Dillon Jones | -0.1 | Enrique Freeman | -0.2 |
| 27 | Terrence Shannon Jr. | 0.1 | Tristen Newton | -0.1 |
| 28 | Ryan Dunn | 0.2 | Zaccharie Risacher | -0.8 |
| 29 | Isaiah Collier | -2.0 | Kevin McCullar Jr. | -0.1 |
| 30 | Baylor Scheierman | 3.2 | Ariel Hukporti | -0.1 |
| 31 | Jonathan Mogbo | 0.4 | Baylor Scheierman | 3.2 |
| 32 | Kyle Filipowski | 2.4 | Terrence Shannon Jr. | 0.1 |
| 33 | Tyler Smith | -0.1 | Pelle Larsson | 2.1 |
| 34 | Tyler Kolek | 1.3 | Harrison Ingram | 0.1 |
| 35 | Johnny Furphy | -0.2 | Quinten Post | 1.6 |
| 36 | Juan Nunez | 0.0 | Tyler Smith | -0.1 |
| 37 | Bobi Klintman | -0.1 | Ron Holland | 1.1 |
| 38 | Ajay Mitchell | 4.5 | Yves Missi | -0.6 |
| 39 | Jaylen Wells | 1.2 | Ajay Mitchell | 4.5 |
| 40 | Oso Ighodaro | 3.3 | Juan Nunez | 0.0 |
| 41 | Adem Bona | -0.1 | Johnny Furphy | -0.2 |
| 42 | KJ Simpson | -1.4 | Kyshawn George | 0.2 |
| 43 | Nikola Djurisic | 0.0 | Jaylen Wells | 1.2 |
| 44 | Pelle Larsson | 2.1 | Nikola Djurisic | 0.0 |
| 45 | Jamal Shead | -0.7 | Ulrich Chomche | -0.1 |
| 46 | Cam Christie | -0.4 | Melvin Ajinça | 0.0 |
| 47 | Antonio Reeves | -0.2 | AJ Johnson | -3.3 |
| 48 | Harrison Ingram | 0.1 | Ja'Kobe Walter | 2.1 |
| 49 | Tristen Newton | -0.1 | Rob Dillingham | -2.4 |
| 50 | Enrique Freeman | -0.2 | Cody Williams | -5.5 |
| 51 | Melvin Ajinça | 0.0 | Bronny James | -0.6 |
| 52 | Quinten Post | 1.6 | Antonio Reeves | -0.2 |
| 53 | Cam Spencer | 4.9 | Isaiah Collier | -2.0 |
| 54 | Anton Watson | -0.1 | Dalton Knecht | 0.3 |
| 55 | Bronny James | -0.6 | Pacôme Dadiet | -0.1 |
| 56 | Kevin McCullar Jr. | -0.1 | Bub Carrington | -6.6 |
| 57 | Ulrich Chomche | -0.1 | Cam Christie | -0.4 |
| 58 | Ariel Hukporti | -0.1 | Bobi Klintman | -0.1 |

### 2025

| # | Actual draft | 5yr WAR | Model redraft | 5yr WAR |
|--:|:-------------|--------:|:--------------|--------:|
| 1 | Cooper Flagg | 5.0 | Collin Murray-Boyles | 2.7 |
| 2 | Dylan Harper | 4.4 | Cooper Flagg | 5.0 |
| 3 | VJ Edgecombe | 4.3 | Thomas Sorber | 0.0 |
| 4 | Kon Knueppel | 7.4 | Kon Knueppel | 7.4 |
| 5 | Ace Bailey | -2.4 | VJ Edgecombe | 4.3 |
| 6 | Tre Johnson | -2.0 | Ryan Kalkbrenner | 3.3 |
| 7 | Jeremiah Fears | -0.6 | Dylan Harper | 4.4 |
| 8 | Egor Dëmin | 0.8 | Johni Broome | -0.3 |
| 9 | Collin Murray-Boyles | 2.7 | Asa Newell | 0.3 |
| 10 | Khaman Maluach | -0.3 | Cedric Coward | 2.3 |
| 11 | Cedric Coward | 2.3 | Adou Thiero | -0.2 |
| 12 | Noa Essengue | -0.1 | Nique Clifford | -2.7 |
| 13 | Derik Queen | 2.3 | Khaman Maluach | -0.3 |
| 14 | Carter Bryant | 0.1 | Noa Essengue | -0.1 |
| 15 | Thomas Sorber | 0.0 | Derik Queen | 2.3 |
| 16 | Yang Hansen | -1.0 | Carter Bryant | 0.1 |
| 17 | Joan Beringer | 0.6 | Rasheer Fleming | -0.2 |
| 18 | Walter Clayton | -1.4 | Kasparas Jakučionis | 1.0 |
| 19 | Nolan Traoré | -3.1 | Jase Richardson | 0.1 |
| 20 | Kasparas Jakučionis | 1.0 | Joan Beringer | 0.6 |
| 21 | Will Riley | -2.0 | Danny Wolf | -0.1 |
| 22 | Drake Powell | -2.5 | Bogoljub Markovic | 0.0 |
| 23 | Asa Newell | 0.3 | Max Shulga | -0.1 |
| 24 | Nique Clifford | -2.7 | Kam Jones | -1.3 |
| 25 | Jase Richardson | 0.1 | Walter Clayton | -1.4 |
| 26 | Ben Saraf | -2.5 | Maxime Raynaud | -0.3 |
| 27 | Danny Wolf | -0.1 | Brooks Barnhizer | -0.1 |
| 28 | Hugo González | 0.6 | Amari Williams | 0.0 |
| 29 | Liam McNeeley | 0.2 | Yang Hansen | -1.0 |
| 30 | Yanic Konan Niederhäuser | 0.2 | Jahmai Mashack | -1.8 |
| 31 | Rasheer Fleming | -0.2 | Micah Peavy | -1.1 |
| 32 | Noah Penda | 0.6 | Sion James | 0.6 |
| 33 | Sion James | 0.6 | Javon Small | 1.3 |
| 34 | Ryan Kalkbrenner | 3.3 | Koby Brea | 0.1 |
| 35 | Johni Broome | -0.3 | Will Richard | 1.0 |
| 36 | Adou Thiero | -0.2 | Rocco Zikarsky | 0.1 |
| 37 | Chaz Lanier | -0.3 | Noah Penda | 0.6 |
| 38 | Kam Jones | -1.3 | Hugo González | 0.6 |
| 39 | Alijah Martin | -0.1 | Alex Toohey | 0.0 |
| 40 | Micah Peavy | -1.1 | Yanic Konan Niederhäuser | 0.2 |
| 41 | Koby Brea | 0.1 | Jeremiah Fears | -0.6 |
| 42 | Maxime Raynaud | -0.3 | Alijah Martin | -0.1 |
| 43 | Jamir Watkins | -0.5 | Kobe Sanders | 0.0 |
| 44 | Brooks Barnhizer | -0.1 | Drake Powell | -2.5 |
| 45 | Rocco Zikarsky | 0.1 | Taelon Peter | -0.6 |
| 46 | Amari Williams | 0.0 | Saliou Niang | 0.0 |
| 47 | Bogoljub Markovic | 0.0 | Mohamed Diawara | 0.3 |
| 48 | Javon Small | 1.3 | Nolan Traoré | -3.1 |
| 49 | Tyrese Proctor | -0.2 | Jamir Watkins | -0.5 |
| 50 | Kobe Sanders | 0.0 | John Tonje | -0.0 |
| 51 | Mohamed Diawara | 0.3 | Tyrese Proctor | -0.2 |
| 52 | Alex Toohey | 0.0 | Ben Saraf | -2.5 |
| 53 | John Tonje | -0.0 | Lachlan Olbrich | -0.2 |
| 54 | Taelon Peter | -0.6 | Egor Dëmin | 0.8 |
| 55 | Lachlan Olbrich | -0.2 | Ace Bailey | -2.4 |
| 56 | Will Richard | 1.0 | Tre Johnson | -2.0 |
| 57 | Max Shulga | -0.1 | Will Riley | -2.0 |
| 58 | Saliou Niang | 0.0 | Liam McNeeley | 0.2 |
| 59 | Jahmai Mashack | -1.8 | Chaz Lanier | -0.3 |
