# nba-redraft

## AI models used

Two walk-forward models, rank-averaged 50/50, then a small international tilt:

| Model | What it is |
|:------|:-----------|
| **CatBoost** | gradient-boosted trees (depth 5, 800 iterations, 10 seeds), context = every draft class from 2003 before the one being scored |
| **TabICL v2** | tabular in-context-learning foundation model (`TabICLRegressor`, 8 estimators, 8 seeds), context from 2010 |
| intl tilt | international rows: 70% model rank + 30% league-relative production x youth (`iz_young_x_eff`) |

Both learn a label that matches how each class is judged: WAR over a player's first max(3, N) NBA seasons, N = seasons the
scored class has played (2019-2021: five, 2022: four, 2023-2025: three), Gaussian-ranked within class. Features, all
knowable before draft night: Torvik final season + trajectory, physicals, NBA combine, international pro / FIBA youth
lines, Torvik team context, a 5-8 board pre-draft mock consensus with its 90/60/30/7-day movement (Wayback captures),
basketball-reference biography (NBA relatives, shooting hand, birthplace, high-school path) and -- CatBoost only --
game-log challenge-response features (rematches, bounce-back, error persistence, close games). Nothing after draft
night is used; the pure blend without the tilt scores 50% too, 0.495 (`winners/catboost_tabicl_market_bio_m3`).

## Holdout (2019-2025)

| AI model accuracy | NBA scouts accuracy | Drafts the AI won |
|:-----------------:|:-------------------:|:-----------------:|
| **50%**           | 26%                 | **7 of 7**        |

Accuracy = Spearman rank correlation between the order (AI redraft, or the real draft) and the players' realised 5-year WAR
(classes that have played fewer than five seasons are judged on the seasons they have played). Rule: `intl a=0.3 on [blend C03 m3 +mo+rs+person x5 | C03 m3 +mo+rs+person x5 s5 | T10 o2 m3 +mo+person x4 | T10 o2 m3 +mo+person x4 s4]`.

## By draft

| Draft | AI model | NBA scouts | Difference |
|:------|---------:|-----------:|-----------:|
| 2019  | 46%      | 40%        | +6         |
| 2020  | 55%      | 35%        | +20         |
| 2021  | 57%      | 41%        | +15         |
| 2022  | 48%      | 24%        | +24         |
| 2023  | 55%      | 10%        | +45         |
| 2024  | 52%      | 15%        | +37         |
| 2025  | 38%      | 17%        | +21         |
| **Mean** | **50%** | **26%**   | **+24**    |

## By player

Every holdout draft, actual order beside the model's reordering of the same names.

### 2019

| # | Actual draft | 5yr WAR | Model redraft | 5yr WAR |
|--:|:-------------|--------:|:--------------|--------:|
| 1 | Zion Williamson | 22.1 | Zion Williamson | 22.1 |
| 2 | Ja Morant | 19.2 | Jarrett Culver | -1.4 |
| 3 | RJ Barrett | 6.1 | Brandon Clarke | 12.3 |
| 4 | De'Andre Hunter | 2.4 | Goga Bitadze | 3.7 |
| 5 | Darius Garland | 15.3 | Coby White | 6.9 |
| 6 | Jarrett Culver | -1.4 | Tyler Herro | 12.5 |
| 7 | Coby White | 6.9 | Matisse Thybulle | 15.6 |
| 8 | Jaxson Hayes | 2.0 | Ja Morant | 19.2 |
| 9 | Rui Hachimura | 0.1 | De'Andre Hunter | 2.4 |
| 10 | Cam Reddish | -0.9 | Chuma Okeke | 2.3 |
| 11 | Cameron Johnson | 17.2 | Grant Williams | 7.2 |
| 12 | P.J. Washington | 7.0 | Ty Jerome | 1.3 |
| 13 | Tyler Herro | 12.5 | Bol Bol | 1.9 |
| 14 | Romeo Langford | 0.9 | Jaxson Hayes | 2.0 |
| 15 | Sekou Doumbouya | -3.6 | P.J. Washington | 7.0 |
| 16 | Chuma Okeke | 2.3 | Nickeil Alexander-Walker | 4.8 |
| 17 | Nickeil Alexander-Walker | 4.8 | RJ Barrett | 6.1 |
| 18 | Goga Bitadze | 3.7 | Cameron Johnson | 17.2 |
| 19 | Luka Šamanić | -0.7 | Sekou Doumbouya | -3.6 |
| 20 | Matisse Thybulle | 15.6 | Dylan Windler | 1.0 |
| 21 | Brandon Clarke | 12.3 | Nic Claxton | 16.3 |
| 22 | Grant Williams | 7.2 | Bruno Fernando | -0.6 |
| 23 | Darius Bazley | -2.2 | Darius Garland | 15.3 |
| 24 | Ty Jerome | 1.3 | Isaiah Roby | -2.1 |
| 25 | Nassir Little | 1.2 | Cam Reddish | -0.9 |
| 26 | Dylan Windler | 1.0 | Keldon Johnson | 7.2 |
| 27 | Mfiondu Kabengele | -0.1 | Luka Šamanić | -0.7 |
| 28 | Jordan Poole | 3.7 | Daniel Gafford | 13.1 |
| 29 | Keldon Johnson | 7.2 | Tremont Waters | -0.3 |
| 30 | Kevin Porter Jr. | 7.0 | Alen Smailagić | -0.2 |
| 31 | Nic Claxton | 16.3 | Jalen McDaniels | 0.3 |
| 32 | KZ Okpala | -1.2 | Carsen Edwards | -0.6 |
| 33 | Carsen Edwards | -0.6 | Terance Mann | 11.8 |
| 34 | Bruno Fernando | -0.6 | Quinndary Weatherspoon | -0.9 |
| 35 | Didi Louzada | -0.8 | Romeo Langford | 0.9 |
| 36 | Cody Martin | 7.4 | Eric Paschall | -2.1 |
| 37 | Deividas Sirvydis | -0.0 | Mfiondu Kabengele | -0.1 |
| 38 | Daniel Gafford | 13.1 | Deividas Sirvydis | -0.0 |
| 39 | Alen Smailagić | -0.2 | Didi Louzada | -0.8 |
| 40 | Justin James | -0.2 | Darius Bazley | -2.2 |
| 41 | Eric Paschall | -2.1 | Kyle Guy | -0.3 |
| 42 | Admiral Schofield | -1.0 | Cody Martin | 7.4 |
| 43 | Jaylen Nowell | 1.0 | Nassir Little | 1.2 |
| 44 | Bol Bol | 1.9 | Vanja Marinković | 0.0 |
| 45 | Isaiah Roby | -2.1 | Dewan Hernandez | 0.1 |
| 46 | Talen Horton-Tucker | 3.7 | Kevin Porter Jr. | 7.0 |
| 47 | Ignas Brazdeikis | -0.8 | Jordan Poole | 3.7 |
| 48 | Terance Mann | 11.8 | Talen Horton-Tucker | 3.7 |
| 49 | Quinndary Weatherspoon | -0.9 | Admiral Schofield | -1.0 |
| 50 | Jarrell Brantley | -0.1 | Jaylen Nowell | 1.0 |
| 51 | Tremont Waters | -0.3 | Ignas Brazdeikis | -0.8 |
| 52 | Jalen McDaniels | 0.3 | Rui Hachimura | 0.1 |
| 53 | Justin Wright-Foreman | -0.2 | Justin James | -0.2 |
| 54 | Marial Shayok | -0.1 | Miye Oni | -0.0 |
| 55 | Kyle Guy | -0.3 | Jarrell Brantley | -0.1 |
| 56 | Jaylen Hands | 0.0 | Marial Shayok | -0.1 |
| 57 | Jordan Bone | -0.8 | Jordan Bone | -0.8 |
| 58 | Miye Oni | -0.0 | Jaylen Hands | 0.0 |
| 59 | Dewan Hernandez | 0.1 | KZ Okpala | -1.2 |
| 60 | Vanja Marinković | 0.0 | Justin Wright-Foreman | -0.2 |

### 2020

| # | Actual draft | 5yr WAR | Model redraft | 5yr WAR |
|--:|:-------------|--------:|:--------------|--------:|
| 1 | Anthony Edwards | 36.5 | James Wiseman | -4.2 |
| 2 | James Wiseman | -4.2 | Tyrese Haliburton | 38.5 |
| 3 | LaMelo Ball | 18.1 | Onyeka Okongwu | 14.5 |
| 4 | Patrick Williams | 1.2 | Devin Vassell | 8.1 |
| 5 | Isaac Okoro | 6.3 | Obi Toppin | 10.2 |
| 6 | Onyeka Okongwu | 14.5 | Xavier Tillman Sr. | 5.3 |
| 7 | Killian Hayes | -4.1 | Jalen Smith | 5.6 |
| 8 | Obi Toppin | 10.2 | Anthony Edwards | 36.5 |
| 9 | Deni Avdija | 13.4 | Josh Green | 2.1 |
| 10 | Jalen Smith | 5.6 | LaMelo Ball | 18.1 |
| 11 | Devin Vassell | 8.1 | Desmond Bane | 26.9 |
| 12 | Tyrese Haliburton | 38.5 | Saddiq Bey | 7.1 |
| 13 | Kira Lewis Jr. | -0.0 | Patrick Williams | 1.2 |
| 14 | Aaron Nesmith | 5.6 | Tyrell Terry | 0.1 |
| 15 | Cole Anthony | 5.8 | Isaac Okoro | 6.3 |
| 16 | Isaiah Stewart | 6.0 | Malachi Flynn | 1.4 |
| 17 | Aleksej Pokusevski | -3.1 | Payton Pritchard | 14.7 |
| 18 | Josh Green | 2.1 | Tyrese Maxey | 21.9 |
| 19 | Saddiq Bey | 7.1 | Tre Jones | 9.4 |
| 20 | Precious Achiuwa | 3.6 | Precious Achiuwa | 3.6 |
| 21 | Tyrese Maxey | 21.9 | Cole Anthony | 5.8 |
| 22 | Zeke Nnaji | -0.4 | Kira Lewis Jr. | -0.0 |
| 23 | Leandro Bolmaro | -0.5 | Udoka Azubuike | 0.1 |
| 24 | R.J. Hampton | -4.5 | Paul Reed | 6.7 |
| 25 | Immanuel Quickley | 19.5 | Deni Avdija | 13.4 |
| 26 | Payton Pritchard | 14.7 | Aaron Nesmith | 5.6 |
| 27 | Udoka Azubuike | 0.1 | Jahmi'us Ramsey | -0.6 |
| 28 | Jaden McDaniels | 12.8 | Isaiah Joe | 10.9 |
| 29 | Malachi Flynn | 1.4 | Killian Hayes | -4.1 |
| 30 | Desmond Bane | 26.9 | Isaiah Stewart | 6.0 |
| 31 | Tyrell Terry | 0.1 | Justinian Jessup | 0.0 |
| 32 | Vernon Carey Jr. | -0.1 | CJ Elleby | -1.7 |
| 33 | Daniel Oturu | -0.5 | Marko Simonovic | -0.0 |
| 34 | Théo Maledon | -6.3 | Vít Krejčí | 0.6 |
| 35 | Xavier Tillman Sr. | 5.3 | Tyler Bey | -0.4 |
| 36 | Tyler Bey | -0.4 | Leandro Bolmaro | -0.5 |
| 37 | Vít Krejčí | 0.6 | R.J. Hampton | -4.5 |
| 38 | Saben Lee | 0.0 | KJ Martin | 0.3 |
| 39 | Elijah Hughes | -1.8 | Zeke Nnaji | -0.4 |
| 40 | Robert Woodard II | -0.0 | Skylar Mays | 0.6 |
| 41 | Tre Jones | 9.4 | Yam Madar | 0.0 |
| 42 | Nick Richards | 0.4 | Reggie Perry | -0.9 |
| 43 | Jahmi'us Ramsey | -0.6 | Elijah Hughes | -1.8 |
| 44 | Marko Simonovic | -0.0 | Daniel Oturu | -0.5 |
| 45 | Jordan Nwora | -1.9 | Vernon Carey Jr. | -0.1 |
| 46 | CJ Elleby | -1.7 | Jalen Harris | 0.3 |
| 47 | Yam Madar | 0.0 | Jaden McDaniels | 12.8 |
| 48 | Nico Mannion | -0.3 | Sam Merrill | 4.0 |
| 49 | Isaiah Joe | 10.9 | Nick Richards | 0.4 |
| 50 | Skylar Mays | 0.6 | Jay Scrubb | -0.5 |
| 51 | Justinian Jessup | 0.0 | Jordan Nwora | -1.9 |
| 52 | KJ Martin | 0.3 | Théo Maledon | -6.3 |
| 53 | Cassius Winston | -0.6 | Grant Riller | 0.0 |
| 54 | Cassius Stanley | -0.1 | Immanuel Quickley | 19.5 |
| 55 | Jay Scrubb | -0.5 | Saben Lee | 0.0 |
| 56 | Grant Riller | 0.0 | Aleksej Pokusevski | -3.1 |
| 57 | Reggie Perry | -0.9 | Robert Woodard II | -0.0 |
| 58 | Paul Reed | 6.7 | Nico Mannion | -0.3 |
| 59 | Jalen Harris | 0.3 | Cassius Winston | -0.6 |
| 60 | Sam Merrill | 4.0 | Cassius Stanley | -0.1 |

### 2021

| # | Actual draft | 5yr WAR | Model redraft | 5yr WAR |
|--:|:-------------|--------:|:--------------|--------:|
| 1 | Cade Cunningham | 22.6 | Cade Cunningham | 22.6 |
| 2 | Jalen Green | 7.5 | Franz Wagner | 25.7 |
| 3 | Evan Mobley | 28.5 | Evan Mobley | 28.5 |
| 4 | Scottie Barnes | 32.0 | Jalen Suggs | 12.8 |
| 5 | Jalen Suggs | 12.8 | Scottie Barnes | 32.0 |
| 6 | Josh Giddey | 19.6 | Chris Duarte | 1.6 |
| 7 | Jonathan Kuminga | 3.5 | Corey Kispert | 2.0 |
| 8 | Franz Wagner | 25.7 | Isaiah Jackson | 3.7 |
| 9 | Davion Mitchell | 5.8 | Jalen Johnson | 17.0 |
| 10 | Ziaire Williams | 1.0 | Kai Jones | 0.6 |
| 11 | James Bouknight | -1.4 | Jared Butler | 0.7 |
| 12 | Joshua Primo | -1.1 | Miles McBride | 6.7 |
| 13 | Chris Duarte | 1.6 | Alperen Şengün | 29.5 |
| 14 | Moses Moody | 5.6 | Neemias Queta | 7.8 |
| 15 | Corey Kispert | 2.0 | Moses Moody | 5.6 |
| 16 | Alperen Şengün | 29.5 | Day'Ron Sharpe | 6.3 |
| 17 | Trey Murphy III | 20.7 | Davion Mitchell | 5.8 |
| 18 | Tre Mann | -1.4 | Filip Petrušev | -0.0 |
| 19 | Kai Jones | 0.6 | Jeremiah Robinson-Earl | -1.0 |
| 20 | Jalen Johnson | 17.0 | James Bouknight | -1.4 |
| 21 | Keon Johnson | -3.7 | Scottie Lewis | 0.0 |
| 22 | Isaiah Jackson | 3.7 | Ayo Dosunmu | 6.6 |
| 23 | Usman Garuba | 0.9 | Quentin Grimes | 9.5 |
| 24 | Josh Christopher | -1.8 | Jaden Springer | 0.2 |
| 25 | Quentin Grimes | 9.5 | Trey Murphy III | 20.7 |
| 26 | Bones Hyland | 6.2 | Bones Hyland | 6.2 |
| 27 | Cam Thomas | 0.7 | Sandro Mamukelashvili | 5.6 |
| 28 | Jaden Springer | 0.2 | Josh Christopher | -1.8 |
| 29 | Day'Ron Sharpe | 6.3 | Jason Preston | 0.0 |
| 30 | Santi Aldama | 11.4 | Luka Garza | 3.7 |
| 31 | Isaiah Todd | -0.6 | Isaiah Livers | -0.5 |
| 32 | Jeremiah Robinson-Earl | -1.0 | Joe Wieskamp | -0.4 |
| 33 | Jason Preston | 0.0 | Marcus Zegarowski | 0.0 |
| 34 | Rokas Jokubaitis | 0.0 | Jalen Green | 7.5 |
| 35 | Herbert Jones | 15.1 | Aaron Wiggins | 7.6 |
| 36 | Miles McBride | 6.7 | Usman Garuba | 0.9 |
| 37 | JT Thor | -2.2 | Herbert Jones | 15.1 |
| 38 | Ayo Dosunmu | 6.6 | Kessler Edwards | -0.2 |
| 39 | Neemias Queta | 7.8 | Jonathan Kuminga | 3.5 |
| 40 | Jared Butler | 0.7 | Josh Giddey | 19.6 |
| 41 | Joe Wieskamp | -0.4 | David Johnson | 0.0 |
| 42 | Isaiah Livers | -0.5 | Sharife Cooper | -1.0 |
| 43 | Greg Brown III | -2.2 | JT Thor | -2.2 |
| 44 | Kessler Edwards | -0.2 | Dalano Banton | 2.0 |
| 45 | Juhann Begarin | 0.0 | Charles Bassey | 1.8 |
| 46 | Dalano Banton | 2.0 | Rokas Jokubaitis | 0.0 |
| 47 | David Johnson | 0.0 | Ziaire Williams | 1.0 |
| 48 | Sharife Cooper | -1.0 | Balša Koprivica | 0.0 |
| 49 | Marcus Zegarowski | 0.0 | RaiQuan Gray | 0.2 |
| 50 | Filip Petrušev | -0.0 | Brandon Boston Jr. | -1.1 |
| 51 | Brandon Boston Jr. | -1.1 | Santi Aldama | 11.4 |
| 52 | Luka Garza | 3.7 | Keon Johnson | -3.7 |
| 53 | Charles Bassey | 1.8 | Georgios Kalaitzakis | -1.1 |
| 54 | Sandro Mamukelashvili | 5.6 | Isaiah Todd | -0.6 |
| 55 | Aaron Wiggins | 7.6 | Jericho Sims | 0.9 |
| 56 | Scottie Lewis | 0.0 | Juhann Begarin | 0.0 |
| 57 | Balša Koprivica | 0.0 | Cam Thomas | 0.7 |
| 58 | Jericho Sims | 0.9 | Joshua Primo | -1.1 |
| 59 | RaiQuan Gray | 0.2 | Tre Mann | -1.4 |
| 60 | Georgios Kalaitzakis | -1.1 | Greg Brown III | -2.2 |

### 2022

| # | Actual draft | 5yr WAR | Model redraft | 5yr WAR |
|--:|:-------------|--------:|:--------------|--------:|
| 1 | Paolo Banchero | 18.5 | Chet Holmgren | 19.9 |
| 2 | Chet Holmgren | 19.9 | Keegan Murray | 8.4 |
| 3 | Jabari Smith Jr. | 7.5 | Jabari Smith Jr. | 7.5 |
| 4 | Keegan Murray | 8.4 | Jalen Duren | 17.1 |
| 5 | Jaden Ivey | -0.5 | Jeremy Sochan | -0.1 |
| 6 | Bennedict Mathurin | -1.3 | Tari Eason | 8.6 |
| 7 | Shaedon Sharpe | 2.7 | Paolo Banchero | 18.5 |
| 8 | Dyson Daniels | 15.4 | Mark Williams | 8.4 |
| 9 | Jeremy Sochan | -0.1 | Walker Kessler | 12.9 |
| 10 | Johnny Davis | -1.9 | Jake LaRavia | 3.5 |
| 11 | Ousmane Dieng | -0.6 | Dalen Terry | 0.8 |
| 12 | Jalen Williams | 22.4 | Bennedict Mathurin | -1.3 |
| 13 | Jalen Duren | 17.1 | Shaedon Sharpe | 2.7 |
| 14 | Ochai Agbaji | -0.8 | Christian Braun | 8.1 |
| 15 | Mark Williams | 8.4 | Kennedy Chandler | -0.6 |
| 16 | AJ Griffin | 0.7 | Christian Koloko | -0.3 |
| 17 | Tari Eason | 8.6 | E.J. Liddell | -0.2 |
| 18 | Dalen Terry | 0.8 | Jalen Williams | 22.4 |
| 19 | Jake LaRavia | 3.5 | Johnny Davis | -1.9 |
| 20 | Malaki Branham | -5.2 | Josh Minott | 3.2 |
| 21 | Christian Braun | 8.1 | Jaden Ivey | -0.5 |
| 22 | Walker Kessler | 12.9 | Karlo Matković | 1.7 |
| 23 | David Roddy | -2.4 | Jaylin Williams | 8.7 |
| 24 | MarJon Beauchamp | -1.2 | Dyson Daniels | 15.4 |
| 25 | Blake Wesley | -3.0 | Wendell Moore Jr. | -0.1 |
| 26 | Wendell Moore Jr. | -0.1 | AJ Griffin | 0.7 |
| 27 | Nikola Jović | 3.2 | Andrew Nembhard | 5.1 |
| 28 | Patrick Baldwin Jr. | -0.1 | Vince Williams Jr. | 2.5 |
| 29 | TyTy Washington Jr. | -0.9 | Ismael Kamagate | 0.0 |
| 30 | Peyton Watson | 2.2 | Khalifa Diop | 0.0 |
| 31 | Andrew Nembhard | 5.1 | David Roddy | -2.4 |
| 32 | Caleb Houstan | 0.1 | TyTy Washington Jr. | -0.9 |
| 33 | Christian Koloko | -0.3 | Ochai Agbaji | -0.8 |
| 34 | Jaylin Williams | 8.7 | Nikola Jović | 3.2 |
| 35 | Max Christie | 0.4 | Gabriele Procida | 0.0 |
| 36 | Gabriele Procida | 0.0 | Kendall Brown | -0.2 |
| 37 | Jaden Hardy | -2.3 | Luke Travers | -0.3 |
| 38 | Kennedy Chandler | -0.6 | Isaiah Mobley | -0.0 |
| 39 | Khalifa Diop | 0.0 | Ousmane Dieng | -0.6 |
| 40 | Bryce McGowens | -2.3 | Malaki Branham | -5.2 |
| 41 | E.J. Liddell | -0.2 | Jabari Walker | -1.2 |
| 42 | Trevor Keels | -0.1 | Hugo Besson | 0.0 |
| 43 | Moussa Diabaté | 6.1 | Gui Santos | 2.4 |
| 44 | Ryan Rollins | 3.8 | Ryan Rollins | 3.8 |
| 45 | Josh Minott | 3.2 | Tyrese Martin | -1.0 |
| 46 | Ismael Kamagate | 0.0 | MarJon Beauchamp | -1.2 |
| 47 | Vince Williams Jr. | 2.5 | JD Davison | 0.1 |
| 48 | Kendall Brown | -0.2 | Matteo Spagnolo | 0.0 |
| 49 | Isaiah Mobley | -0.0 | Trevor Keels | -0.1 |
| 50 | Matteo Spagnolo | 0.0 | Peyton Watson | 2.2 |
| 51 | Tyrese Martin | -1.0 | Max Christie | 0.4 |
| 52 | Karlo Matković | 1.7 | Yannick Nzosa | 0.0 |
| 53 | JD Davison | 0.1 | Moussa Diabaté | 6.1 |
| 54 | Yannick Nzosa | 0.0 | Jaden Hardy | -2.3 |
| 55 | Gui Santos | 2.4 | Caleb Houstan | 0.1 |
| 56 | Luke Travers | -0.3 | Patrick Baldwin Jr. | -0.1 |
| 57 | Jabari Walker | -1.2 | Bryce McGowens | -2.3 |
| 58 | Hugo Besson | 0.0 | Blake Wesley | -3.0 |

### 2023

| # | Actual draft | 5yr WAR | Model redraft | 5yr WAR |
|--:|:-------------|--------:|:--------------|--------:|
| 1 | Victor Wembanyama | 32.1 | Victor Wembanyama | 32.1 |
| 2 | Brandon Miller | 6.0 | Dereck Lively II | 5.7 |
| 3 | Scoot Henderson | -1.9 | Anthony Black | 4.2 |
| 4 | Amen Thompson | 20.4 | Brandon Miller | 6.0 |
| 5 | Ausar Thompson | 10.7 | Cason Wallace | 13.7 |
| 6 | Anthony Black | 4.2 | Jarace Walker | 0.3 |
| 7 | Bilal Coulibaly | -2.9 | Brandin Podziemski | 9.1 |
| 8 | Jarace Walker | 0.3 | Trayce Jackson-Davis | 4.7 |
| 9 | Taylor Hendricks | -0.2 | Jaime Jaquez Jr. | 5.7 |
| 10 | Cason Wallace | 13.7 | Amen Thompson | 20.4 |
| 11 | Jett Howard | -0.4 | Taylor Hendricks | -0.2 |
| 12 | Dereck Lively II | 5.7 | Jaylen Clark | 1.0 |
| 13 | Gradey Dick | -3.5 | Andre Jackson Jr. | -1.6 |
| 14 | Jordan Hawkins | -3.5 | Jalen Pickett | 0.5 |
| 15 | Kobe Bufkin | -0.6 | Gradey Dick | -3.5 |
| 16 | Keyonte George | -1.0 | Jalen Slawson | 0.3 |
| 17 | Jalen Hood-Schifino | -1.2 | Scoot Henderson | -1.9 |
| 18 | Jaime Jaquez Jr. | 5.7 | Kobe Bufkin | -0.6 |
| 19 | Brandin Podziemski | 9.1 | Jordan Miller | 2.2 |
| 20 | Cam Whitmore | 2.1 | Ausar Thompson | 10.7 |
| 21 | Noah Clowney | -1.1 | Cam Whitmore | 2.1 |
| 22 | Dariq Whitehead | -0.8 | Leonard Miller | 0.9 |
| 23 | Kris Murray | 0.4 | Colby Jones | -0.2 |
| 24 | Olivier-Maxence Prosper | 0.8 | Noah Clowney | -1.1 |
| 25 | Marcus Sasser | 0.7 | Kris Murray | 0.4 |
| 26 | Ben Sheppard | 0.3 | Marcus Sasser | 0.7 |
| 27 | Nick Smith Jr. | -4.4 | James Nnaji | 0.0 |
| 28 | Brice Sensabaugh | -0.6 | Kobe Brown | -0.3 |
| 29 | Julian Strawther | -1.6 | Bilal Coulibaly | -2.9 |
| 30 | Kobe Brown | -0.3 | Hunter Tyson | -0.6 |
| 31 | James Nnaji | 0.0 | Mouhamed Gueye | 1.5 |
| 32 | Jalen Pickett | 0.5 | Toumani Camara | 6.8 |
| 33 | Leonard Miller | 0.9 | Julian Phillips | -0.4 |
| 34 | Colby Jones | -0.2 | Jett Howard | -0.4 |
| 35 | Julian Phillips | -0.4 | Ben Sheppard | 0.3 |
| 36 | Andre Jackson Jr. | -1.6 | Sidy Cissoko | -1.1 |
| 37 | Hunter Tyson | -0.6 | Tarik Biberovic | 0.0 |
| 38 | Jordan Walsh | 2.1 | Isaiah Wong | -0.2 |
| 39 | Mouhamed Gueye | 1.5 | Mojave King | 0.0 |
| 40 | Maxwell Lewis | -0.7 | Keyonte George | -1.0 |
| 41 | Amari Bailey | -0.2 | Tristan Vukcevic | 1.1 |
| 42 | Tristan Vukcevic | 1.1 | Seth Lundy | -0.2 |
| 43 | Rayan Rupert | -3.6 | Rayan Rupert | -3.6 |
| 44 | Sidy Cissoko | -1.1 | Jordan Walsh | 2.1 |
| 45 | GG Jackson II | -0.0 | Julian Strawther | -1.6 |
| 46 | Seth Lundy | -0.2 | Jalen Hood-Schifino | -1.2 |
| 47 | Mojave King | 0.0 | Amari Bailey | -0.2 |
| 48 | Jordan Miller | 2.2 | Jordan Hawkins | -3.5 |
| 49 | Emoni Bates | -0.3 | Jalen Wilson | -3.1 |
| 50 | Keyontae Johnson | -0.2 | Keyontae Johnson | -0.2 |
| 51 | Jalen Wilson | -3.1 | Olivier-Maxence Prosper | 0.8 |
| 52 | Toumani Camara | 6.8 | Chris Livingston | -0.4 |
| 53 | Jaylen Clark | 1.0 | Dariq Whitehead | -0.8 |
| 54 | Jalen Slawson | 0.3 | Brice Sensabaugh | -0.6 |
| 55 | Isaiah Wong | -0.2 | Emoni Bates | -0.3 |
| 56 | Tarik Biberovic | 0.0 | Maxwell Lewis | -0.7 |
| 57 | Trayce Jackson-Davis | 4.7 | Nick Smith Jr. | -4.4 |
| 58 | Chris Livingston | -0.4 | GG Jackson II | -0.0 |

### 2024

| # | Actual draft | 5yr WAR | Model redraft | 5yr WAR |
|--:|:-------------|--------:|:--------------|--------:|
| 1 | Zaccharie Risacher | -0.8 | Reed Sheppard | 5.9 |
| 2 | Alex Sarr | 0.9 | Donovan Clingan | 8.6 |
| 3 | Reed Sheppard | 5.9 | Devin Carter | 0.0 |
| 4 | Stephon Castle | 5.5 | Jonathan Mogbo | 0.4 |
| 5 | Ron Holland | 1.1 | Adem Bona | -0.1 |
| 6 | Tidjane Salaün | -1.1 | Zach Edey | 3.1 |
| 7 | Donovan Clingan | 8.6 | Stephon Castle | 5.5 |
| 8 | Rob Dillingham | -2.4 | Ryan Dunn | 0.2 |
| 9 | Zach Edey | 3.1 | Cam Spencer | 4.9 |
| 10 | Cody Williams | -5.5 | Anton Watson | -0.1 |
| 11 | Matas Buzelis | 1.8 | DaRon Holmes | 0.2 |
| 12 | Nikola Topić | -0.4 | Tyler Kolek | 1.3 |
| 13 | Devin Carter | 0.0 | Alex Sarr | 0.9 |
| 14 | Bub Carrington | -6.6 | Oso Ighodaro | 3.3 |
| 15 | Kel'el Ware | 6.5 | Jamal Shead | -0.7 |
| 16 | Jared McCain | 0.7 | Matas Buzelis | 1.8 |
| 17 | Dalton Knecht | 0.3 | Kyle Filipowski | 2.4 |
| 18 | Tristan Da Silva | 1.7 | Kel'el Ware | 6.5 |
| 19 | Ja'Kobe Walter | 2.1 | Tidjane Salaün | -1.1 |
| 20 | Jaylon Tyson | 2.9 | Baylor Scheierman | 3.2 |
| 21 | Yves Missi | -0.6 | Jared McCain | 0.7 |
| 22 | DaRon Holmes | 0.2 | Tristen Newton | -0.1 |
| 23 | AJ Johnson | -3.3 | Ariel Hukporti | -0.1 |
| 24 | Kyshawn George | 0.2 | Nikola Topić | -0.4 |
| 25 | Pacôme Dadiet | -0.1 | Dillon Jones | -0.1 |
| 26 | Dillon Jones | -0.1 | KJ Simpson | -1.4 |
| 27 | Terrence Shannon Jr. | 0.1 | Tristan Da Silva | 1.7 |
| 28 | Ryan Dunn | 0.2 | Kevin McCullar Jr. | -0.1 |
| 29 | Isaiah Collier | -2.0 | Pelle Larsson | 2.1 |
| 30 | Baylor Scheierman | 3.2 | Terrence Shannon Jr. | 0.1 |
| 31 | Jonathan Mogbo | 0.4 | Harrison Ingram | 0.1 |
| 32 | Kyle Filipowski | 2.4 | Jaylon Tyson | 2.9 |
| 33 | Tyler Smith | -0.1 | Tyler Smith | -0.1 |
| 34 | Tyler Kolek | 1.3 | Zaccharie Risacher | -0.8 |
| 35 | Johnny Furphy | -0.2 | Enrique Freeman | -0.2 |
| 36 | Juan Nunez | 0.0 | Ron Holland | 1.1 |
| 37 | Bobi Klintman | -0.1 | Quinten Post | 1.6 |
| 38 | Ajay Mitchell | 4.5 | Kyshawn George | 0.2 |
| 39 | Jaylen Wells | 1.2 | Ajay Mitchell | 4.5 |
| 40 | Oso Ighodaro | 3.3 | Juan Nunez | 0.0 |
| 41 | Adem Bona | -0.1 | Jaylen Wells | 1.2 |
| 42 | KJ Simpson | -1.4 | Johnny Furphy | -0.2 |
| 43 | Nikola Djurisic | 0.0 | Melvin Ajinça | 0.0 |
| 44 | Pelle Larsson | 2.1 | Yves Missi | -0.6 |
| 45 | Jamal Shead | -0.7 | Bub Carrington | -6.6 |
| 46 | Cam Christie | -0.4 | Bronny James | -0.6 |
| 47 | Antonio Reeves | -0.2 | Bobi Klintman | -0.1 |
| 48 | Harrison Ingram | 0.1 | Nikola Djurisic | 0.0 |
| 49 | Tristen Newton | -0.1 | Ja'Kobe Walter | 2.1 |
| 50 | Enrique Freeman | -0.2 | Rob Dillingham | -2.4 |
| 51 | Melvin Ajinça | 0.0 | Ulrich Chomche | -0.1 |
| 52 | Quinten Post | 1.6 | AJ Johnson | -3.3 |
| 53 | Cam Spencer | 4.9 | Dalton Knecht | 0.3 |
| 54 | Anton Watson | -0.1 | Cody Williams | -5.5 |
| 55 | Bronny James | -0.6 | Antonio Reeves | -0.2 |
| 56 | Kevin McCullar Jr. | -0.1 | Cam Christie | -0.4 |
| 57 | Ulrich Chomche | -0.1 | Pacôme Dadiet | -0.1 |
| 58 | Ariel Hukporti | -0.1 | Isaiah Collier | -2.0 |

### 2025

| # | Actual draft | 5yr WAR | Model redraft | 5yr WAR |
|--:|:-------------|--------:|:--------------|--------:|
| 1 | Cooper Flagg | 5.0 | Cooper Flagg | 5.0 |
| 2 | Dylan Harper | 4.4 | VJ Edgecombe | 4.3 |
| 3 | VJ Edgecombe | 4.3 | Collin Murray-Boyles | 2.7 |
| 4 | Kon Knueppel | 7.4 | Kon Knueppel | 7.4 |
| 5 | Ace Bailey | -2.4 | Dylan Harper | 4.4 |
| 6 | Tre Johnson | -2.0 | Thomas Sorber | 0.0 |
| 7 | Jeremiah Fears | -0.6 | Ryan Kalkbrenner | 3.3 |
| 8 | Egor Dëmin | 0.8 | Johni Broome | -0.3 |
| 9 | Collin Murray-Boyles | 2.7 | Cedric Coward | 2.3 |
| 10 | Khaman Maluach | -0.3 | Carter Bryant | 0.1 |
| 11 | Cedric Coward | 2.3 | Kasparas Jakučionis | 1.0 |
| 12 | Noa Essengue | -0.1 | Noa Essengue | -0.1 |
| 13 | Derik Queen | 2.3 | Asa Newell | 0.3 |
| 14 | Carter Bryant | 0.1 | Nique Clifford | -2.7 |
| 15 | Thomas Sorber | 0.0 | Danny Wolf | -0.1 |
| 16 | Yang Hansen | -1.0 | Khaman Maluach | -0.3 |
| 17 | Joan Beringer | 0.6 | Kam Jones | -1.3 |
| 18 | Walter Clayton | -1.4 | Rasheer Fleming | -0.2 |
| 19 | Nolan Traoré | -3.1 | Max Shulga | -0.1 |
| 20 | Kasparas Jakučionis | 1.0 | Walter Clayton | -1.4 |
| 21 | Will Riley | -2.0 | Bogoljub Markovic | 0.0 |
| 22 | Drake Powell | -2.5 | Maxime Raynaud | -0.3 |
| 23 | Asa Newell | 0.3 | Adou Thiero | -0.2 |
| 24 | Nique Clifford | -2.7 | Derik Queen | 2.3 |
| 25 | Jase Richardson | 0.1 | Yang Hansen | -1.0 |
| 26 | Ben Saraf | -2.5 | Jase Richardson | 0.1 |
| 27 | Danny Wolf | -0.1 | Joan Beringer | 0.6 |
| 28 | Hugo González | 0.6 | Micah Peavy | -1.1 |
| 29 | Liam McNeeley | 0.2 | Koby Brea | 0.1 |
| 30 | Yanic Konan Niederhäuser | 0.2 | Amari Williams | 0.0 |
| 31 | Rasheer Fleming | -0.2 | Jahmai Mashack | -1.8 |
| 32 | Noah Penda | 0.6 | Drake Powell | -2.5 |
| 33 | Sion James | 0.6 | Javon Small | 1.3 |
| 34 | Ryan Kalkbrenner | 3.3 | Brooks Barnhizer | -0.1 |
| 35 | Johni Broome | -0.3 | Sion James | 0.6 |
| 36 | Adou Thiero | -0.2 | Rocco Zikarsky | 0.1 |
| 37 | Chaz Lanier | -0.3 | Noah Penda | 0.6 |
| 38 | Kam Jones | -1.3 | Alex Toohey | 0.0 |
| 39 | Alijah Martin | -0.1 | Will Richard | 1.0 |
| 40 | Micah Peavy | -1.1 | Alijah Martin | -0.1 |
| 41 | Koby Brea | 0.1 | Jamir Watkins | -0.5 |
| 42 | Maxime Raynaud | -0.3 | Jeremiah Fears | -0.6 |
| 43 | Jamir Watkins | -0.5 | Hugo González | 0.6 |
| 44 | Brooks Barnhizer | -0.1 | Lachlan Olbrich | -0.2 |
| 45 | Rocco Zikarsky | 0.1 | Kobe Sanders | 0.0 |
| 46 | Amari Williams | 0.0 | Yanic Konan Niederhäuser | 0.2 |
| 47 | Bogoljub Markovic | 0.0 | John Tonje | -0.0 |
| 48 | Javon Small | 1.3 | Ace Bailey | -2.4 |
| 49 | Tyrese Proctor | -0.2 | Taelon Peter | -0.6 |
| 50 | Kobe Sanders | 0.0 | Tyrese Proctor | -0.2 |
| 51 | Mohamed Diawara | 0.3 | Egor Dëmin | 0.8 |
| 52 | Alex Toohey | 0.0 | Mohamed Diawara | 0.3 |
| 53 | John Tonje | -0.0 | Tre Johnson | -2.0 |
| 54 | Taelon Peter | -0.6 | Ben Saraf | -2.5 |
| 55 | Lachlan Olbrich | -0.2 | Saliou Niang | 0.0 |
| 56 | Will Richard | 1.0 | Will Riley | -2.0 |
| 57 | Max Shulga | -0.1 | Nolan Traoré | -3.1 |
| 58 | Saliou Niang | 0.0 | Liam McNeeley | 0.2 |
| 59 | Jahmai Mashack | -1.8 | Chaz Lanier | -0.3 |
