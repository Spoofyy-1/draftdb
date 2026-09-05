# nba-redraft

## AI models used

Two walk-forward models, rank-averaged 50/50, then a small international tilt:

| Model | What it is |
|:------|:-----------|
| **CatBoost** | gradient-boosted trees (depth 5, 800 iterations, 5 seeds), context = every draft class from 2003 before the one being scored |
| **TabICL v2** | tabular in-context-learning foundation model (`TabICLRegressor`, 8 estimators, 4 seeds), context from 2010 |
| intl tilt | international rows: 70% model rank + 30% league-relative production x youth (`iz_young_x_eff`) |

Both learn a label that matches how each class is judged: WAR over a player's first max(3, N) NBA seasons, N = seasons the
scored class has played (2019-2021: five, 2022: four, 2023-2025: three), Gaussian-ranked within class. Features (all
knowable before draft night): Torvik final season + trajectory, physicals, NBA combine, international pro / FIBA youth
lines, Torvik team context, a 5-8 board pre-draft mock consensus with its 90/60/30/7-day movement (Wayback captures), and
-- CatBoost only -- game-log challenge-response features (rematches, bounce-back, error persistence, close games).
Nothing after draft night is used; the pure blend without the tilt scores 49% (`winners/catboost_tabicl_market_m3`).

## Holdout (2019-2025)

| AI model accuracy | NBA scouts accuracy | Drafts the AI won |
|:-----------------:|:-------------------:|:-----------------:|
| **50%**           | 26%                 | **7 of 7**        |

Accuracy = Spearman rank correlation between the order (AI redraft, or the real draft) and the players' realised 5-year WAR
(classes that have played fewer than five seasons are judged on the seasons they have played). Rule: `intl a=0.3 on [blend C03 m3 +mo+rs x5 | T10 o2 m3 +mo x4]`.

## By draft

| Draft | AI model | NBA scouts | Difference |
|:------|---------:|-----------:|-----------:|
| 2019  | 47%      | 40%        | +7         |
| 2020  | 54%      | 35%        | +19         |
| 2021  | 59%      | 41%        | +17         |
| 2022  | 46%      | 24%        | +22         |
| 2023  | 53%      | 10%        | +43         |
| 2024  | 52%      | 15%        | +37         |
| 2025  | 39%      | 17%        | +23         |
| **Mean** | **50%** | **26%**   | **+24**    |

## By player

Every holdout draft, actual order beside the model's reordering of the same names.

### 2019

| # | Actual draft | 5yr WAR | Model redraft | 5yr WAR |
|--:|:-------------|--------:|:--------------|--------:|
| 1 | Zion Williamson | 22.1 | Zion Williamson | 22.1 |
| 2 | Ja Morant | 19.2 | Jarrett Culver | -1.4 |
| 3 | RJ Barrett | 6.1 | Brandon Clarke | 12.3 |
| 4 | De'Andre Hunter | 2.4 | Coby White | 6.9 |
| 5 | Darius Garland | 15.3 | Goga Bitadze | 3.7 |
| 6 | Jarrett Culver | -1.4 | Tyler Herro | 12.5 |
| 7 | Coby White | 6.9 | Matisse Thybulle | 15.6 |
| 8 | Jaxson Hayes | 2.0 | Ja Morant | 19.2 |
| 9 | Rui Hachimura | 0.1 | Jaxson Hayes | 2.0 |
| 10 | Cam Reddish | -0.9 | De'Andre Hunter | 2.4 |
| 11 | Cameron Johnson | 17.2 | Grant Williams | 7.2 |
| 12 | P.J. Washington | 7.0 | Ty Jerome | 1.3 |
| 13 | Tyler Herro | 12.5 | Chuma Okeke | 2.3 |
| 14 | Romeo Langford | 0.9 | P.J. Washington | 7.0 |
| 15 | Sekou Doumbouya | -3.6 | Bol Bol | 1.9 |
| 16 | Chuma Okeke | 2.3 | Nickeil Alexander-Walker | 4.8 |
| 17 | Nickeil Alexander-Walker | 4.8 | Cameron Johnson | 17.2 |
| 18 | Goga Bitadze | 3.7 | Dylan Windler | 1.0 |
| 19 | Luka Šamanić | -0.7 | RJ Barrett | 6.1 |
| 20 | Matisse Thybulle | 15.6 | Sekou Doumbouya | -3.6 |
| 21 | Brandon Clarke | 12.3 | Nic Claxton | 16.3 |
| 22 | Grant Williams | 7.2 | Darius Garland | 15.3 |
| 23 | Darius Bazley | -2.2 | Luka Šamanić | -0.7 |
| 24 | Ty Jerome | 1.3 | Bruno Fernando | -0.6 |
| 25 | Nassir Little | 1.2 | Isaiah Roby | -2.1 |
| 26 | Dylan Windler | 1.0 | Cam Reddish | -0.9 |
| 27 | Mfiondu Kabengele | -0.1 | Keldon Johnson | 7.2 |
| 28 | Jordan Poole | 3.7 | Alen Smailagić | -0.2 |
| 29 | Keldon Johnson | 7.2 | Daniel Gafford | 13.1 |
| 30 | Kevin Porter Jr. | 7.0 | Tremont Waters | -0.3 |
| 31 | Nic Claxton | 16.3 | Terance Mann | 11.8 |
| 32 | KZ Okpala | -1.2 | Jalen McDaniels | 0.3 |
| 33 | Carsen Edwards | -0.6 | Carsen Edwards | -0.6 |
| 34 | Bruno Fernando | -0.6 | Quinndary Weatherspoon | -0.9 |
| 35 | Didi Louzada | -0.8 | Romeo Langford | 0.9 |
| 36 | Cody Martin | 7.4 | Eric Paschall | -2.1 |
| 37 | Deividas Sirvydis | -0.0 | Nassir Little | 1.2 |
| 38 | Daniel Gafford | 13.1 | Deividas Sirvydis | -0.0 |
| 39 | Alen Smailagić | -0.2 | Didi Louzada | -0.8 |
| 40 | Justin James | -0.2 | Darius Bazley | -2.2 |
| 41 | Eric Paschall | -2.1 | Kyle Guy | -0.3 |
| 42 | Admiral Schofield | -1.0 | Cody Martin | 7.4 |
| 43 | Jaylen Nowell | 1.0 | Mfiondu Kabengele | -0.1 |
| 44 | Bol Bol | 1.9 | Vanja Marinković | 0.0 |
| 45 | Isaiah Roby | -2.1 | Dewan Hernandez | 0.1 |
| 46 | Talen Horton-Tucker | 3.7 | Talen Horton-Tucker | 3.7 |
| 47 | Ignas Brazdeikis | -0.8 | Jordan Poole | 3.7 |
| 48 | Terance Mann | 11.8 | Kevin Porter Jr. | 7.0 |
| 49 | Quinndary Weatherspoon | -0.9 | Admiral Schofield | -1.0 |
| 50 | Jarrell Brantley | -0.1 | Rui Hachimura | 0.1 |
| 51 | Tremont Waters | -0.3 | Jaylen Nowell | 1.0 |
| 52 | Jalen McDaniels | 0.3 | Ignas Brazdeikis | -0.8 |
| 53 | Justin Wright-Foreman | -0.2 | Justin James | -0.2 |
| 54 | Marial Shayok | -0.1 | Miye Oni | -0.0 |
| 55 | Kyle Guy | -0.3 | Marial Shayok | -0.1 |
| 56 | Jaylen Hands | 0.0 | Jarrell Brantley | -0.1 |
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
| 10 | Jalen Smith | 5.6 | Desmond Bane | 26.9 |
| 11 | Devin Vassell | 8.1 | Saddiq Bey | 7.1 |
| 12 | Tyrese Haliburton | 38.5 | Patrick Williams | 1.2 |
| 13 | Kira Lewis Jr. | -0.0 | LaMelo Ball | 18.1 |
| 14 | Aaron Nesmith | 5.6 | Malachi Flynn | 1.4 |
| 15 | Cole Anthony | 5.8 | Tyrell Terry | 0.1 |
| 16 | Isaiah Stewart | 6.0 | Isaac Okoro | 6.3 |
| 17 | Aleksej Pokusevski | -3.1 | Payton Pritchard | 14.7 |
| 18 | Josh Green | 2.1 | Kira Lewis Jr. | -0.0 |
| 19 | Saddiq Bey | 7.1 | Tyrese Maxey | 21.9 |
| 20 | Precious Achiuwa | 3.6 | Tre Jones | 9.4 |
| 21 | Tyrese Maxey | 21.9 | Precious Achiuwa | 3.6 |
| 22 | Zeke Nnaji | -0.4 | Udoka Azubuike | 0.1 |
| 23 | Leandro Bolmaro | -0.5 | Aaron Nesmith | 5.6 |
| 24 | R.J. Hampton | -4.5 | Jahmi'us Ramsey | -0.6 |
| 25 | Immanuel Quickley | 19.5 | Paul Reed | 6.7 |
| 26 | Payton Pritchard | 14.7 | Deni Avdija | 13.4 |
| 27 | Udoka Azubuike | 0.1 | Cole Anthony | 5.8 |
| 28 | Jaden McDaniels | 12.8 | Marko Simonovic | -0.0 |
| 29 | Malachi Flynn | 1.4 | Isaiah Joe | 10.9 |
| 30 | Desmond Bane | 26.9 | Isaiah Stewart | 6.0 |
| 31 | Tyrell Terry | 0.1 | Killian Hayes | -4.1 |
| 32 | Vernon Carey Jr. | -0.1 | Justinian Jessup | 0.0 |
| 33 | Daniel Oturu | -0.5 | CJ Elleby | -1.7 |
| 34 | Théo Maledon | -6.3 | Vít Krejčí | 0.6 |
| 35 | Xavier Tillman Sr. | 5.3 | Leandro Bolmaro | -0.5 |
| 36 | Tyler Bey | -0.4 | Elijah Hughes | -1.8 |
| 37 | Vít Krejčí | 0.6 | Skylar Mays | 0.6 |
| 38 | Saben Lee | 0.0 | R.J. Hampton | -4.5 |
| 39 | Elijah Hughes | -1.8 | Zeke Nnaji | -0.4 |
| 40 | Robert Woodard II | -0.0 | Yam Madar | 0.0 |
| 41 | Tre Jones | 9.4 | Vernon Carey Jr. | -0.1 |
| 42 | Nick Richards | 0.4 | Daniel Oturu | -0.5 |
| 43 | Jahmi'us Ramsey | -0.6 | Tyler Bey | -0.4 |
| 44 | Marko Simonovic | -0.0 | Reggie Perry | -0.9 |
| 45 | Jordan Nwora | -1.9 | Jalen Harris | 0.3 |
| 46 | CJ Elleby | -1.7 | KJ Martin | 0.3 |
| 47 | Yam Madar | 0.0 | Immanuel Quickley | 19.5 |
| 48 | Nico Mannion | -0.3 | Jay Scrubb | -0.5 |
| 49 | Isaiah Joe | 10.9 | Sam Merrill | 4.0 |
| 50 | Skylar Mays | 0.6 | Nick Richards | 0.4 |
| 51 | Justinian Jessup | 0.0 | Jaden McDaniels | 12.8 |
| 52 | KJ Martin | 0.3 | Théo Maledon | -6.3 |
| 53 | Cassius Winston | -0.6 | Jordan Nwora | -1.9 |
| 54 | Cassius Stanley | -0.1 | Robert Woodard II | -0.0 |
| 55 | Jay Scrubb | -0.5 | Grant Riller | 0.0 |
| 56 | Grant Riller | 0.0 | Saben Lee | 0.0 |
| 57 | Reggie Perry | -0.9 | Aleksej Pokusevski | -3.1 |
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
| 7 | Jonathan Kuminga | 3.5 | Isaiah Jackson | 3.7 |
| 8 | Franz Wagner | 25.7 | Corey Kispert | 2.0 |
| 9 | Davion Mitchell | 5.8 | Kai Jones | 0.6 |
| 10 | Ziaire Williams | 1.0 | Jared Butler | 0.7 |
| 11 | James Bouknight | -1.4 | Jalen Johnson | 17.0 |
| 12 | Joshua Primo | -1.1 | Miles McBride | 6.7 |
| 13 | Chris Duarte | 1.6 | Alperen Şengün | 29.5 |
| 14 | Moses Moody | 5.6 | Day'Ron Sharpe | 6.3 |
| 15 | Corey Kispert | 2.0 | Moses Moody | 5.6 |
| 16 | Alperen Şengün | 29.5 | Neemias Queta | 7.8 |
| 17 | Trey Murphy III | 20.7 | Davion Mitchell | 5.8 |
| 18 | Tre Mann | -1.4 | Filip Petrušev | -0.0 |
| 19 | Kai Jones | 0.6 | Jeremiah Robinson-Earl | -1.0 |
| 20 | Jalen Johnson | 17.0 | James Bouknight | -1.4 |
| 21 | Keon Johnson | -3.7 | Scottie Lewis | 0.0 |
| 22 | Isaiah Jackson | 3.7 | Ayo Dosunmu | 6.6 |
| 23 | Usman Garuba | 0.9 | Jaden Springer | 0.2 |
| 24 | Josh Christopher | -1.8 | Quentin Grimes | 9.5 |
| 25 | Quentin Grimes | 9.5 | Bones Hyland | 6.2 |
| 26 | Bones Hyland | 6.2 | Trey Murphy III | 20.7 |
| 27 | Cam Thomas | 0.7 | Sandro Mamukelashvili | 5.6 |
| 28 | Jaden Springer | 0.2 | Jason Preston | 0.0 |
| 29 | Day'Ron Sharpe | 6.3 | Luka Garza | 3.7 |
| 30 | Santi Aldama | 11.4 | Joe Wieskamp | -0.4 |
| 31 | Isaiah Todd | -0.6 | Isaiah Livers | -0.5 |
| 32 | Jeremiah Robinson-Earl | -1.0 | Dalano Banton | 2.0 |
| 33 | Jason Preston | 0.0 | Aaron Wiggins | 7.6 |
| 34 | Rokas Jokubaitis | 0.0 | Jalen Green | 7.5 |
| 35 | Herbert Jones | 15.1 | Herbert Jones | 15.1 |
| 36 | Miles McBride | 6.7 | Josh Christopher | -1.8 |
| 37 | JT Thor | -2.2 | Jonathan Kuminga | 3.5 |
| 38 | Ayo Dosunmu | 6.6 | Usman Garuba | 0.9 |
| 39 | Neemias Queta | 7.8 | Kessler Edwards | -0.2 |
| 40 | Jared Butler | 0.7 | David Johnson | 0.0 |
| 41 | Joe Wieskamp | -0.4 | Balša Koprivica | 0.0 |
| 42 | Isaiah Livers | -0.5 | Sharife Cooper | -1.0 |
| 43 | Greg Brown III | -2.2 | JT Thor | -2.2 |
| 44 | Kessler Edwards | -0.2 | Josh Giddey | 19.6 |
| 45 | Juhann Begarin | 0.0 | Charles Bassey | 1.8 |
| 46 | Dalano Banton | 2.0 | Rokas Jokubaitis | 0.0 |
| 47 | David Johnson | 0.0 | Ziaire Williams | 1.0 |
| 48 | Sharife Cooper | -1.0 | Santi Aldama | 11.4 |
| 49 | Marcus Zegarowski | 0.0 | Marcus Zegarowski | 0.0 |
| 50 | Filip Petrušev | -0.0 | RaiQuan Gray | 0.2 |
| 51 | Brandon Boston Jr. | -1.1 | Brandon Boston Jr. | -1.1 |
| 52 | Luka Garza | 3.7 | Keon Johnson | -3.7 |
| 53 | Charles Bassey | 1.8 | Georgios Kalaitzakis | -1.1 |
| 54 | Sandro Mamukelashvili | 5.6 | Isaiah Todd | -0.6 |
| 55 | Aaron Wiggins | 7.6 | Juhann Begarin | 0.0 |
| 56 | Scottie Lewis | 0.0 | Jericho Sims | 0.9 |
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
| 4 | Keegan Murray | 8.4 | Jeremy Sochan | -0.1 |
| 5 | Jaden Ivey | -0.5 | Jalen Duren | 17.1 |
| 6 | Bennedict Mathurin | -1.3 | Tari Eason | 8.6 |
| 7 | Shaedon Sharpe | 2.7 | Paolo Banchero | 18.5 |
| 8 | Dyson Daniels | 15.4 | Bennedict Mathurin | -1.3 |
| 9 | Jeremy Sochan | -0.1 | Jake LaRavia | 3.5 |
| 10 | Johnny Davis | -1.9 | Mark Williams | 8.4 |
| 11 | Ousmane Dieng | -0.6 | Dalen Terry | 0.8 |
| 12 | Jalen Williams | 22.4 | Walker Kessler | 12.9 |
| 13 | Jalen Duren | 17.1 | Christian Braun | 8.1 |
| 14 | Ochai Agbaji | -0.8 | Shaedon Sharpe | 2.7 |
| 15 | Mark Williams | 8.4 | E.J. Liddell | -0.2 |
| 16 | AJ Griffin | 0.7 | Kennedy Chandler | -0.6 |
| 17 | Tari Eason | 8.6 | Josh Minott | 3.2 |
| 18 | Dalen Terry | 0.8 | Christian Koloko | -0.3 |
| 19 | Jake LaRavia | 3.5 | Karlo Matković | 1.7 |
| 20 | Malaki Branham | -5.2 | Jaden Ivey | -0.5 |
| 21 | Christian Braun | 8.1 | Jaylin Williams | 8.7 |
| 22 | Walker Kessler | 12.9 | Jalen Williams | 22.4 |
| 23 | David Roddy | -2.4 | Wendell Moore Jr. | -0.1 |
| 24 | MarJon Beauchamp | -1.2 | Dyson Daniels | 15.4 |
| 25 | Blake Wesley | -3.0 | Johnny Davis | -1.9 |
| 26 | Wendell Moore Jr. | -0.1 | Vince Williams Jr. | 2.5 |
| 27 | Nikola Jović | 3.2 | TyTy Washington Jr. | -0.9 |
| 28 | Patrick Baldwin Jr. | -0.1 | Khalifa Diop | 0.0 |
| 29 | TyTy Washington Jr. | -0.9 | Ismael Kamagate | 0.0 |
| 30 | Peyton Watson | 2.2 | David Roddy | -2.4 |
| 31 | Andrew Nembhard | 5.1 | Andrew Nembhard | 5.1 |
| 32 | Caleb Houstan | 0.1 | Ochai Agbaji | -0.8 |
| 33 | Christian Koloko | -0.3 | Nikola Jović | 3.2 |
| 34 | Jaylin Williams | 8.7 | AJ Griffin | 0.7 |
| 35 | Max Christie | 0.4 | Gabriele Procida | 0.0 |
| 36 | Gabriele Procida | 0.0 | Luke Travers | -0.3 |
| 37 | Jaden Hardy | -2.3 | Malaki Branham | -5.2 |
| 38 | Kennedy Chandler | -0.6 | Ousmane Dieng | -0.6 |
| 39 | Khalifa Diop | 0.0 | Kendall Brown | -0.2 |
| 40 | Bryce McGowens | -2.3 | Isaiah Mobley | -0.0 |
| 41 | E.J. Liddell | -0.2 | Jabari Walker | -1.2 |
| 42 | Trevor Keels | -0.1 | Gui Santos | 2.4 |
| 43 | Moussa Diabaté | 6.1 | JD Davison | 0.1 |
| 44 | Ryan Rollins | 3.8 | Tyrese Martin | -1.0 |
| 45 | Josh Minott | 3.2 | Ryan Rollins | 3.8 |
| 46 | Ismael Kamagate | 0.0 | MarJon Beauchamp | -1.2 |
| 47 | Vince Williams Jr. | 2.5 | Trevor Keels | -0.1 |
| 48 | Kendall Brown | -0.2 | Hugo Besson | 0.0 |
| 49 | Isaiah Mobley | -0.0 | Matteo Spagnolo | 0.0 |
| 50 | Matteo Spagnolo | 0.0 | Peyton Watson | 2.2 |
| 51 | Tyrese Martin | -1.0 | Yannick Nzosa | 0.0 |
| 52 | Karlo Matković | 1.7 | Caleb Houstan | 0.1 |
| 53 | JD Davison | 0.1 | Jaden Hardy | -2.3 |
| 54 | Yannick Nzosa | 0.0 | Moussa Diabaté | 6.1 |
| 55 | Gui Santos | 2.4 | Max Christie | 0.4 |
| 56 | Luke Travers | -0.3 | Patrick Baldwin Jr. | -0.1 |
| 57 | Jabari Walker | -1.2 | Blake Wesley | -3.0 |
| 58 | Hugo Besson | 0.0 | Bryce McGowens | -2.3 |

### 2023

| # | Actual draft | 5yr WAR | Model redraft | 5yr WAR |
|--:|:-------------|--------:|:--------------|--------:|
| 1 | Victor Wembanyama | 32.1 | Victor Wembanyama | 32.1 |
| 2 | Brandon Miller | 6.0 | Dereck Lively II | 5.7 |
| 3 | Scoot Henderson | -1.9 | Brandon Miller | 6.0 |
| 4 | Amen Thompson | 20.4 | Anthony Black | 4.2 |
| 5 | Ausar Thompson | 10.7 | Jarace Walker | 0.3 |
| 6 | Anthony Black | 4.2 | Cason Wallace | 13.7 |
| 7 | Bilal Coulibaly | -2.9 | Brandin Podziemski | 9.1 |
| 8 | Jarace Walker | 0.3 | Trayce Jackson-Davis | 4.7 |
| 9 | Taylor Hendricks | -0.2 | Jaime Jaquez Jr. | 5.7 |
| 10 | Cason Wallace | 13.7 | Amen Thompson | 20.4 |
| 11 | Jett Howard | -0.4 | Andre Jackson Jr. | -1.6 |
| 12 | Dereck Lively II | 5.7 | Jaylen Clark | 1.0 |
| 13 | Gradey Dick | -3.5 | Taylor Hendricks | -0.2 |
| 14 | Jordan Hawkins | -3.5 | Gradey Dick | -3.5 |
| 15 | Kobe Bufkin | -0.6 | Jalen Pickett | 0.5 |
| 16 | Keyonte George | -1.0 | Jordan Miller | 2.2 |
| 17 | Jalen Hood-Schifino | -1.2 | Scoot Henderson | -1.9 |
| 18 | Jaime Jaquez Jr. | 5.7 | Jalen Slawson | 0.3 |
| 19 | Brandin Podziemski | 9.1 | Kobe Bufkin | -0.6 |
| 20 | Cam Whitmore | 2.1 | Cam Whitmore | 2.1 |
| 21 | Noah Clowney | -1.1 | Noah Clowney | -1.1 |
| 22 | Dariq Whitehead | -0.8 | Colby Jones | -0.2 |
| 23 | Kris Murray | 0.4 | Ausar Thompson | 10.7 |
| 24 | Olivier-Maxence Prosper | 0.8 | Leonard Miller | 0.9 |
| 25 | Marcus Sasser | 0.7 | Marcus Sasser | 0.7 |
| 26 | Ben Sheppard | 0.3 | Kris Murray | 0.4 |
| 27 | Nick Smith Jr. | -4.4 | Bilal Coulibaly | -2.9 |
| 28 | Brice Sensabaugh | -0.6 | Kobe Brown | -0.3 |
| 29 | Julian Strawther | -1.6 | James Nnaji | 0.0 |
| 30 | Kobe Brown | -0.3 | Hunter Tyson | -0.6 |
| 31 | James Nnaji | 0.0 | Julian Phillips | -0.4 |
| 32 | Jalen Pickett | 0.5 | Mouhamed Gueye | 1.5 |
| 33 | Leonard Miller | 0.9 | Toumani Camara | 6.8 |
| 34 | Colby Jones | -0.2 | Sidy Cissoko | -1.1 |
| 35 | Julian Phillips | -0.4 | Tarik Biberovic | 0.0 |
| 36 | Andre Jackson Jr. | -1.6 | Ben Sheppard | 0.3 |
| 37 | Hunter Tyson | -0.6 | Isaiah Wong | -0.2 |
| 38 | Jordan Walsh | 2.1 | Mojave King | 0.0 |
| 39 | Mouhamed Gueye | 1.5 | Keyonte George | -1.0 |
| 40 | Maxwell Lewis | -0.7 | Jett Howard | -0.4 |
| 41 | Amari Bailey | -0.2 | Rayan Rupert | -3.6 |
| 42 | Tristan Vukcevic | 1.1 | Tristan Vukcevic | 1.1 |
| 43 | Rayan Rupert | -3.6 | Julian Strawther | -1.6 |
| 44 | Sidy Cissoko | -1.1 | Seth Lundy | -0.2 |
| 45 | GG Jackson II | -0.0 | Jordan Walsh | 2.1 |
| 46 | Seth Lundy | -0.2 | Jalen Hood-Schifino | -1.2 |
| 47 | Mojave King | 0.0 | Amari Bailey | -0.2 |
| 48 | Jordan Miller | 2.2 | Jalen Wilson | -3.1 |
| 49 | Emoni Bates | -0.3 | Keyontae Johnson | -0.2 |
| 50 | Keyontae Johnson | -0.2 | Jordan Hawkins | -3.5 |
| 51 | Jalen Wilson | -3.1 | Olivier-Maxence Prosper | 0.8 |
| 52 | Toumani Camara | 6.8 | Chris Livingston | -0.4 |
| 53 | Jaylen Clark | 1.0 | Dariq Whitehead | -0.8 |
| 54 | Jalen Slawson | 0.3 | Brice Sensabaugh | -0.6 |
| 55 | Isaiah Wong | -0.2 | Emoni Bates | -0.3 |
| 56 | Tarik Biberovic | 0.0 | Nick Smith Jr. | -4.4 |
| 57 | Trayce Jackson-Davis | 4.7 | Maxwell Lewis | -0.7 |
| 58 | Chris Livingston | -0.4 | GG Jackson II | -0.0 |

### 2024

| # | Actual draft | 5yr WAR | Model redraft | 5yr WAR |
|--:|:-------------|--------:|:--------------|--------:|
| 1 | Zaccharie Risacher | -0.8 | Reed Sheppard | 5.9 |
| 2 | Alex Sarr | 0.9 | Donovan Clingan | 8.6 |
| 3 | Reed Sheppard | 5.9 | Zach Edey | 3.1 |
| 4 | Stephon Castle | 5.5 | Jonathan Mogbo | 0.4 |
| 5 | Ron Holland | 1.1 | Adem Bona | -0.1 |
| 6 | Tidjane Salaün | -1.1 | Devin Carter | 0.0 |
| 7 | Donovan Clingan | 8.6 | Ryan Dunn | 0.2 |
| 8 | Rob Dillingham | -2.4 | Stephon Castle | 5.5 |
| 9 | Zach Edey | 3.1 | Anton Watson | -0.1 |
| 10 | Cody Williams | -5.5 | DaRon Holmes | 0.2 |
| 11 | Matas Buzelis | 1.8 | Cam Spencer | 4.9 |
| 12 | Nikola Topić | -0.4 | Tyler Kolek | 1.3 |
| 13 | Devin Carter | 0.0 | Oso Ighodaro | 3.3 |
| 14 | Bub Carrington | -6.6 | Baylor Scheierman | 3.2 |
| 15 | Kel'el Ware | 6.5 | Kyle Filipowski | 2.4 |
| 16 | Jared McCain | 0.7 | Matas Buzelis | 1.8 |
| 17 | Dalton Knecht | 0.3 | Jamal Shead | -0.7 |
| 18 | Tristan Da Silva | 1.7 | Jared McCain | 0.7 |
| 19 | Ja'Kobe Walter | 2.1 | Kel'el Ware | 6.5 |
| 20 | Jaylon Tyson | 2.9 | Tidjane Salaün | -1.1 |
| 21 | Yves Missi | -0.6 | Dillon Jones | -0.1 |
| 22 | DaRon Holmes | 0.2 | Tristen Newton | -0.1 |
| 23 | AJ Johnson | -3.3 | Alex Sarr | 0.9 |
| 24 | Kyshawn George | 0.2 | Nikola Topić | -0.4 |
| 25 | Pacôme Dadiet | -0.1 | Ariel Hukporti | -0.1 |
| 26 | Dillon Jones | -0.1 | Kevin McCullar Jr. | -0.1 |
| 27 | Terrence Shannon Jr. | 0.1 | KJ Simpson | -1.4 |
| 28 | Ryan Dunn | 0.2 | Terrence Shannon Jr. | 0.1 |
| 29 | Isaiah Collier | -2.0 | Pelle Larsson | 2.1 |
| 30 | Baylor Scheierman | 3.2 | Tristan Da Silva | 1.7 |
| 31 | Jonathan Mogbo | 0.4 | Jaylon Tyson | 2.9 |
| 32 | Kyle Filipowski | 2.4 | Harrison Ingram | 0.1 |
| 33 | Tyler Smith | -0.1 | Tyler Smith | -0.1 |
| 34 | Tyler Kolek | 1.3 | Zaccharie Risacher | -0.8 |
| 35 | Johnny Furphy | -0.2 | Enrique Freeman | -0.2 |
| 36 | Juan Nunez | 0.0 | Kyshawn George | 0.2 |
| 37 | Bobi Klintman | -0.1 | Quinten Post | 1.6 |
| 38 | Ajay Mitchell | 4.5 | Ron Holland | 1.1 |
| 39 | Jaylen Wells | 1.2 | Juan Nunez | 0.0 |
| 40 | Oso Ighodaro | 3.3 | Johnny Furphy | -0.2 |
| 41 | Adem Bona | -0.1 | Ajay Mitchell | 4.5 |
| 42 | KJ Simpson | -1.4 | Jaylen Wells | 1.2 |
| 43 | Nikola Djurisic | 0.0 | Bub Carrington | -6.6 |
| 44 | Pelle Larsson | 2.1 | Bobi Klintman | -0.1 |
| 45 | Jamal Shead | -0.7 | Melvin Ajinça | 0.0 |
| 46 | Cam Christie | -0.4 | Yves Missi | -0.6 |
| 47 | Antonio Reeves | -0.2 | Ja'Kobe Walter | 2.1 |
| 48 | Harrison Ingram | 0.1 | Nikola Djurisic | 0.0 |
| 49 | Tristen Newton | -0.1 | Bronny James | -0.6 |
| 50 | Enrique Freeman | -0.2 | Rob Dillingham | -2.4 |
| 51 | Melvin Ajinça | 0.0 | Ulrich Chomche | -0.1 |
| 52 | Quinten Post | 1.6 | Antonio Reeves | -0.2 |
| 53 | Cam Spencer | 4.9 | AJ Johnson | -3.3 |
| 54 | Anton Watson | -0.1 | Dalton Knecht | 0.3 |
| 55 | Bronny James | -0.6 | Cody Williams | -5.5 |
| 56 | Kevin McCullar Jr. | -0.1 | Isaiah Collier | -2.0 |
| 57 | Ulrich Chomche | -0.1 | Pacôme Dadiet | -0.1 |
| 58 | Ariel Hukporti | -0.1 | Cam Christie | -0.4 |

### 2025

| # | Actual draft | 5yr WAR | Model redraft | 5yr WAR |
|--:|:-------------|--------:|:--------------|--------:|
| 1 | Cooper Flagg | 5.0 | Cooper Flagg | 5.0 |
| 2 | Dylan Harper | 4.4 | VJ Edgecombe | 4.3 |
| 3 | VJ Edgecombe | 4.3 | Collin Murray-Boyles | 2.7 |
| 4 | Kon Knueppel | 7.4 | Dylan Harper | 4.4 |
| 5 | Ace Bailey | -2.4 | Kon Knueppel | 7.4 |
| 6 | Tre Johnson | -2.0 | Thomas Sorber | 0.0 |
| 7 | Jeremiah Fears | -0.6 | Ryan Kalkbrenner | 3.3 |
| 8 | Egor Dëmin | 0.8 | Johni Broome | -0.3 |
| 9 | Collin Murray-Boyles | 2.7 | Cedric Coward | 2.3 |
| 10 | Khaman Maluach | -0.3 | Kasparas Jakučionis | 1.0 |
| 11 | Cedric Coward | 2.3 | Carter Bryant | 0.1 |
| 12 | Noa Essengue | -0.1 | Noa Essengue | -0.1 |
| 13 | Derik Queen | 2.3 | Nique Clifford | -2.7 |
| 14 | Carter Bryant | 0.1 | Danny Wolf | -0.1 |
| 15 | Thomas Sorber | 0.0 | Asa Newell | 0.3 |
| 16 | Yang Hansen | -1.0 | Khaman Maluach | -0.3 |
| 17 | Joan Beringer | 0.6 | Kam Jones | -1.3 |
| 18 | Walter Clayton | -1.4 | Rasheer Fleming | -0.2 |
| 19 | Nolan Traoré | -3.1 | Derik Queen | 2.3 |
| 20 | Kasparas Jakučionis | 1.0 | Walter Clayton | -1.4 |
| 21 | Will Riley | -2.0 | Bogoljub Markovic | 0.0 |
| 22 | Drake Powell | -2.5 | Max Shulga | -0.1 |
| 23 | Asa Newell | 0.3 | Adou Thiero | -0.2 |
| 24 | Nique Clifford | -2.7 | Jase Richardson | 0.1 |
| 25 | Jase Richardson | 0.1 | Maxime Raynaud | -0.3 |
| 26 | Ben Saraf | -2.5 | Micah Peavy | -1.1 |
| 27 | Danny Wolf | -0.1 | Yang Hansen | -1.0 |
| 28 | Hugo González | 0.6 | Joan Beringer | 0.6 |
| 29 | Liam McNeeley | 0.2 | Amari Williams | 0.0 |
| 30 | Yanic Konan Niederhäuser | 0.2 | Koby Brea | 0.1 |
| 31 | Rasheer Fleming | -0.2 | Brooks Barnhizer | -0.1 |
| 32 | Noah Penda | 0.6 | Javon Small | 1.3 |
| 33 | Sion James | 0.6 | Sion James | 0.6 |
| 34 | Ryan Kalkbrenner | 3.3 | Drake Powell | -2.5 |
| 35 | Johni Broome | -0.3 | Jahmai Mashack | -1.8 |
| 36 | Adou Thiero | -0.2 | Rocco Zikarsky | 0.1 |
| 37 | Chaz Lanier | -0.3 | Noah Penda | 0.6 |
| 38 | Kam Jones | -1.3 | Will Richard | 1.0 |
| 39 | Alijah Martin | -0.1 | Alex Toohey | 0.0 |
| 40 | Micah Peavy | -1.1 | Alijah Martin | -0.1 |
| 41 | Koby Brea | 0.1 | Lachlan Olbrich | -0.2 |
| 42 | Maxime Raynaud | -0.3 | Jamir Watkins | -0.5 |
| 43 | Jamir Watkins | -0.5 | Hugo González | 0.6 |
| 44 | Brooks Barnhizer | -0.1 | Kobe Sanders | 0.0 |
| 45 | Rocco Zikarsky | 0.1 | Yanic Konan Niederhäuser | 0.2 |
| 46 | Amari Williams | 0.0 | Jeremiah Fears | -0.6 |
| 47 | Bogoljub Markovic | 0.0 | Tyrese Proctor | -0.2 |
| 48 | Javon Small | 1.3 | Ace Bailey | -2.4 |
| 49 | Tyrese Proctor | -0.2 | Tre Johnson | -2.0 |
| 50 | Kobe Sanders | 0.0 | John Tonje | -0.0 |
| 51 | Mohamed Diawara | 0.3 | Mohamed Diawara | 0.3 |
| 52 | Alex Toohey | 0.0 | Taelon Peter | -0.6 |
| 53 | John Tonje | -0.0 | Ben Saraf | -2.5 |
| 54 | Taelon Peter | -0.6 | Egor Dëmin | 0.8 |
| 55 | Lachlan Olbrich | -0.2 | Saliou Niang | 0.0 |
| 56 | Will Richard | 1.0 | Liam McNeeley | 0.2 |
| 57 | Max Shulga | -0.1 | Nolan Traoré | -3.1 |
| 58 | Saliou Niang | 0.0 | Will Riley | -2.0 |
| 59 | Jahmai Mashack | -1.8 | Chaz Lanier | -0.3 |
