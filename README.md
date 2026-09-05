# nba-redraft

## AI models used

One model, two context windows, rank-averaged 50/50:

| Model | What it is |
|:------|:-----------|
| **TabICL v2** | tabular in-context-learning foundation model (`TabICLRegressor`, 8 estimators), context from 2003 |
| **TabICL v2** | same model, context from 2010 |

## Holdout (2019-2025)

| AI model accuracy | NBA scouts accuracy | Drafts the AI won |
|:-----------------:|:-------------------:|:-----------------:|
| **45%**           | 26%                 | **7 of 7**        |

## By draft

| Draft | AI model | NBA scouts | Difference |
|:------|---------:|-----------:|-----------:|
| 2019  | 45%      | 40%        | +5         |
| 2020  | 47%      | 35%        | +12         |
| 2021  | 53%      | 41%        | +12         |
| 2022  | 38%      | 24%        | +14         |
| 2023  | 55%      | 10%        | +45         |
| 2024  | 46%      | 15%        | +31         |
| 2025  | 31%      | 17%        | +14         |
| **Mean** | **45%** | **26%**   | **+19**    |

## By player

Every holdout draft, actual order beside the model's reordering of the same names.

### 2019

| # | Actual draft | 5yr WAR | Model redraft | 5yr WAR |
|--:|:-------------|--------:|:--------------|--------:|
| 1 | Zion Williamson | 22.1 | Zion Williamson | 22.1 |
| 2 | Ja Morant | 19.2 | Jarrett Culver | -1.4 |
| 3 | RJ Barrett | 6.1 | Brandon Clarke | 12.3 |
| 4 | De'Andre Hunter | 2.4 | Chuma Okeke | 2.3 |
| 5 | Darius Garland | 15.3 | Ja Morant | 19.2 |
| 6 | Jarrett Culver | -1.4 | Tyler Herro | 12.5 |
| 7 | Coby White | 6.9 | Bol Bol | 1.9 |
| 8 | Jaxson Hayes | 2.0 | Goga Bitadze | 3.7 |
| 9 | Rui Hachimura | 0.1 | Coby White | 6.9 |
| 10 | Cam Reddish | -0.9 | Grant Williams | 7.2 |
| 11 | Cameron Johnson | 17.2 | Jaxson Hayes | 2.0 |
| 12 | P.J. Washington | 7.0 | Matisse Thybulle | 15.6 |
| 13 | Tyler Herro | 12.5 | Alen Smailagić | -0.2 |
| 14 | Romeo Langford | 0.9 | P.J. Washington | 7.0 |
| 15 | Sekou Doumbouya | -3.6 | Ty Jerome | 1.3 |
| 16 | Chuma Okeke | 2.3 | Dylan Windler | 1.0 |
| 17 | Nickeil Alexander-Walker | 4.8 | De'Andre Hunter | 2.4 |
| 18 | Goga Bitadze | 3.7 | Nic Claxton | 16.3 |
| 19 | Luka Šamanić | -0.7 | Sekou Doumbouya | -3.6 |
| 20 | Matisse Thybulle | 15.6 | Cameron Johnson | 17.2 |
| 21 | Brandon Clarke | 12.3 | Nickeil Alexander-Walker | 4.8 |
| 22 | Grant Williams | 7.2 | RJ Barrett | 6.1 |
| 23 | Darius Bazley | -2.2 | Bruno Fernando | -0.6 |
| 24 | Ty Jerome | 1.3 | Isaiah Roby | -2.1 |
| 25 | Nassir Little | 1.2 | Terance Mann | 11.8 |
| 26 | Dylan Windler | 1.0 | Daniel Gafford | 13.1 |
| 27 | Mfiondu Kabengele | -0.1 | Keldon Johnson | 7.2 |
| 28 | Jordan Poole | 3.7 | Tremont Waters | -0.3 |
| 29 | Keldon Johnson | 7.2 | Jalen McDaniels | 0.3 |
| 30 | Kevin Porter Jr. | 7.0 | Cam Reddish | -0.9 |
| 31 | Nic Claxton | 16.3 | Quinndary Weatherspoon | -0.9 |
| 32 | KZ Okpala | -1.2 | Romeo Langford | 0.9 |
| 33 | Carsen Edwards | -0.6 | Luka Šamanić | -0.7 |
| 34 | Bruno Fernando | -0.6 | Cody Martin | 7.4 |
| 35 | Didi Louzada | -0.8 | Carsen Edwards | -0.6 |
| 36 | Cody Martin | 7.4 | Darius Garland | 15.3 |
| 37 | Deividas Sirvydis | -0.0 | Darius Bazley | -2.2 |
| 38 | Daniel Gafford | 13.1 | Deividas Sirvydis | -0.0 |
| 39 | Alen Smailagić | -0.2 | Kyle Guy | -0.3 |
| 40 | Justin James | -0.2 | Talen Horton-Tucker | 3.7 |
| 41 | Eric Paschall | -2.1 | Vanja Marinković | 0.0 |
| 42 | Admiral Schofield | -1.0 | Admiral Schofield | -1.0 |
| 43 | Jaylen Nowell | 1.0 | Jordan Poole | 3.7 |
| 44 | Bol Bol | 1.9 | Didi Louzada | -0.8 |
| 45 | Isaiah Roby | -2.1 | Eric Paschall | -2.1 |
| 46 | Talen Horton-Tucker | 3.7 | Mfiondu Kabengele | -0.1 |
| 47 | Ignas Brazdeikis | -0.8 | Dewan Hernandez | 0.1 |
| 48 | Terance Mann | 11.8 | Justin James | -0.2 |
| 49 | Quinndary Weatherspoon | -0.9 | Rui Hachimura | 0.1 |
| 50 | Jarrell Brantley | -0.1 | Jarrell Brantley | -0.1 |
| 51 | Tremont Waters | -0.3 | Nassir Little | 1.2 |
| 52 | Jalen McDaniels | 0.3 | Ignas Brazdeikis | -0.8 |
| 53 | Justin Wright-Foreman | -0.2 | Jordan Bone | -0.8 |
| 54 | Marial Shayok | -0.1 | Marial Shayok | -0.1 |
| 55 | Kyle Guy | -0.3 | Kevin Porter Jr. | 7.0 |
| 56 | Jaylen Hands | 0.0 | Miye Oni | -0.0 |
| 57 | Jordan Bone | -0.8 | Jaylen Hands | 0.0 |
| 58 | Miye Oni | -0.0 | Jaylen Nowell | 1.0 |
| 59 | Dewan Hernandez | 0.1 | Justin Wright-Foreman | -0.2 |
| 60 | Vanja Marinković | 0.0 | KZ Okpala | -1.2 |

### 2020

| # | Actual draft | 5yr WAR | Model redraft | 5yr WAR |
|--:|:-------------|--------:|:--------------|--------:|
| 1 | Anthony Edwards | 36.5 | James Wiseman | -4.2 |
| 2 | James Wiseman | -4.2 | Tyrese Haliburton | 38.5 |
| 3 | LaMelo Ball | 18.1 | Onyeka Okongwu | 14.5 |
| 4 | Patrick Williams | 1.2 | Devin Vassell | 8.1 |
| 5 | Isaac Okoro | 6.3 | Anthony Edwards | 36.5 |
| 6 | Onyeka Okongwu | 14.5 | Obi Toppin | 10.2 |
| 7 | Killian Hayes | -4.1 | Xavier Tillman Sr. | 5.3 |
| 8 | Obi Toppin | 10.2 | Jalen Smith | 5.6 |
| 9 | Deni Avdija | 13.4 | Tyrell Terry | 0.1 |
| 10 | Jalen Smith | 5.6 | Josh Green | 2.1 |
| 11 | Devin Vassell | 8.1 | LaMelo Ball | 18.1 |
| 12 | Tyrese Haliburton | 38.5 | Patrick Williams | 1.2 |
| 13 | Kira Lewis Jr. | -0.0 | Isaac Okoro | 6.3 |
| 14 | Aaron Nesmith | 5.6 | Desmond Bane | 26.9 |
| 15 | Cole Anthony | 5.8 | Malachi Flynn | 1.4 |
| 16 | Isaiah Stewart | 6.0 | Kira Lewis Jr. | -0.0 |
| 17 | Aleksej Pokusevski | -3.1 | Jahmi'us Ramsey | -0.6 |
| 18 | Josh Green | 2.1 | Paul Reed | 6.7 |
| 19 | Saddiq Bey | 7.1 | Saddiq Bey | 7.1 |
| 20 | Precious Achiuwa | 3.6 | Precious Achiuwa | 3.6 |
| 21 | Tyrese Maxey | 21.9 | Killian Hayes | -4.1 |
| 22 | Zeke Nnaji | -0.4 | Tre Jones | 9.4 |
| 23 | Leandro Bolmaro | -0.5 | Tyrese Maxey | 21.9 |
| 24 | R.J. Hampton | -4.5 | Isaiah Joe | 10.9 |
| 25 | Immanuel Quickley | 19.5 | Payton Pritchard | 14.7 |
| 26 | Payton Pritchard | 14.7 | Cole Anthony | 5.8 |
| 27 | Udoka Azubuike | 0.1 | Yam Madar | 0.0 |
| 28 | Jaden McDaniels | 12.8 | Udoka Azubuike | 0.1 |
| 29 | Malachi Flynn | 1.4 | Isaiah Stewart | 6.0 |
| 30 | Desmond Bane | 26.9 | Justinian Jessup | 0.0 |
| 31 | Tyrell Terry | 0.1 | Zeke Nnaji | -0.4 |
| 32 | Vernon Carey Jr. | -0.1 | Aaron Nesmith | 5.6 |
| 33 | Daniel Oturu | -0.5 | Deni Avdija | 13.4 |
| 34 | Théo Maledon | -6.3 | Marko Simonovic | -0.0 |
| 35 | Xavier Tillman Sr. | 5.3 | R.J. Hampton | -4.5 |
| 36 | Tyler Bey | -0.4 | Vít Krejčí | 0.6 |
| 37 | Vít Krejčí | 0.6 | CJ Elleby | -1.7 |
| 38 | Saben Lee | 0.0 | Théo Maledon | -6.3 |
| 39 | Elijah Hughes | -1.8 | Skylar Mays | 0.6 |
| 40 | Robert Woodard II | -0.0 | Jalen Harris | 0.3 |
| 41 | Tre Jones | 9.4 | KJ Martin | 0.3 |
| 42 | Nick Richards | 0.4 | Leandro Bolmaro | -0.5 |
| 43 | Jahmi'us Ramsey | -0.6 | Reggie Perry | -0.9 |
| 44 | Marko Simonovic | -0.0 | Vernon Carey Jr. | -0.1 |
| 45 | Jordan Nwora | -1.9 | Sam Merrill | 4.0 |
| 46 | CJ Elleby | -1.7 | Grant Riller | 0.0 |
| 47 | Yam Madar | 0.0 | Jay Scrubb | -0.5 |
| 48 | Nico Mannion | -0.3 | Daniel Oturu | -0.5 |
| 49 | Isaiah Joe | 10.9 | Robert Woodard II | -0.0 |
| 50 | Skylar Mays | 0.6 | Tyler Bey | -0.4 |
| 51 | Justinian Jessup | 0.0 | Elijah Hughes | -1.8 |
| 52 | KJ Martin | 0.3 | Aleksej Pokusevski | -3.1 |
| 53 | Cassius Winston | -0.6 | Jaden McDaniels | 12.8 |
| 54 | Cassius Stanley | -0.1 | Nick Richards | 0.4 |
| 55 | Jay Scrubb | -0.5 | Jordan Nwora | -1.9 |
| 56 | Grant Riller | 0.0 | Nico Mannion | -0.3 |
| 57 | Reggie Perry | -0.9 | Immanuel Quickley | 19.5 |
| 58 | Paul Reed | 6.7 | Cassius Winston | -0.6 |
| 59 | Jalen Harris | 0.3 | Saben Lee | 0.0 |
| 60 | Sam Merrill | 4.0 | Cassius Stanley | -0.1 |

### 2021

| # | Actual draft | 5yr WAR | Model redraft | 5yr WAR |
|--:|:-------------|--------:|:--------------|--------:|
| 1 | Cade Cunningham | 22.6 | Cade Cunningham | 22.6 |
| 2 | Jalen Green | 7.5 | Evan Mobley | 28.5 |
| 3 | Evan Mobley | 28.5 | Franz Wagner | 25.7 |
| 4 | Scottie Barnes | 32.0 | Jalen Suggs | 12.8 |
| 5 | Jalen Suggs | 12.8 | Isaiah Jackson | 3.7 |
| 6 | Josh Giddey | 19.6 | Scottie Barnes | 32.0 |
| 7 | Jonathan Kuminga | 3.5 | Chris Duarte | 1.6 |
| 8 | Franz Wagner | 25.7 | Miles McBride | 6.7 |
| 9 | Davion Mitchell | 5.8 | Alperen Şengün | 29.5 |
| 10 | Ziaire Williams | 1.0 | Day'Ron Sharpe | 6.3 |
| 11 | James Bouknight | -1.4 | Jared Butler | 0.7 |
| 12 | Joshua Primo | -1.1 | Jalen Johnson | 17.0 |
| 13 | Chris Duarte | 1.6 | Corey Kispert | 2.0 |
| 14 | Moses Moody | 5.6 | Moses Moody | 5.6 |
| 15 | Corey Kispert | 2.0 | Neemias Queta | 7.8 |
| 16 | Alperen Şengün | 29.5 | Kai Jones | 0.6 |
| 17 | Trey Murphy III | 20.7 | Davion Mitchell | 5.8 |
| 18 | Tre Mann | -1.4 | Usman Garuba | 0.9 |
| 19 | Kai Jones | 0.6 | Jaden Springer | 0.2 |
| 20 | Jalen Johnson | 17.0 | Jeremiah Robinson-Earl | -1.0 |
| 21 | Keon Johnson | -3.7 | Ayo Dosunmu | 6.6 |
| 22 | Isaiah Jackson | 3.7 | James Bouknight | -1.4 |
| 23 | Usman Garuba | 0.9 | Josh Giddey | 19.6 |
| 24 | Josh Christopher | -1.8 | Bones Hyland | 6.2 |
| 25 | Quentin Grimes | 9.5 | Luka Garza | 3.7 |
| 26 | Bones Hyland | 6.2 | Juhann Begarin | 0.0 |
| 27 | Cam Thomas | 0.7 | Sharife Cooper | -1.0 |
| 28 | Jaden Springer | 0.2 | Joe Wieskamp | -0.4 |
| 29 | Day'Ron Sharpe | 6.3 | Scottie Lewis | 0.0 |
| 30 | Santi Aldama | 11.4 | Quentin Grimes | 9.5 |
| 31 | Isaiah Todd | -0.6 | Jason Preston | 0.0 |
| 32 | Jeremiah Robinson-Earl | -1.0 | Marcus Zegarowski | 0.0 |
| 33 | Jason Preston | 0.0 | Charles Bassey | 1.8 |
| 34 | Rokas Jokubaitis | 0.0 | Jalen Green | 7.5 |
| 35 | Herbert Jones | 15.1 | Josh Christopher | -1.8 |
| 36 | Miles McBride | 6.7 | Isaiah Livers | -0.5 |
| 37 | JT Thor | -2.2 | Rokas Jokubaitis | 0.0 |
| 38 | Ayo Dosunmu | 6.6 | Balša Koprivica | 0.0 |
| 39 | Neemias Queta | 7.8 | Sandro Mamukelashvili | 5.6 |
| 40 | Jared Butler | 0.7 | Filip Petrušev | -0.0 |
| 41 | Joe Wieskamp | -0.4 | Herbert Jones | 15.1 |
| 42 | Isaiah Livers | -0.5 | David Johnson | 0.0 |
| 43 | Greg Brown III | -2.2 | Trey Murphy III | 20.7 |
| 44 | Kessler Edwards | -0.2 | Dalano Banton | 2.0 |
| 45 | Juhann Begarin | 0.0 | Jonathan Kuminga | 3.5 |
| 46 | Dalano Banton | 2.0 | Georgios Kalaitzakis | -1.1 |
| 47 | David Johnson | 0.0 | Aaron Wiggins | 7.6 |
| 48 | Sharife Cooper | -1.0 | Brandon Boston Jr. | -1.1 |
| 49 | Marcus Zegarowski | 0.0 | Isaiah Todd | -0.6 |
| 50 | Filip Petrušev | -0.0 | JT Thor | -2.2 |
| 51 | Brandon Boston Jr. | -1.1 | Kessler Edwards | -0.2 |
| 52 | Luka Garza | 3.7 | RaiQuan Gray | 0.2 |
| 53 | Charles Bassey | 1.8 | Keon Johnson | -3.7 |
| 54 | Sandro Mamukelashvili | 5.6 | Jericho Sims | 0.9 |
| 55 | Aaron Wiggins | 7.6 | Joshua Primo | -1.1 |
| 56 | Scottie Lewis | 0.0 | Santi Aldama | 11.4 |
| 57 | Balša Koprivica | 0.0 | Ziaire Williams | 1.0 |
| 58 | Jericho Sims | 0.9 | Tre Mann | -1.4 |
| 59 | RaiQuan Gray | 0.2 | Cam Thomas | 0.7 |
| 60 | Georgios Kalaitzakis | -1.1 | Greg Brown III | -2.2 |

### 2022

| # | Actual draft | 5yr WAR | Model redraft | 5yr WAR |
|--:|:-------------|--------:|:--------------|--------:|
| 1 | Paolo Banchero | 18.5 | Chet Holmgren | 19.9 |
| 2 | Chet Holmgren | 19.9 | Jeremy Sochan | -0.1 |
| 3 | Jabari Smith Jr. | 7.5 | Keegan Murray | 8.4 |
| 4 | Keegan Murray | 8.4 | Jabari Smith Jr. | 7.5 |
| 5 | Jaden Ivey | -0.5 | Walker Kessler | 12.9 |
| 6 | Bennedict Mathurin | -1.3 | Jalen Duren | 17.1 |
| 7 | Shaedon Sharpe | 2.7 | Paolo Banchero | 18.5 |
| 8 | Dyson Daniels | 15.4 | Mark Williams | 8.4 |
| 9 | Jeremy Sochan | -0.1 | Jake LaRavia | 3.5 |
| 10 | Johnny Davis | -1.9 | Tari Eason | 8.6 |
| 11 | Ousmane Dieng | -0.6 | Karlo Matković | 1.7 |
| 12 | Jalen Williams | 22.4 | Bennedict Mathurin | -1.3 |
| 13 | Jalen Duren | 17.1 | Christian Braun | 8.1 |
| 14 | Ochai Agbaji | -0.8 | Dalen Terry | 0.8 |
| 15 | Mark Williams | 8.4 | Johnny Davis | -1.9 |
| 16 | AJ Griffin | 0.7 | Shaedon Sharpe | 2.7 |
| 17 | Tari Eason | 8.6 | E.J. Liddell | -0.2 |
| 18 | Dalen Terry | 0.8 | Ismael Kamagate | 0.0 |
| 19 | Jake LaRavia | 3.5 | Kennedy Chandler | -0.6 |
| 20 | Malaki Branham | -5.2 | Jaden Ivey | -0.5 |
| 21 | Christian Braun | 8.1 | Josh Minott | 3.2 |
| 22 | Walker Kessler | 12.9 | AJ Griffin | 0.7 |
| 23 | David Roddy | -2.4 | Khalifa Diop | 0.0 |
| 24 | MarJon Beauchamp | -1.2 | TyTy Washington Jr. | -0.9 |
| 25 | Blake Wesley | -3.0 | Wendell Moore Jr. | -0.1 |
| 26 | Wendell Moore Jr. | -0.1 | Christian Koloko | -0.3 |
| 27 | Nikola Jović | 3.2 | Vince Williams Jr. | 2.5 |
| 28 | Patrick Baldwin Jr. | -0.1 | Nikola Jović | 3.2 |
| 29 | TyTy Washington Jr. | -0.9 | Jaylin Williams | 8.7 |
| 30 | Peyton Watson | 2.2 | David Roddy | -2.4 |
| 31 | Andrew Nembhard | 5.1 | Gabriele Procida | 0.0 |
| 32 | Caleb Houstan | 0.1 | Luke Travers | -0.3 |
| 33 | Christian Koloko | -0.3 | Dyson Daniels | 15.4 |
| 34 | Jaylin Williams | 8.7 | Ochai Agbaji | -0.8 |
| 35 | Max Christie | 0.4 | Jalen Williams | 22.4 |
| 36 | Gabriele Procida | 0.0 | MarJon Beauchamp | -1.2 |
| 37 | Jaden Hardy | -2.3 | Andrew Nembhard | 5.1 |
| 38 | Kennedy Chandler | -0.6 | Kendall Brown | -0.2 |
| 39 | Khalifa Diop | 0.0 | Isaiah Mobley | -0.0 |
| 40 | Bryce McGowens | -2.3 | Jabari Walker | -1.2 |
| 41 | E.J. Liddell | -0.2 | Malaki Branham | -5.2 |
| 42 | Trevor Keels | -0.1 | Ousmane Dieng | -0.6 |
| 43 | Moussa Diabaté | 6.1 | Matteo Spagnolo | 0.0 |
| 44 | Ryan Rollins | 3.8 | Gui Santos | 2.4 |
| 45 | Josh Minott | 3.2 | Jaden Hardy | -2.3 |
| 46 | Ismael Kamagate | 0.0 | Tyrese Martin | -1.0 |
| 47 | Vince Williams Jr. | 2.5 | Hugo Besson | 0.0 |
| 48 | Kendall Brown | -0.2 | Trevor Keels | -0.1 |
| 49 | Isaiah Mobley | -0.0 | Peyton Watson | 2.2 |
| 50 | Matteo Spagnolo | 0.0 | JD Davison | 0.1 |
| 51 | Tyrese Martin | -1.0 | Yannick Nzosa | 0.0 |
| 52 | Karlo Matković | 1.7 | Ryan Rollins | 3.8 |
| 53 | JD Davison | 0.1 | Moussa Diabaté | 6.1 |
| 54 | Yannick Nzosa | 0.0 | Patrick Baldwin Jr. | -0.1 |
| 55 | Gui Santos | 2.4 | Caleb Houstan | 0.1 |
| 56 | Luke Travers | -0.3 | Max Christie | 0.4 |
| 57 | Jabari Walker | -1.2 | Blake Wesley | -3.0 |
| 58 | Hugo Besson | 0.0 | Bryce McGowens | -2.3 |

### 2023

| # | Actual draft | 5yr WAR | Model redraft | 5yr WAR |
|--:|:-------------|--------:|:--------------|--------:|
| 1 | Victor Wembanyama | 32.1 | Dereck Lively II | 5.7 |
| 2 | Brandon Miller | 6.0 | Trayce Jackson-Davis | 4.7 |
| 3 | Scoot Henderson | -1.9 | Brandon Miller | 6.0 |
| 4 | Amen Thompson | 20.4 | Victor Wembanyama | 32.1 |
| 5 | Ausar Thompson | 10.7 | Anthony Black | 4.2 |
| 6 | Anthony Black | 4.2 | Cason Wallace | 13.7 |
| 7 | Bilal Coulibaly | -2.9 | Jarace Walker | 0.3 |
| 8 | Jarace Walker | 0.3 | Jaime Jaquez Jr. | 5.7 |
| 9 | Taylor Hendricks | -0.2 | Taylor Hendricks | -0.2 |
| 10 | Cason Wallace | 13.7 | Colby Jones | -0.2 |
| 11 | Jett Howard | -0.4 | Brandin Podziemski | 9.1 |
| 12 | Dereck Lively II | 5.7 | Jalen Pickett | 0.5 |
| 13 | Gradey Dick | -3.5 | Gradey Dick | -3.5 |
| 14 | Jordan Hawkins | -3.5 | Jaylen Clark | 1.0 |
| 15 | Kobe Bufkin | -0.6 | Ausar Thompson | 10.7 |
| 16 | Keyonte George | -1.0 | Leonard Miller | 0.9 |
| 17 | Jalen Hood-Schifino | -1.2 | Jalen Slawson | 0.3 |
| 18 | Jaime Jaquez Jr. | 5.7 | Amen Thompson | 20.4 |
| 19 | Brandin Podziemski | 9.1 | Scoot Henderson | -1.9 |
| 20 | Cam Whitmore | 2.1 | James Nnaji | 0.0 |
| 21 | Noah Clowney | -1.1 | Noah Clowney | -1.1 |
| 22 | Dariq Whitehead | -0.8 | Andre Jackson Jr. | -1.6 |
| 23 | Kris Murray | 0.4 | Jordan Miller | 2.2 |
| 24 | Olivier-Maxence Prosper | 0.8 | Cam Whitmore | 2.1 |
| 25 | Marcus Sasser | 0.7 | Kobe Bufkin | -0.6 |
| 26 | Ben Sheppard | 0.3 | Kris Murray | 0.4 |
| 27 | Nick Smith Jr. | -4.4 | Bilal Coulibaly | -2.9 |
| 28 | Brice Sensabaugh | -0.6 | Marcus Sasser | 0.7 |
| 29 | Julian Strawther | -1.6 | Kobe Brown | -0.3 |
| 30 | Kobe Brown | -0.3 | Toumani Camara | 6.8 |
| 31 | James Nnaji | 0.0 | Hunter Tyson | -0.6 |
| 32 | Jalen Pickett | 0.5 | Keyonte George | -1.0 |
| 33 | Leonard Miller | 0.9 | Mouhamed Gueye | 1.5 |
| 34 | Colby Jones | -0.2 | Mojave King | 0.0 |
| 35 | Julian Phillips | -0.4 | Tarik Biberovic | 0.0 |
| 36 | Andre Jackson Jr. | -1.6 | Julian Phillips | -0.4 |
| 37 | Hunter Tyson | -0.6 | Jordan Walsh | 2.1 |
| 38 | Jordan Walsh | 2.1 | Sidy Cissoko | -1.1 |
| 39 | Mouhamed Gueye | 1.5 | Amari Bailey | -0.2 |
| 40 | Maxwell Lewis | -0.7 | Rayan Rupert | -3.6 |
| 41 | Amari Bailey | -0.2 | Tristan Vukcevic | 1.1 |
| 42 | Tristan Vukcevic | 1.1 | Jett Howard | -0.4 |
| 43 | Rayan Rupert | -3.6 | Seth Lundy | -0.2 |
| 44 | Sidy Cissoko | -1.1 | Keyontae Johnson | -0.2 |
| 45 | GG Jackson II | -0.0 | Julian Strawther | -1.6 |
| 46 | Seth Lundy | -0.2 | Isaiah Wong | -0.2 |
| 47 | Mojave King | 0.0 | Chris Livingston | -0.4 |
| 48 | Jordan Miller | 2.2 | Jordan Hawkins | -3.5 |
| 49 | Emoni Bates | -0.3 | Ben Sheppard | 0.3 |
| 50 | Keyontae Johnson | -0.2 | Brice Sensabaugh | -0.6 |
| 51 | Jalen Wilson | -3.1 | Jalen Wilson | -3.1 |
| 52 | Toumani Camara | 6.8 | Dariq Whitehead | -0.8 |
| 53 | Jaylen Clark | 1.0 | Olivier-Maxence Prosper | 0.8 |
| 54 | Jalen Slawson | 0.3 | Emoni Bates | -0.3 |
| 55 | Isaiah Wong | -0.2 | Maxwell Lewis | -0.7 |
| 56 | Tarik Biberovic | 0.0 | Jalen Hood-Schifino | -1.2 |
| 57 | Trayce Jackson-Davis | 4.7 | GG Jackson II | -0.0 |
| 58 | Chris Livingston | -0.4 | Nick Smith Jr. | -4.4 |

### 2024

| # | Actual draft | 5yr WAR | Model redraft | 5yr WAR |
|--:|:-------------|--------:|:--------------|--------:|
| 1 | Zaccharie Risacher | -0.8 | Reed Sheppard | 5.9 |
| 2 | Alex Sarr | 0.9 | Donovan Clingan | 8.6 |
| 3 | Reed Sheppard | 5.9 | Jonathan Mogbo | 0.4 |
| 4 | Stephon Castle | 5.5 | Zach Edey | 3.1 |
| 5 | Ron Holland | 1.1 | Devin Carter | 0.0 |
| 6 | Tidjane Salaün | -1.1 | DaRon Holmes | 0.2 |
| 7 | Donovan Clingan | 8.6 | Stephon Castle | 5.5 |
| 8 | Rob Dillingham | -2.4 | Adem Bona | -0.1 |
| 9 | Zach Edey | 3.1 | Nikola Topić | -0.4 |
| 10 | Cody Williams | -5.5 | Ryan Dunn | 0.2 |
| 11 | Matas Buzelis | 1.8 | Tyler Kolek | 1.3 |
| 12 | Nikola Topić | -0.4 | Anton Watson | -0.1 |
| 13 | Devin Carter | 0.0 | Jared McCain | 0.7 |
| 14 | Bub Carrington | -6.6 | Kel'el Ware | 6.5 |
| 15 | Kel'el Ware | 6.5 | Alex Sarr | 0.9 |
| 16 | Jared McCain | 0.7 | Kyle Filipowski | 2.4 |
| 17 | Dalton Knecht | 0.3 | Dillon Jones | -0.1 |
| 18 | Tristan Da Silva | 1.7 | Jamal Shead | -0.7 |
| 19 | Ja'Kobe Walter | 2.1 | Jaylon Tyson | 2.9 |
| 20 | Jaylon Tyson | 2.9 | Oso Ighodaro | 3.3 |
| 21 | Yves Missi | -0.6 | Zaccharie Risacher | -0.8 |
| 22 | DaRon Holmes | 0.2 | Tyler Smith | -0.1 |
| 23 | AJ Johnson | -3.3 | Cam Spencer | 4.9 |
| 24 | Kyshawn George | 0.2 | Matas Buzelis | 1.8 |
| 25 | Pacôme Dadiet | -0.1 | Ron Holland | 1.1 |
| 26 | Dillon Jones | -0.1 | Tristan Da Silva | 1.7 |
| 27 | Terrence Shannon Jr. | 0.1 | Kevin McCullar Jr. | -0.1 |
| 28 | Ryan Dunn | 0.2 | Ariel Hukporti | -0.1 |
| 29 | Isaiah Collier | -2.0 | Kyshawn George | 0.2 |
| 30 | Baylor Scheierman | 3.2 | Baylor Scheierman | 3.2 |
| 31 | Jonathan Mogbo | 0.4 | Juan Nunez | 0.0 |
| 32 | Kyle Filipowski | 2.4 | Harrison Ingram | 0.1 |
| 33 | Tyler Smith | -0.1 | Rob Dillingham | -2.4 |
| 34 | Tyler Kolek | 1.3 | Pelle Larsson | 2.1 |
| 35 | Johnny Furphy | -0.2 | Tidjane Salaün | -1.1 |
| 36 | Juan Nunez | 0.0 | KJ Simpson | -1.4 |
| 37 | Bobi Klintman | -0.1 | Tristen Newton | -0.1 |
| 38 | Ajay Mitchell | 4.5 | Johnny Furphy | -0.2 |
| 39 | Jaylen Wells | 1.2 | Bub Carrington | -6.6 |
| 40 | Oso Ighodaro | 3.3 | Bobi Klintman | -0.1 |
| 41 | Adem Bona | -0.1 | Quinten Post | 1.6 |
| 42 | KJ Simpson | -1.4 | Terrence Shannon Jr. | 0.1 |
| 43 | Nikola Djurisic | 0.0 | Nikola Djurisic | 0.0 |
| 44 | Pelle Larsson | 2.1 | Ja'Kobe Walter | 2.1 |
| 45 | Jamal Shead | -0.7 | Ajay Mitchell | 4.5 |
| 46 | Cam Christie | -0.4 | Enrique Freeman | -0.2 |
| 47 | Antonio Reeves | -0.2 | Ulrich Chomche | -0.1 |
| 48 | Harrison Ingram | 0.1 | Yves Missi | -0.6 |
| 49 | Tristen Newton | -0.1 | Melvin Ajinça | 0.0 |
| 50 | Enrique Freeman | -0.2 | Jaylen Wells | 1.2 |
| 51 | Melvin Ajinça | 0.0 | AJ Johnson | -3.3 |
| 52 | Quinten Post | 1.6 | Bronny James | -0.6 |
| 53 | Cam Spencer | 4.9 | Cody Williams | -5.5 |
| 54 | Anton Watson | -0.1 | Cam Christie | -0.4 |
| 55 | Bronny James | -0.6 | Dalton Knecht | 0.3 |
| 56 | Kevin McCullar Jr. | -0.1 | Pacôme Dadiet | -0.1 |
| 57 | Ulrich Chomche | -0.1 | Isaiah Collier | -2.0 |
| 58 | Ariel Hukporti | -0.1 | Antonio Reeves | -0.2 |

### 2025

| # | Actual draft | 5yr WAR | Model redraft | 5yr WAR |
|--:|:-------------|--------:|:--------------|--------:|
| 1 | Cooper Flagg | 5.0 | Cooper Flagg | 5.0 |
| 2 | Dylan Harper | 4.4 | Collin Murray-Boyles | 2.7 |
| 3 | VJ Edgecombe | 4.3 | VJ Edgecombe | 4.3 |
| 4 | Kon Knueppel | 7.4 | Dylan Harper | 4.4 |
| 5 | Ace Bailey | -2.4 | Noa Essengue | -0.1 |
| 6 | Tre Johnson | -2.0 | Kon Knueppel | 7.4 |
| 7 | Jeremiah Fears | -0.6 | Thomas Sorber | 0.0 |
| 8 | Egor Dëmin | 0.8 | Johni Broome | -0.3 |
| 9 | Collin Murray-Boyles | 2.7 | Nique Clifford | -2.7 |
| 10 | Khaman Maluach | -0.3 | Khaman Maluach | -0.3 |
| 11 | Cedric Coward | 2.3 | Carter Bryant | 0.1 |
| 12 | Noa Essengue | -0.1 | Jase Richardson | 0.1 |
| 13 | Derik Queen | 2.3 | Ryan Kalkbrenner | 3.3 |
| 14 | Carter Bryant | 0.1 | Cedric Coward | 2.3 |
| 15 | Thomas Sorber | 0.0 | Asa Newell | 0.3 |
| 16 | Yang Hansen | -1.0 | Kasparas Jakučionis | 1.0 |
| 17 | Joan Beringer | 0.6 | Danny Wolf | -0.1 |
| 18 | Walter Clayton | -1.4 | Derik Queen | 2.3 |
| 19 | Nolan Traoré | -3.1 | Jeremiah Fears | -0.6 |
| 20 | Kasparas Jakučionis | 1.0 | Joan Beringer | 0.6 |
| 21 | Will Riley | -2.0 | Kam Jones | -1.3 |
| 22 | Drake Powell | -2.5 | Bogoljub Markovic | 0.0 |
| 23 | Asa Newell | 0.3 | Walter Clayton | -1.4 |
| 24 | Nique Clifford | -2.7 | Adou Thiero | -0.2 |
| 25 | Jase Richardson | 0.1 | Max Shulga | -0.1 |
| 26 | Ben Saraf | -2.5 | Egor Dëmin | 0.8 |
| 27 | Danny Wolf | -0.1 | Brooks Barnhizer | -0.1 |
| 28 | Hugo González | 0.6 | Drake Powell | -2.5 |
| 29 | Liam McNeeley | 0.2 | Amari Williams | 0.0 |
| 30 | Yanic Konan Niederhäuser | 0.2 | Rasheer Fleming | -0.2 |
| 31 | Rasheer Fleming | -0.2 | Maxime Raynaud | -0.3 |
| 32 | Noah Penda | 0.6 | Jahmai Mashack | -1.8 |
| 33 | Sion James | 0.6 | Will Richard | 1.0 |
| 34 | Ryan Kalkbrenner | 3.3 | Yang Hansen | -1.0 |
| 35 | Johni Broome | -0.3 | Micah Peavy | -1.1 |
| 36 | Adou Thiero | -0.2 | Noah Penda | 0.6 |
| 37 | Chaz Lanier | -0.3 | Ace Bailey | -2.4 |
| 38 | Kam Jones | -1.3 | Rocco Zikarsky | 0.1 |
| 39 | Alijah Martin | -0.1 | Alijah Martin | -0.1 |
| 40 | Micah Peavy | -1.1 | Lachlan Olbrich | -0.2 |
| 41 | Koby Brea | 0.1 | Sion James | 0.6 |
| 42 | Maxime Raynaud | -0.3 | Tre Johnson | -2.0 |
| 43 | Jamir Watkins | -0.5 | Saliou Niang | 0.0 |
| 44 | Brooks Barnhizer | -0.1 | Hugo González | 0.6 |
| 45 | Rocco Zikarsky | 0.1 | Alex Toohey | 0.0 |
| 46 | Amari Williams | 0.0 | Ben Saraf | -2.5 |
| 47 | Bogoljub Markovic | 0.0 | Koby Brea | 0.1 |
| 48 | Javon Small | 1.3 | Javon Small | 1.3 |
| 49 | Tyrese Proctor | -0.2 | Nolan Traoré | -3.1 |
| 50 | Kobe Sanders | 0.0 | Will Riley | -2.0 |
| 51 | Mohamed Diawara | 0.3 | Jamir Watkins | -0.5 |
| 52 | Alex Toohey | 0.0 | Mohamed Diawara | 0.3 |
| 53 | John Tonje | -0.0 | Kobe Sanders | 0.0 |
| 54 | Taelon Peter | -0.6 | Taelon Peter | -0.6 |
| 55 | Lachlan Olbrich | -0.2 | Tyrese Proctor | -0.2 |
| 56 | Will Richard | 1.0 | Yanic Konan Niederhäuser | 0.2 |
| 57 | Max Shulga | -0.1 | John Tonje | -0.0 |
| 58 | Saliou Niang | 0.0 | Chaz Lanier | -0.3 |
| 59 | Jahmai Mashack | -1.8 | Liam McNeeley | 0.2 |
