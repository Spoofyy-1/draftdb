# Text features from dated pre-draft Wikipedia articles

**Source.** For every drafted player 2010–2026 (934), the English Wikipedia article revision that existed the night before his draft, fetched through the Wikipedia API by revision timestamp (`rvstart = draft date`, `rvdir = older`). 880 players had an article by then; 49 did not (`txt_exists = 0`, which is itself a fact known on draft night); 5 have no article at all.

**No language model is used anywhere in these features.** An LLM reading a prospect's name may know how his career turned out, which would leak outcomes into the inputs. Every feature below is a count or a regular-expression match against the fixed word lists printed here, computed by `code/text_features_rules.py`, so anyone can recompute them from the same revision and get the same numbers.

## Features

| column | definition |
|---|---|
| `txt_exists` | 1 if an article existed the night before the draft |
| `txt_words` | words in the cleaned article text (templates, references, markup removed) |
| `txt_refs` | number of `<ref` citations in the raw wikitext — a proxy for media coverage |
| `txt_sections` | number of section headings |
| `txt_size_bytes` | size of the revision in bytes |
| `txt_projected_pick` | the smallest pick number in phrases like 'projected as a top-N pick' or 'projected No. N', if any |
| `txt_<list>` | number of matches of the word list `<list>` (below) in the text; `txt_<list>_per1k` is the same per 1,000 words |
| `txt_pro_family`, `txt_nba_family`, `txt_left`, `txt_prep`, `txt_transfer` | 1 if the list matched at least once |
| `txt_tone` | (positive − negative) / (positive + negative + 3) from the two tone lists |

## The word lists (regular expressions, case-insensitive)

```python
LEX={
 "hype":      [r"\btop[- ]\d+ (prospect|pick|player)",r"\blottery\b",r"\bfive[- ]star\b",r"\bconsensus\b",r"\bprojected\b",r"\bfranchise\b",r"\bphenom\b",r"\bgenerational\b",r"\belite\b",r"\bbest (player|prospect)\b",r"\bnumber[- ]one\b",r"\bno\. ?1\b",r"\bhighly[- ]touted\b",r"\bmcdonald'?s all[- ]american\b",r"\bplayer of the year\b",r"\ball[- ]american\b",r"\bnational champion",r"\bmost valuable player\b|\bmvp\b"],
 "injury":    [r"\binjur(y|ies|ed)\b",r"\bsurger(y|ies)\b",r"\btorn\b",r"\btear\b",r"\bfractur",r"\bsprain",r"\bconcussion",r"\bsidelined\b",r"\bmissed .{0,20}(games|season|weeks|months)",r"\bstress (fracture|reaction)\b",r"\bacl\b",r"\bmeniscus\b",r"\bachilles\b",r"\bredshirt",r"\bout for the season\b"],
 "character": [r"\bsuspend",r"\barrest",r"\bdismiss",r"\bineligib",r"\bacademic",r"\bcontrovers",r"\bcharged with\b",r"\bviolat",r"\bkicked off\b",r"\bdisciplin",r"\bfailed (a )?drug test\b",r"\bmisdemeanor|\bfelony\b"],
 "pro_family":[r"\b(father|mother|dad|mom|brother|sister|uncle|cousin|nephew|son|daughter|twin)\b.{0,120}\b(played|plays|career|professional|nba|wnba|euroleague|drafted)\b",r"\b(son|daughter|brother|sister|nephew|cousin) of\b.{0,80}\b(basketball|nba|wnba|player)\b"],
 "nba_family":[r"\b(father|mother|dad|mom|brother|sister|uncle|cousin|nephew|son|daughter|twin)\b.{0,120}\b(nba|wnba)\b",r"\b(former|ex-)?(nba|wnba) (player|guard|forward|center)\b.{0,60}\b(father|mother|brother|sister|uncle|cousin)\b"],
 "breakout":  [r"\bbreakout\b",r"\bimprov(ed|ement|ing)\b",r"\btook a leap\b",r"\bemerg(ed|ing)\b",r"\bris(er|ing|en)\b",r"\bdevelop(ed|ment|ing)\b",r"\bgrowth spurt\b",r"\bgrew (from|to|\d)",r"\blate bloomer\b",r"\bunranked\b",r"\bwalk[- ]on\b",r"\blightly recruited\b",r"\bunder[- ]the[- ]radar\b"],
 "shooter":   [r"\bshoot(er|ing)\b",r"\bthree[- ]point",r"\b3[- ]point",r"\bjump ?shot\b",r"\bcatch[- ]and[- ]shoot\b",r"\bfree[- ]throw",r"\brange\b",r"\bstroke\b"],
 "athlete":   [r"\bathletic",r"\bexplosiv",r"\bvertical\b",r"\bleap",r"\bdunk",r"\bwingspan\b",r"\bspeed\b",r"\bquickness\b",r"\bbounce\b",r"\bmotor\b"],
 "playmaker": [r"\bpassing\b",r"\bpasser\b",r"\bplaymak",r"\bassists?\b",r"\bcourt vision\b",r"\bvision\b",r"\bball[- ]handl",r"\bpoint guard\b",r"\bfacilitat"],
 "defender":  [r"\bdefen(se|sive|der)\b",r"\brim protect",r"\bblocks?\b",r"\bsteals?\b",r"\bshot[- ]block",r"\blockdown\b",r"\bversatil"],
 "size":      [r"\b(7|six|seven)[- ]foot",r"\bwingspan\b",r"\blength\b",r"\bundersized\b",r"\bsize for (his|the) position\b",r"\bstanding reach\b"],
 "national":  [r"\bnational team\b",r"\bfiba\b",r"\bu1[6-9]\b|\bunder-1[6-9]\b",r"\bworld cup\b",r"\beurobasket\b",r"\bolympi",r"\bgold medal\b|\bsilver medal\b|\bbronze medal\b"],
 "transfer":  [r"\btransferr?(ed|ing)\b",r"\btransfer portal\b"],
 "left":      [r"\bleft[- ]handed\b",r"\bsouthpaw\b"],
 "prep":      [r"\bimg academy\b",r"\bmontverde\b",r"\boak hill\b",r"\bsunrise christian\b",r"\bfindlay prep\b",r"\bla lumiere\b",r"\bprolific prep\b",r"\bhuntington prep\b",r"\bbrewster academy\b",r"\bwasatch academy\b",r"\bnba academy\b",r"\bovertime elite\b"],
 "positive":  [r"\bbest\b",r"\boutstanding\b",r"\bexceptional\b",r"\bdominant\b",r"\bstar\b",r"\bstandout\b",r"\bremarkable\b",r"\bimpressive\b",r"\bhonou?rs?\b",r"\bawards?\b",r"\brecord\b",r"\bled (the|his)\b"],
 "negative":  [r"\bstruggl",r"\bdisappoint",r"\binconsisten",r"\bconcern",r"\bquestion(s|ed|able)?\b",r"\bdecline",r"\bpoor\b",r"\bbench(ed)?\b",r"\bcriticis",r"\bslump\b",r"\blimited\b",r"\bweakness"],
}
```

Known limitations, on purpose: `pro_family` and `nba_family` are co-occurrence rules (a family word within 120 characters of a league/career word), so they can fire on a relative who played in another league or on a sentence about the player's own career near a family mention; `hype` counts award and ranking language and is therefore correlated with article length; `tone` is a crude ratio. These are documented data, not judgments.