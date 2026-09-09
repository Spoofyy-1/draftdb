#!/usr/bin/env python3
"""Build team_cards.json + team_cards_audit.csv for the DraftDB team pages.

For every team and every draft 2019-2025: the team's actual pick(s) beside the
player the model ranked highest among those still available at that slot, with
both players' realized WAR.

Read-only with respect to every existing site file; writes only:
  site/data/team_cards.json
  site/data/team_cards_audit.csv
"""
import csv, json, os, re, sys, unicodedata, collections, datetime

SITE = '/Users/kennakao/nba/site'
DRAFT_DIR = f'{SITE}/data/draft_history'
TEAMS_DIR = f'{SITE}/teams'
IDENT = '/Users/kennakao/Downloads/nba_redraft_handoff/identity_KEEP_SEPARATE/tabular_names.csv'
ANSWERS = '/Users/kennakao/nba/draftdb_stage/data/answers/answers_{}.csv'
BOARD_EXPANDING = '/Users/kennakao/nba/results/board_v421_gen99_expanding.csv'
BOARD_STRICT = '/Users/kennakao/nba/results/board_v421_gen99_strict.csv'
OUT_JSON = f'{SITE}/data/team_cards.json'
OUT_CSV = f'{SITE}/data/team_cards_audit.csv'
YEARS = list(range(2019, 2026))
EPS = 0.05

log = collections.defaultdict(list)


def norm_name(s):
    s = unicodedata.normalize('NFKD', s or '')
    s = ''.join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace('.', ' ').replace('-', ' ').replace("'", '')
    s = re.sub(r'\b(jr|sr|ii|iii|iv|v)\b', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


def fnum(v, default=None):
    try:
        v = str(v).strip()
        return default if v == '' else float(v)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------- board
if os.path.exists(BOARD_EXPANDING) and os.path.getsize(BOARD_EXPANDING) > 0:
    board_path, boards_used = BOARD_EXPANDING, 'expanding'
else:
    board_path, boards_used = BOARD_STRICT, 'strict'

board_score = {}                                   # (pid, year) -> score
by_season = collections.defaultdict(list)
for r in csv.DictReader(open(board_path)):
    y = int(float(r['season']))
    if y not in YEARS:
        continue
    sc = fnum(r['score'])
    key = (r['pid'], y)
    if key in board_score:
        log['duplicate_board_rows'].append(f'{r["pid"]} {y}')
        continue
    board_score[key] = sc
    by_season[y].append((r['pid'], sc))

model_rank = {}                                    # (pid, year) -> rank (1 = best)
for y, rows in by_season.items():
    for i, (pid, sc) in enumerate(sorted(rows, key=lambda t: (-t[1], t[0])), 1):
        model_rank[(pid, y)] = i

# ---------------------------------------------------------------- identity
ident_rows = [r for r in csv.DictReader(open(IDENT))
              if r['draft_year'].strip().isdigit() and int(r['draft_year']) in YEARS]
by_nba, by_name = {}, collections.defaultdict(list)
for r in ident_rows:
    v = fnum(r['nba_id'])
    if v is not None:
        by_nba[int(v)] = r
    by_name[(norm_name(r['player_name']), int(r['draft_year']))].append(r)

# ---------------------------------------------------------------- answers (WAR)
war = {}                                           # (pid, year) -> dict
for y in YEARS:
    for r in csv.DictReader(open(ANSWERS.format(y))):
        war[(r['pid'], y)] = {
            'war': fnum(r.get('y_early_war'), 0.0) or 0.0,
            'seasons': int(fnum(r.get('y_partial_seasons'), 0) or 0),
            'complete': str(r.get('y_window_complete', '')).strip().lower() == 'true',
            'games': fnum(r.get('y_games_5yr'), 0.0) or 0.0,
            'minutes': fnum(r.get('y_minutes_5yr'), 0.0) or 0.0,
        }

# ---------------------------------------------------------------- draft history
teams = {}                                         # team_id -> (city, name, abbr)
picks = collections.defaultdict(list)              # year -> [pick dict]
for y in YEARS:
    rs = json.load(open(f'{DRAFT_DIR}/draft_{y}.json'))['resultSets'][0]
    H = {h: i for i, h in enumerate(rs['headers'])}
    for row in rs['rowSet']:
        tid = row[H['TEAM_ID']]
        teams[tid] = (row[H['TEAM_CITY']], row[H['TEAM_NAME']], row[H['TEAM_ABBREVIATION']])
        nba_id, nm = int(row[H['PERSON_ID']]), row[H['PLAYER_NAME']]
        rec, how = by_nba.get(nba_id), 'nba_id'
        if rec is None:
            cand = by_name.get((norm_name(nm), y), [])
            if len(cand) == 1:
                rec, how = cand[0], 'name'
                log['join_by_name'].append(f'{y} #{row[H["OVERALL_PICK"]]} {nm}')
            else:
                how = 'none'
                log['join_failed'].append(f'{y} #{row[H["OVERALL_PICK"]]} {nm}'
                                          + (f' ({len(cand)} name candidates)' if cand else ''))
        picks[y].append({
            'pick': int(row[H['OVERALL_PICK']]), 'round': int(row[H['ROUND_NUMBER']]),
            'team_id': tid, 'abbr': row[H['TEAM_ABBREVIATION']],
            'team_name': f'{row[H["TEAM_CITY"]]} {row[H["TEAM_NAME"]]}',
            'nba_id': nba_id, 'name': nm, 'org': row[H['ORGANIZATION']],
            'pid': rec['pid'] if rec else None, 'join': how,
        })
    picks[y].sort(key=lambda p: p['pick'])

# ---------------------------------------------------------------- slugs
slug_files = {f[:-5] for f in os.listdir(TEAMS_DIR) if f.endswith('.html') and f != 'index.html'}
slug_of = {}
for tid, (city, name, abbr) in teams.items():
    cands = [f'{city} {name}', f'{city} {name}'.replace('LA ', 'Los Angeles '), abbr]
    for c in cands:
        s = re.sub(r'[^a-z0-9]+', '-', c.lower()).strip('-')
        if s in slug_files:
            slug_of[tid] = s
            break
    else:
        log['slug_failed'].append(f'{tid} {city} {name}')
missing_slugs = slug_files - set(slug_of.values())
if missing_slugs:
    log['slug_unused'].extend(sorted(missing_slugs))

# ---------------------------------------------------------------- per-year lookups
pick_by_pid = {}                                   # (pid, year) -> pick dict
for y in YEARS:
    for p in picks[y]:
        if p['pid']:
            pick_by_pid[(p['pid'], y)] = p


def war_of(pid, y):
    return war.get((pid, y))


def player_war(pid, y):
    w = war_of(pid, y)
    return w['war'] if w else 0.0


# board candidates per year, sorted best-model-score first
board_by_year = {y: sorted(
    [p for p in picks[y] if p['pid'] and (p['pid'], y) in board_score],
    key=lambda p: (-board_score[(p['pid'], y)], p['pick'])) for y in YEARS}
# every drafted player that year with a WAR record, for best_hindsight
hind_by_year = {y: [p for p in picks[y] if p['pid'] and (p['pid'], y) in war] for y in YEARS}

# ---------------------------------------------------------------- build cards
audit = []
team_out = {}
for tid, (city, name, abbr) in teams.items():
    team_out[tid] = {
        'slug': slug_of.get(tid), 'name': f'{city} {name}', 'abbr': abbr, 'team_id': tid,
        'summary': None, 'drafts': [],
    }

for tid, (city, name, abbr) in teams.items():
    drafts = []
    for y in YEARS:
        mine = [p for p in picks[y] if p['team_id'] == tid]
        if not mine:
            drafts.append({'year': y, 'no_pick': True, 'picks': []})
            continue
        year_picks = []
        for p in mine:
            pk = p['pick']
            # model's choice: highest score among board players still on the
            # table (actual_pick > pk), excluding the actual pick itself.
            choice = next((c for c in board_by_year[y]
                           if c['pick'] > pk and c['pid'] != p['pid']), None)
            if choice is None:
                log['no_model_choice'].append(f'{y} #{pk} {abbr}')
                continue
            aw = war_of(p['pid'], y) if p['pid'] else None
            a_war = aw['war'] if aw else 0.0
            c_war = player_war(choice['pid'], y)
            hind = max((h for h in hind_by_year[y] if h['pick'] > pk and h['pid'] != p['pid']),
                       key=lambda h: (player_war(h['pid'], y), -h['pick']), default=None)
            if p['pid'] is None:
                log['actual_no_outcome_record'].append(f'{y} #{pk} {p["name"]} ({abbr})')
            elif aw is None:
                log['actual_no_answers_row'].append(f'{y} #{pk} {p["name"]} ({abbr})')

            raw_delta = c_war - a_war
            verdict = 'model' if raw_delta > EPS else ('team' if raw_delta < -EPS else 'tie')
            cw = war_of(choice['pid'], y) or {}
            card = {
                'pick': pk, 'round': p['round'],
                'actual': {
                    'pid': p['pid'], 'name': p['name'], 'org': p['org'],
                    'war': round(a_war, 1),
                    'seasons_scored': aw['seasons'] if aw else 0,
                    'model_rank': model_rank.get((p['pid'], y)) if p['pid'] else None,
                    'games': round(aw['games'], 1) if aw else 0.0,
                    'minutes': round(aw['minutes'], 1) if aw else 0.0,
                },
                'model_choice': {
                    'pid': choice['pid'], 'name': choice['name'], 'org': choice['org'],
                    'war': round(c_war, 1),
                    'seasons_scored': cw.get('seasons', 0),
                    'model_rank': model_rank[(choice['pid'], y)],
                    'actual_pick': choice['pick'], 'drafted_by': choice['team_name'],
                },
                'best_hindsight': ({
                    'pid': hind['pid'], 'name': hind['name'],
                    'war': round(player_war(hind['pid'], y), 1),
                    'actual_pick': hind['pick'], 'drafted_by': hind['team_name'],
                } if hind else None),
                'delta_war': round(raw_delta, 1),
                'verdict': verdict,
            }
            year_picks.append(card)
            audit.append({
                'slug': slug_of.get(tid), 'team': f'{city} {name}', 'abbr': abbr, 'year': y,
                'pick': pk, 'round': p['round'],
                'actual_name': p['name'], 'actual_pid': p['pid'] or '',
                'actual_org': p['org'], 'actual_war': round(a_war, 3),
                'actual_seasons_scored': aw['seasons'] if aw else 0,
                'actual_model_rank': model_rank.get((p['pid'], y), '') if p['pid'] else '',
                'actual_on_board': bool(p['pid'] and (p['pid'], y) in board_score),
                'actual_join': p['join'],
                'model_name': choice['name'], 'model_pid': choice['pid'],
                'model_org': choice['org'], 'model_war': round(c_war, 3),
                'model_seasons_scored': cw.get('seasons', 0),
                'model_rank': model_rank[(choice['pid'], y)],
                'model_score': round(board_score[(choice['pid'], y)], 6),
                'model_actual_pick': choice['pick'], 'model_drafted_by': choice['abbr'],
                'hindsight_name': hind['name'] if hind else '',
                'hindsight_war': round(player_war(hind['pid'], y), 3) if hind else '',
                'hindsight_actual_pick': hind['pick'] if hind else '',
                'hindsight_drafted_by': hind['abbr'] if hind else '',
                'delta_war': round(raw_delta, 3), 'verdict': verdict,
                'war_note': ('actual pick has no pre-draft/outcome record; WAR imputed 0'
                             if not p['pid'] else ''),
                'drafting_team_note': 'team that made the selection on draft night; '
                                      'draft-night trades are not reflected',
            })
        drafts.append({'year': y, 'no_pick': False,
                       'picks': sorted(year_picks, key=lambda c: c['pick'])})
    team_out[tid]['drafts'] = drafts

    n_picks = sum(1 for y in YEARS for p in picks[y] if p['team_id'] == tid)
    cards = [c for d in drafts for c in d.get('picks', [])]

    def summarize(cs, npk):
        return {
            'n_picks': npk, 'n_cards': len(cs),
            'war_actual_total': round(sum(c['actual']['war'] for c in cs), 1),
            'war_model_total': round(sum(c['model_choice']['war'] for c in cs), 1),
            'model_better': sum(1 for c in cs if c['verdict'] == 'model'),
            'team_better': sum(1 for c in cs if c['verdict'] == 'team'),
            'ties': sum(1 for c in cs if c['verdict'] == 'tie'),
        }

    s = summarize(cards, n_picks)
    fr = summarize([c for c in cards if c['round'] == 1], 0)
    s['first_round_only'] = {k: fr[k] for k in
                             ('war_actual_total', 'war_model_total',
                              'model_better', 'team_better')}
    team_out[tid]['summary'] = s

out = {
    'generated': datetime.date.today().isoformat(),
    'boards_used': boards_used,
    'war_definition': (
        'Realized WAR to date: y_early_war from the answers files, summed over the '
        "player's NBA seasons scored so far, at most five (2019-2021 classes have "
        'complete five-season windows; 2022-2025 are partial and still accruing, so '
        'compare within a draft year, not across). Players who never appeared in the '
        'NBA score 0. A handful of draft-and-stash picks have no pre-draft or outcome '
        'record in our data at all; they are shown with WAR 0 and flagged in '
        'team_cards_audit.csv (war_note).'),
    'model': {
        'label': 'EVO gen11 + international-line mask + NBADraft.net grades + '
                 'Torvik league context + ridge on top columns',
        'expanding_accuracy': 50.6, 'strict_accuracy': 41.8,
        'draft_accuracy_expanding': 24.0, 'draft_accuracy_strict': 26.3,
    },
    'teams': sorted(team_out.values(), key=lambda t: t['name']),
}
with open(OUT_JSON, 'w') as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
    f.write('\n')

cols = list(audit[0].keys())
with open(OUT_CSV, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in sorted(audit, key=lambda r: (r['team'], r['year'], r['pick'])):
        w.writerow(r)

# ---------------------------------------------------------------- report
print(f'boards_used            : {boards_used}  ({os.path.basename(board_path)})')
print(f'teams                  : {len(out["teams"])}')
print(f'total picks 2019-2025  : {sum(len(picks[y]) for y in YEARS)}')
print(f'total cards            : {len(audit)}')
print('\ncards per year (cards / picks):')
for y in YEARS:
    n = sum(1 for r in audit if r['year'] == y)
    print(f'  {y}: {n:3d} / {len(picks[y]):3d}   board size {len(board_by_year[y])}')

print('\njoins:')
print(f'  joined by nba_id     : {sum(1 for y in YEARS for p in picks[y] if p["join"]=="nba_id")}')
print(f'  joined by name+year  : {len(log["join_by_name"])}')
for m in log['join_by_name']:
    print(f'      {m}')
print(f'  join failures        : {len(log["join_failed"])}')
for m in log['join_failed']:
    print(f'      {m}')
for k in ('slug_failed', 'slug_unused', 'duplicate_board_rows',
          'actual_no_answers_row', 'no_model_choice'):
    if log[k]:
        print(f'  {k:21s}: {len(log[k])}')
        for m in log[k][:10]:
            print(f'      {m}')

vc = collections.Counter(r['verdict'] for r in audit)
print(f'\nverdicts overall       : model {vc["model"]}  team {vc["team"]}  tie {vc["tie"]}'
      f'   ({100*vc["model"]/len(audit):.1f}% model)')
for rd in (1, 2):
    sub = [r for r in audit if r['round'] == rd]
    c = collections.Counter(r['verdict'] for r in sub)
    print(f'  round {rd} (n={len(sub):3d})     : model {c["model"]}  team {c["team"]}  tie {c["tie"]}'
          f'   ({100*c["model"]/len(sub):.1f}% model)')

srt = sorted(audit, key=lambda r: r['delta_war'])
print('\nfive biggest model wins (model choice minus actual):')
for r in reversed(srt[-5:]):
    print(f'  {r["delta_war"]:+7.1f}  {r["year"]} #{r["pick"]:<3d} {r["abbr"]}  '
          f'{r["actual_name"]} ({r["actual_war"]:.1f}) -> {r["model_name"]} ({r["model_war"]:.1f})')
print('\nfive biggest model losses:')
for r in srt[:5]:
    print(f'  {r["delta_war"]:+7.1f}  {r["year"]} #{r["pick"]:<3d} {r["abbr"]}  '
          f'{r["actual_name"]} ({r["actual_war"]:.1f}) -> {r["model_name"]} ({r["model_war"]:.1f})')

tot_a = sum(r['actual_war'] for r in audit)
tot_m = sum(r['model_war'] for r in audit)
print(f'\nleague-wide WAR        : actual {tot_a:.1f}   model {tot_m:.1f}   '
      f'delta {tot_m-tot_a:+.1f}')
print(f'\nwrote {OUT_JSON}\nwrote {OUT_CSV}')
