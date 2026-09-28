"""Read-only source audit; outputs are derived under article_xg/audit.

No predictive models, no feature selection using outcome performance.
Run with the bundled Python runtime from any directory.
"""
from pathlib import Path
import csv
import hashlib
import json
import platform
import re
import sys
from collections import Counter
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'article_xg' / 'audit'
OUT.mkdir(parents=True, exist_ok=True)
MISSING = {'', 'null', 'none', 'nan', 'n/a', 'na', 'unknown', 'undefined'}


def present(s):
    return ~s.fillna('').astype(str).str.strip().str.lower().isin(MISSING)


def canonical(v):
    if v is None:
        return ''
    if isinstance(v, list):
        return ','.join(canonical(x) for x in v)
    if isinstance(v, dict):
        return json.dumps(v, sort_keys=True)
    s = str(v).strip()
    if s.lower() in {'null', 'none', 'nan'}:
        return ''
    # Normalize numbers only; leave date/time strings intact.
    if re.fullmatch(r'-?\d+(\.\d+)?', s):
        return format(float(s), '.12g')
    return s


def numeric(s):
    return pd.to_numeric(s, errors='coerce')


def roster_ids(v):
    s = str(v).strip()
    if s.startswith('['):
        try:
            return [canonical(x) for x in json.loads(s)]
        except (ValueError, TypeError):
            return []
    return [canonical(x) for x in s.split(',') if x.strip()]


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + '\n', encoding='utf-8')


relational_only = '--relational-only' in sys.argv
tables, inventory, parity, profiles, manifest = {}, [], [], [], []
for path in sorted((ROOT / 'CSV').glob('*.csv')):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        header = stream.readline()
    sep = ';' if header.count(';') > header.count(',') else ','
    frame = pd.read_csv(path, sep=sep, dtype=str, encoding='utf-8-sig', keep_default_na=False)
    tables[path.stem] = frame
    if relational_only:
        continue
    with path.open(encoding='utf-8-sig', newline='') as stream:
        second_count = sum(1 for _ in csv.DictReader(stream, delimiter=sep))
    inventory.append({'table': path.stem, 'rows_pandas': len(frame), 'rows_csv_reader': second_count,
                      'n_columns': len(frame.columns), 'delimiter': sep,
                      'identical_rows_beyond_first': int(frame.duplicated().sum())})
    assert len(frame) == second_count
    for col in frame:
        profiles.append({'table': path.stem, 'field': col, 'rows': len(frame),
                         'present': int(present(frame[col]).sum()),
                         'presence_percent': round(100 * present(frame[col]).mean(), 4)})
    json_path = ROOT / 'JSON' / (path.stem + '.json')
    for source in [path, json_path]:
        if source.exists():
            manifest.append({'file': str(source.relative_to(ROOT)), 'bytes': source.stat().st_size,
                             'sha256': hashlib.file_digest(source.open('rb'), 'sha256').hexdigest()})
    if not json_path.exists():
        parity.append({'table': path.stem, 'status': 'missing JSON'})
        continue
    with json_path.open(encoding='utf-8-sig') as stream:
        data = json.load(stream)
    if not isinstance(data, list) or (data and not isinstance(data[0], dict)):
        parity.append({'table': path.stem, 'status': 'JSON structure requires interpretation',
                       'type': type(data).__name__})
        continue
    keys = set().union(*(d.keys() for d in data)) if data else set()
    differences = Counter()
    examples = []
    for row_number, (row, ref) in enumerate(zip(frame.itertuples(index=False, name=None), data), 2):
        for col, val in zip(frame.columns, row):
            # CSV roster may flatten a JSON array into comma separated IDs.
            if canonical(val) != canonical(ref.get(col)):
                differences[col] += 1
                if len(examples) < 5:
                    examples.append({'csv_line': row_number, 'field': col,
                                     'csv': str(val)[:160], 'json': str(ref.get(col))[:160]})
    parity.append({'table': path.stem, 'csv_rows': len(frame), 'json_rows': len(data),
                   'csv_only_columns': sorted(set(frame.columns) - keys),
                   'json_only_columns': sorted(keys - set(frame.columns)),
                   'positionwise_value_differences': dict(differences), 'examples': examples,
                   'status': 'equivalent' if len(frame) == len(data) and not differences
                   and set(frame.columns) == keys else 'review required'})
    del data

if relational_only:
    previous_manifest = json.loads((OUT/'manifest.json').read_text())
    for source in previous_manifest['sources']:
        with (ROOT/source['file']).open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == source['sha256'], 'Source changed'
    parity = json.loads((OUT/'csv_json_comparison.json').read_text())
else:
    pd.DataFrame(inventory).to_csv(OUT / 'inventory.csv', index=False)
    pd.DataFrame(profiles).to_csv(OUT / 'field_presence_all_tables.csv', index=False)
    write_json(OUT / 'manifest.json', {'python': sys.version, 'platform': platform.platform(),
                                      'pandas': pd.__version__, 'sources': manifest})
    write_json(OUT / 'csv_json_comparison.json', parity)

m = tables['match_infos'].copy()
s = tables['match_shots'].copy()
l = tables['team_lineup'].copy()
for df, cols in [(m, ['match_id', 'league_id', 'home_team_id', 'away_team_id']),
                 (s, ['match_id', 'player_id']), (l, ['match_id', 'team_id'])]:
    for col in cols:
        df[col] = df[col].map(canonical)
m.loc[m.match_id.duplicated(keep=False)].to_csv(OUT/'duplicate_match_metadata.csv',index=False)
# Only completely identical metadata rows are collapsed in this analytical view.
# Conflicting records must stop the audit; shot records remain untouched here.
m = m.drop_duplicates()
assert not m['match_id'].duplicated().any(), 'Conflicting match IDs require resolution before joins.'
m['parsed_date'] = pd.to_datetime(m['date'], dayfirst=True, errors='coerce')
joined = s.merge(m[['match_id', 'league_id', 'season', 'parsed_date', 'home_team_id', 'away_team_id']],
                 on='match_id', how='left', validate='many_to_one', indicator=True)
assert len(joined) == len(s)

roster = []
malformed = 0
for row in l.itertuples(index=False):
    ids = roster_ids(row.player_ids)
    malformed += not bool(ids)
    roster.extend((row.match_id, player_id, row.team_id) for player_id in ids)
r = pd.DataFrame(roster, columns=['match_id', 'player_id', 'team_id'])
group = r.groupby(['match_id', 'player_id'])['team_id'].agg(lambda a: sorted(set(a)))
ambiguous = group[group.map(len) != 1]
mapping = group[group.map(len) == 1].map(lambda a: a[0]).rename('shooting_team_id').reset_index()
joined = joined.merge(mapping, on=['match_id', 'player_id'], how='left', validate='many_to_one')
assert len(joined) == len(s)
joined['resolved_team'] = ((joined.shooting_team_id == joined.home_team_id) |
                           (joined.shooting_team_id == joined.away_team_id))

valid = pd.DataFrame(index=joined.index)
valid['match_id'] = joined['_merge'].eq('both')
valid['player_id'] = present(joined.player_id) & joined.player_id.isin(tables['players'].player_id.map(canonical))
valid['team_membership'] = joined.resolved_team
valid['league_id'] = joined.league_id.isin(['47','87','54','55','53'])
valid['season'] = joined.season.isin([f'{y}/{y+1}' for y in range(2020, 2025)])
valid['date'] = joined.parsed_date.notna()
for col, upper in [('x1',105), ('y1',68), ('x2',105), ('y2',68)]:
    a = numeric(joined[col])
    valid[col] = a.notna() & np.isfinite(a) & a.between(0,upper)
valid['minute'] = numeric(joined.minute).between(0,120) & numeric(joined.minute).mod(1).eq(0)
valid['time_added'] = numeric(joined.time_added).between(0,30) & numeric(joined.time_added).mod(1).eq(0)
for col in ['shot_type','situation','result']:
    valid[col] = present(joined[col])  # vocabulary semantics reviewed separately

coverage = []
groups = [('ALL','ALL', joined.index)]
groups.extend((league, season, idx) for (league,season),idx in joined.groupby(['league_id','season']).groups.items())
for league, season, idx in groups:
    for col in valid:
        coverage.append({'league_id':league,'season':season,'field':col,'n_shots':len(idx),
                         'n_valid_or_present':int(valid.loc[idx,col].sum()),
                         'percent':round(100*valid.loc[idx,col].mean(),4),
                         'criterion':'presence only; vocabulary pending' if col in ['shot_type','situation','result']
                         else 'syntactic/range/referential validity'})
pd.DataFrame(coverage).to_csv(OUT / 'shot_coverage_by_league_season.csv', index=False)

cohort = m.groupby(['league_id','season']).agg(n_matches=('match_id','size'),
    first_date=('parsed_date','min'), last_date=('parsed_date','max')).reset_index()
shots_per = joined.groupby(['league_id','season']).size().rename('n_shots').reset_index()
cohort.merge(shots_per, on=['league_id','season'], how='outer').to_csv(OUT / 'cohort_by_league_season.csv', index=False)

categories = {col: joined[col].value_counts(dropna=False).to_dict() for col in ['shot_type','situation','result']}
write_json(OUT / 'shot_categories.json', categories)

duplicate_mask = s.duplicated(keep=False)
s.loc[duplicate_mask].to_csv(OUT / 'exact_duplicate_shot_candidates.csv', index=False)
natural_collision = s.duplicated(['match_id','player_id','minute'],keep=False)
joined.loc[~joined.resolved_team, list(s.columns)+['shooting_team_id','home_team_id','away_team_id']].to_csv(
    OUT / 'unresolved_shooter_teams.csv', index=False)

# Diagnostic only: does independent team orientation agree with endpoint-based direction?
x1,x2 = numeric(joined.x1),numeric(joined.x2)
has_geometry = valid.x1 & valid.y1 & valid.x2 & valid.y2 & valid.team_membership
away = joined.shooting_team_id.eq(joined.away_team_id)
disagree = has_geometry & ((away & (x2 < x1)) | (~away & (x2 > x1)))
joined.loc[disagree, list(s.columns)+['shooting_team_id','home_team_id','away_team_id']].to_csv(
    OUT / 'orientation_diagnostic_disagreements.csv', index=False)
orientation = []
for (league,season),idx in joined.groupby(['league_id','season']).groups.items():
    orientation.append({'league_id':league,'season':season,'eligible':int(has_geometry.loc[idx].sum()),
                        'direction_disagreements':int(disagree.loc[idx].sum())})
pd.DataFrame(orientation).to_csv(OUT / 'orientation_diagnostic_by_stratum.csv', index=False)

# Check goal totals against match scores for data quality, never predictor construction.
goal = joined.result.astype(str).str.strip().str.lower().eq('goal')
goal_count = joined.loc[goal].groupby('match_id').size()
goal_check = m[['match_id','league_id','season','home_score','away_score']].copy()
goal_check['goals_from_shots'] = goal_check.match_id.map(goal_count).fillna(0).astype(int)
goal_check['final_score_goals'] = numeric(goal_check.home_score)+numeric(goal_check.away_score)
goal_check['difference'] = goal_check.goals_from_shots-goal_check.final_score_goals
goal_check.loc[goal_check.difference.ne(0)].to_csv(OUT / 'shot_goal_score_differences.csv',index=False)

required = ['match_id','player_id','team_membership','league_id','season','date','x1','y1','result']
coverage_df = pd.DataFrame(coverage)
summary = {
 'n_match_rows_raw':len(tables['match_infos']),'n_matches':len(m),'n_shots':len(s),'n_player_matches':len(tables['players_match']),
 'n_events':len(tables['match_events']), 'league_ids':sorted(m.league_id.unique()),
 'seasons':sorted(m.season.unique()), 'n_league_season_cells':len(cohort),
 'missing_match_references':int((~valid.match_id).sum()),
 'unknown_player_references':int((~valid.player_id).sum()),
 'malformed_or_empty_rosters':int(malformed),'ambiguous_match_player_teams':len(ambiguous),
 'unresolved_shot_teams':int((~valid.team_membership).sum()),
 'exact_duplicate_shot_excess':int(s.duplicated().sum()),
 'rows_in_natural_key_collisions':int(natural_collision.sum()),
 'direction_diagnostic_eligible':int(has_geometry.sum()),
 'direction_diagnostic_disagreements':int(disagree.sum()),
 'equal_endpoint_origin_x':int((has_geometry & x1.eq(x2)).sum()),
 'matches_with_shot_goal_score_difference':int(goal_check.difference.ne(0).sum()),
 'required_preliminary_checks_pass':int(valid[required].all(axis=1).sum()),
 'minimum_stratum_coverage_percent':coverage_df[coverage_df.league_id.ne('ALL')].groupby('field')['percent'].min().to_dict(),
 'csv_json_tables_equivalent':sum(p['status']=='equivalent' for p in parity),
 'csv_json_tables_total':len(parity),
 'limitations':['No semantic validation of shot categories yet.',
 'No correction or deletion of source records.',
 'Direction diagnostic is not an inclusion rule and is not used to build predictors.',
 'This script reports schema-level audit metadata, not predictive model performance.']}
write_json(OUT/'summary.json',summary)
print(json.dumps(summary, indent=2, ensure_ascii=False))
