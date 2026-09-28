"""Independent stdlib JSON check and provisional xG preparation.

Uses neither pandas nor the first audit's joining or missingness functions.
No model fitting. All source files remain unchanged.
"""
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'article_xg'/'audit'
DERIVED = ROOT/'article_xg'/'data_derived'
DERIVED.mkdir(exist_ok=True)


def read(name):
    with (ROOT/'JSON'/f'{name}.json').open(encoding='utf-8-sig') as stream:
        return json.load(stream)


def key(value):
    return str(int(value)) if value is not None and str(value).strip() else ''


def signature(row):
    return json.dumps(row,sort_keys=True,separators=(',',':'))


def dump(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


raw_matches,shots,lineups,events = [read(n) for n in ['match_infos','match_shots','team_lineup','match_events']]
matches = {}
for row in raw_matches:
    mid=key(row['match_id'])
    assert mid not in matches or row == matches[mid], f'Conflicting metadata {mid}'
    matches[mid]=row
members=defaultdict(set)
for row in lineups:
    ids=row['player_ids']
    if isinstance(ids,str):
        ids=json.loads(ids) if ids.startswith('[') else ids.split(',')
    for pid in ids:
        members[(key(row['match_id']),key(pid))].add(key(row['team_id']))

by_match=defaultdict(list)
for idx,row in enumerate(shots):
    by_match[key(row['match_id'])].append((idx,row))

duplicate_ids={mid for mid,n in Counter(key(m['match_id']) for m in raw_matches).items() if n>1}
block_duplicates={}
removed_duplicate_indices=set()
for mid in sorted(duplicate_ids):
    source=by_match[mid]
    counter=Counter(signature(row) for _,row in source)
    assert set(counter.values())=={2}, f'Nonuniform duplicated match shots {mid}'
    seen=set()
    for idx,row in source:
        sig=signature(row)
        if sig in seen:
            removed_duplicate_indices.add(idx)
        seen.add(sig)
    block_duplicates[mid]={'raw_shots':len(source),'unique_shots':len(counter),
                           'all_shot_signatures_occur_twice':True}

other_identical=defaultdict(list)
for idx,row in enumerate(shots):
    if key(row['match_id']) not in duplicate_ids:
        other_identical[signature(row)].append(idx)
ambiguous_indices=set(i for indices in other_identical.values() if len(indices)>1 for i in indices)

normal_goal=Counter()
all_goal=Counter()
event_goals=Counter()
for idx,row in enumerate(shots):
    if idx in removed_duplicate_indices:
        continue
    if row['result']=='Goal':
        normal_goal[key(row['match_id'])]+=1
    if row['result'] in {'Goal','Own goal'}:
        all_goal[key(row['match_id'])]+=1
for row in events:
    if row['type']=='goal':
        event_goals[key(row['match_id'])]+=1
# For duplicated matches, verify identical event copies before collapsing.
for mid in duplicate_ids:
    source=[e for e in events if key(e['match_id'])==mid and e['type']=='goal']
    event_goals[mid]=len(set(signature(e) for e in source))
score_differences=[]
for mid,m in matches.items():
    final=int(m['home_score'])+int(m['away_score'])
    if all_goal[mid]!=final or event_goals[mid]!=final:
        score_differences.append({'match_id':mid,'league_id':m['league_id'],'season':m['season'],
                                  'final_score_goals':final,'shot_goals_including_own_goals':all_goal[mid],
                                  'event_goals':event_goals[mid]})
dump('score_reconciliation_including_own_goals.json',score_differences)

# Independent match-level count consistency: own goals are not attacking shots.
team_stats={}
for row in read('match_stats'):
    k=(key(row['match_id']),key(row['team_id']))
    assert k not in team_stats or team_stats[k]==row, f'Conflicting team stats {k}'
    team_stats[k]=row
shot_count=Counter()
team_shot_count=Counter()
team_scored=Counter()
for idx,row in enumerate(shots):
    if idx in removed_duplicate_indices:
        continue
    mid=key(row['match_id']); m=matches[mid]
    teams=members.get((mid,key(row['player_id'])),set())
    if row['result']!='Own goal':
        shot_count[mid]+=1
    if len(teams)==1:
        team=next(iter(teams))
        if row['result']!='Own goal':
            team_shot_count[(mid,team)]+=1
        if row['result']=='Goal':
            team_scored[(mid,team)]+=1
        elif row['result']=='Own goal' and team in {key(m['home_team_id']),key(m['away_team_id'])}:
            benefiting_team=key(m['away_team_id']) if team==key(m['home_team_id']) else key(m['home_team_id'])
            team_scored[(mid,benefiting_team)]+=1
count_differences=[]
team_goal_differences=[]
team_count_differences=[]
for mid,m in matches.items():
    home,away=key(m['home_team_id']),key(m['away_team_id'])
    hs,aws=team_stats.get((mid,home)),team_stats.get((mid,away))
    if hs is None or aws is None:
        count_differences.append({'match_id':mid,'issue':'missing team stats'})
        continue
    expected=int(hs['total_shots'])+int(aws['total_shots'])
    if shot_count[mid]!=expected:
        count_differences.append({'match_id':mid,'observed_non_own_goal_shots':shot_count[mid],
                                  'team_stats_total_shots':expected,'difference':shot_count[mid]-expected})
    if team_shot_count[(mid,home)]!=int(hs['total_shots']) or team_shot_count[(mid,away)]!=int(aws['total_shots']):
        team_count_differences.append({'match_id':mid,'home_observed':team_shot_count[(mid,home)],
            'away_observed':team_shot_count[(mid,away)],'home_stats':int(hs['total_shots']),
            'away_stats':int(aws['total_shots'])})
    if team_scored[(mid,home)]!=int(m['home_score']) or team_scored[(mid,away)]!=int(m['away_score']):
        team_goal_differences.append({'match_id':mid,'home_score':int(m['home_score']),
            'away_score':int(m['away_score']),'resolved_home_goals':team_scored[(mid,home)],
            'resolved_away_goals':team_scored[(mid,away)]})
dump('shot_counts_against_team_stats.json',count_differences)
dump('team_goal_reconciliation.json',team_goal_differences)
dump('team_shot_count_reconciliation.json',team_count_differences)

body={'Right foot','Left foot','Header','Other body parts'}
situations={'Regular play','From corner','Set piece','Fast break','Free kick','Penalty','Throw in set piece','Individual play'}
outcomes={'Goal','Miss','Blocked','Attempt saved','Post','Own goal'}
assert all(s['shot_type'] in body and s['situation'] in situations and s['result'] in outcomes for s in shots)

exclusions=Counter()
unresolved=0
out_rows=[]
own_direction=Counter()
metadata_failures=[]
date_bad=set()
eligible_stats=Counter()
for mid,m in matches.items():
    try:
        datetime.strptime(m['date'],'%d/%m/%Y')
    except (ValueError,TypeError):
        date_bad.add(mid)

for idx,row in enumerate(shots):
    mid=key(row['match_id']); pid=key(row['player_id']); m=matches[mid]
    teamset=members.get((mid,pid),set())
    okteam=len(teamset)==1 and next(iter(teamset)) in {key(m['home_team_id']),key(m['away_team_id'])}
    unresolved+=not okteam
    if okteam and row.get('x2') is not None:
        try:
            isaway=next(iter(teamset))==key(m['away_team_id'])
            bad=(isaway and float(row['x2'])<float(row['x1'])) or (not isaway and float(row['x2'])>float(row['x1']))
            if bad:
                own_direction[row['result']]+=1
        except (TypeError,ValueError):
            pass
    # Sequential, mutually exclusive flow counts. Non-model audit fields may use outcomes.
    reason=None
    if idx in removed_duplicate_indices:
        reason='verified_duplicate_match_extraction_copy'
    elif idx in ambiguous_indices:
        reason='unresolved_identical_shot_pair_both_quarantined'
    elif row['result']=='Own goal':
        reason='own_goal_not_a_standard_attacking_attempt'
    elif row['situation']=='Penalty':
        reason='penalty_reserved_for_secondary_analysis'
    elif mid in date_bad:
        reason='invalid_match_date_no_external_imputation'
    elif not okteam:
        reason='unresolved_shooter_team'
    if reason:
        exclusions[reason]+=1
        metadata_failures.append({'source_row_1based':idx+1,'match_id':mid,'reason':reason})
        continue
    x=float(row['x1']);y=float(row['y1'])
    assert math.isfinite(x) and math.isfinite(y) and 0<=x<=105 and 0<=y<=68
    away=next(iter(teamset))==key(m['away_team_id'])
    if away:
        x,y=105-x,68-y
    lower,upper=34-7.32/2,34+7.32/2
    cross=abs((-x)*(upper-y)-(-x)*(lower-y))
    dot=x*x+(lower-y)*(upper-y)
    angle=math.atan2(cross,dot)
    distance=math.hypot(x,y-34)
    # Independent cosine-law check for non-degenerate locations.
    a=math.hypot(x,lower-y); b=math.hypot(x,upper-y)
    if a*b>1e-12:
        alternative=math.acos(max(-1,min(1,(a*a+b*b-7.32**2)/(2*a*b))))
        assert abs(angle-alternative)<1e-6, 'Geometry implementation disagreement'
    else:
        exclusions['degenerate_goalpost_origin']+=1
        metadata_failures.append({'source_row_1based':idx+1,'match_id':mid,'reason':'degenerate_goalpost_origin'})
        continue
    date=datetime.strptime(m['date'],'%d/%m/%Y').date().isoformat()
    season=m['season']
    partition='development' if season in ['2020/2021','2021/2022','2022/2023'] else ('calibration' if season=='2023/2024' else 'test_reserved')
    out_rows.append({'shot_id':f'local_row_{idx+1:06d}','match_id':mid,'player_id':pid,
        'league_id':m['league_id'],'season':season,'date':date,'partition':partition,
        'shooting_team_id':next(iter(teamset)),'home':int(not away),
        'x_origin_norm':x,'y_origin_norm':y,'distance_m':distance,'angle_rad':angle,
        'body_part':row['shot_type'],'situation':row['situation'],'minute':row['minute'],
        # Explicit exemptions from the coverage threshold: preserve raw values.
        # Endpoint coordinates are audit/post-shot fields, not pre-shot xG inputs.
        'time_added':row.get('time_added'),
        'time_added_missing':int(row.get('time_added') is None or str(row.get('time_added')).strip()==''),
        'x2':row.get('x2'),'y2':row.get('y2'),
        'goal':int(row['result']=='Goal')})
    eligible_stats[(str(m['league_id']),season)]+=1

assert len(shots)==len(out_rows)+sum(exclusions.values())
assert len(set(r['shot_id'] for r in out_rows))==len(out_rows)
match_partitions=defaultdict(set)
dates_by_partition=defaultdict(list)
for r in out_rows:
    match_partitions[r['match_id']].add(r['partition'])
    dates_by_partition[r['partition']].append(r['date'])
assert all(len(v)==1 for v in match_partitions.values())
assert max(dates_by_partition['development'])<min(dates_by_partition['calibration'])
assert max(dates_by_partition['calibration'])<min(dates_by_partition['test_reserved'])

with (DERIVED/'shots_provisional.csv').open('w',newline='',encoding='utf-8') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(out_rows[0]));writer.writeheader();writer.writerows(out_rows)
with (OUT/'exclusion_log_provisional.csv').open('w',newline='',encoding='utf-8') as stream:
    writer=csv.DictWriter(stream,fieldnames=['source_row_1based','match_id','reason']);writer.writeheader();writer.writerows(metadata_failures)

# Conservative internally consistent cohort. No relabelling or inferred repairs.
# Keep the larger provisional cohort for a declared sensitivity analysis.
match_flags=defaultdict(set)
for rows,flag in [(count_differences,'total_shot_count_difference'),
                  (team_count_differences,'team_shot_count_difference'),
                  (team_goal_differences,'team_score_difference'),
                  (score_differences,'score_or_goal_event_difference')]:
    for item in rows:
        match_flags[item['match_id']].add(flag)
for mid in date_bad:
    match_flags[mid].add('invalid_date')
for mid,m in matches.items():
    upper=34 if str(m['league_id'])=='54' or (str(m['league_id'])=='53' and m['season'] in {'2023/2024','2024/2025'}) else 38
    try:
        regular=1<=int(m['round'])<=upper
    except (ValueError,TypeError):
        regular=False
    if not regular:
        match_flags[mid].add('outside_regular_season_round_range')
for item in metadata_failures:
    if item['reason'] in {'unresolved_shooter_team','unresolved_identical_shot_pair_both_quarantined'}:
        match_flags[item['match_id']].add(item['reason'])
strict_rows=[row for row in out_rows if row['match_id'] not in match_flags]
assert strict_rows
assert all(r['match_id'] not in match_flags for r in strict_rows)
with (DERIVED/'shots_primary_candidate.csv').open('w',newline='',encoding='utf-8') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(strict_rows[0]));writer.writeheader();writer.writerows(strict_rows)
dump('match_quarantine_flags.json',{mid:sorted(flags) for mid,flags in sorted(match_flags.items())})
strata=[]
for league,season in sorted({(str(m['league_id']),m['season']) for m in matches.values()}):
    raw_ids={mid for mid,m in matches.items() if str(m['league_id'])==league and m['season']==season}
    retained=[r for r in strict_rows if str(r['league_id'])==league and r['season']==season]
    retained_ids={r['match_id'] for r in retained}
    strata.append({'league_id':league,'season':season,'raw_matches':len(raw_ids),
        'retained_matches':len(retained_ids),'match_retention_percent':round(100*len(retained_ids)/len(raw_ids),4),
        'retained_non_penalty_shots':len(retained)})
with (OUT/'primary_candidate_retention_by_stratum.csv').open('w',newline='',encoding='utf-8') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(strata[0]));writer.writeheader();writer.writerows(strata)

first=json.loads((OUT/'summary.json').read_text())
assert len(matches)==first['n_matches']
assert len(shots)==first['n_shots']
assert unresolved==first['unresolved_shot_teams']
assert sum(own_direction.values())==first['direction_diagnostic_disagreements']
report={'independent_crosscheck':'PASS for counts, team resolution and orientation diagnostics',
 'verified_duplicate_match_blocks':block_duplicates,'verified_duplicate_shot_copies_removed':len(removed_duplicate_indices),
 'other_identical_shot_rows_quarantined':len(ambiguous_indices),
 'invalid_date_match_ids':sorted(date_bad), 'n_match_score_or_event_discrepancies_after_own_goals':len(score_differences),
 'n_match_shot_score_discrepancies':sum(r['final_score_goals']!=r['shot_goals_including_own_goals'] for r in score_differences),
 'n_match_shot_count_discrepancies':len(count_differences),
 'n_match_team_goal_discrepancies':len(team_goal_differences),
 'n_match_team_shot_count_discrepancies':len(team_count_differences),
 'endpoint_direction_disagreements_by_result':dict(own_direction),
 'raw_shots':len(shots),'sequential_exclusions':dict(exclusions),'provisional_shots':len(out_rows),
 'provisional_matches':len(match_partitions),
 'partition_shots':dict(Counter(r['partition'] for r in out_rows)),
 'partition_date_ranges':{p:[min(d),max(d)] for p,d in dates_by_partition.items()},
 'n_geometry_cosine_crosschecks':len(out_rows),
 'primary_candidate_shots':len(strict_rows),
 'primary_candidate_matches':len({r['match_id'] for r in strict_rows}),
 'primary_candidate_partition_shots':dict(Counter(r['partition'] for r in strict_rows)),
 'quarantined_match_ids':len(match_flags),
 'minimum_primary_match_retention_percent':min(r['match_retention_percent'] for r in strata),
 'cohort_status':'Conservative primary candidate passes internal consistency; final protocol freeze and independent export check pending',
 'predictive_test_metrics_computed':False}
dump('independent_verification.json',report)
print(json.dumps(report,indent=2,ensure_ascii=False))
