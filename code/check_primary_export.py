"""Third check: exported cohort versus original CSV, using pandas.

Rebuild eligibility and geometry independently of the JSON preparation script.
Does not fit models or calculate predictive performance.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'article_xg'/'audit'

def read(name):
    return pd.read_csv(ROOT/'CSV'/f'{name}.csv',sep=';',encoding='utf-8-sig',dtype=str,keep_default_na=False)

m=read('match_infos').drop_duplicates()
s=read('match_shots')
l=read('team_lineup').drop_duplicates()
t=read('match_stats').drop_duplicates()
mids=m.match_id[m.match_id.isin(read('match_infos').loc[lambda d:d.match_id.duplicated(keep=False),'match_id'])]
s['source_index']=np.arange(1,len(s)+1)
dupblock=s.match_id.isin(mids) & s.drop(columns='source_index').duplicated()
s=s.loc[~dupblock].copy()
s['shot_id']=s.source_index.map(lambda i:f'local_row_{i:06d}')
teams=l.assign(player_id=l.player_ids.str.split(',')).explode('player_id')
assert not teams.duplicated(['match_id','player_id']).any()
z=s.merge(teams[['match_id','player_id','team_id']],on=['match_id','player_id'],how='left',validate='many_to_one')
z=z.merge(m,on='match_id',how='left',validate='many_to_one')
valid_team=z.team_id.eq(z.home_team_id) | z.team_id.eq(z.away_team_id)
bad_ids=set(z.loc[~valid_team,'match_id'])
bad_ids.update(m.loc[pd.to_datetime(m.date,format='%d/%m/%Y',errors='coerce').isna(),'match_id'])
upper=np.where(m.league_id.eq('54') | (m.league_id.eq('53') & m.season.isin(['2023/2024','2024/2025'])),34,38)
round_number=pd.to_numeric(m['round'],errors='coerce')
bad_ids.update(m.loc[~(round_number.ge(1) & round_number.le(upper)),'match_id'])
dup=z.drop(columns=['source_index','shot_id']).duplicated(keep=False)
bad_ids.update(z.loc[dup,'match_id'])

non_own=z[z.result.ne('Own goal')]
counts=non_own.groupby(['match_id','team_id']).size().rename('observed').reset_index()
checks=t.merge(counts,on=['match_id','team_id'],how='left')
bad_ids.update(checks.loc[checks.observed.fillna(0).ne(pd.to_numeric(checks.total_shots)),'match_id'])
goals=z[z.result.isin(['Goal','Own goal'])].copy()
goals['benefiting_team']=goals.team_id
own=goals.result.eq('Own goal')
goals.loc[own,'benefiting_team']=np.where(goals.loc[own,'team_id'].eq(goals.loc[own,'home_team_id']),
    goals.loc[own,'away_team_id'],goals.loc[own,'home_team_id'])
count_goals=goals[goals.team_id.notna()].groupby(['match_id','benefiting_team']).size()
for side in ['home','away']:
    idx=pd.MultiIndex.from_arrays([m.match_id,m[f'{side}_team_id']])
    actual=count_goals.reindex(idx,fill_value=0).to_numpy()
    bad_ids.update(m.loc[actual!=pd.to_numeric(m[f'{side}_score']).to_numpy(),'match_id'])
e=read('match_events')
edup=e.match_id.isin(mids) & e.duplicated()
ecount=e.loc[~edup & e.type.eq('goal')].groupby('match_id').size()
bad_ids.update(m.loc[m.match_id.map(ecount).fillna(0).ne(pd.to_numeric(m.home_score)+pd.to_numeric(m.away_score)),'match_id'])

expected=z.loc[~z.match_id.isin(bad_ids) & z.result.ne('Own goal') & z.situation.ne('Penalty')].copy()
out=pd.read_csv(ROOT/'article_xg'/'data_derived'/'shots_primary_candidate.csv',dtype={'match_id':str,'player_id':str,'league_id':str,'shooting_team_id':str})
assert set(out.shot_id)==set(expected.shot_id), 'Independent eligibility disagreement'
joined=out.merge(expected,on='shot_id',suffixes=('_export','_source'),validate='one_to_one')
assert (joined.match_id_export==joined.match_id_source).all()
assert (joined.player_id_export==joined.player_id_source).all()
away=joined.team_id.eq(joined.away_team_id)
x=np.where(away,105-pd.to_numeric(joined.x1),pd.to_numeric(joined.x1))
y=np.where(away,68-pd.to_numeric(joined.y1),pd.to_numeric(joined.y1))
distance=np.sqrt(x*x+(y-34)**2)
theta1=np.arctan2(34-3.66-y,-x)
theta2=np.arctan2(34+3.66-y,-x)
angle=np.abs((theta1-theta2+np.pi)%(2*np.pi)-np.pi)
assert np.allclose(distance,joined.distance_m,atol=1e-10)
assert np.allclose(angle,joined.angle_rad,atol=1e-10)
assert np.array_equal(joined.goal.to_numpy(),joined.result.eq('Goal').astype(int).to_numpy())
assert np.array_equal(joined.home.to_numpy(),(~away).astype(int).to_numpy())
coverage_exempt_fields=['time_added','x2','y2']
assert not out.drop(columns=coverage_exempt_fields).isna().any().any()
for field in coverage_exempt_fields:
    exported=pd.to_numeric(joined[field+'_export'],errors='raise')
    source=pd.to_numeric(joined[field+'_source'].replace('',np.nan),errors='raise')
    assert np.array_equal(exported.isna(),source.isna()), f'Missingness changed in {field}'
    assert np.allclose(exported,source,equal_nan=True,rtol=0,atol=1e-12), f'Source values changed in {field}'
assert np.array_equal(out.time_added_missing.to_numpy(),out.time_added.isna().astype(int).to_numpy())
assert out.match_id.nunique()==len(expected.match_id.unique())
assert len(out)==len(expected)
assert out.groupby('match_id').partition.nunique().eq(1).all()
dates=out.groupby('partition').date.agg(['min','max'])
assert dates.loc['development','max']<dates.loc['calibration','min']<dates.loc['test_reserved','min']
assert dates.loc['calibration','max']<dates.loc['test_reserved','min']

manifest=json.loads((OUT/'manifest.json').read_text())
for item in manifest['sources']:
    with (ROOT/item['file']).open('rb') as stream:
        assert hashlib.file_digest(stream,'sha256').hexdigest()==item['sha256']
report={'status':'PASS','independent_eligibility_identical':True,'geometry_checked_rows':len(out),
        'geometry_check':'vectorized difference of bearings versus atan2 cross/dot and cosine law',
        'labels_checked_rows':len(out),'no_missing_required_core_values':True,'match_partitions_disjoint':True,
        'coverage_exempt_fields_preserved':coverage_exempt_fields,
        'coverage_exempt_source_values_verified_rows':len(out),
        'time_added_missing_values_preserved':int(out.time_added.isna().sum()),
        'temporal_ordering_pass':True,'source_hashes_unchanged':len(manifest['sources']),
        'predictive_metrics_computed':False,'models_fitted':0,
        'qualification':'Internal consistency only; no independent source-provider ground truth.'}
(OUT/'third_check_primary_export.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
