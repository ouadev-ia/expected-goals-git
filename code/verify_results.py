"""Independent verification from persisted predictions, without training selection."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import rankdata
import joblib

BASE=Path(__file__).resolve().parents[1]; R=BASE/'results'
d=pd.read_csv(BASE/'data_derived/shots_primary_candidate.csv')
p=pd.read_csv(R/'predictions_test.csv.gz'); scores=pd.read_csv(R/'test_metrics.csv')
t=d[d.partition=='test_reserved']; checked=0; maxdiff=0.
assert set(p.shot_id)==set(t.shot_id)
for (exp,model),g in p.groupby(['experiment','model']):
    assert not g.shot_id.duplicated().any(); assert set(g.shot_id)==set(t.shot_id)
    z=g.merge(t[['shot_id','goal','match_id','league_id']],on='shot_id',validate='one_to_one',suffixes=('','_source'))
    for col in ['goal','match_id','league_id']: assert z[col].eq(z[col+'_source']).all()
    variants=['raw','calibrated'] if exp=='A' else ['B0','B1','B2']
    if model=='M3': variants+=['calibrated_beta','calibrated_isotonic'] if exp=='A' else ['B1_beta','B1_isotonic','B2_beta','B2_isotonic']
    for v in variants:
        for lg,gg in [('pooled',g)]+list(g.groupby('league_id')):
            a=gg[v].to_numpy(); y=gg.goal.to_numpy(); assert np.isfinite(a).all() and ((a>=1e-6)&(a<=1-1e-6)).all()
            manual_ll=float(np.where(y==1,-np.log(a),-np.log(1-a)).mean())
            manual_br=float(np.dot(a-y,a-y)/len(y))
            ranks=rankdata(a); n1=int(y.sum()); n0=len(y)-n1
            manual_auc=float((ranks[y==1].sum()-n1*(n1+1)/2)/(n1*n0))
            row=scores[(scores.experiment==exp)&(scores.model==model)&(scores.variant==v)&(scores.league_id==str(lg))]
            assert len(row)==1
            diffs=np.abs(row[['log_loss','brier','auc']].iloc[0].to_numpy(dtype=float)-[manual_ll,manual_br,manual_auc])
            assert diffs.max()<1e-12
            maxdiff=max(maxdiff,float(diffs.max())); checked+=1
            if v not in ['raw','B0','B1_isotonic','B2_isotonic','calibrated_isotonic'] and model!='M0' and lg!='pooled':
                basev='raw' if exp=='A' else 'B0'
                assert abs(manual_auc-float(scores[(scores.experiment==exp)&(scores.model==model)&(scores.variant==basev)&(scores.league_id==str(lg))].auc.iloc[0]))<1e-12

for (exp,model,v),gg in scores[~scores.league_id.isin(['pooled','macro'])].groupby(['experiment','model','variant']):
    assert len(gg)==5
    macro=scores[(scores.experiment==exp)&(scores.model==model)&(scores.variant==v)&(scores.league_id=='macro')]
    assert np.allclose(gg[['log_loss','brier','auc']].mean(),macro[['log_loss','brier','auc']].iloc[0],atol=1e-13)

feature_sets=[]
for target in ['all',47,53,54,55,87]:
    train=d[d.partition=='development']; train=train if target=='all' else train[train.league_id!=target]
    sample=t if target=='all' else t[t.league_id==target]
    sample=sample.sample(120,random_state=56)
    for kind in ['M0','M1','M2','M3']:
        m=joblib.load(R/'models'/f'{target}_{kind}.joblib')
        features=[c for name,tr,cols in m.named_steps['pre'].transformers_ if name!='remainder' for c in cols]
        assert set(features)=={'distance_m','angle_rad','body_part','situation','minute','home','time_added','time_added_missing'}
        assert not {'x2','y2','goal','player_id','season','league_id','match_id'}.intersection(features)
        imp=m.named_steps['pre'].named_transformers_['other'].named_steps['missing']
        assert np.allclose(imp.statistics_,train[['minute','home','time_added','time_added_missing']].median().to_numpy())
        pp=p[(p.experiment==('A' if target=='all' else 'B'))&(p.model==kind)].set_index('shot_id').loc[sample.shot_id]
        assert np.allclose(np.clip(m.predict_proba(sample)[:,1],1e-6,1-1e-6),pp['raw' if target=='all' else 'B0'],atol=1e-13)
        feature_sets.append(features)

main=p[(p.experiment=='B')&(p.model=='M3')]
bs=pd.read_csv(R/'bootstrap_B2_vs_B0.csv')
delta=[]
for lg,g in main.groupby('league_id'):
    y=g.goal.to_numpy(); a=g.B2.to_numpy(); b=g.B0.to_numpy()
    dif=np.where(y==1,np.log(b/a),np.log((1-b)/(1-a))).mean(); delta.append(dif)
    assert abs(dif-bs[bs.league_id==str(lg)].delta_log_loss.iloc[0])<1e-12
assert abs(np.mean(delta)-bs[bs.league_id=='macro'].delta_log_loss.iloc[0])<1e-12
manifest=json.loads((BASE/'audit/manifest.json').read_text())
for item in manifest['sources']:
    with (BASE.parent/item['file']).open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==item['sha256']
lock=json.loads((R/'computational_lock.json').read_text())
with (BASE/'EXPERIMENT_LOCK.md').open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==lock['protocol_sha256']
report={'status':'PASS','independent_metric_rows_checked':checked,'maximum_absolute_metric_difference':maxdiff,'models_reload_checked':24,'predictions_recomputed':24*120,'calibration_monotone_auc_invariance':True,'macro_averages_verified':True,'primary_effect_independently_verified':True,'training_only_medians_verified':True,'prohibited_predictors_absent':True,'original_source_hashes_unchanged':len(manifest['sources']),'protocol_hash_unchanged':True,'limits':'Numerical and pipeline verification, not external ground-truth validation or independent peer review.'}
(R/'independent_results_verification.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
