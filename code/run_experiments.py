"""Locked, leakage-controlled xG benchmark. --stage fit then --stage evaluate.

Run with the project virtualenv and OMP_NUM_THREADS=2. Original data untouched.
"""
from pathlib import Path
import argparse, hashlib, json, platform, time, warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize, brentq
from scipy.special import expit, logit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
import sklearn, scipy, joblib
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]; BASE=ROOT/'article_xg'; OUT=BASE/'results'
OUT.mkdir(exist_ok=True); (OUT/'models').mkdir(exist_ok=True)
SEED=20260910; EPS=1e-6; LEAGUES=[47,53,54,55,87]
NAMES={47:'Premier League',53:'Ligue 1',54:'Bundesliga',55:'Serie A',87:'LaLiga'}
SEASONS=['2020/2021','2021/2022','2022/2023']
GRIDS={'M0':[{}], 'M1':[{'C':.1},{'C':1.}], 'M2':[{'knots':4},{'knots':6}],
       'M3':[{'leaves':l,'iterations':i} for l in [7,15] for i in [100,200]]}

def dump(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def clip(p): return np.clip(np.asarray(p,dtype=float),EPS,1-EPS)
def loss(y,p):
    p=clip(p); y=np.asarray(y); return -(y*np.log(p)+(1-y)*np.log1p(-p))

def make_model(kind,param,variant='full'):
    geom=['distance_m','angle_rad']; cats=[] if variant=='geometry' else ['body_part','situation']
    other=[] if variant in ['geometry','context'] else ['minute','home']
    if variant=='full': other+=['time_added','time_added_missing']
    transforms=[]
    if kind=='M2':
        transforms.append(('geometry',Pipeline([('spline',SplineTransformer(n_knots=param['knots'],degree=3,knots='quantile',include_bias=False,extrapolation='linear')),('scale',StandardScaler())]),geom))
    else: transforms.append(('geometry',StandardScaler(),geom))
    if cats: transforms.append(('category',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats))
    if other: transforms.append(('other',Pipeline([('missing',SimpleImputer(strategy='median')),('scale',StandardScaler())]),other))
    pre=ColumnTransformer(transforms,remainder='drop',sparse_threshold=0)
    if kind=='M0': est=DummyClassifier(strategy='prior')
    elif kind in ['M1','M2']: est=LogisticRegression(C=param.get('C',1.),solver='lbfgs',max_iter=1500,tol=1e-8)
    else: est=HistGradientBoostingClassifier(learning_rate=.05,max_iter=param['iterations'],max_leaf_nodes=param['leaves'],min_samples_leaf=100,l2_regularization=1.,early_stopping=False,random_state=SEED,categorical_features=None)
    return Pipeline([('pre',pre),('model',est)])

def fit_cal(p,y,method='logistic'):
    p=clip(p); y=np.asarray(y)
    if method=='isotonic': return {'method':method,'model':IsotonicRegression(out_of_bounds='clip').fit(p,y)}
    if method=='logistic': X=np.column_stack([np.ones(len(p)),logit(p)]); initial=[0.,1.]; bounds=[(None,None),(0,None)]
    else: X=np.column_stack([np.ones(len(p)),np.log(p),-np.log1p(-p)]); initial=[0.,1.,1.]; bounds=[(None,None),(0,None),(0,None)]
    def f(b):
        z=X@b; return np.mean(np.logaddexp(0,z)-y*z),X.T@(expit(z)-y)/len(y)
    res=minimize(f,initial,jac=True,method='L-BFGS-B',bounds=bounds,options={'ftol':1e-13,'gtol':1e-9,'maxiter':1000})
    assert res.success, res.message
    return {'method':method,'coef':res.x.tolist(),'n':len(y),'goals':int(y.sum())}
def apply_cal(cal,p):
    p=clip(p)
    if cal['method']=='isotonic': return clip(cal['model'].predict(p))
    X=np.column_stack([np.ones(len(p)),logit(p)]) if cal['method']=='logistic' else np.column_stack([np.ones(len(p)),np.log(p),-np.log1p(-p)])
    return clip(expit(X@np.asarray(cal['coef'])))

def metrics(y,p):
    p=clip(p); y=np.asarray(y); z=logit(p)
    off=brentq(lambda a:np.mean(expit(z+a)-y),-30,30)
    if np.std(z)>1e-10:
        cal=fit_cal(p,y); intercept,slope=cal['coef']
    else: intercept=slope=None
    return dict(n=len(y),goals=int(y.sum()),prevalence=float(y.mean()),mean_xg=float(p.mean()),log_loss=float(loss(y,p).mean()),brier=float(np.mean((p-y)**2)),auc=float(roc_auc_score(y,p)),calibration_in_large=off,joint_intercept=intercept,calibration_slope=slope)

def load_data(broad=False):
    d=pd.read_csv(BASE/'data_derived'/('shots_provisional.csv' if broad else 'shots_primary_candidate.csv'))
    if broad:
        m=pd.read_csv(ROOT/'CSV/match_infos.csv',sep=';',encoding='utf-8-sig').drop_duplicates('match_id')
        upper=np.where((m.league_id==54)|((m.league_id==53)&m.season.isin(['2023/2024','2024/2025'])),34,38)
        d=d[d.match_id.isin(m.loc[m['round'].between(1,upper),'match_id'])].copy()
    assert not d.shot_id.duplicated().any(); assert d.groupby('match_id').partition.nunique().max()==1
    return d

def stage_fit():
    d=load_data(); dev=d[d.partition=='development']; records=[]; choices={}
    lock={'created_utc':pd.Timestamp.now(tz='UTC').isoformat(),'seed':SEED,'grids':GRIDS,'code_sha256':sha(Path(__file__)),'protocol_sha256':sha(BASE/'EXPERIMENT_LOCK.md'),'data_sha256':sha(BASE/'data_derived/shots_primary_candidate.csv'),'python':platform.python_version(),'sklearn':sklearn.__version__,'scipy':scipy.__version__,'numpy':np.__version__,'test_predictive_metrics_seen':False}
    assert not (OUT/'predictions_test.csv.gz').exists(), 'Do not reselect after test evaluation'
    dump('computational_lock.json',lock)
    for target in ['all']+LEAGUES:
        train=dev if target=='all' else dev[dev.league_id!=target]
        choices[str(target)]={}
        for kind,grid in GRIDS.items():
            candidates=[]
            for param in grid:
                scores=[]
                for j in [1,2]:
                    a=train[train.season.isin(SEASONS[:j])]; b=train[train.season==SEASONS[j]]
                    model=make_model(kind,param).fit(a,a.goal)
                    p=clip(model.predict_proba(b)[:,1]); rowloss=loss(b.goal,p)
                    score=pd.DataFrame({'league':b.league_id.to_numpy(),'loss':rowloss}).groupby('league').loss.mean().mean()
                    scores.append(float(score)); records.append(dict(target=target,model=kind,param=json.dumps(param),fold=j,macro_log_loss=score,n_train=len(a),n_valid=len(b)))
                candidates.append((np.mean(scores),param))
            best=min(candidates,key=lambda t:t[0]); choices[str(target)][kind]=best[1]
            model=make_model(kind,best[1]).fit(train,train.goal)
            joblib.dump(model,OUT/'models'/f'{target}_{kind}.joblib')
            pd.DataFrame(records).to_csv(OUT/'validation_scores.csv',index=False)
            dump('selected_hyperparameters.json',choices)
            print(f'FIT target={target} model={kind} macro-CV={best[0]:.6f} parameters={best[1]}',flush=True)
    dump('fit_completed.json',{'utc':pd.Timestamp.now(tz='UTC').isoformat(),'models':24,'test_predictions_computed':False})

def bootstrap_difference(frame,col='B2',reference='B0',cluster='match_id',reps=2000):
    rng=np.random.default_rng(SEED); draws=[]; point=[]; league_rows=[]
    for lg in LEAGUES:
        a=frame[frame.league_id==lg].copy()
        a['dl']=loss(a.goal,a[col])-loss(a.goal,a[reference]); a['db']=(a.goal-a[col])**2-(a.goal-a[reference])**2
        if cluster=='week': a['unit']=pd.to_datetime(a.date).dt.to_period('W').astype(str)
        else: a['unit']=a[cluster]
        g=a.groupby('unit').agg(n=('goal','size'),dl=('dl','sum'),db=('db','sum')).to_numpy()
        # Identical sampled clusters for both competing predictions and metrics.
        boot=np.empty((reps,2))
        for start in range(0,reps,100):
            inds=rng.integers(0,len(g),size=(min(100,reps-start),len(g)))
            sums=g[inds].sum(axis=1); boot[start:start+len(inds)]=sums[:,1:]/sums[:,:1]
        draws.append(boot); point.append([a.dl.mean(),a.db.mean()])
        league_rows.append({'league_id':lg,'league':NAMES[lg],'clusters':len(g),'delta_log_loss':a.dl.mean(),'ll_low':np.quantile(boot[:,0],.025),'ll_high':np.quantile(boot[:,0],.975),'delta_brier':a.db.mean(),'brier_low':np.quantile(boot[:,1],.025),'brier_high':np.quantile(boot[:,1],.975)})
    macro=np.mean(draws,axis=0); p=np.mean(point,axis=0)
    league_rows.append({'league_id':'macro','league':'Macro-average','clusters':sum(x['clusters'] for x in league_rows),'delta_log_loss':p[0],'ll_low':np.quantile(macro[:,0],.025),'ll_high':np.quantile(macro[:,0],.975),'delta_brier':p[1],'brier_low':np.quantile(macro[:,1],.025),'brier_high':np.quantile(macro[:,1],.975)})
    return league_rows

def stage_evaluate():
    assert (OUT/'fit_completed.json').exists()
    d=load_data(); cal=d[d.partition=='calibration']; test=d[d.partition=='test_reserved']; dev=d[d.partition=='development']
    lock=json.loads((OUT/'computational_lock.json').read_text()); assert sha(BASE/'data_derived/shots_primary_candidate.csv')==lock['data_sha256']
    choices=json.loads((OUT/'selected_hyperparameters.json').read_text()); rows=[]; preds=[]; calibrators=[]
    def record(frame,experiment,kind,variant,p):
        for lg,g in frame.assign(p=p).groupby('league_id'):
            rows.append(dict(experiment=experiment,model=kind,variant=variant,league_id=lg,**metrics(g.goal,g.p)))
        if experiment=='A': rows.append(dict(experiment=experiment,model=kind,variant=variant,league_id='pooled',**metrics(frame.goal,p)))
    for target in ['all']+LEAGUES:
        t=test if target=='all' else test[test.league_id==target]
        c=cal if target=='all' else cal[cal.league_id!=target]
        for kind in GRIDS:
            model=joblib.load(OUT/'models'/f'{target}_{kind}.joblib'); p=clip(model.predict_proba(t)[:,1])
            variants={'raw' if target=='all' else 'B0':p}
            methods=['logistic','beta','isotonic'] if kind=='M3' else ['logistic']
            for method in methods:
                cc=fit_cal(model.predict_proba(c)[:,1],c.goal,method)
                label=('calibrated' if target=='all' else 'B1')+('' if method=='logistic' else '_'+method)
                variants[label]=apply_cal(cc,p)
                joblib.dump(cc,OUT/'models'/f'{target}_{kind}_{label}.joblib')
                if method!='isotonic': calibrators.append(dict(target=target,model=kind,variant=label,**cc))
                if target!='all':
                    tc=cal[cal.league_id==target]; cc=fit_cal(model.predict_proba(tc)[:,1],tc.goal,method)
                    label='B2'+('' if method=='logistic' else '_'+method); variants[label]=apply_cal(cc,p)
                    joblib.dump(cc,OUT/'models'/f'{target}_{kind}_{label}.joblib')
                    if method!='isotonic': calibrators.append(dict(target=target,model=kind,variant=label,**cc))
            part=t[['shot_id','match_id','league_id','date','goal','time_added_missing','body_part','situation','distance_m']].copy()
            part['experiment']='A' if target=='all' else 'B'; part['model']=kind
            for v,pp in variants.items():
                part[v]=pp; record(t,part.experiment.iloc[0],kind,v,pp)
            preds.append(part)
        print(f'EVALUATED target={target}',flush=True)
    ptable=pd.concat(preds,ignore_index=True); ptable.to_csv(OUT/'predictions_test.csv.gz',index=False)
    for kind in GRIDS:
        pp=ptable[(ptable.experiment=='B')&(ptable.model==kind)]
        for v in ['B0','B1','B2']+(['B1_beta','B2_beta','B1_isotonic','B2_isotonic'] if kind=='M3' else []):
            rows.append(dict(experiment='B',model=kind,variant=v,league_id='pooled',**metrics(pp.goal,pp[v])))
    dump('calibration_parameters.json',calibrators)
    # Macro-average is average of metrics, not calibration fitted to mixed predictions.
    scores=pd.DataFrame(rows); macro=scores[scores.league_id!='pooled'].groupby(['experiment','model','variant'])[['log_loss','brier','auc']].mean().reset_index(); macro['league_id']='macro'
    scores=pd.concat([scores,macro],ignore_index=True); scores.to_csv(OUT/'test_metrics.csv',index=False)
    main=ptable[(ptable.experiment=='B')&(ptable.model=='M3')]
    for col in ['B2','B1','B2_beta','B2_isotonic']:
        pd.DataFrame(bootstrap_difference(main,col)).to_csv(OUT/f'bootstrap_{col}_vs_B0.csv',index=False)
    pd.DataFrame(bootstrap_difference(main,cluster='week')).to_csv(OUT/'bootstrap_week_B2_vs_B0.csv',index=False)
    print('MAIN BOOTSTRAP COMPLETE',flush=True)
    # Temporal feature and season ablations use frozen all-league M3 parameters.
    sensitivity=[]
    for variant in ['geometry','context','without_time_added','exclude_2020']:
        train=dev[dev.season!='2020/2021'] if variant=='exclude_2020' else dev
        model=make_model('M3',choices['all']['M3'],'full' if variant=='exclude_2020' else variant).fit(train,train.goal)
        p=clip(model.predict_proba(test)[:,1]); cc=fit_cal(model.predict_proba(cal)[:,1],cal.goal)
        pp=apply_cal(cc,p)
        for v,z in [('raw',p),('calibrated',pp)]:
            sensitivity.append(dict(analysis=variant,variant=v,**metrics(test.goal,z)))
        joblib.dump(model,OUT/'models'/f'ablation_{variant}.joblib')
        print(f'ABLATION {variant}',flush=True)
    pd.DataFrame(sensitivity).to_csv(OUT/'temporal_ablations.csv',index=False)
    # Broad-cohort replication, same hyperparameters. Retains coherent individual shots.
    broad=load_data(True); bd=broad[broad.partition=='development']; bc=broad[broad.partition=='calibration']; bt=broad[broad.partition=='test_reserved']; broadparts=[]; broadrows=[]
    for target in ['all']+LEAGUES:
        train=bd if target=='all' else bd[bd.league_id!=target]
        t=bt if target=='all' else bt[bt.league_id==target]
        c=bc if target=='all' else bc[bc.league_id==target]
        model=make_model('M3',choices[str(target)]['M3']).fit(train,train.goal)
        p=clip(model.predict_proba(t)[:,1]); cc=fit_cal(model.predict_proba(c)[:,1],c.goal); pp=apply_cal(cc,p)
        for v,z in [('raw',p),('adapted',pp)]: broadrows.append(dict(target=target,variant=v,**metrics(t.goal,z)))
        if target!='all': broadparts.append(t[['shot_id','match_id','league_id','date','goal']].assign(B0=p,B2=pp))
        print(f'BROAD target={target}',flush=True)
    pd.DataFrame(broadrows).to_csv(OUT/'broad_cohort_metrics.csv',index=False)
    bp=pd.concat(broadparts,ignore_index=True); bp.to_csv(OUT/'predictions_broad_test.csv.gz',index=False)
    pd.DataFrame(bootstrap_difference(bp)).to_csv(OUT/'bootstrap_broad_B2_vs_B0.csv',index=False)
    d.groupby(['league_id','season','partition']).agg(shots=('goal','size'),goals=('goal','sum'),matches=('match_id','nunique'),time_recorded=('time_added','count')).reset_index().to_csv(OUT/'cohort_description.csv',index=False)
    for name,dd in [('primary',d),('broad',broad)]:
        print(name,len(dd),dd.match_id.nunique(),int(dd.goal.sum()),flush=True)
    dump('evaluation_completed.json',{'utc':pd.Timestamp.now(tz='UTC').isoformat(),'primary_shots':len(d),'primary_matches':d.match_id.nunique(),'test_shots':len(test),'test_matches':test.match_id.nunique(),'broad_shots':len(broad),'broad_matches':broad.match_id.nunique(),'bootstrap_repetitions':2000,'evaluation_code_sha256':sha(Path(__file__))})

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--stage',choices=['fit','evaluate'],required=True); args=parser.parse_args()
    with threadpool_limits(limits=2):
        if args.stage=='fit': stage_fit()
        else: stage_evaluate()
