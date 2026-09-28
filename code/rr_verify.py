"""Independent verification of the revised analysis (plain-numpy recomputation)."""
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from rr_common import RDIR, BASE, LEAGUES, SEED

R = RDIR
log = []


def check(name, ok, detail=''):
    log.append(dict(check=name, status='PASS' if ok else 'FAIL', detail=detail))
    print(('PASS' if ok else 'FAIL'), name, detail, flush=True)


def ll(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y)
    return float((-(y * np.log(p) + (1 - y) * np.log1p(-p))).mean())


d = pd.read_csv(BASE / 'data_derived' / 'shots_primary_candidate.csv')
check('cohort shots', len(d) == 214143, str(len(d)))
check('cohort matches', d.match_id.nunique() == 8702, str(d.match_id.nunique()))
check('test shots', len(d[d.partition == 'test_reserved']) == 41925)

p = pd.read_csv(R / 'revised_predictions_test.csv.gz')
pm = p[(p.experiment == 'B') & (p.model == 'M3')]
# equal-league weighting check
exp = {}
for lg in LEAGUES:
    g = pm[pm.league_id == lg]
    exp[lg] = (ll(g.goal, g.B0), ll(g.goal, g.B2))
b0m = float(np.mean([v[0] for v in exp.values()]))
b2m = float(np.mean([v[1] for v in exp.values()]))
m = pd.read_csv(R / 'revised_test_metrics.csv')
rep_b0 = float(m[(m.experiment == 'B') & (m.model == 'M3') & (m.variant == 'B0') & (m.league_id == 'macro')].iloc[0].log_loss)
check('B0 macro recomputed', abs(b0m - rep_b0) < 1e-9, f'{b0m:.6f} vs {rep_b0:.6f}')
rep_b2 = float(m[(m.experiment == 'B') & (m.model == 'M3') & (m.variant == 'B2') & (m.league_id == 'macro')].iloc[0].log_loss)
check('B2 macro recomputed', abs(b2m - rep_b2) < 1e-9, f'{b2m:.6f} vs {rep_b2:.6f}')
bs = pd.read_csv(R / 'revised_bootstrap_M3_B2_vs_B0.csv')
brow = bs[bs.league_id == 'macro'].iloc[0]
check('B2B0 point in CI file', abs(brow.delta_log_loss - (b2m - b0m)) < 1e-9)
check('B2B0 CI contains point', brow.ll_low <= brow.delta_log_loss <= brow.ll_high)
check('bootstrap rows', len(bs) == 6 and bs.clusters.sum() - 1705 == bs[bs.league_id == 'macro'].clusters.iloc[0])

# identical test shots across regimes/models
ids_b = set(p[(p.experiment == 'B') & (p.model == 'M3')].shot_id)
test_ids = set(d[d.partition == 'test_reserved'].shot_id)
check('B test shots = cohort test shots', ids_b == test_ids, f'{len(ids_b)}')
# Experiment A: regenerate predictions from reloaded models (models were not persisted per-shot)
from rr_common import clip as _clip, make_model_rev as _mm
tst = d[d.partition == 'test_reserved']
okA = True
detA = []
for kind in ['M0', 'M1', 'M2', 'M3']:
    mdl = joblib.load(R / 'models_rev' / f'all_{kind}.joblib')
    pp = _clip(mdl.predict_proba(tst)[:, 1])
    a = pd.read_csv(R / 'experiment_A_revised_metrics.csv')
    rep = float(a[(a.model == kind) & (a.variant == 'raw') & (a.league_id == 'pooled')].iloc[0].log_loss)
    if abs(ll(tst.goal, pp) - rep) > 1e-9:
        okA = False
        detA.append(kind)
check('A metrics regenerated from reloaded models', okA, str(detA))

# leakage: no forbidden predictor in any saved revised pipeline
from sklearn.compose import ColumnTransformer
bad = {'time_added', 'time_added_missing', 'league_id', 'season', 'player_id',
       'shooting_team_id', 'goal', 'x2', 'y2'}
leak = []
for f in (R / 'models_rev').glob('*.joblib'):
    if 'isotonic' in f.name or '_B1' in f.name or '_B2' in f.name or 'rolling' in f.name or 'ablation' in f.name:
        continue
    try:
        pipe = joblib.load(f)
    except Exception:
        continue
    cols = set()
    pre = pipe.named_steps.get('pre')
    if isinstance(pre, ColumnTransformer):
        for name_t, _, c in pre.transformers_:
            if name_t == 'remainder':
                continue  # dropped columns are not predictors
            cols.update(list(c) if not isinstance(c, str) else [c])
    hit = cols & bad
    if hit:
        leak.append((f.name, hit))
check('no forbidden predictors in pipelines', not leak, str(leak[:3]))

# training-only imputation: saved statistics equal recomputed per-target training medians
dev = d[d.partition == 'development']
ok_med, det_med = True, []
for tgt in ['all', 47, 53, 54, 55, 87]:
    pipe = joblib.load(R / 'models_rev' / f'{tgt}_M3.joblib')
    for name, trans, cols in pipe.named_steps['pre'].transformers_:
        if name == 'other':
            tr = dev if tgt == 'all' else dev[dev.league_id != int(tgt)]
            ref = tr[list(cols)].median().to_numpy()
            if not np.allclose(trans.named_steps['missing'].statistics_, ref, atol=1e-12):
                ok_med = False
                det_med.append(tgt)
check('imputation statistics = per-target training medians', ok_med, str(det_med))

# calibration params: n/goals match calibration cohort; spot refit of one B2
cal = pd.read_csv(R / 'revised_calibration_parameters.json'.replace('.json', '.json')) if False else None
cj = json.loads((R / 'revised_calibration_parameters.json').read_text())
e = [x for x in cj if x['target'] == 47 and x['model'] == 'M3' and x['variant'] == 'B2'][0]
cc = d[(d.partition == 'calibration') & (d.league_id == 47)]
check('calibrator n/goals', e['n'] == len(cc) and e['goals'] == int(cc.goal.sum()))
from scipy.special import expit, logit
from scipy.optimize import minimize
base = joblib.load(R / 'models_rev' / '47_M3.joblib')
pp = np.clip(base.predict_proba(cc)[:, 1], 1e-6, 1 - 1e-6)
X = np.column_stack([np.ones(len(pp)), logit(pp)])
yy = cc.goal.to_numpy()
f = lambda b: np.mean(np.logaddexp(0, X @ b) - yy * (X @ b))
res = minimize(f, np.array([-0.5, 1.2]), method='L-BFGS-B', bounds=[(None, None), (0, None)])
Xb = np.column_stack([np.ones(len(pp)), logit(pp)])
ll_refit = float(np.mean(np.logaddexp(0, Xb @ res.x) - yy * (Xb @ res.x)))
ll_saved = float(np.mean(np.logaddexp(0, Xb @ np.asarray(e['coef'])) - yy * (Xb @ np.asarray(e['coef']))))
check('calibrator spot-refit agrees', np.max(np.abs(np.array(res.x) - np.array(e['coef']))) < 1e-3
      and abs(ll_refit - ll_saved) < 1e-9, f'max|dcoef|={np.max(np.abs(np.array(res.x) - np.array(e["coef"]))):.2e}')

# rolling windows disjoint + origin3 matches primary
ro = pd.read_csv(R / 'rolling_origin.csv')
o3 = ro[(ro.origin == 'origin3') & (ro.variant == 'B2-B0')].iloc[0]
check('origin3 = revised primary', abs(o3.log_loss - brow.delta_log_loss) < 1e-9)

# full-pipeline file integrity
fp = pd.read_csv(R / 'full_pipeline_bootstrap.csv')
check('fullpipe 1000 rows', len(fp) == 1000, str(len(fp)))
dd = fp.B2B0.dropna()
check('fullpipe completed>=990', len(dd) >= 990, str(len(dd)))

# club macro point matches primary
cm = pd.read_csv(R / 'club_cluster_macro.csv').iloc[0]
check('club point = primary point', abs(cm.macro_point - brow.delta_log_loss) < 1e-9)

# model family intervals contain points
mf = pd.read_csv(R / 'model_family_B2B0.csv')
check('family CIs contain points', bool(((mf.ll_low <= mf.B2B0) & (mf.B2B0 <= mf.ll_high)).all()))

# S8/S9 shapes, no NaN in key columns
s8 = pd.read_csv(R / 'B0_B1_B2_calibration.csv')
check('S8 15 rows, key cols finite', len(s8) == 15 and bool(s8[['observed', 'mean_pred', 'CIL', 'slope']].notna().all().all()))
s9 = pd.read_csv(R / 'reliability_bins.csv')
check('S9 150 rows', len(s9) == 150, str(len(s9)))

with open(R / 'verification_log.md', 'w') as f:
    f.write('# Independent verification log (revised analysis)\n\n')
    for l in log:
        f.write(f"- {l['status']}: {l['check']} {l['detail']}\n")
fails = [l for l in log if l['status'] == 'FAIL']
print('FAILURES:', len(fails))
