"""Shared code for reviewer-response revised analysis (no-time-added budget).

Mirrors scripts/run_experiments.py exactly, except the predictor budget:
REVISED 'full' = distance, angle, body_part, situation, minute, home
(no time_added / time_added_missing anywhere).
"""
from pathlib import Path
import hashlib
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
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).resolve().parents[1]
RDIR = BASE / 'results' / 'reviewer_response'
SEED = 20260910
EPS = 1e-6
LEAGUES = [47, 53, 54, 55, 87]
NAMES = {47: 'Premier League', 53: 'Ligue 1', 54: 'Bundesliga', 55: 'Serie A', 87: 'LaLiga'}
SEASONS = ['2020/2021', '2021/2022', '2022/2023']
GRIDS = {'M0': [{}], 'M1': [{'C': .1}, {'C': 1.}], 'M2': [{'knots': 4}, {'knots': 6}],
         'M3': [{'leaves': l, 'iterations': i} for l in [7, 15] for i in [100, 200]]}
PRIMARY_REVISED_FEATURES = ['distance_m', 'angle_rad', 'body_part', 'situation', 'minute', 'home']


def clip(p):
    return np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)


def loss(y, p):
    p = clip(p)
    y = np.asarray(y)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def make_model_rev(kind, param, variant='full'):
    geom = ['distance_m', 'angle_rad']
    cats = [] if variant == 'geometry' else ['body_part', 'situation']
    other = [] if variant in ['geometry', 'context'] else ['minute', 'home']
    # NOTE: revised primary budget NEVER adds time_added / time_added_missing.
    # variant 'full_with_time' is provided ONLY for the T1-vs-T2 incremental
    # sensitivity and is never used for any primary endpoint.
    if variant == 'full_with_time':
        other += ['time_added', 'time_added_missing']
    transforms = []
    if kind == 'M2':
        transforms.append(('geometry', Pipeline([('spline', SplineTransformer(
            n_knots=param['knots'], degree=3, knots='quantile',
            include_bias=False, extrapolation='linear')), ('scale', StandardScaler())]), geom))
    else:
        transforms.append(('geometry', StandardScaler(), geom))
    if cats:
        transforms.append(('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cats))
    if other:
        transforms.append(('other', Pipeline([('missing', SimpleImputer(strategy='median')),
                                              ('scale', StandardScaler())]), other))
    pre = ColumnTransformer(transforms, remainder='drop', sparse_threshold=0)
    if kind == 'M0':
        est = DummyClassifier(strategy='prior')
    elif kind in ['M1', 'M2']:
        est = LogisticRegression(C=param.get('C', 1.), solver='lbfgs', max_iter=1500, tol=1e-8)
    else:
        est = HistGradientBoostingClassifier(learning_rate=.05, max_iter=param['iterations'],
                                             max_leaf_nodes=param['leaves'], min_samples_leaf=100,
                                             l2_regularization=1., early_stopping=False,
                                             random_state=SEED, categorical_features=None)
    return Pipeline([('pre', pre), ('model', est)])


def fit_cal(p, y, method='logistic'):
    p = clip(p)
    y = np.asarray(y)
    if method == 'isotonic':
        return {'method': method, 'model': IsotonicRegression(out_of_bounds='clip').fit(p, y)}
    if method == 'logistic':
        X = np.column_stack([np.ones(len(p)), logit(p)])
        initial = [0., 1.]
        bounds = [(None, None), (0, None)]
    else:
        X = np.column_stack([np.ones(len(p)), np.log(p), -np.log1p(-p)])
        initial = [0., 1., 1.]
        bounds = [(None, None), (0, None), (0, None)]

    def f(b):
        z = X @ b
        return np.mean(np.logaddexp(0, z) - y * z), X.T @ (expit(z) - y) / len(y)
    res = minimize(f, initial, jac=True, method='L-BFGS-B', bounds=bounds,
                   options={'ftol': 1e-13, 'gtol': 1e-9, 'maxiter': 1000})
    assert res.success, res.message
    return {'method': method, 'coef': res.x.tolist(), 'n': len(y), 'goals': int(y.sum())}


def apply_cal(cal, p):
    p = clip(p)
    if cal['method'] == 'isotonic':
        return clip(cal['model'].predict(p))
    if cal['method'] == 'logistic':
        X = np.column_stack([np.ones(len(p)), logit(p)])
    else:
        X = np.column_stack([np.ones(len(p)), np.log(p), -np.log1p(-p)])
    return clip(expit(X @ np.asarray(cal['coef'])))


def metrics(y, p):
    p = clip(p)
    y = np.asarray(y)
    z = logit(p)
    off = brentq(lambda a: np.mean(expit(z + a) - y), -30, 30)
    if np.std(z) > 1e-10:
        cal = fit_cal(p, y)
        intercept, slope = cal['coef']
    else:
        intercept, slope = None, None
    return dict(n=len(y), goals=int(y.sum()), prevalence=float(y.mean()),
                mean_xg=float(p.mean()), log_loss=float(loss(y, p).mean()),
                brier=float(np.mean((p - y) ** 2)), auc=float(roc_auc_score(y, p)),
                calibration_in_large=off, joint_intercept=intercept, calibration_slope=slope)


def bootstrap_difference(frame, col='B2', reference='B0', cluster='match_id', reps=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    draws = []
    point = []
    league_rows = []
    for lg in LEAGUES:
        a = frame[frame.league_id == lg].copy()
        a['dl'] = loss(a.goal, a[col]) - loss(a.goal, a[reference])
        a['db'] = (a.goal - a[col]) ** 2 - (a.goal - a[reference]) ** 2
        if cluster == 'week':
            a['unit'] = pd.to_datetime(a.date).dt.to_period('W').astype(str)
        else:
            a['unit'] = a[cluster]
        g = a.groupby('unit').agg(n=('goal', 'size'), dl=('dl', 'sum'), db=('db', 'sum')).to_numpy()
        boot = np.empty((reps, 2))
        for start in range(0, reps, 100):
            inds = rng.integers(0, len(g), size=(min(100, reps - start), len(g)))
            sums = g[inds].sum(axis=1)
            boot[start:start + len(inds)] = sums[:, 1:] / sums[:, :1]
        draws.append(boot)
        point.append([a.dl.mean(), a.db.mean()])
        league_rows.append({'league_id': lg, 'league': NAMES[lg], 'clusters': len(g),
                            'delta_log_loss': a.dl.mean(),
                            'll_low': np.quantile(boot[:, 0], .025), 'll_high': np.quantile(boot[:, 0], .975),
                            'delta_brier': a.db.mean(),
                            'brier_low': np.quantile(boot[:, 1], .025), 'brier_high': np.quantile(boot[:, 1], .975)})
    macro = np.mean(draws, axis=0)
    p = np.mean(point, axis=0)
    league_rows.append({'league_id': 'macro', 'league': 'Macro-average',
                        'clusters': sum(x['clusters'] for x in league_rows),
                        'delta_log_loss': p[0],
                        'll_low': np.quantile(macro[:, 0], .025), 'll_high': np.quantile(macro[:, 0], .975),
                        'delta_brier': p[1],
                        'brier_low': np.quantile(macro[:, 1], .025), 'brier_high': np.quantile(macro[:, 1], .975)})
    return league_rows


def load_primary():
    return pd.read_csv(BASE / 'data_derived' / 'shots_primary_candidate.csv')


def sha(p):
    with open(p, 'rb') as f:
        h = hashlib.sha256()
        for ch in iter(lambda: f.read(1 << 20), b''):
            h.update(ch)
    return h.hexdigest()
