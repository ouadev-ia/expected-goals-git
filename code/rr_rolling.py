"""Rolling-origin temporal robustness (revised budget, M3, frozen primary settings).

Origin 1: dev 2020/21, calib 2021/22, test 2022/23.
Origin 2: dev 2020/21-21/22, calib 2022/23, test 2023/24.
Origin 3: primary windows (re-tabulated from revised results, no refit).

Secondary analysis. No test outcome enters fitting; hyperparameters are the
frozen revised-primary selections (no new selection rule for Origin 1).
"""
import sys
import json
from pathlib import Path
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).parent))
from rr_common import (RDIR, LEAGUES, NAMES, make_model_rev, fit_cal, apply_cal,
                       clip, metrics, load_primary)

ORIGINS = [
    dict(name='origin1', dev=['2020/2021'], calib=['2021/2022'], test=['2022/2023']),
    dict(name='origin2', dev=['2020/2021', '2021/2022'], calib=['2022/2023'], test=['2023/2024']),
]


def run_origin(o):
    d = load_primary()
    dev = d[d.season.isin(o['dev'])]
    cal = d[d.season.isin(o['calib'])]
    tst = d[d.season.isin(o['test'])]
    assert set(dev.match_id).isdisjoint(cal.match_id)
    assert set(cal.match_id).isdisjoint(tst.match_id)
    assert set(dev.match_id).isdisjoint(tst.match_id)
    choices = json.loads((RDIR / 'revised_selected_hyperparameters.json').read_text())
    rows = []
    # Experiment A (all leagues)
    ma = make_model_rev('M3', choices['all']['M3']).fit(dev, dev.goal)
    pa = clip(ma.predict_proba(tst)[:, 1])
    ca = fit_cal(ma.predict_proba(cal)[:, 1], cal.goal)
    qa = apply_cal(ca, pa)
    for v, pp in [('B0_raw', pa), ('B2', qa)]:
        r = dict(origin=o['name'], experiment='A', variant=v, n=len(tst), goals=int(tst.goal.sum()),
                 **{k: metrics(tst.goal, pp)[k] for k in
                    ['log_loss', 'brier', 'auc', 'calibration_in_large', 'calibration_slope']})
        rows.append(r)
    # Experiment B (leave-one-league-out macro)
    lg_rows = []
    for target in LEAGUES:
        tr = dev[dev.league_id != target]
        tt = tst[tst.league_id == target]
        cc_src = cal[cal.league_id != target]
        cc_dst = cal[cal.league_id == target]
        m = make_model_rev('M3', choices[str(target)]['M3']).fit(tr, tr.goal)
        p0 = clip(m.predict_proba(tt)[:, 1])
        q2 = apply_cal(fit_cal(m.predict_proba(cc_dst)[:, 1], cc_dst.goal), p0)
        mb0 = metrics(tt.goal, p0)
        mb2 = metrics(tt.goal, q2)
        lg_rows.append(dict(league_id=target, n=len(tt), goals=int(tt.goal.sum()),
                            b0=mb0['log_loss'], b2=mb2['log_loss'], d=mb2['log_loss'] - mb0['log_loss'],
                            brier0=mb0['brier'], brier2=mb2['brier'],
                            auc0=mb0['auc'], cil2=mb2['calibration_in_large'], slope2=mb2['calibration_slope']))
    L = pd.DataFrame(lg_rows)
    rows.append(dict(origin=o['name'], experiment='B_macro', variant='B0', n=int(L.n.sum()),
                     goals=int(L.goals.sum()), log_loss=L.b0.mean(), brier=L.brier0.mean(),
                     auc=L.auc0.mean(), calibration_in_large=None, calibration_slope=None))
    rows.append(dict(origin=o['name'], experiment='B_macro', variant='B2', n=int(L.n.sum()),
                     goals=int(L.goals.sum()), log_loss=L.b2.mean(), brier=L.brier2.mean(),
                     auc=L.auc0.mean(), calibration_in_large=None, calibration_slope=None))
    rows.append(dict(origin=o['name'], experiment='B_macro', variant='B2-B0', n=int(L.n.sum()),
                     goals=int(L.goals.sum()), log_loss=L.d.mean(),
                     brier=(L.brier2 - L.brier0).mean(), auc=None,
                     calibration_in_large=None, calibration_slope=None))
    L.to_csv(RDIR / f"rolling_{o['name']}_leagues.csv", index=False)
    joblib.dump(ma, RDIR / 'models_rev' / f"rolling_{o['name']}_A_M3.joblib")
    print(f"ROLLING {o['name']} B2-B0={L.d.mean():.6f} n={L.n.sum()}", flush=True)
    return rows


def main():
    all_rows = []
    for o in ORIGINS:
        all_rows += run_origin(o)
    # Origin 3 = revised primary windows, re-tabulated (no refit).
    m = pd.read_csv(RDIR / 'revised_test_metrics.csv')
    b = m[(m.experiment == 'B') & (m.model == 'M3')]
    g = lambda v, c: b[(b.variant == v) & (b.league_id == 'macro')].iloc[0][c]
    gl = b[b.variant == 'B0']
    all_rows.append(dict(origin='origin3', experiment='B_macro', variant='B2-B0',
                         n=41925, goals=4296, log_loss=g('B2', 'log_loss') - g('B0', 'log_loss'),
                         brier=g('B2', 'brier') - g('B0', 'brier'), auc=None,
                         calibration_in_large=None, calibration_slope=None))
    pd.DataFrame(all_rows).to_csv(RDIR / 'rolling_origin.csv', index=False)
    print('ROLLING COMPLETE', flush=True)


if __name__ == '__main__':
    main()
