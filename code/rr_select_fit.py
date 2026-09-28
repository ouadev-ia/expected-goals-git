"""Revised primary analysis (no-time-added budget). Stages: select | evalA | evalB.

Selection uses ONLY development partitions (no test outcomes anywhere).
Outputs go to results/reviewer_response/ (original files untouched).
"""
import sys
import json
import argparse
from pathlib import Path
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).parent))
from rr_common import (BASE, RDIR, SEED, LEAGUES, NAMES, SEASONS, GRIDS,
                       make_model_rev, fit_cal, apply_cal, clip, loss, metrics, load_primary)

MDIR = RDIR / 'models_rev'


def stage_select():
    d = load_primary()
    dev = d[d.partition == 'development']
    records = []
    choices = {}
    for target in ['all'] + LEAGUES:
        train = dev if target == 'all' else dev[dev.league_id != target]
        choices[str(target)] = {}
        for kind, grid in GRIDS.items():
            cands = []
            for param in grid:
                scores = []
                for j in [1, 2]:
                    a = train[train.season.isin(SEASONS[:j])]
                    b = train[train.season == SEASONS[j]]
                    model = make_model_rev(kind, param).fit(a, a.goal)
                    p = clip(model.predict_proba(b)[:, 1])
                    rowloss = loss(b.goal, p)
                    score = pd.DataFrame({'league': b.league_id.to_numpy(), 'loss': rowloss}
                                         ).groupby('league').loss.mean().mean()
                    scores.append(float(score))
                    records.append(dict(target=target, model=kind, param=json.dumps(param),
                                        fold=j, macro_log_loss=score, n_train=len(a), n_valid=len(b)))
                cands.append((float(sum(scores) / len(scores)), param))
            best = min(cands, key=lambda t: t[0])
            choices[str(target)][kind] = best[1]
            model = make_model_rev(kind, best[1]).fit(train, train.goal)
            joblib.dump(model, MDIR / f'{target}_{kind}.joblib')
            print(f'SELECT target={target} model={kind} macro-CV={best[0]:.6f} params={best[1]}', flush=True)
    (RDIR / 'revised_validation_scores.csv').write_text(
        pd.DataFrame(records).to_csv(index=False))
    (RDIR / 'revised_selected_hyperparameters.json').write_text(json.dumps(choices, indent=2))
    print('SELECTION COMPLETE', flush=True)


def stage_evalA():
    d = load_primary()
    cal = d[d.partition == 'calibration']
    test = d[d.partition == 'test_reserved']
    choices = json.loads((RDIR / 'revised_selected_hyperparameters.json').read_text())
    rows = []
    for kind in GRIDS:
        model = joblib.load(MDIR / f'all_{kind}.joblib')
        p = clip(model.predict_proba(test)[:, 1])
        variants = {'raw': p}
        for method in (['logistic', 'beta', 'isotonic'] if kind == 'M3' else ['logistic']):
            cc = fit_cal(model.predict_proba(cal)[:, 1], cal.goal, method)
            variants['calibrated' if method == 'logistic' else 'calibrated_' + method] = apply_cal(cc, p)
        for v, pp in variants.items():
            r = dict(experiment='A', model=kind, variant=v, league_id='pooled', **metrics(test.goal, pp))
            rows.append(r)
            for lg, g in test.assign(p=pp).groupby('league_id'):
                rows.append(dict(experiment='A', model=kind, variant=v, league_id=lg,
                                 **metrics(g.goal, g.p)))
    # macro = mean of league means for log_loss/brier/auc
    scores = pd.DataFrame(rows)
    macro = scores[~scores.league_id.isin(['pooled'])].groupby(
        ['experiment', 'model', 'variant'])[['log_loss', 'brier', 'auc']].mean().reset_index()
    macro['league_id'] = 'macro'
    pd.concat([scores, macro], ignore_index=True).to_csv(RDIR / 'experiment_A_revised_metrics.csv', index=False)
    print('EVAL-A COMPLETE', flush=True)


def stage_evalB():
    d = load_primary()
    cal = d[d.partition == 'calibration']
    test = d[d.partition == 'test_reserved']
    choices = json.loads((RDIR / 'revised_selected_hyperparameters.json').read_text())
    rows = []
    preds = []
    calibrators = []
    for target in LEAGUES:
        t = test[test.league_id == target]
        c_src = cal[cal.league_id != target]
        c_dst = cal[cal.league_id == target]
        for kind in GRIDS:
            model = joblib.load(MDIR / f'{target}_{kind}.joblib')
            p = clip(model.predict_proba(t)[:, 1])
            variants = {'B0': p}
            for method in (['logistic', 'beta', 'isotonic'] if kind == 'M3' else ['logistic']):
                cc = fit_cal(model.predict_proba(c_src)[:, 1], c_src.goal, method)
                lab = 'B1' + ('' if method == 'logistic' else '_' + method)
                variants[lab] = apply_cal(cc, p)
                joblib.dump(cc, MDIR / f'{target}_{kind}_{lab}.joblib')
                if method != 'isotonic':
                    calibrators.append(dict(target=target, model=kind, variant=lab, **cc))
                cc2 = fit_cal(model.predict_proba(c_dst)[:, 1], c_dst.goal, method)
                lab2 = 'B2' + ('' if method == 'logistic' else '_' + method)
                variants[lab2] = apply_cal(cc2, p)
                joblib.dump(cc2, MDIR / f'{target}_{kind}_{lab2}.joblib')
                if method != 'isotonic':
                    calibrators.append(dict(target=target, model=kind, variant=lab2, **cc2))
            part = t[['shot_id', 'match_id', 'league_id', 'date', 'goal']].copy()
            part['experiment'] = 'B'
            part['model'] = kind
            for v, pp in variants.items():
                part[v] = pp
                g = t.assign(p=pp)
                rows.append(dict(experiment='B', model=kind, variant=v, league_id=target,
                                 **metrics(g.goal, g.p)))
            preds.append(part)
        print(f'EVALUATED B target={target}', flush=True)
    ptable = pd.concat(preds, ignore_index=True)
    ptable.to_csv(RDIR / 'revised_predictions_test.csv.gz', index=False)
    for kind in GRIDS:
        pp = ptable[ptable.model == kind]
        vlist = ['B0', 'B1', 'B2'] + (['B1_beta', 'B2_beta', 'B1_isotonic', 'B2_isotonic'] if kind == 'M3' else [])
        for v in vlist:
            rows.append(dict(experiment='B', model=kind, variant=v, league_id='pooled',
                             **metrics(pp.goal, pp[v])))
    (RDIR / 'revised_calibration_parameters.json').write_text(json.dumps(calibrators, indent=2))
    scores = pd.DataFrame(rows)
    macro = scores[scores.league_id != 'pooled'].groupby(
        ['experiment', 'model', 'variant'])[['log_loss', 'brier', 'auc']].mean().reset_index()
    macro['league_id'] = 'macro'
    pd.concat([scores, macro], ignore_index=True).to_csv(RDIR / 'revised_test_metrics.csv', index=False)
    print('EVAL-B COMPLETE', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['select', 'evalA', 'evalB'], required=True)
    args = ap.parse_args()
    MDIR.mkdir(parents=True, exist_ok=True)
    if args.stage == 'select':
        stage_select()
    elif args.stage == 'evalA':
        stage_evalA()
    else:
        stage_evalB()
