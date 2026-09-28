"""End-to-end (full-pipeline) bootstrap sensitivity, revised budget, M3 B2-B0.

Per replicate: resample development matches (within league) -> refit 5
leave-one-league-out M3 models (FROZEN revised-primary hyperparameters) ->
fit B2 logistic recalibration on resampled destination-calibration matches ->
resample test matches -> paired B0/B2 scoring -> macro B2-B0.

DOCUMENTED DEVIATION: chronological hyperparameter selection is NOT rerun
inside replicates (40 HGB selection fits per replicate x 1000 = infeasible);
selection is frozen to the revised-primary locked choices. Selection
uncertainty is therefore not captured; refit + calibration + evaluation
uncertainty is. Calibration uses logistic only (B2 needed for the contrast).

Usage: rr_fullpipe.py --start 0 --end 20   (appends; header written once)
"""
import sys
import time
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).parent))
from rr_common import (RDIR, SEED, LEAGUES, make_model_rev, fit_cal, apply_cal,
                       clip, loss, load_primary)


def one_replicate(d, dev_idx, cal_idx, test_idx, choices):
    dev = d.iloc[dev_idx]
    cal = d.iloc[cal_idx]
    tst = d.iloc[test_idx]
    diffs = []
    b0s = []
    b2s = []
    for target in LEAGUES:
        tr = dev[dev.league_id != target]
        tt = tst[tst.league_id == target]
        cc = cal[cal.league_id == target]
        m = make_model_rev('M3', choices[str(target)]['M3']).fit(tr, tr.goal)
        p0 = clip(m.predict_proba(tt)[:, 1])
        q2 = apply_cal(fit_cal(m.predict_proba(cc)[:, 1], cc.goal), p0)
        y = tt.goal.to_numpy()
        l0 = float(loss(y, p0).mean())
        l2 = float(loss(y, q2).mean())
        b0s.append(l0)
        b2s.append(l2)
        diffs.append(l2 - l0)
    return float(np.mean(b0s)), float(np.mean(b2s)), float(np.mean(diffs)), diffs


def main(start, end):
    d = load_primary().reset_index(drop=True)
    dev = d[d.partition == 'development']
    cal = d[d.partition == 'calibration']
    tst = d[d.partition == 'test_reserved']
    choices = json.loads((RDIR / 'revised_selected_hyperparameters.json').read_text())
    out = RDIR / 'full_pipeline_bootstrap.csv'
    # per-league match index pools (positions into d)
    dev_pool = {lg: np.where((d.partition == 'development').to_numpy() &
                             (d.league_id == lg).to_numpy())[0] for lg in LEAGUES}
    # map match_id -> positions for resampling by match
    dev_match = {lg: dev[dev.league_id == lg].match_id.unique() for lg in LEAGUES}
    cal_match = {lg: cal[cal.league_id == lg].match_id.unique() for lg in LEAGUES}
    tst_match = {lg: tst[tst.league_id == lg].match_id.unique() for lg in LEAGUES}
    pos_of_match = {}
    for mm in pd.concat([dev.match_id, cal.match_id, tst.match_id]).unique():
        pos_of_match[mm] = np.where(d.match_id.to_numpy() == mm)[0]

    def draw(pool, rng):
        ix = []
        for lg in LEAGUES:
            ms = pool[lg]
            samp = rng.choice(ms, size=len(ms), replace=True)
            for m in samp:
                ix.extend(pos_of_match[m].tolist())
        return np.array(ix, dtype=int)

    header = ['replicate', 'B0', 'B2', 'B2B0'] + [f'd_{lg}' for lg in LEAGUES]
    write_header = not out.exists()
    # Per-replicate streams seeded by replicate index: chunks are
    # order-independent and the full 0..N run is canonical and deterministic.
    t0 = time.time()
    with open(out, 'a') as f:
        if write_header:
            f.write(','.join(header) + '\n')
        failed = 0
        for r in range(start, end):
            rr = np.random.default_rng(SEED + 999983 + r)  # per-replicate stream: chunk-order independent
            try:
                b0, b2, dd, diffs = one_replicate(
                    d, draw(dev_match, rr), draw(cal_match, rr), draw(tst_match, rr), choices)
                f.write(','.join([str(r), f'{b0:.6f}', f'{b2:.6f}', f'{dd:.6f}'] +
                                 [f'{x:.6f}' for x in diffs]) + '\n')
                msg = f'{dd:.6f}'
            except Exception as e:  # degenerate resample: calibration optimizer failure
                f.write(','.join([str(r)] + [''] * (3 + len(LEAGUES))) + '\n')
                failed += 1
                msg = f'FAILED ({type(e).__name__})'
            f.flush()
            print(f'FULLPIPE rep={r} B2B0={msg} elapsed={time.time()-t0:.0f}s', flush=True)
        if failed:
            print(f'FULLPIPE CHUNK {start}-{end}: {failed} failed replicates written as empty rows', flush=True)
    print(f'FULLPIPE CHUNK {start}-{end} done in {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', type=int, required=True)
    ap.add_argument('--end', type=int, required=True)
    a = ap.parse_args()
    main(a.start, a.end)
