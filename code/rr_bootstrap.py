"""Revised bootstraps, S8 calibration table, S9 reliability bins, club-cluster macro.

All resampling is paired match-cluster (or team-cluster), seed 20260910.
"""
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from rr_common import (RDIR, SEED, LEAGUES, NAMES, loss, clip, bootstrap_difference, load_primary)


def main(which):
    pred = pd.read_csv(RDIR / 'revised_predictions_test.csv.gz')
    if which == 'contrasts':
        for kind in ['M0', 'M1', 'M2', 'M3']:
            pp = pred[(pred.experiment == 'B') & (pred.model == kind)]
            pd.DataFrame(bootstrap_difference(pp, 'B2')).to_csv(
                RDIR / f'revised_bootstrap_{kind}_B2_vs_B0.csv', index=False)
            print(f'BOOT {kind} B2vB0', flush=True)
        main_m3 = pred[(pred.experiment == 'B') & (pred.model == 'M3')]
        pd.DataFrame(bootstrap_difference(main_m3, 'B2', reference='B1')).to_csv(
            RDIR / 'revised_bootstrap_M3_B2_vs_B1.csv', index=False)
        pd.DataFrame(bootstrap_difference(main_m3, 'B2_beta')).to_csv(
            RDIR / 'revised_bootstrap_M3_B2beta_vs_B0.csv', index=False)
        pd.DataFrame(bootstrap_difference(main_m3, 'B2_isotonic')).to_csv(
            RDIR / 'revised_bootstrap_M3_B2isotonic_vs_B0.csv', index=False)
        pd.DataFrame(bootstrap_difference(main_m3, cluster='week')).to_csv(
            RDIR / 'revised_bootstrap_M3_week.csv', index=False)
        print('BOOT M3 extras', flush=True)
    elif which == 'rates':
        # CIs for observed goal rate and mean predicted probability per league/regime.
        rng = np.random.default_rng(SEED)
        reps = 2000
        rows = []
        pm3 = pred[(pred.experiment == 'B') & (pred.model == 'M3')]
        for lg in LEAGUES:
            g = pm3[pm3.league_id == lg]
            units = g['match_id'].unique()
            idx_all = [np.where(g['match_id'].to_numpy() == u)[0] for u in units]
            y = g['goal'].to_numpy()
            boot_obs, boot_m = {v: [] for v in ['B0', 'B1', 'B2']}, {v: [] for v in ['B0', 'B1', 'B2']}
            for r in range(reps):
                samp = rng.integers(0, len(units), len(units))
                ix = np.concatenate([idx_all[k] for k in samp])
                boot_obs['B0'].append(y[ix].mean())
                for v in ['B0', 'B1', 'B2']:
                    boot_m[v].append(g[v].to_numpy()[ix].mean())
            for v in ['B0', 'B1', 'B2']:
                rows.append(dict(league_id=lg, league=NAMES[lg], regime=v,
                                 observed=y.mean(),
                                 obs_lo=np.quantile(boot_obs['B0'], .025),
                                 obs_hi=np.quantile(boot_obs['B0'], .975),
                                 mean_pred=g[v].mean(),
                                 mp_lo=np.quantile(boot_m[v], .025), mp_hi=np.quantile(boot_m[v], .975)))
        pd.DataFrame(rows).to_csv(RDIR / 'revised_rates_ci.csv', index=False)
        print('RATES COMPLETE', flush=True)
    elif which == 'bins':
        # S9: deciles of B0 (raw) per league; stats per regime; CI for observed via within-bin match resampling.
        rng = np.random.default_rng(SEED)
        reps = 2000
        pm3 = pred[(pred.experiment == 'B') & (pred.model == 'M3')]
        out = []
        for lg in LEAGUES:
            g = pm3[pm3.league_id == lg].copy()
            g['bin'] = pd.qcut(g['B0'], 10, labels=False, duplicates='drop')
            for b in sorted(g['bin'].unique()):
                gb = g[g['bin'] == b]
                units = gb['match_id'].unique()
                idx_all = [np.where(gb['match_id'].to_numpy() == u)[0] for u in units]
                y = gb['goal'].to_numpy()
                boots = np.empty(reps)
                for s in range(0, reps, 200):
                    inds = rng.integers(0, len(units), size=(min(200, reps - s), len(units)))
                    for k, draw in enumerate(inds):
                        ix = np.concatenate([idx_all[j] for j in draw])
                        boots[s + k] = y[ix].mean()
                for v in ['B0', 'B1', 'B2']:
                    out.append(dict(league_id=lg, league=NAMES[lg], regime=v, bin=int(b),
                                    n=len(gb), goals=int(y.sum()), mean_pred=float(gb[v].mean()),
                                    observed=float(y.mean()),
                                    ci_lower=float(np.quantile(boots, .025)),
                                    ci_upper=float(np.quantile(boots, .975))))
        pd.DataFrame(out).to_csv(RDIR / 'reliability_bins.csv', index=False)
        print('BINS COMPLETE', flush=True)
    elif which == 'club':
        shots = load_primary()[['shot_id', 'shooting_team_id']]
        pm3 = pred[(pred.experiment == 'B') & (pred.model == 'M3')].merge(shots, on='shot_id', how='left')
        assert pm3['shooting_team_id'].notna().all()
        rng = np.random.default_rng(SEED)
        reps = 2000
        macro_draws = []
        league_draws = {lg: [] for lg in LEAGUES}
        for lg in LEAGUES:
            g = pm3[pm3.league_id == lg].copy()
            g['dl'] = loss(g.goal, g.B2) - loss(g.goal, g.B0)
            teams = g['shooting_team_id'].unique()
            idx_all = [np.where(g['shooting_team_id'].to_numpy() == t)[0] for t in teams]
            dl = g['dl'].to_numpy()
            boot = np.empty(reps)
            for s in range(0, reps, 100):
                inds = rng.integers(0, len(teams), size=(min(100, reps - s), len(teams)))
                for k, draw in enumerate(inds):
                    ix = np.concatenate([idx_all[j] for j in draw])
                    boot[s + k] = dl[ix].mean()
            league_draws[lg] = boot
        macro = np.mean([league_draws[lg] for lg in LEAGUES], axis=0)
        rep = pd.DataFrame({'replicate': np.arange(reps), 'macro_B2B0': macro,
                            **{f'league_{lg}': league_draws[lg] for lg in LEAGUES}})
        rep.to_csv(RDIR / 'club_cluster_macro_replicates.csv', index=False)
        point = float(pm3.assign(dl=loss(pm3.goal, pm3.B2) - loss(pm3.goal, pm3.B0)
                                   ).groupby('league_id').dl.mean().mean())
        pd.DataFrame([dict(macro_point=point,
                           macro_lo=float(np.quantile(macro, .025)),
                           macro_hi=float(np.quantile(macro, .975)))]
                     ).to_csv(RDIR / 'club_cluster_macro.csv', index=False)
        print('CLUB COMPLETE', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--which', choices=['contrasts', 'rates', 'bins', 'club'], required=True)
    args = ap.parse_args()
    main(args.which)
