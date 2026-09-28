"""Build all reviewer-response summary tables (CSV + MD) from computed outputs."""
import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from rr_common import RDIR, BASE, LEAGUES, NAMES

R = RDIR
OLD = BASE / 'results'


def get(df, exp, mo, va, lg, col):
    return df[(df.experiment == exp) & (df.model == mo) & (df.variant == va) &
              (df.league_id == str(lg))].iloc[0][col]


# ---- 1. Experiment A summary ----
a = pd.read_csv(R / 'experiment_A_revised_metrics.csv')
ao = pd.read_csv(OLD / 'test_metrics.csv')
rows = []
for v in ['raw', 'calibrated']:
    r = a[(a.model == 'M3') & (a.variant == v) & (a.league_id == 'macro')].iloc[0]
    ro = ao[(ao.experiment == 'A') & (ao.model == 'M3') & (ao.variant == v) & (ao.league_id == 'macro')].iloc[0]
    rows.append(dict(variant=v, ll_new=r.log_loss, ll_old=ro.log_loss,
                     brier_new=r.brier, brier_old=ro.brier, auc_new=r.auc, auc_old=ro.auc,
                     cil_new=r.calibration_in_large, cil_old=ro.calibration_in_large,
                     slope_new=r.calibration_slope, slope_old=ro.calibration_slope))
pd.DataFrame(rows).to_csv(R / 'experiment_A_revised_features.csv', index=False)
with open(R / 'experiment_A_revised_summary.md', 'w') as f:
    r = {x['variant']: x for x in rows}
    f.write('# Experiment A revised (no-time-added budget)\n\n')
    f.write(f"M3 raw: log-loss {r['raw']['ll_new']:.6f} (old {r['raw']['ll_old']:.6f}), "
            f"Brier {r['raw']['brier_new']:.6f}, AUC {r['raw']['auc_new']:.6f}.\n\n")
    f.write(f"M3 calibrated: log-loss {r['calibrated']['ll_new']:.6f} "
            f"(old {r['calibrated']['ll_old']:.6f}).\n")
open(R / 'experiment_A_revised_summary.md', 'a').write(
    '\nClassification: revised-primary candidate (Experiment A temporal benchmark).\n')

# ---- 2. Experiment B summary ----
m = pd.read_csv(R / 'revised_test_metrics.csv')
bm = m[(m.experiment == 'B') & (m.model == 'M3') & (m.league_id == 'macro')]
b0, b1, b2 = (bm[bm.variant == v].iloc[0] for v in ['B0', 'B1', 'B2'])
bs = pd.read_csv(R / 'revised_bootstrap_M3_B2_vs_B0.csv')
bsm = bs[bs.league_id == 'macro'].iloc[0]
b21 = pd.read_csv(R / 'revised_bootstrap_M3_B2_vs_B1.csv')
b21m = b21[b21.league_id == 'macro'].iloc[0]
with open(R / 'revised_experiment_B_summary.md', 'w') as f:
    f.write('# Experiment B revised (no-time-added budget, M3)\n\n')
    f.write(f'B0 log-loss {b0.log_loss:.6f}, B1 {b1.log_loss:.6f}, B2 {b2.log_loss:.6f}.\n\n')
    f.write(f"Primary B2-B0 = {bsm.delta_log_loss:.6f} "
            f"[{bsm.ll_low:.6f}, {bsm.ll_high:.6f}]; "
            f'Brier diff {bsm.delta_brier:.7f}.\n\n')
    f.write(f"Secondary B2-B1 = {b21m.delta_log_loss:.6f} [{b21m.ll_low:.6f}, {b21m.ll_high:.6f}].\n\n")
    f.write('Per-league B2-B0:\n')
    for _, r in bs[bs.league_id != 'macro'].iterrows():
        f.write(f"- {r.league}: {r.delta_log_loss:.6f} [{r.ll_low:.6f}, {r.ll_high:.6f}]\n")
    f.write('\nClassification: revised-primary candidate.\n')

# ---- 3. Old vs revised ----
mo = pd.read_csv(OLD / 'test_metrics.csv')
def gm(df, v, c):
    return df[(df.experiment == 'B') & (df.model == 'M3') & (df.variant == v) &
              (df.league_id == 'macro')].iloc[0][c]
rows = []
for metric, col in [('B0 log-loss', 'log_loss'), ('Brier', 'brier'), ('AUC', 'auc')]:
    pass
comp = [
    ('B0 log-loss', gm(mo, 'B0', 'log_loss'), get(m, 'B', 'M3', 'B0', 'macro', 'log_loss')),
    ('B2 log-loss', gm(mo, 'B2', 'log_loss'), get(m, 'B', 'M3', 'B2', 'macro', 'log_loss')),
]
oro = pd.read_csv(OLD / 'bootstrap_B2_vs_B0.csv').query("league_id=='macro'").iloc[0]
nro = bsm
out = pd.DataFrame([
    dict(metric='B0 log-loss', old=comp[0][1], new=comp[0][2], change=comp[0][2] - comp[0][1]),
    dict(metric='B2 log-loss', old=comp[1][1], new=comp[1][2], change=comp[1][2] - comp[1][1]),
    dict(metric='B2-B0', old=oro.delta_log_loss, new=nro.delta_log_loss,
         change=nro.delta_log_loss - oro.delta_log_loss),
    dict(metric='B2-B1', old=pd.read_csv(OLD / 'bootstrap_B2_vs_B1_exploratory.csv').query("league_id=='macro'").iloc[0].delta_log_loss,
         new=b21m.delta_log_loss, change=b21m.delta_log_loss - pd.read_csv(OLD / 'bootstrap_B2_vs_B1_exploratory.csv').query("league_id=='macro'").iloc[0].delta_log_loss),
    dict(metric='Brier B2-B0', old=oro.delta_brier, new=nro.delta_brier, change=nro.delta_brier - oro.delta_brier),
    dict(metric='AUC B0', old=gm(mo, 'B0', 'auc'), new=get(m, 'B', 'M3', 'B0', 'macro', 'auc'),
         change=get(m, 'B', 'M3', 'B0', 'macro', 'auc') - gm(mo, 'B0', 'auc')),
])
out.to_csv(R / 'old_vs_revised_feature_budget.csv', index=False)
with open(R / 'old_vs_revised_feature_budget.md', 'w') as f:
    f.write('# Old vs revised feature budget (M3, macro)\n\n' + out.to_string(index=False) +
            '\n\nNo model was chosen for a favorable test result; both budgets use the locked '
            'selection rule. Classification: sensitivity analysis.\n')

# ---- 4. time_added incremental (T1 revised vs T2 original, Experiment A M3) ----
t1 = a[(a.model == 'M3') & (a.variant == 'raw') & (a.league_id == 'macro')].iloc[0]
t2 = ao[(ao.experiment == 'A') & (ao.model == 'M3') & (ao.variant == 'raw') & (ao.league_id == 'macro')].iloc[0]
abl = pd.read_csv(OLD / 'temporal_ablations.csv')
ab = abl[abl.analysis == 'without_time_added']
ti = pd.DataFrame([
    dict(model='T1 revised (no time_added)', log_loss=t1.log_loss, brier=t1.brier, auc=t1.auc),
    dict(model='T2 original (with time_added)', log_loss=t2.log_loss, brier=t2.brier, auc=t2.auc),
    dict(model='difference T2-T1', log_loss=t2.log_loss - t1.log_loss,
         brier=t2.brier - t1.brier, auc=t2.auc - t1.auc),
])
ti.to_csv(R / 'time_added_incremental_effect.csv', index=False)
with open(R / 'time_added_incremental_effect.md', 'w') as f:
    f.write('# Incremental effect of time_added beyond minute (Experiment A M3, pooled test)\n\n')
    f.write(ti.to_string(index=False))
    f.write('\n\nFor reference, the frozen original ablation with identical hyperparameters '
            f"(geometry+context+minute+home vs full): without_time_added raw pooled log-loss "
            f"{ab[ab.variant == 'raw'].iloc[0]['log_loss']:.6f} vs full-model raw "
            f"{t2.log_loss:.6f}.\n\nClassification: sensitivity analysis; does not redefine any endpoint.\n")

# ---- 5. time_added coverage audit ----
d = pd.read_csv(BASE / 'data_derived' / 'shots_primary_candidate.csv',
                usecols=['league_id', 'season', 'minute', 'time_added', 'goal'])
rows = []
for (lg, se), g in d.groupby(['league_id', 'season']):
    v = g.time_added.dropna()
    rows.append(dict(league_id=lg, league=NAMES[lg], season=se, n=len(g),
                     recorded=int(len(v)), coverage=round(float(len(v) / len(g) * 100), 2),
                     nunique=int(v.nunique()), min=float(v.min()) if len(v) else None,
                     max=float(v.max()) if len(v) else None, median=float(v.median()) if len(v) else None,
                     q25=float(v.quantile(.25)) if len(v) else None, q75=float(v.quantile(.75)) if len(v) else None,
                     minute_not_45_90=int(((g.minute != 45) & (g.minute != 90) & g.time_added.notna()).sum())))
pd.DataFrame(rows).to_csv(R / 'time_added_coverage.csv', index=False)
cov = pd.DataFrame(rows)
with open(R / 'time_added_coverage.md', 'w') as f:
    f.write('# time_added coverage audit (primary cohort, 25 league-season cells)\n\n')
    f.write(f"Overall recorded: {int(d.time_added.notna().sum())}/{len(d)} "
            f"({100 * d.time_added.notna().mean():.2f}%).\n")
    f.write(f"Recorded values with minute not in (45, 90): "
            f"{int(((d.minute != 45) & (d.minute != 90) & d.time_added.notna()).sum())}.\n")
    f.write(f"Recorded value range: {d.time_added.min():.0f}-{d.time_added.max():.0f}.\n\n")
    f.write('Classification: descriptive audit (no causal or provider-bias inference).\n')

# ---- 6. rolling summary ----
ro = pd.read_csv(R / 'rolling_origin.csv')
with open(R / 'rolling_origin.md', 'w') as f:
    f.write('# Rolling-origin temporal robustness (revised budget, M3, frozen settings)\n\n')
    f.write(ro.to_string(index=False))
    f.write('\n\nClassification: secondary analysis.\n')

# ---- 7. full-pipeline summary ----
fp = pd.read_csv(R / 'full_pipeline_bootstrap.csv')
dd = fp.B2B0.dropna()
with open(R / 'full_pipeline_bootstrap.md', 'w') as f:
    f.write('# Full-pipeline bootstrap sensitivity (revised M3 B2-B0, frozen selection)\n\n')
    f.write(f"Completed replicates: {int(dd.notna().sum())}/1000 "
            f"({int(fp.B2B0.isna().sum())} degenerate calibration failure, documented).\n")
    f.write(f"Mean {dd.mean():.6f}; 2.5% {dd.quantile(.025):.6f}; 97.5% {dd.quantile(.975):.6f}.\n")
    f.write(f"Conditional primary interval: [{bsm.ll_low:.6f}, {bsm.ll_high:.6f}].\n\n")
    f.write('Deviation (loud): hyperparameter selection frozen to revised-primary locked '
            'choices; selection uncertainty not captured. Logistic B2 calibration only.\n\n')
    f.write('Classification: sensitivity analysis.\n')

# ---- 8. model family ----
fam = []
for kind in ['M0', 'M1', 'M2', 'M3']:
    bb = pd.read_csv(R / f'revised_bootstrap_{kind}_B2_vs_B0.csv')
    r = bb[bb.league_id == 'macro'].iloc[0]
    mm = m[(m.experiment == 'B') & (m.model == kind)]
    fam.append(dict(model=kind,
                    B0=get(m, 'B', kind, 'B0', 'macro', 'log_loss'),
                    B2=get(m, 'B', kind, 'B2', 'macro', 'log_loss'),
                    B2B0=r.delta_log_loss, ll_low=r.ll_low, ll_high=r.ll_high,
                    brier_diff=r.delta_brier,
                    auc=get(m, 'B', kind, 'B0', 'macro', 'auc')))
pd.DataFrame(fam).to_csv(R / 'model_family_B2B0.csv', index=False)
with open(R / 'model_family_B2B0.md', 'w') as f:
    f.write('# Model-family B2-B0 (revised budget, Experiment B macro)\n\n')
    f.write(pd.DataFrame(fam).to_string(index=False))
    f.write('\n\nGoal: test whether the transfer/recalibration pattern is M3-specific, not model selection.\n\n'
            'Classification: sensitivity analysis.\n')

# ---- 9. club summary ----
cc = pd.read_csv(R / 'club_cluster_macro.csv').iloc[0]
with open(R / 'club_cluster_macro.md', 'w') as f:
    f.write('# Club-cluster macro sensitivity (revised M3 B2-B0)\n\n')
    f.write(f"Macro point {cc.macro_point:.6f} [{cc.macro_lo:.6f}, {cc.macro_hi:.6f}] "
            f"vs match-cluster [{bsm.ll_low:.6f}, {bsm.ll_high:.6f}].\n\n"
            'Classification: sensitivity analysis.\n')

# ---- 10. S8 calibration table ----
rates = pd.read_csv(R / 'revised_rates_ci.csv')
s8 = []
for lg_int in LEAGUES:
    lg = str(lg_int)
    for v in ['B0', 'B1', 'B2']:
        mm = m[(m.experiment == 'B') & (m.model == 'M3') & (m.variant == v) & (m.league_id == lg)].iloc[0]
        rr = rates[(rates.league_id == lg_int) & (rates.regime == v)].iloc[0]
        s8.append(dict(league=NAMES[int(lg)], regime=v, observed=mm.prevalence, mean_pred=mm.mean_xg,
                       obs_lo=rr.obs_lo, obs_hi=rr.obs_hi, mp_lo=rr.mp_lo, mp_hi=rr.mp_hi,
                       CIL=mm.calibration_in_large, intercept=mm.joint_intercept, slope=mm.calibration_slope))
pd.DataFrame(s8).to_csv(R / 'B0_B1_B2_calibration.csv', index=False)
print('TABLES COMPLETE')
