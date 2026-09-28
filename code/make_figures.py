"""Scientific figures from saved results. No model selection or fitting."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from run_experiments import bootstrap_difference, metrics, NAMES, LEAGUES, SEED

B=Path(__file__).resolve().parents[1]; R=B/'results'; F=B/'figures'; F.mkdir(exist_ok=True)
d=pd.read_csv(B/'data_derived/shots_primary_candidate.csv'); p=pd.read_csv(R/'predictions_test.csv.gz'); s=pd.read_csv(R/'test_metrics.csv')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','axes.labelsize':10,'savefig.dpi':350,'svg.fonttype':'none'})
COL=['#196B91','#B25420','#49843D','#88539B','#6B6963']
def save(fig,name):
    fig.savefig(F/(name+'.png'),bbox_inches='tight',facecolor='white'); fig.savefig(F/(name+'.svg'),bbox_inches='tight',facecolor='white'); plt.close(fig)

co=d.groupby(['league_id','season']).agg(n=('goal','size'),goals=('goal','sum'),matches=('match_id','nunique')).reset_index()
co['rate']=co.goals/co.n
fig,ax=plt.subplots(figsize=(8.1,3.5)); vals=co.pivot(index='league_id',columns='season',values='n').loc[LEAGUES]
im=ax.imshow(vals,cmap='Blues',vmin=5000,vmax=11500,aspect='auto')
ax.set_xticks(range(5),[x.replace('202','2') for x in vals.columns]); ax.set_yticks(range(5),[NAMES[x] for x in LEAGUES])
for i,lg in enumerate(LEAGUES):
    for j,se in enumerate(vals.columns):
        row=co[(co.league_id==lg)&(co.season==se)].iloc[0]
        ax.text(j,i,f'{row.n:,.0f}\n{100*row.rate:.1f}% goals',ha='center',va='center',fontsize=9,color='white' if row.n>9200 else '#192E40')
ax.axvline(2.5,color='white',lw=3);ax.axvline(3.5,color='white',lw=3)
ax.set_title('Audited non-penalty shots by league and season',pad=12)
ax.set_xlabel('2020/21-2022/23: development     2023/24: calibration     2024/25: evaluation',labelpad=12)
fig.tight_layout();save(fig,'Figure_1_cohort')

bs=pd.read_csv(R/'bootstrap_B2_vs_B0.csv'); fig,axes=plt.subplots(1,2,figsize=(8.1,3.7),sharey=True)
for ax,est,lo,hi,title in [(axes[0],'delta_log_loss','ll_low','ll_high','Log-loss difference'),(axes[1],'delta_brier','brier_low','brier_high','Brier-score difference')]:
    for i,row in bs.iterrows():
        ax.errorbar(row[est]*1000,i,xerr=[[1000*(row[est]-row[lo])],[1000*(row[hi]-row[est])]],fmt='D' if i==5 else 'o',color='#196B91' if i<5 else '#131313',capsize=3,ms=5)
    ax.axvline(0,color='#888888',ls='--',lw=1);ax.axhline(4.5,color='#dddddd',lw=1)
    ax.set_title(title);ax.set_xlabel('B2 minus B0 (x 1,000)');ax.grid(axis='x',alpha=.18)
axes[0].set_yticks(range(6),bs.league);axes[0].invert_yaxis();fig.tight_layout();save(fig,'Figure_2_adaptation_effect')

main=p[(p.experiment=='B')&(p.model=='M3')].copy()
fig,axes=plt.subplots(2,3,figsize=(8.2,5.8),sharex=True,sharey=True); binrows=[]
for ax,(lg,g) in zip(axes.flat,list(main.groupby('league_id'))+[('pooled',main)]):
    g=g.copy(); g['bin']=pd.qcut(g.B0,10,labels=False,duplicates='drop'); pts=[]
    rng=np.random.default_rng(SEED)
    for b,gg in g.groupby('bin'):
        # Pointwise intervals: cluster resample within the bin, preserving contributing matches.
        agg=gg.groupby('match_id').goal.agg(['size','sum']).to_numpy(); sims=[]
        for k in range(20):
            z=agg[rng.integers(0,len(agg),size=(100,len(agg)))].sum(axis=1); sims.extend(z[:,1]/z[:,0])
        pts.append({'league_id':lg,'bin':b,'n':len(gg),'matches':len(agg),'mean_raw':gg.B0.mean(),'mean_adapted':gg.B2.mean(),'observed':gg.goal.mean(),'low':np.quantile(sims,.025),'high':np.quantile(sims,.975)})
    q=pd.DataFrame(pts);binrows+=pts
    ax.plot([0,.45],[0,.45],ls='--',c='#999999',lw=1)
    ax.plot(q.mean_raw,q.observed,'o-',c=COL[0],label='B0: raw',ms=3,lw=1.2)
    ax.errorbar(q.mean_adapted,q.observed,yerr=[q.observed-q.low,q.high-q.observed],fmt='s-',c=COL[1],label='B2: target calibrated',ms=3,lw=1.2,capsize=2)
    ax.set_title(NAMES.get(lg,'Pooled'),fontsize=10);ax.set_xlim(0,.42);ax.set_ylim(0,.42);ax.grid(alpha=.15)
    ax.xaxis.set_major_formatter(PercentFormatter(1,0));ax.yaxis.set_major_formatter(PercentFormatter(1,0))
axes[0,0].legend(fontsize=7,loc='upper left');fig.supxlabel('Mean predicted goal probability');fig.supylabel('Observed goal proportion');fig.tight_layout();save(fig,'Figure_3_calibration')
pd.DataFrame(binrows).to_csv(R/'calibration_plot_data.csv',index=False)

fig,ax=plt.subplots(figsize=(8.0,3.5))
for c,lg in zip(COL,LEAGUES):
    q=co[co.league_id==lg];ax.plot(range(5),q.rate,'o-',label=NAMES[lg],c=c,lw=1.6,ms=5)
ax.axvspan(2.5,3.5,color='#F2F0EA',alpha=.6);ax.axvspan(3.5,4.2,color='#E7EFF2',alpha=.7)
ax.set_xticks(range(5),['2020/21','2021/22','2022/23','2023/24','2024/25']);ax.yaxis.set_major_formatter(PercentFormatter(1,1));ax.set_ylabel('Observed non-penalty goal proportion');ax.set_xlim(-.15,4.2);ax.grid(axis='y',alpha=.2);ax.legend(ncol=3,fontsize=8,loc='upper center',bbox_to_anchor=(.5,1.23));fig.tight_layout();save(fig,'Figure_4_seasonal_rates')

a=pd.read_csv(R/'temporal_ablations.csv');full=s[(s.experiment=='A')&(s.model=='M3')&(s.league_id=='pooled')&(s.variant.isin(['raw','calibrated']))].copy();full['analysis']='full';a=pd.concat([a,full],ignore_index=True)
order=['geometry','context','without_time_added','full','exclude_2020'];labels=['Distance + angle','+ body part + situation','+ minute + home','+ time_added + missingness','Full, excluding 2020/21']
fig,ax=plt.subplots(figsize=(8.0,3.6))
for shift,v,c,label in [(-.10,'raw',COL[0],'Raw'),(.10,'calibrated',COL[1],'Prior-season calibrated')]:
    z=a[a.variant==v].set_index('analysis').loc[order]; ax.scatter(z.log_loss,np.arange(5)+shift,c=c,label=label,s=35)
ax.set_yticks(range(5),labels);ax.invert_yaxis();ax.set_xlabel('Pooled temporal test log-loss (lower is better)');ax.legend(fontsize=8);ax.grid(axis='x',alpha=.2);fig.tight_layout();save(fig,'Figure_5_ablations')

# Additional descriptive contrasts, not new confirmatory tests or model selection.
pd.DataFrame(bootstrap_difference(main,'B2','B1')).to_csv(R/'bootstrap_B2_vs_B1_exploratory.csv',index=False)
at=p[(p.experiment=='A')&(p.model=='M3')][['shot_id','raw']].rename(columns={'raw':'temporal_raw'})
joint=main.merge(at,on='shot_id',validate='one_to_one')
pd.DataFrame(bootstrap_difference(joint,'B0','temporal_raw')).to_csv(R/'bootstrap_transfer_gap_exploratory.csv',index=False)
sub=[]
for fld in ['time_added_missing','body_part','situation']:
    for val,g in main.groupby(fld):
        for v in ['B0','B2']: sub.append(dict(field=fld,level=val,variant=v,**metrics(g.goal,g[v])))
pd.DataFrame(sub).to_csv(R/'subgroup_metrics_exploratory.csv',index=False)
(R/'figure_manifest.json').write_text(json.dumps({'figures':sorted(x.name for x in F.glob('*.png')),'source':'Persisted real test predictions and audited cohort','raster_dpi':350,'vector_format':'SVG','calibration_bins':'10 test-prediction quantile groups per league; no labels used for bin boundaries','calibration_intervals':'2000 within-bin match bootstrap replicates; pointwise, conditional on fixed bins and fitted models'},indent=2)+'\n')
print('Created five figures as PNG and SVG; data and exploratory contrasts saved.')
