# -*- coding: utf-8 -*-
"""Recompute paper reliability definitions and paired block-length checks."""
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from GEFcom2014.external_datasets import load_daily_bundle
from GEFcom2014.forecast_quality.compare_scenarios import load_scenarios
from GEFcom2014.forecast_quality.neso_wind_results import MODEL_KEYS, MAIN_MODELS, SEEDS, VARIANTS, _scenario_file, _pair_bootstrap
from GEFcom2014.forecast_quality.unified_postprocessing import HourlyAffineCalibrator

ROOT=Path(__file__).resolve().parents[2]
V2=ROOT/'export/fa_bm_vae_neso_wind_classical_v2'
OUT=V2/'paper_checks_v1'
MODELS=MAIN_MODELS+('J=0 / independent prior',)
LEVELS=np.arange(1,100)/100
MAQCE_LEVELS=np.linspace(.05,.95,19)
NOMINALS=(.5,.8,.9,.95)

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def curves(samples, y):
    # rows are forecast dates x 24 hours x reads; reliability uses strict <,
    # MAQCE uses <=, matching the existing implementations.
    flat=samples.reshape(-1,samples.shape[-1]); obs_y=y.reshape(-1)
    rel=np.mean(obs_y[:,None] < np.quantile(flat,LEVELS,axis=1).T,axis=0)
    pooled_rel=float(np.mean(np.abs(rel-LEVELS))*100)
    obs=np.mean(obs_y[:,None] <= np.quantile(flat,MAQCE_LEVELS,axis=1).T,axis=0)
    pooled_maqce=float(np.mean(np.abs(obs-MAQCE_LEVELS)))
    cov=[]; widths=[]
    for c in NOMINALS:
        a=(1-c)/2; lo,hi=np.quantile(flat,[a,1-a],axis=1)
        cov.append(float(np.mean((obs_y>=lo)&(obs_y<=hi)))); widths.append(float(np.mean(hi-lo)))
    return pooled_rel,pooled_maqce,cov,widths

def block_ci(diff, dates, length, reps, seed):
    dates=np.asarray(dates,dtype='datetime64[D]'); diff=np.asarray(diff,float)
    if len(diff)!=len(dates) or not np.isfinite(diff).all() or np.any(np.diff(dates).astype(int)<=0): raise ValueError('incomplete paired dates')
    cuts=np.r_[0,np.where(np.diff(dates).astype(int)!=1)[0]+1,len(dates)]
    segs=[np.arange(a,b) for a,b in zip(cuts[:-1],cuts[1:])]
    # Sample within each contiguous segment, preserving each segment's date count;
    # short segments use circular blocks confined to that segment.
    rng=np.random.default_rng(seed); draws=[]
    for _ in range(reps):
        got=[]
        for seg in segs:
            n=len(seg); b=min(length,n); idx=[]
            while len(idx)<n:
                start=int(rng.integers(n))
                idx.extend(seg[(start+np.arange(b))%n].tolist())
            got.extend(idx[:n])
        draws.append(float(diff[np.asarray(got)].mean()))
    return float(diff.mean()),np.quantile(draws,[.025,.975]).tolist()

def run_a():
    daily_rows=[]; pooled_inputs={}
    for wi in range(1,5):
        home=V2/'rolling'/f'quarter_{wi}'; bundle=home/'data/neso_wind_daily.npz'; arr,_,dates=load_daily_bundle(bundle)
        result=home/'formal/results'; daily=pd.read_csv(result/'metrics_daily.csv')
        yvs=np.asarray(arr['y_vs'],float).reshape(-1); ytest=np.asarray(arr['y_test'],float).reshape(-1)
        for model in MODELS:
          key=MODEL_KEYS[model]
          for seed in SEEDS:
            rawvs=load_scenarios(_scenario_file(home/'formal',key,seed,'VS',100),len(yvs))
            rawte=load_scenarios(_scenario_file(home/'formal',key,seed,'TEST',100),len(ytest))
            cal=HourlyAffineCalibrator(limits=(0.,float('inf'))).fit(rawvs,yvs)
            rawtest=rawte.reshape(-1,24,100); caltest=cal.transform(rawte).reshape(-1,24,100)
            for variant,samples in [('Raw',rawtest),('Cal',caltest),('Cal+ECC',caltest)]:
              part=daily[(daily.model==model)&(daily.seed==seed)&(daily.variant==variant)]
              if set(part.date.astype(str))!=set(dates['test'].astype(str)): raise ValueError('daily grid mismatch')
              rel,maq,cov,width=curves(samples,ytest.reshape(-1,24))
              check=part.sort_values('date')
              # Existing daily rows retain the old date-wise definitions.
              daily_rel=float(check.Reliability_MAE_pp.mean()); daily_maq=float(check.MAQCE.mean())
              if variant=='Cal+ECC' and (not np.allclose(samples,caltest,atol=0,rtol=0)): raise ValueError('ECC marginal reuse invalid')
              row={'window':f'quarter_{wi}','model':model,'seed':seed,'variant':variant,
                   'mean_daily_reliability_mae_pp':daily_rel,'pooled_reliability_mae_pp':rel,
                   'mean_daily_maqce':daily_maq,'pooled_maqce':maq,'n_dates':len(ytest)//24}
              for j,n in enumerate(NOMINALS): row[f'coverage_{int(n*100)}']=cov[j]; row[f'width_{int(n*100)}_mw']=width[j]
              daily_rows.append(row); pooled_inputs[(wi,model,seed,variant)]=(samples,ytest.reshape(-1,24))
    frame=pd.DataFrame(daily_rows); frame.to_csv(OUT/'calibration_metrics_per_seed.csv',index=False)
    summary=frame.groupby(['window','model','variant']).agg({c:['mean','std'] for c in frame.columns if c.endswith('_pp') or c.endswith('_maqce') or c.startswith('coverage_') or c.startswith('width_')}).reset_index()
    summary.to_csv(OUT/'calibration_metrics_summary.csv',index=False)
    # True annual pooled metrics are recomputed over all four windows per seed.
    annual=[]
    for model in MODELS:
      for seed in SEEDS:
       for variant in VARIANTS:
        ss=np.concatenate([pooled_inputs[(w,model,seed,variant)][0] for w in range(1,5)])
        yy=np.concatenate([pooled_inputs[(w,model,seed,variant)][1] for w in range(1,5)])
        rel,maq,cov,width=curves(ss,yy)
        r=frame[(frame.model==model)&(frame.seed==seed)&(frame.variant==variant)]
        row={'window':'FULL_YEAR','model':model,'seed':seed,'variant':variant,
             'mean_daily_reliability_mae_pp':float(np.average(r.mean_daily_reliability_mae_pp,weights=r.n_dates)),
             'pooled_reliability_mae_pp':rel,'mean_daily_maqce':float(np.average(r.mean_daily_maqce,weights=r.n_dates)),
             'pooled_maqce':maq,'n_dates':int(r.n_dates.sum())}
        for j,n in enumerate(NOMINALS): row[f'coverage_{int(n*100)}']=cov[j]; row[f'width_{int(n*100)}_mw']=width[j]
        annual.append(row)
    pd.DataFrame(annual).to_csv(OUT/'calibration_metrics_per_seed.csv',mode='a',header=False,index=False)
    return frame,pd.DataFrame(annual)

def seed_direction_tables(byseed, window_counts):
    """Keep seasonal changes separate from seed variation; weight by actual dates."""
    keys = ['model', 'comparator', 'variant', 'metric']
    frame = byseed[byseed.window != 'FULL_YEAR'].copy()
    if frame.duplicated(keys + ['window', 'seed']).any():
        raise ValueError('Duplicate window/seed difference')
    expected = {(w, s) for w in window_counts for s in SEEDS}
    annual = []
    for key, part in frame.groupby(keys, sort=False):
        if set(zip(part.window, part.seed)) != expected:
            raise ValueError('Incomplete window/seed difference grid')
        if not np.isfinite(part.mean_difference).all():
            raise ValueError('Non-finite seed difference')
        for seed, sample in part.groupby('seed', sort=True):
            weights = sample.window.map(window_counts).to_numpy()
            annual.append(dict(zip(keys, key), window='FULL_YEAR', seed=int(seed),
                               mean_difference=float(np.average(sample.mean_difference, weights=weights)),
                               n_dates=int(weights.sum())))
    frame['n_dates'] = frame.window.map(window_counts)
    combined = pd.concat([frame, pd.DataFrame(annual)], ignore_index=True)
    summary = []
    for key, part in combined.groupby(['window'] + keys, sort=False):
        values = part.mean_difference.to_numpy()
        negative = int(np.count_nonzero(values < -1e-12))
        positive = int(np.count_nonzero(values > 1e-12))
        summary.append(dict(zip(['window'] + keys, key), n_seeds=len(values),
                            n_negative=negative, n_positive=positive,
                            n_zero=len(values)-negative-positive,
                            mixed_seed_signs=bool(negative and positive)))
    return combined, pd.DataFrame(summary)


def run_c():
    rows=[]; byseed=[]; comparisons=[(MAIN_MODELS[0],m,'Cal',['CRPS_MW']) for m in MAIN_MODELS[1:]]
    comparisons += [(MAIN_MODELS[0],'J=0 / independent prior',v,['CRPS_MW','Energy_Score_MW','Variogram_Score','Ramp_CRPS_MW']) for v in ('Raw','Cal')]
    for model,other,variant,metrics in comparisons:
      for metric in metrics:
       win_diffs=[]; win_dates=[]
       for wi in range(1,5):
        d=pd.read_csv(V2/'rolling'/f'quarter_{wi}'/'formal/results/metrics_daily.csv')
        a=d[(d.model==model)&(d.variant==variant)].groupby('date')[metric].mean()
        b=d[(d.model==other)&(d.variant==variant)].groupby('date')[metric].mean()
        if not a.index.equals(b.index): raise ValueError('date pairing mismatch')
        dates=a.index.to_numpy(); diff=a.to_numpy()-b.to_numpy(); win_diffs.append(diff); win_dates.append(dates)
        for seed in SEEDS:
          x=d[(d.model==model)&(d.seed==seed)&(d.variant==variant)].set_index('date')[metric]
          z=d[(d.model==other)&(d.seed==seed)&(d.variant==variant)].set_index('date')[metric]
          byseed.append({'window':f'quarter_{wi}','seed':seed,'model':model,'comparator':other,'variant':variant,'metric':metric,'mean_difference':float((x-z).mean())})
       for length in (3,7,14):
        reps=[]; weights=[]
        for i,(diff,dates) in enumerate(zip(win_diffs,win_dates)):
          weights.append(len(diff))
        # Re-run a stratified bootstrap directly so annual CI is over full-year mean with fixed window weights.
        rng=np.random.default_rng(20261023+len(rows)); draws=[]
        for _ in range(5000):
          estimates=[]
          for i,(diff,dates) in enumerate(zip(win_diffs,win_dates)):
            estimates.append(block_ci_once(diff,dates,length,rng))
          draws.append(float(np.average(estimates,weights=weights)))
        point=float(np.average([x.mean() for x in win_diffs],weights=weights)); comparator_means=[]
        for wi in range(1,5):
          d=pd.read_csv(V2/'rolling'/f'quarter_{wi}'/'formal/results/metrics_daily.csv')
          comparator_means.append(float(d[(d.model==other)&(d.variant==variant)][metric].mean()))
        ref=float(np.average(comparator_means,weights=weights)); ci=np.quantile(draws,[.025,.975])
        rows.append({'model':model,'comparator':other,'variant':variant,'metric':metric,'block_days':length,'estimate_main_minus_comparator':point,'relative_difference_percent':point/max(abs(ref),1e-12)*100,'ci95_low':float(ci[0]),'ci95_high':float(ci[1]),'repetitions':5000,'seed':20261023+len(rows),'statistical_unit':'date; seed means before bootstrap'})
    pd.DataFrame(rows).to_csv(OUT/'paired_block_sensitivity.csv',index=False)
    counts = {f'quarter_{i}': pd.read_csv(V2/'rolling'/f'quarter_{i}'/'formal/results/metrics_daily.csv').date.nunique()
              for i in range(1, 5)}
    differences, directions = seed_direction_tables(pd.DataFrame(byseed), counts)
    differences.to_csv(OUT/'paired_differences_by_seed.csv', index=False)
    directions.to_csv(OUT/'seed_direction_summary.csv', index=False)

def block_ci_once(diff,dates,length,rng):
    dates=np.asarray(dates,dtype='datetime64[D]'); diff=np.asarray(diff,float); cuts=np.r_[0,np.where(np.diff(dates).astype(int)!=1)[0]+1,len(dates)]; result=[]
    for a,b in zip(cuts[:-1],cuts[1:]):
      seg=np.arange(a,b); n=len(seg); block=min(length,n); idx=[]
      while len(idx)<n:
        st=int(rng.integers(n)); idx.extend(seg[(st+np.arange(block))%n].tolist())
      result.extend(idx[:n])
    return float(diff[np.asarray(result)].mean())

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    run_a(); run_c()
