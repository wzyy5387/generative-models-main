"""Compare the frozen original NESO results with five separately trained baselines."""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from GEFcom2014.external_datasets import load_daily_bundle
from GEFcom2014.models.run_neso_extra_baselines import ROOT, V2, OUT, verify, sha, write_json, validate_task
from GEFcom2014.models.neso_extra_baselines import LABELS
from GEFcom2014.forecast_quality.neso_wind_results import (
    MAIN_MODELS, DAILY_METRICS, _metrics_for, _scenario_file, _write_table, _table_rows,
)
from GEFcom2014.forecast_quality.neso_paper_checks import curves
from GEFcom2014.forecast_quality.unified_postprocessing import HourlyAffineCalibrator
from GEFcom2014.forecast_quality.qbm_temporal_ecc import ensemble_copula_coupling

RESULTS = OUT/'results'
VARIANTS = ('Raw','Cal','Cal+ECC')


def circular_indices(dates, windows, block_days, repetitions, seed):
    dates = np.asarray(dates,dtype='datetime64[D]')
    windows = np.asarray(windows)
    if len(dates)!=len(windows) or np.any(np.diff(dates).astype(int)<=0):
        raise ValueError('Dates must be unique and ordered')
    cuts=np.r_[0,np.where((np.diff(dates).astype(int)!=1)|(windows[1:]!=windows[:-1]))[0]+1,len(dates)]
    rng=np.random.default_rng(seed)
    chunks=[]
    for left,right in zip(cuts[:-1],cuts[1:]):
        n=right-left;block=min(block_days,n)
        starts=rng.integers(n,size=(repetitions,(n+block-1)//block))
        indices=(starts[:,:,None]+np.arange(block))%n+left
        chunks.append(indices.reshape(repetitions,-1)[:,:n])
    return np.concatenate(chunks,axis=1)


def calibration_row(model,seed,variant,window,samples,y,daily):
    rel,maq,cov,width=curves(samples,y)
    return dict(window=window,model=model,seed=seed,variant=variant,n_dates=len(y),
                mean_daily_reliability_mae_pp=float(daily['Reliability_MAE_pp'].mean()),
                pooled_reliability_mae_pp=rel,mean_daily_maqce=float(daily['MAQCE'].mean()),pooled_maqce=maq,
                **{f'coverage_{int(c*100)}':v for c,v in zip([.5,.8,.9,.95],cov)},
                **{f'width_{int(c*100)}_mw':v for c,v in zip([.5,.8,.9,.95],width)})


def evaluate_new(protocol):
    state=json.loads((OUT/'formal/task_manifest.json').read_text(encoding='utf-8'))
    if state['status']!='COMPLETED' or len(state['tasks'])!=60:
        raise ValueError('All 60 formal tasks must finish before TEST evaluation')
    inputs={str((OUT/'protocol.json').relative_to(ROOT)):sha(OUT/'protocol.json')}
    for record in state['tasks']:
        directory=ROOT/record['directory']
        hashes=validate_task(directory,record['model'],record['quarter'],record['seed'],False,protocol)
        if hashes!=record['artifacts']:raise ValueError('Task artifacts changed')
        inputs.update({str((directory/name).relative_to(ROOT)):h for name,h in hashes.items()})
    rows,calibration,runtimes=[],[],[]
    bundles={q:load_daily_bundle(ROOT/protocol['windows'][str(q)]['bundle']) for q in range(1,5)}
    for key,label in LABELS.items():
        for seed in range(3):
            full={v:[] for v in VARIANTS};full_y=[];cell_rows=[]
            for quarter in range(1,5):
                raw,_,dates=bundles[quarter]
                directory=OUT/f'formal/quarter_{quarter}/{key}/seed_{seed}'
                meta=json.loads((directory/'metadata.json').read_text())
                runtimes.append(dict(window=f'quarter_{quarter}',model=label,seed=seed,
                                     fit_seconds=meta['fit_seconds'],sampling_vs_seconds=meta['sampling_seconds']['VS'],
                                     sampling_test_seconds=meta['sampling_seconds']['TEST'],
                                     model_parameters=meta['model_parameters'],
                                     conditioner_parameters=meta.get('conditioner_parameters',0),
                                     critic_parameters=meta.get('critic_parameters',0),device=meta['device']))
                with (directory/'scenarios_100_VS.pickle').open('rb') as f:vs=np.asarray(pickle.load(f),dtype=float)
                with (directory/'scenarios_100_TEST.pickle').open('rb') as f:test=np.asarray(pickle.load(f),dtype=float)
                y=np.asarray(raw['y_test'],dtype=float)
                calibrator=HourlyAffineCalibrator(limits=(0.,float('inf'))).fit(vs,np.asarray(raw['y_vs'],dtype=float).reshape(-1))
                cal=calibrator.transform(test)
                template_path=_scenario_file(V2/f'rolling/quarter_{quarter}/formal','fa_bm_vae',seed,'TEST',100)
                inputs[str(template_path.relative_to(ROOT))]=sha(template_path)
                with template_path.open('rb') as f:template=pickle.load(f)
                ecc=ensemble_copula_coupling(cal,template)
                if not np.array_equal(np.sort(cal,axis=1),np.sort(ecc,axis=1)):
                    raise ValueError('ECC altered marginal samples')
                full_y.append(y)
                for variant,samples in [('Raw',test),('Cal',cal),('Cal+ECC',ecc)]:
                    metrics=_metrics_for(samples,y)
                    full[variant].append(samples)
                    for i,date in enumerate(dates['test']):
                        day=samples[i*24:(i+1)*24]
                        row=dict(window=f'quarter_{quarter}',model=label,seed=seed,date=str(date),variant=variant,
                                 n_scenarios=100,statistical_unit='forecast date',
                                 scenario_min_mw=float(day.min()),scenario_max_mw=float(day.max()),
                                 scenario_negative_fraction=float((day<0).mean()),
                                 **{metric:float(values[i]) for metric,values in metrics.items()})
                        rows.append(row);cell_rows.append(row)
                    calibration.append(calibration_row(label,seed,variant,f'quarter_{quarter}',samples,y,pd.DataFrame(metrics)))
                print(f'evaluated {key} seed{seed} q{quarter}',flush=True)
            cell=pd.DataFrame(cell_rows)
            for variant in VARIANTS:
                calibration.append(calibration_row(label,seed,variant,'FULL_YEAR',np.concatenate(full[variant]),
                                                   np.concatenate(full_y),cell[cell.variant==variant]))
    return pd.DataFrame(rows),pd.DataFrame(calibration),pd.DataFrame(runtimes),inputs


def tables(daily):
    all_frames=[]
    windows=list(daily.window.unique())+['FULL_YEAR']
    main=list(MAIN_MODELS)+list(LABELS.values())
    for window in windows:
        frame=daily if window=='FULL_YEAR' else daily[daily.window==window]
        per=frame.groupby(['model','seed','variant'])[list(DAILY_METRICS)].mean().reset_index()
        per.insert(0,'window',window);all_frames.append(per)
        summary=[]
        for (model,variant),group in per.groupby(['model','variant']):
            if set(group.seed)!={0,1,2}:raise ValueError('Missing training seeds')
            for metric in DAILY_METRICS:
                summary.append(dict(window=window,model=model,variant=variant,metric=metric,
                                    mean_seed_means=group[metric].mean(),sd_between_seeds=group[metric].std(ddof=1),
                                    n_seeds=3,n_dates_per_seed=frame.date.nunique()))
        folder=RESULTS if window=='FULL_YEAR' else RESULTS/window
        folder.mkdir(exist_ok=True)
        pd.DataFrame(summary).to_csv(folder/'summary.csv',index=False)
        for variant,name in [('Raw','raw'),('Cal','cal'),('Cal+ECC','cal_ecc_shared_template')]:
            _write_table(folder,'main_table_'+name,_table_rows(summary,main,variant),DAILY_METRICS)
            tex=folder/f'main_table_{name}.tex'
            tex.write_text(tex.read_text(encoding='utf-8').replace('_',r'\_').replace('±',r'$\pm$'),encoding='utf-8')
        _write_table(folder,'j0_ablation_raw',_table_rows(summary,['J=0 / independent prior'],'Raw'),DAILY_METRICS)
    pd.concat(all_frames,ignore_index=True).to_csv(RESULTS/'metrics_per_seed.csv',index=False)


def paired(daily):
    metrics=['CRPS_MW','Energy_Score_MW','Variogram_Score','Ramp_CRPS_MW']
    base=daily[daily.model=='FA-BM-VAE']
    rows,byseed=[],[]
    dates=sorted(base.date.unique())
    window_for_date=base.drop_duplicates('date').set_index('date').loc[dates,'window'].to_numpy()
    resamples={b:circular_indices(dates,window_for_date,b,5000,20261023+b) for b in [3,7,14]}
    for model in [*MAIN_MODELS[1:],*LABELS.values(),'J=0 / independent prior']:
        other=daily[daily.model==model]
        for variant in VARIANTS:
            left=base[base.variant==variant].set_index(['seed','date'])[metrics].sort_index()
            right=other[other.variant==variant].set_index(['seed','date'])[metrics].sort_index()
            if not left.index.equals(right.index):raise ValueError('Unpaired observations')
            diff=left-right
            means=diff.groupby('date').mean().loc[dates]
            for metric in metrics:
                values=means[metric].to_numpy()
                for block,indices in resamples.items():
                    low,high=np.quantile(values[indices].mean(axis=1),[.025,.975])
                    rows.append(dict(model='FA-BM-VAE',comparator=model,variant=variant,metric=metric,
                                     estimate_main_minus_comparator=values.mean(),ci95_low=low,ci95_high=high,
                                     block_days=block,repetitions=5000,bootstrap_seed=20261023+block,
                                     statistical_unit='date; mean across three seeds; quarter/segment stratified',
                                     interval_scope='conditional on trained models; pointwise, no multiplicity adjustment'))
                for seed in range(3):
                    sample=diff.loc[seed].loc[dates,metric]
                    for window in ['FULL_YEAR',*sorted(base.window.unique())]:
                        chosen=sample if window=='FULL_YEAR' else sample[np.asarray(window_for_date)==window]
                        byseed.append(dict(window=window,comparator=model,variant=variant,metric=metric,
                                           seed=seed,n_dates=len(chosen),mean_difference=chosen.mean()))
    pd.DataFrame(rows).to_csv(RESULTS/'paired_block_sensitivity.csv',index=False)
    pd.DataFrame(byseed).to_csv(RESULTS/'paired_differences_by_seed.csv',index=False)


def main():
    protocol=verify(protected=True)
    RESULTS.mkdir(exist_ok=True)
    code_files=[Path(__file__),ROOT/'GEFcom2014/forecast_quality/neso_wind_results.py',
                ROOT/'GEFcom2014/forecast_quality/neso_paper_checks.py',
                ROOT/'GEFcom2014/forecast_quality/compare_scenarios.py',
                ROOT/'GEFcom2014/forecast_quality/unified_postprocessing.py',
                ROOT/'GEFcom2014/forecast_quality/qbm_temporal_ecc.py']
    new,calibration,runtimes,inputs=evaluate_new(protocol)
    old_path=V2/'rolling_results/metrics_daily.csv'
    old_cal_path=V2/'paper_checks_v1/calibration_metrics_per_seed.csv'
    old=pd.read_csv(old_path)
    combined=pd.concat([old,new],ignore_index=True)
    key=['window','model','seed','variant','date']
    if combined.duplicated(key).any():raise ValueError('Duplicate metric rows')
    expected_models=set(MAIN_MODELS)|set(LABELS.values())|{'J=0 / independent prior'}
    if set(combined.model)!=expected_models:raise ValueError('Unexpected models')
    for _,part in combined.groupby(['model','seed','variant']):
        if len(part)!=361 or set(part.date)!=set(old.date):raise ValueError('Incomplete test calendar')
    if len(combined)!=12*3*3*361:raise ValueError('Incomplete metric grid')
    combined.to_csv(RESULTS/'metrics_daily.csv',index=False)
    new.to_csv(RESULTS/'new_metrics_daily.csv',index=False)
    pd.concat([pd.read_csv(old_cal_path),calibration],ignore_index=True).to_csv(RESULTS/'calibration_metrics_per_seed.csv',index=False)
    runtimes.to_csv(RESULTS/'new_model_runtimes.csv',index=False)
    tables(combined);paired(combined)
    inputs.update({str(p.relative_to(ROOT)):sha(p) for p in [old_path,old_cal_path]})
    write_json(RESULTS/'evaluation_identity.json',dict(protocol_sha256=sha(OUT/'protocol.json'),
               code_sha256={str(p.relative_to(ROOT)):sha(p) for p in code_files},input_sha256=inputs,
               output_sha256={str(p.relative_to(RESULTS)):sha(p) for p in RESULTS.rglob('*') if p.is_file() and p.name!='evaluation_identity.json'},
               new_tasks=60,original_tasks_reused=84,bootstrap_recomputed=True,timestamp_status='UNRESOLVED',
               interpretation='Conditional historical comparison; Cal+ECC uses same-seed/window FA-BM-VAE template'))
    verify(protected=True)
    print('All new results evaluated; protected original artifacts unchanged',flush=True)


if __name__=='__main__':main()
