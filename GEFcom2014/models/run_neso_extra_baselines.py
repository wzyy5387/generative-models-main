"""Freeze and run the explicitly authorized conditional NESO historical extension."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT/'export/fa_bm_vae_neso_wind_classical_v2'
OUT = ROOT/'export/fa_bm_vae_neso_wind_baseline_extension_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def check_hashes(hashes, root=ROOT):
    for relative, expected in hashes.items():
        path = root/relative
        if not path.is_file() or sha(path) != expected:
            raise ValueError('Frozen file changed: '+str(path))


def freeze():
    from GEFcom2014.models.neso_extra_baselines import configurations, LABELS
    if (OUT/'protocol.json').exists():
        return verify()
    OUT.mkdir(parents=True, exist_ok=True)
    configs = OUT/'configs'
    configs.mkdir(exist_ok=True)
    config_hashes = {}
    for key, values in configurations().items():
        path = configs/(key+'.json')
        write_json(path, dict(values, model=key))
        config_hashes[str(path.relative_to(ROOT))] = sha(path)
    sources = [
        'GEFcom2014/models/neso_extra_baselines.py', 'GEFcom2014/models/run_neso_extra_baselines.py',
        'GEFcom2014/models/probabilistic_baselines.py', 'GEFcom2014/models/strong_probabilistic_baselines.py',
        'GEFcom2014/models/legacy_baselines.py', 'GEFcom2014/models/GAN/utils_gan_wasserstein.py',
        'GEFcom2014/models/VAE/utils_vae.py', 'GEFcom2014/external_datasets.py',
        'GEFcom2014/models/utils_models.py', 'tests/test_neso_extra_baselines.py',
    ]
    protected = {str(p.relative_to(ROOT)): sha(p) for p in V2.rglob('*') if p.is_file()}
    # Validate the source experiment against its own manifests before reusing it.
    identity = json.loads((V2/'paper_checks_v1/analysis_identity.json').read_text(encoding='utf-8'))
    check_hashes(identity['input_sha256'])
    windows = {}
    all_test = []
    for quarter in range(1,5):
        bundle = V2/f'rolling/quarter_{quarter}/data/neso_wind_daily.npz'
        with np.load(bundle, allow_pickle=False) as data:
            dates = {s: data['dates_'+s].astype(str).tolist() for s in ['ls','vs','test']}
        if not (max(dates['ls']) < min(dates['vs']) <= max(dates['vs']) < min(dates['test'])):
            raise ValueError('Nonchronological split')
        windows[str(quarter)] = dict(bundle=str(bundle.relative_to(ROOT)), bundle_sha256=sha(bundle), dates=dates)
        all_test.extend(dates['test'])
    if len(all_test) != 361 or len(set(all_test)) != 361:
        raise ValueError('Wrong frozen test calendar')
    protocol = dict(version=1, frozen_at_unix=time.time(), models=LABELS, seeds=[0,1,2],
                    n_scenarios=100, formal_tasks=60, smoke_tasks=5, windows=windows,
                    source_protocol_sha256=sha(V2/'protocol_v2.json'),
                    timestamp_status='UNRESOLVED; user explicitly chose to proceed without external confirmation',
                    research_scope='Conditional historical comparison under naive timestamps interpreted as UTC and previous London day 09:15 cutoff; no verified real-time availability claim; post-result extension, not independent confirmation',
                    config_sha256=config_hashes, code_sha256={p:sha(ROOT/p) for p in sources},
                    protected_v2_sha256=protected,
                    selection='LS fit and VS selection only; Treeffuser internal early stopping uses LS-only holdout',
                    source_adaptations='CVAE normal prior plus Gaussian decoder; WGAN linear critic and scalar interpolation; D3U supplied context adapter; DDPM split-specific random stream; no TEST tuning',
                    bootstrap='Date-paired seed averages; quarter and contiguous-segment stratified circular blocks 3/7/14; 5000 repeats; fixed date counts; no multiplicity claims',
                    python=sys.version, platform=platform.platform(),
                    packages={p:importlib.metadata.version(p) for p in ['torch','numpy','pandas','scikit-learn','diffusers','treeffuser','lightgbm']})
    for source in sources:
        dest = OUT/'source_snapshot'/source
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT/source).read_bytes())
    write_json(OUT/'protocol.json', protocol)
    (OUT/'protocol.sha256').write_text(sha(OUT/'protocol.json'), encoding='ascii')
    (OUT/'TIMESTAMP_DECISION.md').write_text(
        '# 时间戳决定\n\n用户明确选择不等待外部确认，继续补齐实验。保留UNRESOLVED，沿用冻结UTC解释和前一伦敦日09:15截止。'
        '本实验是有条件的历史模型比较，不代表已验证历史实时可用性。既有暂停建议已被本次用户授权取代。\n', encoding='utf-8')
    return protocol


def verify(protected=False):
    if sha(OUT/'protocol.json') != (OUT/'protocol.sha256').read_text().strip():
        raise ValueError('Protocol digest mismatch')
    protocol = json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    check_hashes(protocol['code_sha256']); check_hashes(protocol['config_sha256'])
    for window in protocol['windows'].values():
        check_hashes({window['bundle']:window['bundle_sha256']})
    if protected: check_hashes(protocol['protected_v2_sha256'])
    return protocol


def validate_task(directory, key, quarter, seed, smoke, protocol):
    import pickle
    meta = json.loads((directory/'metadata.json').read_text())
    window = protocol['windows'][str(quarter)]
    splits, count = (['VS'],5) if smoke else (['VS','TEST'],100)
    config = json.loads((OUT/'configs'/f'{key}.json').read_text())
    expected = dict(config, seed=seed)
    if smoke:
        expected.update(epochs=1,n_scenarios=5)
        if key=='d3u':expected['anchor_epochs']=1
        if key=='treeffuser':expected.update(tree_n_estimators=10,tree_n_repeats=1)
    if (meta['model'] != key or meta['seed'] != seed or meta['config'] != expected
            or meta['scenario_splits'] != splits or meta['test_used_for_selection']
            or Path(meta['dataset_bundle']).resolve() != (ROOT/window['bundle']).resolve()):
        raise ValueError('Wrong task identity')
    for split in splits:
        if meta['dates'][split] != window['dates'][split.lower()]:
            raise ValueError('Scenario date mapping changed')
        with (directory/f'scenarios_{count}_{split}.pickle').open('rb') as f: x=np.asarray(pickle.load(f))
        if x.shape != (len(meta['dates'][split])*24,count) or not np.isfinite(x).all():
            raise ValueError('Wrong scenario matrix')
    return {str(p.relative_to(directory)):sha(p) for p in directory.iterdir() if p.is_file()}


def run(mode):
    protocol = freeze()
    verify(protected=True)
    if mode == 'formal':
        smoke = json.loads((OUT/'smoke/task_manifest.json').read_text(encoding='utf-8'))
        if smoke['status'] != 'COMPLETED' or len(smoke['tasks']) != 5:
            raise ValueError('Smoke has not completed')
        for record in smoke['tasks']:
            check_hashes(record['artifacts'], ROOT/record['directory'])
    run_root = OUT/mode
    run_root.mkdir(exist_ok=True)
    path = run_root/'task_manifest.json'
    state = json.loads(path.read_text(encoding='utf-8')) if path.exists() else dict(status='RUNNING', tasks=[])
    tasks = [(q,k,s) for q in ([1] if mode=='smoke' else range(1,5))
             for k in protocol['models'] for s in ([0] if mode=='smoke' else [0,1,2])]
    for index, (quarter,key,seed) in enumerate(tasks):
        directory = run_root/f'quarter_{quarter}'/key/f'seed_{seed}'
        if index < len(state['tasks']):
            old = state['tasks'][index]
            if (old['quarter'],old['model'],old['seed']) != (quarter,key,seed) or old['status'] != 'COMPLETED':
                raise ValueError('Prior task incomplete or identity changed; inspect before resuming')
            hashes = validate_task(directory,key,quarter,seed,mode=='smoke',protocol)
            if hashes != old['artifacts']:raise ValueError('Prior task artifacts changed')
            continue
        verify()
        log = run_root/f'{index+1:02d}_q{quarter}_{key}_s{seed}.log'
        command = [sys.executable,'-u','-m','GEFcom2014.models.neso_extra_baselines',
                   '--config',str(OUT/'configs'/f'{key}.json'), '--bundle',str(ROOT/protocol['windows'][str(quarter)]['bundle']),
                   '--output',str(directory),'--seed',str(seed)]
        if mode=='smoke':command.append('--smoke')
        record = dict(quarter=quarter,model=key,seed=seed,status='RUNNING',started_at=time.time(),
                      directory=str(directory.relative_to(ROOT)),command=command,log=str(log.relative_to(ROOT)))
        state['status']='RUNNING';state['tasks'].append(record);write_json(path,state)
        print(f'{mode} {index+1}/{len(tasks)} q{quarter} {key} seed={seed}',flush=True)
        env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',WANDB_MODE='disabled')
        with log.open('w',encoding='utf-8') as handle:
            process=subprocess.Popen(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,env=env)
            record['pid']=process.pid;write_json(path,state)
            code=process.wait()
        record['elapsed_seconds']=time.time()-record['started_at'];record['exit_code']=code
        if code:
            record['status']=state['status']='FAILED';write_json(path,state)
            raise RuntimeError(f'Task failed: {log}')
        record['artifacts']=validate_task(directory,key,quarter,seed,mode=='smoke',protocol)
        record['log_sha256']=sha(log);record['status']='COMPLETED';write_json(path,state)
        print(f'completed in {record["elapsed_seconds"]:.1f}s',flush=True)
    verify(protected=True)
    state['status']='COMPLETED';write_json(path,state)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['freeze','smoke','formal','verify'])
    args=parser.parse_args()
    if args.mode=='freeze':freeze()
    elif args.mode=='verify':verify(protected=True);print('Frozen source and output hashes verified')
    else:run(args.mode)
