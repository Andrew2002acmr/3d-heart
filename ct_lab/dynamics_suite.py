"""Prepare33 explicit jobs; execute only by manual CLI after baseline fidelity gate."""
import argparse
from ct_lab.dynamics_core import *

def jobs():
    cfg=protocol()['config'];st=cfg['stage06'];result=[]
    for seed in st['seeds']:
        result.append(dict(group='baseline',seed=seed,n=48,steps=st['full_steps'],sampling='historical',loss_name='ce_dice',family='baseline'))
        for sampling in ['ROI05_pool','equal_case','foreground_boost']:
            result.append(dict(group='sampling',seed=seed,n=48,steps=st['ablation_steps'],sampling=sampling,loss_name='ce_dice',family='sampling'))
        for n in [8,16,24,32]:result.append(dict(group='data',seed=seed,n=n,steps=st['learning_curve_steps'],sampling='historical',loss_name='ce_dice',family='data'))
        for loss in ['ce','weighted_ce_dice','focal_dice']:result.append(dict(group='losses',seed=seed,n=48,steps=st['ablation_steps'],sampling='historical',loss_name=loss,family='loss'))
    return sorted(result,key=lambda j:({'baseline':0,'sampling':1,'data':2,'losses':3}[j['group']],j['seed'],j['n'],j['sampling'],j['loss_name']))

def prepare():
    matrix=jobs();path=REPO/'configs/experiments/training_dynamics_06_runpod.json'
    value={'execution':'manual existing RTX4090 only; no infrastructure provisioning','jobs':matrix,'total_optimizer_steps':sum(j['steps'] for j in matrix),
      'priority':['baseline','sampling','baseline additional seeds','data','losses'],'three_initialization_seeds_per_main_configuration':True,
      'same_nested_composition_seed':20261029,'subset_sizes':[8,16,24,32,48],
      'reused_prefixes':'historical sampler/CE+Dice/N48 at7740 uses each full baseline step007740; no duplicate training',
      'ablation_selection':'fixed last7740 for sampling/loss/data to avoid unequal validation checkpoint-selection opportunities',
      'baseline_selection':'historical max prepared development Dice at epoch ends; native reported separately',
      'local_policy':'user limited local work to one full baseline; every other training job is manual RunPod pending'}
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=value:
        # Only execution ordering may be clarified before any manual job starts.
        previous=json.loads(path.read_text(encoding='utf-8'));assert {json.dumps(j,sort_keys=True) for j in previous['jobs']}=={json.dumps(j,sort_keys=True) for j in value['jobs']}
        write(OUT/'pre_smoke_drafts'/f'manual_order_{time.time_ns()}.json',previous)
    write_if_changed(path,value)
    write_if_changed(OUT/'manual_runpod_plan.json',value);print('MANUAL PLAN READY',len(matrix),'jobs',value['total_optimizer_steps'],'optimizer steps; not started',flush=True)

def fidelity_gate():
    directory=OUT/'runs/baseline_s20261009_n48';result=json.loads((directory/'result.json').read_text());history=json.loads((directory/'history.json').read_text())
    native=result['evaluation']['by_role']['development']['native_Dice'];st=protocol()['config']['stage06'];valid=abs(native-st['reference_native_development_Dice'])<=st['reproduction_tolerance_native_mean_Dice']
    valid=valid and result['steps']==23220 and result['processed_examples']==371430 and history[-1]['training_loss']<history[0]['training_loss'] and all(np.isfinite(r['training_loss']) for r in history)
    evidence={'baseline_native_development_Dice':native,'historical_reference':st['reference_native_development_Dice'],'absolute_tolerance':.03,'passed':bool(valid),
      'meaning':'engineering gate for further fixed experiments, not independent validation or proof of exact replication','initialization_seed_count':1}
    write(OUT/'baseline_fidelity_gate.json',evidence)
    if not valid:raise ValueError('Baseline fidelity gate failed; inspect curves/provenance before more training. No automatic hyperparameter tuning.')
    return evidence

def execute(group):
    prepare()
    from ct_lab.dynamics_train import fit
    # Always one complete baseline first; no dozens of runs without checking it.
    fit();fidelity_gate()
    if group not in ('baseline','all'):
        for seed in protocol()['config']['stage06']['seeds'][1:]:
            if not (OUT/f'runs/baseline_s{seed}_n48/result.json').exists():raise ValueError('Run --group baseline first: matched7740-step reference prefixes are required for all3 seeds')
    for job in jobs():
        if job['seed']==20261009 and job['family']=='baseline':continue
        if group!='all' and job['group']!=group:continue
        if group=='baseline' and job['group']!='baseline':continue
        fit(job['seed'],job['n'],job['steps'],job['sampling'],job['loss_name'],job['family'])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--group',choices=['baseline','sampling','data','losses','all'],default='baseline');args=parser.parse_args()
    if args.prepare_only:prepare()
    else:execute(args.group)
