"""Evidence-based historical training audit. Never reads test images or test metrics."""
import math,subprocess
from ct_lab.dynamics_core import *

def run():
    initialize();records=[]
    for variant in ('v1','v2'):
        path=WEIGHTS/f'chd68_{variant}_best.pt';saved=torch.load(path,map_location='cpu',weights_only=True)
        history=[{k:v for k,v in h.items() if k!='validation_cases'} for h in saved['history']]
        # Validation history may be inspected; no historical test metrics are loaded.
        steps=sorted({int(s['step'].item() if torch.is_tensor(s['step']) else s['step']) for s in saved['optimizer']['state'].values() if 'step' in s})
        files=['model.py','losses.py','augmentation.py','cardiac_data.py','cardiac_train.py','runtime.py'];source=[]
        for name in files:
            relative='heart3d/ml/'+name
            exported=OUT/'historical_source'/variant/name;fallback=REPO/'outputs/training_dynamics_06/historical_source'/variant/name
            try:old=subprocess.run(['git','show',saved['git_revision']+':'+relative],cwd=REPO,capture_output=True,text=True);text=old.stdout if old.returncode==0 else None
            except FileNotFoundError:text=None
            origin='local Git commit'
            if text is None:
                origin='exported historical Git source snapshot';available=exported if exported.exists() else fallback;text=available.read_text(encoding='utf-8') if available.exists() else None
            source.append({'path':relative,'historical_source_available':text is not None,'historical_source_origin':origin,'working_tree_matches_historical':text is not None and text==(REPO/relative).read_text(encoding='utf-8')})
            if text is not None:write_text_if_changed(exported,text)
        samples=sorted({h['samples'] for h in history});cfg=saved['config'];per_epoch=[math.ceil(n/cfg['batch_size']) for n in samples]
        assert len(samples)==1 and len(steps)==1 and steps[0]==saved['epoch']*per_epoch[0]
        record={'variant':variant,'checkpoint_SHA256':fingerprint(path),'checkpoint_epoch':saved['epoch'],'checkpoint_optimizer_steps':steps,
                'configured_epochs':cfg['epochs'],'samples_per_epoch':samples[0],'optimizer_steps_per_epoch':per_epoch[0],
                'configured_total_steps':cfg['epochs']*per_epoch[0],'configured_total_examples':cfg['epochs']*samples[0],
                'history_epochs_available':len(history),'history_scope':'history contained in immutable best checkpoint, not necessarily complete run history',
                'parameters':sum(v.numel() for v in saved['model'].values() if torch.is_tensor(v)),
                'config':cfg,'preprocessing':saved['preprocessing'],'optimizer_groups':[{k:v for k,v in group.items() if k!='params'} for group in saved['optimizer']['param_groups']],
                'history':history,'source_comparison':source,'historical_environment':'Python3.12.3/Torch2.8.0+cu128/RTX4090 per saved public benchmark receipt'}
        metadata=REPO/('metadata/pediatric/cardiac_chd68_results_v1.json' if variant=='v1' else 'metadata/pediatric/cardiac_chd68_resolution384_results_v2.json')
        if metadata.exists():
            training_only=json.loads(metadata.read_text(encoding='utf-8'))['training']
            record.update(reported_completed_epochs=training_only['epochs'],reported_full_training_seconds=training_only['seconds'],
                completed_steps_derived_from_reported_epochs_and_invariant_epoch_size=training_only['epochs']*per_epoch[0],
                historical_training_summary_SHA256=fingerprint(metadata))
        records.append(record)
    roi_path=REPO/'outputs/heart_localization_roi_05/protocol.json'
    if not roi_path.exists():roi_path=OUT/'inputs/ROI05_protocol.json'
    if not roi_path.exists():roi_path=REPO/'outputs/training_dynamics_06/inputs/ROI05_protocol.json'
    roi=json.loads(roi_path.read_text());write_if_changed(OUT/'inputs/ROI05_protocol.json',roi)
    differences=[
      ['Architecture','HeartUNet25D 489112 parameters','Same base architecture'],['Input','5 source Z slices,256,clip0..2015','Same offsets/clip/size for B_full256'],
      ['Batch size',16,2],['Optimizer','Adam weight_decay=0','AdamW weight_decay=.0001'],['LR/scheduler','.001/fixed','.001/fixed'],
      ['Loss','CE + batchwise foreground soft Dice','Same code; different batch size changes batch-Dice objective'],
      ['Precision','FP32','AMP float16 forward/F32 loss; F16 input cache'],['Epochs',30,20],['Optimizer steps',records[0]['configured_total_steps'],3840],['Examples',records[0]['configured_total_examples'],7680],
      ['Unique Z centers/epoch','All prepared foreground + capped negative per case','12 GT-bbox uniform Z + up to4 negative; fixed small pool'],
      ['Case weighting','Global shuffled slice table; long cases contribute more','Equal cyclic case opportunities'],
      ['Augmentation','New per-example/epoch normalized shift/noise after spatial transform; clamp; no translation','One fixed source noise variant/case,50% clean,extra3%translation; different scale convention'],
      ['Checkpoint selection','Maximum development prepared-grid mean case Dice;30 epochs, no early stopping','Fixed last epoch'],
      ['Reported historical training Dice','Prepared-grid case macro','ROI05 native-grid metrics; separate grids'],
      ['Empty/ignore rules','255 excluded; reference-present foreground macro','Same helper'],['Native output','Inverse probabilities then argmax','Same mapping within tested interpolation tolerance']]
    write(OUT/'training_audit.json',{'historical':records,'ROI05':roi,'differences':differences,'sealed_test_images_read':False,'historical_test_metrics_used':False})
    # Baseline budget/factors fixed before new training or any ablation selection.
    historical=records[0];cfg=dict(historical['config']);cfg.update(data_root='outputs/training_dynamics_06',cache_relative_path='prepared256',allowed_partitions=['train','validation'],
      full_training_authorized=True,outputs_relative_path='runs',checkpoint_relative_path='runs',cpu_threads=1)
    cfg.update(stage06={'epoch_steps':774,'full_steps':23220,'control_steps':[480,3840,7740,15480,23220],
         'seeds':[20261009,20261010,20261011],'subset_seed':20261029,'subset_sizes':[8,16,24,32,48],
         'train_monitor':training_cases(8),'ablation_steps':7740,'learning_curve_steps':7740,'sampling_variants':['historical','ROI05_pool','equal_case','foreground_boost'],
         'foreground_boost_rule':'equal cases; halve each case eligible background probability relative to equal_case; not an assumed70% target',
         'loss_variants':['ce','ce_dice','weighted_ce_dice','focal_dice'],'focal_gamma':2.,
         'reproduction_tolerance_native_mean_Dice':.03,'reference_native_development_Dice':.7978329276179655,
         'selection':'historical max prepared-grid development case Dice; fixed-control checkpoints also reported',
         'no_early_stopping':True,'paid_infrastructure_started':False,
         'execution_gate':'one full historical-budget baseline must finish and be inspected before additional training jobs',
         'execution_differences':['RTX2060 instead of RTX4090','Python/Torch/CUDA versions differ','CPU threads1 instead of6 for bounded local execution; arithmetic/parity tested','cache preparation restricted to58 train/development cases; test sealed'],
         'preserve_original_sampling_loss_and_FP32':True})
    CONFIG.parent.mkdir(parents=True,exist_ok=True)
    if CONFIG.exists() and json.loads(CONFIG.read_text(encoding='utf-8'))!=cfg:
        assert not (OUT/'runs').exists(),'Protocol cannot change after training starts'
        write(OUT/'pre_smoke_drafts'/f'config_{time.time_ns()}.json',json.loads(CONFIG.read_text(encoding='utf-8')))
    write_if_changed(CONFIG,cfg)
    plan={'config':cfg,'config_SHA256':fingerprint(CONFIG),'train_cases':cohort()[0]['train'],'validation_cases':cohort()[0]['validation'],
          'sealed_test_read':False,'validation_status':'reused development, not independent test','preregistered_after_existing05_results_before_new06_training':True}
    if (OUT/'protocol.json').exists() and protocol()!=plan:
        assert not (OUT/'runs').exists(),'Protocol cannot change after training starts'
        write(OUT/'pre_smoke_drafts'/f'protocol_{time.time_ns()}.json',protocol())
    write(OUT/'protocol.json',plan)
    print('AUDIT/PROTOCOL READY',historical['configured_total_steps'],'steps',flush=True)

if __name__=='__main__':run()
