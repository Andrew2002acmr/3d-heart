"""Historical-budget FP32 baseline with exact original sampling/augmentation and resume."""
import argparse,gc,math,shutil,time
from collections import Counter
from torch.utils.data import DataLoader
from ct_lab.dynamics_core import *
from heart3d.ml.cardiac_data import CardiacDataset
from heart3d.ml.cardiac_train import validation_score
from heart3d.ml.model import HeartUNet25D
from heart3d.ml.losses import MulticlassSegmentationLoss
from heart3d.ml.runtime import step

class IndexedDataset(CardiacDataset):
    def __getitem__(self,index):return {**super().__getitem__(index),'sample_index':index}

def subset_dataset(config_path,cases,augmentation=True,partition='train'):
    split,_=cohort()
    if not set(cases)<=set(split[partition]):raise ValueError('Dataset partition/group leakage')
    data=IndexedDataset(config_path,partition,OUT,augmentation)
    data.rows=[r for r in data.rows if r['case_id'] in cases];data.provenance={c:p for c,p in data.provenance.items() if c in cases};data.set_epoch(0);return data

def fit(seed=20261009,n=48,max_steps=None,sampling='historical',loss_name='ce_dice',family='baseline'):
    plan=protocol();assert fingerprint(CONFIG)==plan['config_SHA256'],'Frozen stage06 configuration changed'
    base=plan['config'];cfg=json.loads(json.dumps(base));cfg['seed']=seed;st=cfg['stage06'];max_steps=max_steps or st['full_steps'];cases=training_cases(n)
    assert sampling in st['sampling_variants'] and loss_name in st['loss_variants']
    if family=='baseline' and (n!=48 or max_steps!=st['full_steps']):raise ValueError('Baseline has the full predeclared48-case historical budget; use suite data group for subsets')
    if family!='baseline' and max_steps!=st['ablation_steps']:raise ValueError('Ablation budget differs from preregistration')
    cfg['loss']={'variant':loss_name,'ignore_index':255}
    if loss_name=='focal_dice':cfg['loss']['gamma']=2.
    name=f'baseline_s{seed}_n{n}' if family=='baseline' else f'{family}_{sampling}_{loss_name}_s{seed}_n{n}'
    cfg['stage06_job']={'family':family,'sampling':sampling,'loss':loss_name,'max_steps':max_steps,'n_train':n,'selection':'historical best prepared' if family=='baseline' else 'fixed final step budget'}
    if family!='baseline':cfg['stage06']['control_steps']=[max_steps]
    directory=OUT/'runs'/name;receipt=directory/'result.json'
    if receipt.exists():return json.loads(receipt.read_text())
    if not (OUT/'smoke.json').exists() or not (OUT/'preparation.json').exists():raise ValueError('Preparation/smoke gate missing')
    if not torch.cuda.is_available():raise ValueError('This full-budget local run requires the verified CUDA gate')
    assert n in st['subset_sizes'] and seed in st['seeds']
    device='cuda';seed_runtime(seed,cfg['cpu_threads'],device);directory.mkdir(parents=True,exist_ok=True);write(directory/'config.json',cfg)
    from heart3d.ml.cardiac_train import device_info
    write(directory/'environment.json',device_info(device))
    # Preserve the exact implementation seen by every invocation/resume.
    snapshot=directory/'implementation'/str(time.time_ns());implementation={}
    for relative in ['ct_lab/dynamics_train.py','ct_lab/dynamics_core.py','ct_lab/dynamics_evaluate.py','heart3d/ml/model.py','heart3d/ml/cardiac_data.py','heart3d/ml/losses.py','heart3d/ml/augmentation.py','heart3d/ml/runtime.py']:
        destination=snapshot/relative;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/relative,destination);implementation[relative]=fingerprint(destination)
    write(snapshot/'SHA256.json',implementation)
    training=subset_dataset(directory/'config.json',cases);validation=CardiacDataset(directory/'config.json','validation',OUT,augmentation=False)
    cfg['epochs']=math.ceil(max_steps/math.ceil(len(training)/16));write(directory/'config.json',cfg)
    monitor_cases=[c for c in st['train_monitor'] if c in cases];monitor=subset_dataset(directory/'config.json',monitor_cases,False)
    model=HeartUNet25D(**cfg['model']).to(device)
    if loss_name=='ce_dice':criterion=MulticlassSegmentationLoss(**cfg['loss'])
    else:
        from ct_lab.dynamics_losses import criterion as make_criterion
        criterion,fitted=make_criterion(loss_name,cases,device)
        if fitted:write(directory/'loss_weights.json',fitted)
    opt=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
    progress=0;epoch0=0;skip=0;history=[];log=[];best=-1.;exposures=Counter();processed=0;running_sum=0.;running_count=0;start=time.perf_counter();last=directory/'last.pt'
    if last.exists():
        checkpoint=torch.load(last,map_location='cpu',weights_only=True);model.load_state_dict(checkpoint['model']);opt.load_state_dict(checkpoint['optimizer']);progress=checkpoint['steps']
        epoch0=checkpoint['epoch_index'];skip=checkpoint['batches_completed'];history=checkpoint['history'];log=checkpoint['step_history'];best=checkpoint['best_prepared_Dice']
        exposures=Counter(checkpoint['exposures']);processed=checkpoint['processed_examples'];running_sum=checkpoint['running_loss_sum'];running_count=checkpoint['running_count']
    torch.cuda.reset_peak_memory_stats()
    def saved(epoch,batches):
        return {'format':'training_dynamics06','model':model.state_dict(),'optimizer':opt.state_dict(),'config':cfg,'preprocessing':training.preproc,
          'split_SHA256':fingerprint(REPO/'configs/splits/cardiac_chd68_v1.json'),'steps':progress,'epoch_index':epoch,'batches_completed':batches,
          'seed':seed,'train_cases':cases,'processed_examples':processed,'history':history,'step_history':log,'best_prepared_Dice':best,
          'exposures':dict(exposures),'running_loss_sum':running_sum,'running_count':running_count,'selection':cfg['stage06_job']['selection']}
    def control(epoch,batches):
        path=directory/f'step{progress:06d}.pt'
        if not path.exists():atomic_checkpoint(path,saved(epoch,batches))
        atomic_checkpoint(last,saved(epoch,batches))
        from ct_lab.dynamics_evaluate import evaluate
        eval_cases=list(dict.fromkeys(monitor_cases+(['ct_1001'] if 'ct_1001' in cases else [])+plan['validation_cases']))
        evaluate(path,eval_cases,directory/'evaluation'/f'step{progress:06d}')
    epochs_needed=math.ceil(max_steps/max(1,math.ceil(len(training)/16)))
    # Resume incomplete native evaluations before progressing past their immutable weights.
    for c in st['control_steps']:
        path=directory/f'step{c:06d}.pt';destination=directory/'evaluation'/f'step{c:06d}'
        if path.exists() and not (destination/'summary.json').exists():
            from ct_lab.dynamics_evaluate import evaluate
            evaluate(path,list(dict.fromkeys(monitor_cases+(['ct_1001'] if 'ct_1001' in cases else [])+plan['validation_cases'])),destination)
    for epoch in range(epoch0,epochs_needed+1):
        training.set_epoch(epoch)
        if sampling=='historical':loader=DataLoader(training,batch_size=16,shuffle=True,num_workers=0,pin_memory=True,generator=torch.Generator().manual_seed(seed+epoch))
        else:
            from ct_lab.dynamics_sampling import sampling_plan
            statistics={r['case']:r for r in json.loads((OUT/'case_statistics.json').read_text())}
            indices,_=sampling_plan(training,sampling,seed,epoch,statistics)
            loader=DataLoader(training,batch_size=16,sampler=indices,num_workers=0,pin_memory=True,generator=torch.Generator().manual_seed(seed+epoch))
        for j,batch in enumerate(loader):
            if epoch==epoch0 and j<skip:continue
            tick=time.perf_counter();loss,norm=step(model,opt,criterion,batch,device);progress+=1;count=len(batch['image']);processed+=count;running_sum+=loss*count;running_count+=count
            for case,z in zip(batch['case_id'],batch['center_index'].tolist()):exposures[f'{case}:{z}']+=1
            log.append({'step':progress,'loss':loss,'learning_rate':opt.param_groups[0]['lr'],'gradient_norm':norm,'processed_examples':processed,'unique_slices':len(exposures),'epoch_index':epoch,'seconds_update':time.perf_counter()-tick})
            if progress%50==0:
                write(directory/'live.json',{'step':progress,'loss':loss,'processed_examples':processed,'unique_slices':len(exposures),'seconds_this_invocation':time.perf_counter()-start})
                print('DYNAMICS TRAIN',name,progress,'loss',loss,'examples',processed,flush=True)
            if progress in st['control_steps']:control(epoch,j+1)
            if progress>=max_steps:break
        score,values=validation_score(model,validation,cfg,device);train_score,_=validation_score(model,monitor,cfg,device)
        history.append({'epoch':epoch+1,'step':progress,'training_loss':running_sum/max(1,running_count),'processed_examples':processed,'unique_slices':len(exposures),
           'prepared_development_Dice':score,'prepared_train_monitor_Dice':train_score,'prepared_gap':train_score-score,'validation_cases':values,'learning_rate':opt.param_groups[0]['lr']})
        is_best=score>best;best=max(best,score);write(directory/'history.json',history);write(directory/'steps.json',log)
        if is_best:atomic_checkpoint(directory/'best_candidate.pt',saved(epoch+1,0))
        running_sum=0.;running_count=0;atomic_checkpoint(last,saved(epoch+1,0));print('DYNAMICS EPOCH',epoch+1,progress,'val prepared',score,'train monitor',train_score,flush=True)
        if progress>=max_steps:break
    selected=directory/'selected.pt'
    shutil.copyfile(directory/'best_candidate.pt' if family=='baseline' else directory/f'step{max_steps:06d}.pt',selected)
    from ct_lab.dynamics_evaluate import evaluate
    final=evaluate(selected,list(dict.fromkeys(monitor_cases+(['ct_1001'] if 'ct_1001' in cases else [])+plan['validation_cases'])),directory/'evaluation'/'selected')
    result={'name':name,'seed':seed,'n_train':n,'family':family,'sampling':sampling,'loss':loss_name,'selection':cfg['stage06_job']['selection'],'train_cases':cases,'steps':progress,'processed_examples':processed,'unique_slices':len(exposures),'selected_checkpoint_SHA256':fingerprint(selected),
      'prepared_best_development_Dice':best,'evaluation':final,'peak_GPU_bytes':torch.cuda.max_memory_allocated(),'seconds_this_invocation':time.perf_counter()-start}
    write(directory/'sampling_counts.json',dict(exposures));write(receipt,result);training.close();validation.close();monitor.close();print('DYNAMICS FULL BASELINE COMPLETE',name,flush=True);return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,default=20261009);parser.add_argument('--cases',type=int,default=48);parser.add_argument('--steps',type=int);args=parser.parse_args();fit(args.seed,args.cases,args.steps)
