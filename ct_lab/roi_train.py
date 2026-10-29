"""Matched-budget scratch 2.5D training; train-only group CV before main jobs."""
import gc
import json
import time
import numpy as np
import torch
from ct_lab.roi_core import *
from ct_lab.roi_models import ROISegmenter
from ct_lab.roi_data import TrainingData,augment_seg
from heart3d.ml.losses import MulticlassSegmentationLoss


def load_segmenter(path,device='cpu'):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved.get('format')!='roi_segmenter05' or saved['split_sha256']!=fingerprint(REPO/'configs/splits/cardiac_chd68_v1.json'):raise ValueError('Checkpoint provenance differs')
    plan=json.loads((OUT/'protocol.json').read_text())
    if not set(saved['train_cases'])<=set(plan['train_cases']) or set(saved['train_cases'])&set(saved['heldout_cases']):raise ValueError('Segmentation case leakage')
    model=ROISegmenter(saved['job']['branch']).to(device);model.load_state_dict(saved['model'],strict=True);model.eval();return model,saved


@torch.no_grad()
def batch_dice(logits,target):
    pred=logits.argmax(1);valid=target<8;values=[]
    for c in range(1,8):
        truth=(target==c);p=(pred==c)&valid;den=p.sum()+truth.sum()
        if truth.any():values.append(float((2*(p&truth).sum()+1e-6)/(den+1e-6)))
    return float(np.mean(values)) if values else 0.


def fit(job,train_cases,heldout,plan,context='all_train'):
    directory=OUT/'segmenters'/context/job['name'];receipt=directory/'result.json'
    if not (OUT/'smoke.json').exists():raise ValueError('Resource smoke gate missing')
    device='cuda' if torch.cuda.is_available() else 'cpu';seed_runtime(SEED,1,device);cfg=plan['segmentation'];cv=context!='all_train'
    epochs=cfg['CV_epochs'] if cv else cfg['epochs'];steps=cfg['CV_steps_per_epoch'] if cv else cfg['steps_per_epoch'];batch=cfg['batch_size']
    if receipt.exists():
        previous=json.loads(receipt.read_text())
        if previous['epochs']>=epochs:return previous
        write(directory/f'pilot_epoch{previous["epochs"]}_result.json',previous)
    data=TrainingData(train_cases,job['mode'],job['size'],context,job['branch'])
    model=ROISegmenter(job['branch']).to(device);criterion=MulticlassSegmentationLoss();optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    scaler=torch.amp.GradScaler(device,enabled=device=='cuda');history=[];start_epoch=0;directory.mkdir(parents=True,exist_ok=True);last=directory/'last.pt'
    if last.exists():
        saved=torch.load(last,map_location=device,weights_only=True);model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
        start_epoch=saved['epoch'];history=saved['history']
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    for epoch in range(start_epoch,epochs):
        model.train();rng=np.random.default_rng(SEED+epoch*100003);losses=[];dices=[]
        # Cyclic shuffled case IDs guarantee equal case opportunities independent of ROI and input size.
        order=np.concatenate([rng.permutation(len(train_cases)) for _ in range(int(np.ceil(steps*batch/len(train_cases))))])[:steps*batch]
        for step in range(steps):
            torch.manual_seed(SEED+epoch*100003+step);case_ids=[train_cases[i] for i in order[step*batch:(step+1)*batch]]
            x,y,core=data.batch(case_ids,rng);x=x.to(device);y=y.to(device);core=core.to(device) if core is not None else None
            x,y,core=augment_seg(x,y,core,rng,cfg['augmentation']);optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device,dtype=torch.float16,enabled=device=='cuda'):logits=model(x,core)
            loss=criterion(logits.float(),y)
            if not torch.isfinite(loss):raise ValueError('Nonfinite segmentation loss')
            scaler.scale(loss).backward();scaler.step(optimizer);scaler.update();losses.append(float(loss.detach()))
            if step%8==0:dices.append(batch_dice(logits,y))
        if device=='cuda':torch.cuda.synchronize()
        history.append({'epoch':epoch+1,'loss':float(np.mean(losses)),'train_augmented_grid_foreground_Dice':float(np.mean(dices)),
                        'steps':steps,'samples':steps*batch,'seconds_total_this_invocation':time.perf_counter()-start})
        saved={'format':'roi_segmenter05','model':model.state_dict(),'job':job,'train_cases':train_cases,'heldout_cases':heldout,'epoch':epoch+1,
               'seed':SEED,'split_sha256':plan['split_sha256'],'config':cfg,'context':context,'input_channels':5,'output_classes':8}
        torch.save(saved,directory/'best.pt') # fixed last epoch, never outer-validation selection
        torch.save({**saved,'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'history':history},last);write(directory/'history.json',history)
        print('SEGMENTER',context,job['name'],epoch+1,history[-1],flush=True)
    result={'job':job,'context':context,'train_cases':train_cases,'heldout_cases':heldout,'epochs':epochs,'steps_per_epoch':steps,'batch_size':batch,
            'parameters':sum(p.numel() for p in model.parameters()),'checkpoint':str((directory/'best.pt').relative_to(OUT)),
            'checkpoint_sha256':fingerprint(directory/'best.pt'),'last_epoch_loss':history[-1]['loss'],'last_epoch_training_Dice':history[-1]['train_augmented_grid_foreground_Dice'],
            'seconds_this_invocation':time.perf_counter()-start,'peak_GPU_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None,
            'selection':'fixed last epoch; no outer development metrics read for selection'}
    write(receipt,result);del model,optimizer,data;gc.collect()
    if device=='cuda':torch.cuda.empty_cache()
    return result


def run():
    plan,_=freeze();cv_results=[]
    if not (OUT/'training_gate.json').exists():
        write(OUT/'training_gate.json',json.loads((Path(__file__).with_name('roi_execution_gate.json')).read_text()))
    if (OUT/'training_gate.json').exists():
        gate=json.loads((OUT/'training_gate.json').read_text());plan['segmentation']={**plan['segmentation'],'CV_epochs':gate['CV_epochs']}
    from ct_lab.roi_evaluate import evaluate_model
    for fold,heldout in enumerate(plan['CV_folds']):
        cases=[c for c in plan['train_cases'] if c not in heldout];context=f'fold{fold}'
        for name in ('B_full_256','D_predicted_256'):
            job=next(j for j in plan['main_jobs'] if j['name']==name);result=fit(job,cases,heldout,plan,context)
            scores=evaluate_model(result,heldout,plan,topology=False);cv_results.append({'training':result,'evaluation':scores})
    write(OUT/'segmentation_CV.json',cv_results);print('TRAIN-ONLY SEGMENTATION CV COMPLETE',flush=True)
    means={name:float(np.mean([r['evaluation']['mean_case_foreground_Dice'] for r in cv_results if r['training']['job']['name']==name])) for name in ('B_full_256','D_predicted_256')}
    write(OUT/'segmentation_CV_selection.json',{'train_only_CV_means':means,'primary_pipeline':max(means,key=means.get),'primary_resolution':256,
          'scope':'selected before outer development comparisons; 384 and core branch remain predeclared diagnostic ablations, not outer-selected defaults'})
    main=[]
    for name in ('B_full_256','D_predicted_256','C_oracle_256','B_full_384','D_predicted_384','D_core_256'):
        job=next(j for j in plan['main_jobs'] if j['name']==name);result=fit(job,plan['train_cases'],[],plan)
        scores=evaluate_model(result,plan['evaluation_cases'],plan,topology=False,save_predictions=True)
        main.append({'training':result,'evaluation':scores});write(OUT/'segmentation_progress.json',main)
    write(OUT/'segmentation_results.json',{'protocol':plan,'CV':cv_results,'main':main});print('ALL ROI SEGMENTERS COMPLETE',flush=True)


if __name__=='__main__':run()
