"""Bounded training benchmark on TRAIN ONLY; no epoch loop or test evaluation."""
import argparse
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import random
import re
import subprocess
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from heart3d.ml.dataset import HeartDataset
from heart3d.ml.losses import SegmentationLoss
from heart3d.ml.resources import ResourceMonitor,environment
from heart3d.pediatric import write_json
from heart3d.storage import require_space,sha256_file


def seed_everything(seed,threads):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(threads);torch.use_deterministic_algorithms(True)


def train_step(model,optimizer,loss_function,batch):
    model.train();optimizer.zero_grad(set_to_none=True)
    loss=loss_function(model(batch['image']),batch['target'])
    if not torch.isfinite(loss):raise ValueError('Non-finite training loss')
    loss.backward()
    gradients=[p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or any(not torch.isfinite(g).all() for g in gradients):raise ValueError('Invalid gradients')
    norm=float(torch.sqrt(sum((g.detach()**2).sum() for g in gradients)))
    if norm<=0:raise ValueError('Zero gradients')
    optimizer.step()
    return float(loss.detach()),norm


def run(config_path,batches,run_name,data_root=None):
    if not 1<=batches<=200:raise ValueError('This entry point permits at most 200 measured batches')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',run_name) or '..' in run_name:
        raise ValueError('Run name must be a safe single directory component')
    config=json.loads(Path(config_path).read_text())
    if not 0<=config['benchmark_warmup_batches']<=20 or not 1<=config['sanity_batches']<=50:
        raise ValueError('Warmup/sanity exceed bounded benchmark limits')
    from heart3d.ml.model import HeartUNet25D
    seed_everything(config['seed'],config['cpu_threads'])
    root=Path(data_root or config['data_root']);output=root/config['outputs_relative_path']/run_name
    if output.exists():raise ValueError('Run output exists; use a new run name')
    checkpoint=root/config['checkpoint_relative_path']/run_name/'last_benchmark.pt'
    if checkpoint.exists():raise ValueError('Benchmark checkpoint exists; use a new run name')
    require_space(root,100_000_000,int(config['reserve_GB']*1e9));output.mkdir(parents=True)
    dataset=HeartDataset(config_path,'train',data_root);sanity=HeartDataset(config_path,'train',data_root,augmentation=False)
    loader=DataLoader(dataset,batch_size=config['batch_size'],shuffle=True,num_workers=0,
        generator=torch.Generator().manual_seed(config['seed']),drop_last=False)
    if batches+config['benchmark_warmup_batches']>len(loader):raise ValueError('Benchmark exceeds one epoch; choose fewer batches')
    model=HeartUNet25D(**config['model']).cpu()
    loss_function=SegmentationLoss(**config['loss']);optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
    initial_weights=[p.detach().clone() for p in model.parameters()]
    # Two positive central slices from distinct train patients, no augmentation.
    patients=list(sanity.provenance)[:2];fixed_indices=[]
    for pid in patients:
        positive=sanity.provenance[pid]['positive_indices']
        fixed_indices.append(sanity.indices.index((pid,positive[len(positive)//2])))
    fixed=default_collate([sanity[i] for i in fixed_indices])
    def fixed_loss():
        model.eval()
        with torch.no_grad():return float(loss_function(model(fixed['image']),fixed['target']))
    fixed_before=fixed_loss();history=[];timings=[];sample_count=0
    monitor=ResourceMonitor()
    with monitor:
        iterator=iter(loader)
        for step in range(batches+config['benchmark_warmup_batches']):
            start=time.perf_counter();batch=next(iterator)
            loss,norm=train_step(model,optimizer,loss_function,batch)
            elapsed=time.perf_counter()-start
            measured=step>=config['benchmark_warmup_batches']
            if measured:timings.append(elapsed);sample_count+=len(batch['image'])
            history.append({'step':step,'phase':'measured' if measured else 'warmup',
                'seconds_including_data':elapsed,'loss':loss,'gradient_norm':norm,
                'patient_ids':batch['patient_id'],'positive_targets':int(batch['target'].flatten(1).any(1).sum())})
            if (step+1)%10==0:print('BENCHMARK',step+1,'loss',round(loss,5),'seconds',round(elapsed,3),flush=True)
        fixed_after_benchmark=fixed_loss()
        # A separate technical overfit check, never a validation/test quality result.
        for step in range(config['sanity_batches']):
            loss,norm=train_step(model,optimizer,loss_function,fixed)
            history.append({'step':step,'phase':'fixed_train_sanity','loss':loss,'gradient_norm':norm})
        fixed_after_sanity=fixed_loss()
    delta=sum(float((p.detach()-old).abs().sum()) for p,old in zip(model.parameters(),initial_weights))
    learning={'fixed_train_patient_ids':patients,'fixed_loss_before':fixed_before,
        'fixed_loss_after_random_benchmark':fixed_after_benchmark,'fixed_loss_after_sanity':fixed_after_sanity,
        'sanity_batches':config['sanity_batches'],'weights_absolute_delta':delta,
        'finite_losses_and_gradients':True,'sanity_passed':bool(fixed_after_sanity<fixed_before and delta>0),
        'interpretation':'technical training-subset learning check; not a model quality result'}
    env=environment();revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'torch_rng_state':torch.get_rng_state(),
        'config':config,'preprocessing':json.loads(Path(config['preprocessing']).read_text()),
        'split_SHA256':sha256_file(config['split']),'model_version':'own_25d_unet_v1','git_revision':revision,
        'training_scope':'bounded train-only resource benchmark and fixed-train sanity, not a finished baseline'},checkpoint)
    per_batch=float(np.mean(timings));epoch_batches=math.ceil(len(dataset)/config['batch_size'])
    summary={'version':1,'created_utc':datetime.now(timezone.utc).isoformat(),'git_revision':revision,
        'config_SHA256':sha256_file(config_path),'split_SHA256':sha256_file(config['split']),
        'preprocessing_SHA256':sha256_file(config['preprocessing']),'hardware':env,
        'model_parameters':sum(p.numel() for p in model.parameters()),
        'measured_batches':batches,'warmup_batches':config['benchmark_warmup_batches'],
        'mean_seconds_per_batch':per_batch,'median_seconds_per_batch':float(np.median(timings)),
        'p95_seconds_per_batch':float(np.quantile(timings,.95)),'samples_per_second':sample_count/sum(timings),
        'dataset_train_samples_per_epoch':len(dataset),'train_batches_per_epoch':epoch_batches,
        'estimated_train_epoch_seconds':epoch_batches*per_batch,'estimated_30_train_epochs_seconds':30*epoch_batches*per_batch,
        'estimate_scope':'linear extrapolation of measured train batches; excludes validation, checkpoint I/O and scheduling contention',
        'resources':monitor.summary(),'learning':learning,'checkpoint_bytes':checkpoint.stat().st_size,
        'checkpoint_relative_path':checkpoint.relative_to(root).as_posix(),'checkpoint_SHA256':sha256_file(checkpoint),
        'full_training_executed':False,'validation_evaluated':False,'test_evaluated':False,
        'all_benchmark_patient_ids':sorted({pid for h in history for pid in h.get('patient_ids',[])})}
    write_json(output/'config.json',config);write_json(output/'environment.json',env)
    write_json(output/'history.json',history);write_json(output/'resource_samples.json',monitor.samples)
    write_json(output/'summary.json',summary)
    print('BENCHMARK SUMMARY',json.dumps(summary,ensure_ascii=False),flush=True)
    if not learning['sanity_passed']:raise ValueError('Fixed-train learning sanity failed; investigate before full baseline')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--batches',type=int,default=50);p.add_argument('--run-name',required=True);p.add_argument('--data',type=Path)
    a=p.parse_args();run(a.config,a.batches,a.run_name,a.data)
