"""Bounded real-label technical learning check; no cohort metrics or full training."""
import argparse
import json
from pathlib import Path
import time
import nibabel as nib
import numpy as np
import torch
from torch.nn import functional as F
from heart3d.ml.model import HeartUNet25D
from heart3d.ml.losses import MulticlassSegmentationLoss
from heart3d.ml.runtime import seed_runtime,revision
from heart3d.ml.resources import environment,ResourceMonitor
from heart3d.storage import require_space,sha256_file

def cardiac_target(plane):
    discrete=np.asarray(plane,dtype=np.int64)
    return np.where(discrete<=7,discrete,255)

def run(a):
    if not 1<=a.batches<=50:raise ValueError('Bounded sanity batches required')
    if a.output.exists():raise ValueError('Choose a new output version')
    require_space(a.output,50_000_000,int(a.reserve_GB*1e9));a.output.mkdir(parents=True)
    seed_runtime(20261009,4,'cpu')
    ct=nib.load(a.ct);label=nib.load(a.mask)
    if ct.shape!=label.shape or not np.allclose(ct.affine,label.affine):raise ValueError('Pair grid mismatch')
    raw=np.asanyarray(label.dataobj)
    if not np.isfinite(raw).all() or not np.equal(raw,np.rint(raw)).all() or raw.min()<0:raise ValueError('Discrete original labels required')
    counts=((raw>=1)&(raw<=7)).sum(axis=(0,1));center=int(counts.argmax())
    indices=np.clip(center+np.array([-2,-1,0,1,2]),0,ct.shape[2]-1)
    # Explicit voxel-index context because public physical units are unverified.
    full=ct.get_fdata(dtype=np.float32)
    slices=np.stack([full[:,:,i].T for i in indices]);del full
    if not np.isfinite(slices).all():raise ValueError('Nonfinite source image')
    low,high=np.percentile(slices,[1,99.5])
    if high<=low:raise ValueError('Degenerate input distribution')
    slices=(np.clip((slices-low)/(high-low),0,1)*2-1).astype(np.float32)
    x=F.interpolate(torch.from_numpy(slices[None]),size=(128,128),mode='bilinear',align_corners=False)
    target=cardiac_target(raw[:,:,center]).T
    y=F.interpolate(torch.from_numpy(target.copy())[None,None].float(),size=(128,128),mode='nearest')[:,0].long()
    model=HeartUNet25D(output_channels=8)
    criterion=MulticlassSegmentationLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=.002)
    initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    with torch.no_grad():before=float(criterion(model(x),y))
    history=[];start=time.perf_counter()
    with ResourceMonitor() as monitor:
        for batch in range(a.batches):
            tick=time.perf_counter();optimizer.zero_grad(set_to_none=True);loss=criterion(model(x),y)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward();gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
            if gradient<=0:raise ValueError('No gradient')
            optimizer.step();history.append({'batch':batch,'loss':float(loss.detach()),'gradient_norm':float(gradient),'seconds':time.perf_counter()-tick})
        with torch.no_grad():after=float(criterion(model(x),y))
    updated=any(not torch.equal(v,initial[k]) for k,v in model.state_dict().items())
    if not updated or not np.isfinite(after) or after>=before:raise ValueError('Technical learning sanity failed')
    checkpoint=a.output/'sanity_only.pt'
    torch.save({'model':model.state_dict(),'training_scope':'multiclass_sanity_only','git_revision':revision()},checkpoint)
    result={'scope':'single development CT plane repeated; technical sanity only',
        'source_CT_SHA256':sha256_file(a.ct),'source_mask_SHA256':sha256_file(a.mask),
        'patient_split':None,'cohort_training':False,'validation_or_test_used':False,'physical_geometry_used':False,
        'context_voxel_indices':indices.tolist(),'input_size':[128,128],'training_subset_percentiles':[float(low),float(high)],
        'preprocessing_approved_for_baseline':False,'parameters':sum(p.numel() for p in model.parameters()),
        'input_channels':5,'output_channels':8,'loss':'CE + own foreground multiclass Dice; ignore_index=255',
        'batches':a.batches,'loss_before':before,'loss_after':after,'weights_updated':updated,
        'elapsed_seconds':time.perf_counter()-start,'median_sec_per_batch':float(np.median([r['seconds'] for r in history])),
        'history':history,'resources':monitor.summary(),'environment':environment('cpu'),
        'checkpoint_SHA256':sha256_file(checkpoint),'checkpoint_bytes':checkpoint.stat().st_size,
        'not_an_accuracy_experiment':True,'full_training_started':False}
    (a.output/'sanity_report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('MULTICLASS_SANITY',json.dumps({k:result[k] for k in ['parameters','batches','loss_before','loss_after','weights_updated','median_sec_per_batch']}),flush=True)
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['ct','mask','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--batches',type=int,default=12);p.add_argument('--reserve-GB',type=float,default=80)
    run(p.parse_args())
