"""Bounded real-train CPU/GPU sanity; these smoke weights never initialize models."""
import time
import nibabel as nib
import numpy as np
import torch
from ct_lab.roi_core import *
from ct_lab.roi_models import ROISegmenter,HeartLocalizer,localizer_loss
from heart3d.ml.losses import MulticlassSegmentationLoss


def run():
    plan,_=freeze();seed_runtime(SEED,1,'cpu');case='ct_1001';image=nib.load(DEFAULT_DATA/f'{case}_image.nii.gz');x=image.get_fdata(dtype=np.float32)
    y=cardiac_target(np.asanyarray(nib.load(DEFAULT_DATA/f'{case}_label.nii.gz').dataobj));z=x.shape[2]//2;criterion=MulticlassSegmentationLoss();rows=[]
    for device,size,batch in [('cpu',64,1)]+([('cuda',256,2),('cuda',384,2)] if torch.cuda.is_available() else []):
        seed_runtime(SEED,1,device);t=transform(x.shape,[[0,0,0],x.shape],size);context=np.stack([prepare_plane(x[:,:,np.clip(z+o,0,x.shape[2]-1)],t) for o in (-2,-1,0,1,2)])
        inp=torch.from_numpy(np.repeat((np.clip(context,0,2015)/2015*2-1)[None],batch,0)).to(device)
        gt=torch.from_numpy(np.repeat(prepare_plane(y[:,:,z],t,0,255).astype(np.int64)[None],batch,0)).to(device)
        model=ROISegmenter().to(device);opt=torch.optim.AdamW(model.parameters(),lr=.001);amp=device=='cuda';scaler=torch.amp.GradScaler(device,enabled=amp);losses=[];times=[]
        if amp:torch.cuda.reset_peak_memory_stats()
        for step in range(10 if amp else 3):
            if amp:torch.cuda.synchronize()
            start=time.perf_counter();opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device,dtype=torch.float16,enabled=amp):logits=model(inp)
            loss=criterion(logits.float(),gt);assert torch.isfinite(loss);scaler.scale(loss).backward();scaler.step(opt);scaler.update()
            if amp:torch.cuda.synchronize()
            times.append(time.perf_counter()-start);losses.append(float(loss.detach()))
        assert losses[-1]<losses[0],'Learning sanity failed'
        rows.append({'device':device,'size':size,'batch':batch,'steps':len(losses),'loss_before':losses[0],'loss_after':losses[-1],
                     'mean_seconds_step_after_warmup':float(np.mean(times[2:])), 'parameters':sum(p.numel() for p in model.parameters()),
                     'peak_GPU_bytes':torch.cuda.max_memory_allocated() if amp else None,'real_case':case,'used_for_initialization':False})
        del model,opt,inp,gt
        if amp:torch.cuda.empty_cache()
        print('SMOKE',rows[-1],flush=True)
    assert len(rows)==3,'Local GPU unavailable; use provided cloud config before any full training'
    write(OUT/'smoke.json',rows)


if __name__=='__main__':run()
