"""Real-CT CPU smoke and bounded historical FP32/batch16 GPU resource gate."""
import time
import psutil
from torch.utils.data import DataLoader,default_collate
from torch.nn import functional as F
from ct_lab.dynamics_core import *
from heart3d.ml.model import HeartUNet25D
from heart3d.ml.losses import MulticlassSegmentationLoss
from heart3d.ml.runtime import step
from heart3d.ml.cardiac_data import CardiacDataset

def run():
    cfg=protocol()['config'];seed_runtime(cfg['seed'],cfg['cpu_threads'],'cpu')
    clean=CardiacDataset(CONFIG,'train',OUT,augmentation=False);fixed=[]
    for case in list(clean.provenance)[:2]:
        z=clean.provenance[case]['positive_indices'];fixed.append(clean[clean.indices.index((case,z[len(z)//2]))])
    batch=default_collate(fixed);small={'image':F.interpolate(batch['image'],size=(64,64),mode='bilinear',align_corners=False),
                                 'target':F.interpolate(batch['target'][:,None].float(),size=(64,64),mode='nearest-exact')[:,0].long()}
    model=HeartUNet25D(**cfg['model']);loss=MulticlassSegmentationLoss();opt=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
    model.eval();before=float(loss(model(small['image']),small['target']).detach())
    for _ in range(8):step(model,opt,loss,small,'cpu')
    model.eval();after=float(loss(model(small['image']),small['target']).detach());assert after<before
    results={'CPU':{'size':64,'batch':2,'loss_before':before,'loss_after':after,'parameters':sum(p.numel() for p in model.parameters())}}
    del model,opt
    if torch.cuda.is_available():
        seed_runtime(cfg['seed'],cfg['cpu_threads'],'cuda');dataset=CardiacDataset(CONFIG,'train',OUT)
        loader=DataLoader(dataset,batch_size=16,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(cfg['seed']))
        model=HeartUNet25D(**cfg['model']).cuda();opt=torch.optim.Adam(model.parameters(),lr=.001);model.eval()
        with torch.no_grad():before=float(loss(model(batch['image'].cuda()),batch['target'].cuda()))
        torch.cuda.reset_peak_memory_stats();iterator=iter(loader);times=[];losses=[];start=time.perf_counter()
        for i in range(15):
            torch.cuda.synchronize();tick=time.perf_counter();item=next(iterator);v,_=step(model,opt,loss,item,'cuda');torch.cuda.synchronize()
            if i>=5:times.append(time.perf_counter()-tick)
            losses.append(v);print('HISTORICAL GPU SMOKE',i+1,v,flush=True)
        for _ in range(8):step(model,opt,loss,batch,'cuda')
        model.eval()
        with torch.no_grad():after=float(loss(model(batch['image'].cuda()),batch['target'].cuda()))
        assert after<before and np.isfinite(losses).all()
        results['GPU']={'device':torch.cuda.get_device_name(0),'precision':'FP32','batch':16,'size':256,'measured_batches':10,'warmup':5,
           'seconds_per_step_including_data':float(np.mean(times)),'full23220_train_seconds_estimate':float(np.mean(times)*23220),
           'estimate_excludes':'validation, native metrics, checkpoint I/O, thermal/OS contention','peak_allocated_bytes':torch.cuda.max_memory_allocated(),
           'loss_before':before,'loss_after':after,'process_RSS_bytes':psutil.Process().memory_info().rss,'available_RAM_bytes':psutil.virtual_memory().available}
        dataset.close();del model,opt;torch.cuda.empty_cache()
    else:results['GPU']={'available':False,'scope':'CPU smoke only; historical budget CPU run remains possible but not GPU-benchmarked'}
    clean.close();write(OUT/'smoke.json',results);print('SMOKE GATE PASSED',results,flush=True)

if __name__=='__main__':run()
