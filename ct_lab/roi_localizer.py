"""Train-only group OOF heart localization, coverage-driven margin, no GT inference."""
import json
import time
import nibabel as nib
import numpy as np
import torch
from scipy import ndimage as ndi
from ct_lab.roi_core import *
from ct_lab.roi_models import HeartLocalizer,localizer_loss


def fixed_roi(train_cases):
    bounds=np.array([json.loads((OUT/'coarse'/f'{c}.json').read_text())['normalized_bbox'] for c in train_cases])
    return np.stack([np.maximum(0,np.quantile(bounds[:,0],.02,axis=0)-.05),np.minimum(1,np.quantile(bounds[:,1],.98,axis=0)+.05)]).tolist()


def fixed_box(normalized,shape):return clip_box(np.stack([np.floor(np.array(normalized)[0]*shape),np.ceil(np.array(normalized)[1]*shape)]),shape)


def augment_batch(image,target,rng,config):
    b=len(image);theta=np.zeros((b,3,4),np.float32)
    for i in range(b):
        angle=np.deg2rad(rng.uniform(-config['axial_rotation_degrees'],config['axial_rotation_degrees']));scale=rng.uniform(1-config['scale_delta'],1+config['scale_delta'])
        # Tensor spatial axes are native XYZ; affine_grid coordinate components are Z,Y,X.
        rotation=np.array([[1,0,0],[0,np.cos(angle),-np.sin(angle)],[0,np.sin(angle),np.cos(angle)]],np.float32)
        theta[i,:,:3]=rotation*scale;theta[i,:,3]=rng.uniform(-config['translation_fraction'],config['translation_fraction'],3)*2
    grid=torch.nn.functional.affine_grid(torch.from_numpy(theta).to(image.device),image.shape,align_corners=False)
    image=torch.nn.functional.grid_sample(image+1,grid,mode='bilinear',padding_mode='zeros',align_corners=False)-1
    target=torch.nn.functional.grid_sample(target[:,None].float()+1,grid,mode='nearest',padding_mode='zeros',align_corners=False)[:,0]-1
    target=torch.where(target<0,255,target).to(torch.uint8)
    image=image+float(rng.uniform(-config['intensity_shift'],config['intensity_shift']))+torch.randn_like(image)*config['noise_std']
    return image,target


def arrays(cases):
    images=[];targets=[]
    for case in cases:
        with np.load(OUT/'coarse'/f'{case}.npz') as f:images.append(f['image']);targets.append(f['target'])
    return torch.from_numpy(np.stack(images)[:,None]),torch.from_numpy(np.stack(targets))


def binary_dice(p,t):
    valid=t!=255;truth=t==1;pred=(p>.4)&valid
    return float((2*(pred&truth).sum()+1e-6)/(pred.sum()+truth.sum()+1e-6))


def train(train_cases,heldout,plan,name,epochs):
    directory=OUT/'localizers'/name;receipt=directory/'result.json'
    if receipt.exists():return json.loads(receipt.read_text())
    if not (OUT/'smoke.json').exists():raise ValueError('CPU/GPU resource gate missing')
    device='cuda' if torch.cuda.is_available() else 'cpu';seed_runtime(SEED,1,device);cfg=plan['localizer'];model=HeartLocalizer().to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr']);scaler=torch.amp.GradScaler(device,enabled=device=='cuda');images,targets=arrays(train_cases)
    vx,vy=arrays(heldout) if heldout else (None,None);history=[];best=-1;start_epoch=0
    last=directory/'last.pt'
    if last.exists():
        saved=torch.load(last,map_location=device,weights_only=True);model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
        history=saved['history'];start_epoch=saved['epoch'];best=saved['best']
    start=time.perf_counter();directory.mkdir(parents=True,exist_ok=True)
    for epoch in range(start_epoch,epochs):
        model.train();rng=np.random.default_rng(SEED+epoch*100003);order=rng.permutation(len(images));losses=[]
        for step in range(0,len(order),cfg['batch_size']):
            torch.manual_seed(SEED+epoch*100003+step);index=order[step:step+cfg['batch_size']]
            x=images[index].to(device);y=targets[index].to(device);x,y=augment_batch(x,y,rng,cfg['augmentation'])
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device,dtype=torch.float16,enabled=device=='cuda'):logits=model(x)
            loss=localizer_loss(logits,y);assert torch.isfinite(loss);scaler.scale(loss).backward();scaler.step(opt);scaler.update();losses.append(float(loss.detach()))
        score=None
        if heldout:
            model.eval();values=[]
            with torch.inference_mode():
                for i in range(len(vx)):
                    p=model(vx[i:i+1].to(device)).sigmoid()[0,0].cpu().numpy();values.append(binary_dice(p,vy[i].numpy()))
            score=float(np.mean(values))
        history.append({'epoch':epoch+1,'train_loss':float(np.mean(losses)),'heldout_binary_Dice':score})
        saved={'format':'heart_localizer05','model':model.state_dict(),'train_cases':train_cases,'heldout_cases':heldout,'epoch':epoch+1,'seed':SEED,
               'split_sha256':plan['split_sha256'],'config':cfg,'fixed_roi':fixed_roi(train_cases)}
        if not heldout or score>best:
            best=score if heldout else -1;torch.save(saved,directory/'best.pt')
        torch.save({**saved,'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'history':history,'best':best},last)
        write(directory/'history.json',history)
        print('LOCALIZER',name,epoch+1,history[-1],flush=True)
    selected=torch.load(directory/'best.pt',map_location='cpu',weights_only=True)
    result={'name':name,'train_cases':train_cases,'heldout_cases':heldout,'best_epoch':selected['epoch'],'epochs':epochs,'seconds':time.perf_counter()-start,
            'checkpoint':str((directory/'best.pt').relative_to(OUT)),'checkpoint_sha256':fingerprint(directory/'best.pt'),'parameters':sum(p.numel() for p in model.parameters())}
    write(receipt,result);return result


def load_localizer(path):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved.get('format')!='heart_localizer05' or saved['split_sha256']!=fingerprint(REPO/'configs/splits/cardiac_chd68_v1.json'):raise ValueError('Localizer provenance differs')
    plan=json.loads((OUT/'protocol.json').read_text())
    if not set(saved['train_cases'])<=set(plan['train_cases']) or set(saved['train_cases'])&set(saved['heldout_cases']):raise ValueError('Localizer leakage')
    model=HeartLocalizer();model.load_state_dict(saved['model'],strict=True);model.eval();return model,saved


def locate(coarse,shape,model,fixed_normalized,margin=.2,extra=8):
    """Image-only inference API; neither mask nor target is accepted."""
    with torch.inference_mode():p=model(torch.from_numpy(np.asarray(coarse,np.float32))[None,None]).sigmoid()[0,0].numpy()
    cc,n=ndi.label(p>.4,ndi.generate_binary_structure(3,1));counts=np.bincount(cc.ravel());counts[0]=0
    if n:foreground=cc==int(counts.argmax());confidence=float(p[foreground].mean())
    else:foreground=np.zeros(p.shape,bool);confidence=0.
    small=bbox(foreground);fallback=small is None or confidence<.5
    if fallback:raw=fixed_box(fixed_normalized,np.array(shape))
    else:raw=clip_box(np.stack([np.floor(small[0]/p.shape*np.array(shape)),np.ceil(small[1]/p.shape*np.array(shape))]),shape)
    box=expand_box(raw,shape,margin,extra)
    return {'raw_bbox':raw.tolist(),'bbox':box.tolist(),'confidence':confidence,'confidence_calibrated':False,'fallback_fixed_ROI':fallback,
            'margin_fraction':margin,'extra_margin_indices':extra,'source_shape_xyz':np.asarray(shape).tolist(),'coarse_shape':list(p.shape),'GT_input':False},p


def run():
    plan,records=freeze();results=[];oof=[]
    for fold,heldout in enumerate(plan['CV_folds']):
        train_cases=[c for c in plan['train_cases'] if c not in heldout]
        result=train(train_cases,heldout,plan,f'fold{fold}',plan['localizer']['epochs']);results.append(result)
        model,saved=load_localizer(OUT/result['checkpoint'])
        for case in heldout:
            with np.load(OUT/'coarse'/f'{case}.npz') as f:
                for margin in plan['localizer']['margin_candidates']:
                    prediction,_=locate(f['image'],f['shape_xyz'],model,saved['fixed_roi'],margin,8)
                    target=cardiac_target(np.asanyarray(nib.load(DEFAULT_DATA/f'{case}_label.nii.gz').dataobj))
                    oof.append({'case':case,'fold':fold,'margin':margin,'prediction':prediction,'metrics':coverage(target,prediction['bbox'])})
    chosen=.3
    for margin in plan['localizer']['margin_candidates']:
        selected=[r for r in oof if r['margin']==margin];mean=np.mean([r['metrics']['foreground_coverage'] for r in selected]);per=[]
        for c in range(1,8):per.append(np.mean([r['metrics']['per_class_coverage'][str(c)] for r in selected if r['metrics']['per_class_coverage'][str(c)] is not None]))
        if mean>=.995 and min(per)>=.98:chosen=margin;break
    epochs=int(np.median([r['best_epoch'] for r in results]));selection={'margin':chosen,'final_localizer_epochs':epochs,'selection_scope':'train OOF only','fixed_roi':fixed_roi(plan['train_cases'])}
    write(OUT/'localization_OOF.json',oof);write(OUT/'selection.json',selection)
    final=train(plan['train_cases'],[],plan,'all_train',epochs);model,saved=load_localizer(OUT/final['checkpoint']);predictions={};evaluations=[]
    for role in ('train','validation'):
        for case in plan[role+'_cases']:
            with np.load(OUT/'coarse'/f'{case}.npz') as f:prediction,p=locate(f['image'],f['shape_xyz'],model,saved['fixed_roi'],chosen,8);shape=f['shape_xyz']
            predictions[case]=prediction;np.save(OUT/'localizers'/f'{case}_probability.npy',p.astype(np.float16))
            target=cardiac_target(np.asanyarray(nib.load(DEFAULT_DATA/f'{case}_label.nii.gz').dataobj))
            evaluations.append({'case':case,'role':role,'method':'predicted',**coverage(target,prediction['bbox'])})
            evaluations.append({'case':case,'role':role,'method':'fixed',**coverage(target,fixed_box(saved['fixed_roi'],shape))})
    write(OUT/'predicted_boxes.json',predictions);write(OUT/'localization_results.json',{'CV':results,'final':final,'selection':selection,'metrics':evaluations})
    print('LOCALIZATION COMPLETE margin',chosen,'epochs',epochs,flush=True)


if __name__=='__main__':run()
