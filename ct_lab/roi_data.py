"""Aligned 2.5D crop/cache and paired augmentation; GT never chooses predicted ROI."""
from collections import OrderedDict
import json
import nibabel as nib
import numpy as np
import torch
from ct_lab.roi_core import *
from ct_lab.roi_localizer import load_localizer,locate


def case_box(case,mode,context='all_train',oracle_diagnostic=False):
    plan=json.loads((OUT/'protocol.json').read_text())
    if case not in plan['train_cases']+plan['validation_cases']:raise ValueError('Sealed or unknown case')
    with np.load(OUT/'coarse'/f'{case}.npz') as f:shape=f['shape_xyz'];coarse=f['image']
    if mode=='full':return np.array([[0,0,0],shape],int)
    if mode=='oracle':
        if case not in plan['train_cases'] and not oracle_diagnostic:raise ValueError('Outer validation oracle needs explicit diagnostic flag')
        target=cardiac_target(np.asanyarray(nib.load(DEFAULT_DATA/f'{case}_label.nii.gz').dataobj))
        return expand_box(bbox((target>0)&(target<8)),shape,json.loads((OUT/'selection.json').read_text())['margin'],8)
    if mode!='predicted':raise ValueError('Unknown ROI mode')
    if context=='all_train':return np.array(json.loads((OUT/'predicted_boxes.json').read_text())[case]['bbox'],int)
    path=OUT/'box_cache'/f'{context}_{case}.json'
    if path.exists():return np.array(json.loads(path.read_text())['bbox'],int)
    model,saved=load_localizer(OUT/'localizers'/context/'best.pt')
    # Conservative predeclared max margin in segmentation CV; no heldout GT margin tuning.
    result,_=locate(coarse,shape,model,saved['fixed_roi'],.3,8);write(path,result);return np.array(result['bbox'],int)


def resize_stack(volume,indices,t,mask=False):
    b=np.array(t['bbox_xyz']);rx,ry=t['resize']['resized_xy'];px,py=t['resize']['pad_xy'];n=t['size']
    crop=volume[slice(b[0,0],b[1,0]),slice(b[0,1],b[1,1]),indices]
    planes=np.ascontiguousarray(crop.transpose(2,1,0).astype(np.float32));tensor=torch.from_numpy(planes)[:,None]
    resized=torch.nn.functional.interpolate(tensor,size=(ry,rx),mode='nearest-exact' if mask else 'bilinear',**({} if mask else {'align_corners':False}))[:,0].numpy()
    output=np.full((len(indices),n,n),255 if mask else -1,dtype=np.uint8 if mask else np.float32)
    if not mask:resized=np.clip(resized,0,2015)/2015*2-1
    output[:,py:py+ry,px:px+rx]=resized.astype(output.dtype);return output


def centers_for(case):
    info=json.loads((OUT/'coarse'/f'{case}.json').read_text());b=np.array(info['GT_bbox']);nz=info['shape_xyz'][2]
    positive=np.linspace(b[0,2],b[1,2]-1,12).round().astype(int)
    outside=np.array([z for z in np.linspace(0,nz-1,12).round().astype(int) if not b[0,2]<=z<b[1,2]])
    extra=outside[np.linspace(0,len(outside)-1,min(4,len(outside))).round().astype(int)] if len(outside) else np.array([],int)
    return np.array(sorted(set(positive.tolist()+extra.tolist())),int)


def build_case(case,mode,size,context='all_train',branch=False):
    if case not in json.loads((OUT/'protocol.json').read_text())['train_cases']:raise ValueError('Training cache must be train only')
    directory=OUT/'cache'/('shared_full' if mode=='full' else context)/f'{mode}_{size}';path=directory/f'{case}.npz';info=path.with_suffix('.json')
    if path.exists() and info.exists():
        metadata=json.loads(info.read_text())
        if not branch or metadata.get('branch_ready'):return path
    source=nib.load(DEFAULT_DATA/f'{case}_image.nii.gz');volume=source.get_fdata(dtype=np.float32)
    target=cardiac_target(np.asanyarray(nib.load(DEFAULT_DATA/f'{case}_label.nii.gz').dataobj));box=case_box(case,mode,context);t=transform(volume.shape,box,size)
    centers=centers_for(case);offsets=np.array([-2,-1,0,1,2]);wanted=np.clip(centers[:,None]+offsets,0,volume.shape[2]-1)
    unique=np.unique(wanted);planes=resize_stack(volume,unique,t);channels=np.searchsorted(unique,wanted);masks=resize_stack(target,centers,t,mask=True)
    # One deterministic image-space intensity/noise realization per case, shared by every mode/resolution.
    rng=np.random.default_rng(SEED+int(case.split('_')[1])*17);aug=volume.astype(np.float32,order='C',copy=True)
    cfg=json.loads((OUT/'protocol.json').read_text())['segmentation']['augmentation'];shift=rng.uniform(-cfg['intensity_shift'],cfg['intensity_shift'])*2015/2
    flat=aug.ravel()
    for start in range(0,len(flat),1_000_000):
        stop=min(len(flat),start+1_000_000);flat[start:stop]+=rng.normal(shift,cfg['noise_std']*2015/2,stop-start).astype(np.float32)
    aug_planes=resize_stack(aug,unique,t)
    data={'images':planes.astype(np.float16),'images_aug':aug_planes.astype(np.float16),'context_indices':channels,'targets':masks,'centers_source_z':centers}
    if branch:
        from ct_lab.multiclass_dense import dense_features
        os.environ['NUMBA_CACHE_DIR']=str(OUT/'numba_cache');names=['3D Mean','3D Std','3D Gradient','3D LoG'];core=[]
        for condition,source_volume in [('core',volume),('core_aug',aug)]:
            core=[]
            for z in centers:
                lo=np.array([box[0,0],box[0,1],z]);hi=np.array([box[1,0],box[1,1],z+1]);shape=tuple(hi-lo)
                values=dense_features(source_volume,lo,hi,source.affine,names).reshape(*shape,4)[:,:,0]
                features=[]
                for channel in range(4):
                    plane=values[:,:,channel]/(2015 if channel==0 else 512)
                    if channel==0:plane=plane*2-1
                    resized=resize_plane(plane,t['resize'],1,-1 if channel==0 else 0);features.append(resized)
                core.append(np.stack(features))
            data[condition]=np.stack(core).astype(np.float16)
    directory.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**data)
    write(info,{'case':case,'mode':mode,'context':context,'size':size,'transform':t,'branch_ready':branch,'GT_ROI':mode=='oracle',
                'source_z_centers':centers.tolist(),'image_sha256':fingerprint(DEFAULT_DATA/f'{case}_image.nii.gz'),
                'intensity_augmentation':'one fixed source CT shift/noise per case, shared across configurations; 50% clean/aug samples; native core recomputed before joint spatial warp'})
    return path


class TrainingData:
    def __init__(self,cases,mode,size,context='all_train',branch=False):
        plan=json.loads((OUT/'protocol.json').read_text())
        if not set(cases)<=set(plan['train_cases']):raise ValueError('Only train cases may enter optimization')
        if context.startswith('fold') and set(cases)&set(plan['CV_folds'][int(context[4:])]):raise ValueError('Heldout CV case in training loader')
        self.cases=list(cases);self.mode=mode;self.size=size;self.context=context;self.branch=branch;self.cache=OrderedDict()
        from concurrent.futures import ThreadPoolExecutor,as_completed
        from threadpoolctl import threadpool_limits
        self.paths={}
        with threadpool_limits(limits=1),ThreadPoolExecutor(max_workers=2) as executor:
            jobs={executor.submit(build_case,case,mode,size,context,branch):case for case in cases}
            for future in as_completed(jobs):
                case=jobs[future];self.paths[case]=future.result();print('ROI CACHE',context,mode,size,case,flush=True)
    def arrays(self,case):
        if case not in self.cache:
            with np.load(self.paths[case]) as f:self.cache[case]={k:f[k] for k in f.files}
            while len(self.cache)>1 and sum(v.nbytes for item in self.cache.values() for v in item.values())>2*1024**3:self.cache.popitem(last=False)
        self.cache.move_to_end(case);return self.cache[case]
    def batch(self,case_ids,rng):
        images=[];targets=[];cores=[]
        for case in case_ids:
            f=self.arrays(case);i=int(rng.integers(len(f['targets'])));augmented=rng.random()<.5
            images.append(f['images_aug' if augmented else 'images'][f['context_indices'][i]].astype(np.float32));targets.append(f['targets'][i].astype(np.int64))
            if self.branch:cores.append(f['core_aug' if augmented else 'core'][i].astype(np.float32))
        return torch.from_numpy(np.stack(images)),torch.from_numpy(np.stack(targets)),torch.from_numpy(np.stack(cores)) if self.branch else None


def augment_seg(image,target,core,rng,cfg):
    b=len(image);theta=np.zeros((b,2,3),np.float32)
    for i in range(b):
        angle=np.deg2rad(rng.uniform(-cfg['rotation_degrees'],cfg['rotation_degrees']));scale=rng.uniform(1-cfg['scale_delta'],1+cfg['scale_delta'])
        theta[i,:,:2]=[[np.cos(angle)*scale,-np.sin(angle)*scale],[np.sin(angle)*scale,np.cos(angle)*scale]];theta[i,:,2]=rng.uniform(-cfg['shift_fraction'],cfg['shift_fraction'],2)*2
    grid=torch.nn.functional.affine_grid(torch.from_numpy(theta).to(image.device),image.shape,align_corners=False)
    image=torch.nn.functional.grid_sample(image+1,grid,mode='bilinear',align_corners=False)-1
    y=torch.nn.functional.grid_sample(target[:,None].float()+1,grid,mode='nearest',align_corners=False)[:,0]-1;target=torch.where(y<0,255,y).long()
    if core is not None:
        core=torch.nn.functional.grid_sample(core,grid,mode='bilinear',align_corners=False)
    return image,target,core
