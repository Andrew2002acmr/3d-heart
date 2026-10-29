"""GT-free block/halo inference on ROI or full native grid; comparable CHD68 eval."""
import gc
import hashlib
import itertools
import json
import time
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from ct_lab.multiclass_core import *
from ct_lab.multiclass_metrics import dense_metrics
from ct_lab.multiclass_train import slug
from ct_lab.spatial_features import reflect_indices


def roi_bounds(shape,config):
    center=np.rint(np.array(shape)*config['center_fraction']).astype(int);size=np.minimum(config['size_xyz'],shape)
    lower=np.clip(center-size//2,0,np.array(shape)-size);return lower,lower+size


def blocks(lower,upper,block=(64,64,16)):
    for start in itertools.product(*[range(int(a),int(b),int(s)) for a,b,s in zip(lower,upper,block)]):
        yield np.array(start),np.minimum(np.array(start)+block,upper)


def dense_features(volume,start,stop,affine,names):
    spacing=check_affine(affine);r=max(3,*[int(8/s+.5) for s in spacing]);shape=np.asarray(stop)-start
    if r>24:raise ValueError('Bounded halo limit')
    indices=[reflect_indices(np.arange(a-r,b+r),n) for a,b,n in zip(start,stop,volume.shape)]
    patch=volume[np.ix_(*indices)].astype(np.float64);crop=tuple(slice(r,r+int(s)) for s in shape);center=patch[crop];bank={'Intensity':center}
    for prefix,axes in [('XY',(0,1)),('3D',(0,1,2))]:
        if not any(n.startswith(prefix+' ') for n in names):continue
        size=tuple(5 if a in axes else 1 for a in range(3));mean=ndi.uniform_filter(patch,size,mode='reflect')[crop]
        second=ndi.uniform_filter(patch*patch,size,mode='reflect')[crop]
        gm=np.zeros(tuple(shape));log=np.zeros(tuple(shape));sigma=tuple(1/spacing[a] if a in axes else 0 for a in range(3))
        for a in axes:
            plus=list(crop);minus=list(crop);plus[a]=slice(r+1,r+int(shape[a])+1);minus[a]=slice(r-1,r+int(shape[a])-1)
            p=patch[tuple(plus)];m=patch[tuple(minus)];g=(p-m)/(2*spacing[a])
            if start[a]==0:
                edge=[slice(None)]*3;edge[a]=0;g[tuple(edge)]=(p[tuple(edge)]-center[tuple(edge)])/spacing[a]
            if stop[a]==volume.shape[a]:
                edge=[slice(None)]*3;edge[a]=-1;g[tuple(edge)]=(center[tuple(edge)]-m[tuple(edge)])/spacing[a]
            gm+=g*g;order=[0,0,0];order[a]=2
            log+=ndi.gaussian_filter(patch,sigma,order=order,mode='reflect',truncate=4)[crop]/spacing[a]**2
        bank.update({prefix+' Mean':mean,prefix+' Std':np.sqrt(np.maximum(0,second-mean*mean)),prefix+' Gradient':np.sqrt(gm),prefix+' LoG':log})
    for s in (.5,1.,2.):
        if f'3D Gaussian {s:g}' in names:bank[f'3D Gaussian {s:g}']=ndi.gaussian_filter(patch,np.array([s,s,s])/spacing,mode='reflect',truncate=4)[crop]
    if any(n.startswith('Z') for n in names):
        minus=list(crop);plus=list(crop);minus[2]=slice(r-1,r+int(shape[2])-1);plus[2]=slice(r+1,r+int(shape[2])+1)
        zm=patch[tuple(minus)];zp=patch[tuple(plus)]
        bank.update({'Z I -1':zm,'Z I +1':zp,'Z1 Std':np.stack([zm,center,zp],-1).std(-1)})
    return np.column_stack([bank[n].ravel() for n in names]).astype(np.float32)


def inference_signature(model,scaler,names,lower,upper,shape):
    h=hashlib.sha256()
    for key,value in model.state_dict().items():h.update(key.encode());h.update(value.detach().cpu().numpy().tobytes())
    return {'model_weights_sha256':h.hexdigest(),'scaler':scaler,'features':list(names),'lower_xyz':np.asarray(lower).tolist(),
            'upper_xyz':np.asarray(upper).tolist(),'source_shape_xyz':list(shape)}


def cached_region(directory,model,scaler,names,lower,upper,shape):
    path=directory/'inference.json'
    if not path.exists():return None
    meta=json.loads(path.read_text())
    if meta.get('complete') and meta['signature']==inference_signature(model,scaler,names,lower,upper,shape):
        return np.load(directory/'prediction.npy',mmap_mode='r')
    raise ValueError('Cached dense inference signature differs')


def infer_region(volume,affine,model,scaler,names,lower,upper,directory,block=(64,64,16)):
    """No mask input. Probability maps are disk-backed, never dense feature tensor."""
    directory.mkdir(parents=True,exist_ok=True);shape=tuple(map(int,np.asarray(upper)-lower))
    probabilities=np.lib.format.open_memmap(directory/'probabilities.npy',mode='w+',dtype=np.float16,shape=(*shape,8))
    prediction=np.lib.format.open_memmap(directory/'prediction.npy',mode='w+',dtype=np.uint8,shape=shape)
    entropy=np.lib.format.open_memmap(directory/'entropy.npy',mode='w+',dtype=np.float32,shape=shape)
    for i,(start,stop) in enumerate(blocks(lower,upper,block)):
        values=dense_features(volume,start,stop,affine,names);prob,_=infer(model,normalize(values,scaler),batch=8192)
        bshape=tuple(stop-start);slices=tuple(slice(int(a),int(b)) for a,b in zip(start-lower,stop-lower))
        probabilities[slices]=prob.reshape(*bshape,8).astype(np.float16);prediction[slices]=prob.argmax(1).reshape(bshape)
        entropy[slices]=(-np.sum(prob*np.log(np.maximum(prob,1e-12)),1)).reshape(bshape)
        if i%128==0:print('DENSE BLOCK',directory.name,i,flush=True)
    probabilities.flush();prediction.flush();entropy.flush()
    write(directory/'inference.json',{'complete':True,'signature':inference_signature(model,scaler,names,lower,upper,volume.shape)})
    return prediction


def save_native(array,source,lower,path):
    affine=source.affine.copy();affine[:3,3]=nib.affines.apply_affine(source.affine,lower)
    header=source.header.copy();header.set_data_dtype(array.dtype);out=nib.Nifti1Image(np.asarray(array),affine,header)
    out.header.set_xyzt_units('unknown');nib.save(out,path)


def evaluate_region(directory,prediction,target,lower,upper,source,metadata):
    # GT first enters AFTER predictions have been computed without a target argument.
    slices=tuple(slice(int(a),int(b)) for a,b in zip(lower,upper));metric=dense_metrics(np.asarray(prediction),target[slices])
    metric.update(metadata);metric.update(lower_xyz=np.asarray(lower).tolist(),upper_xyz=np.asarray(upper).tolist(),ROI_GT_input=False,
                                        scope='full native volume' if np.array_equal(lower,[0,0,0]) and np.array_equal(upper,source.shape) else 'predeclared fixed ROI; boundary-truncated components')
    write(directory/'metrics.json',metric);save_native(prediction,source,lower,directory/'prediction.nii.gz')
    # Compact largest components for the interactive 3D view; counts above use ALL voxels/components.
    points=[]
    for c in range(1,8):
        cc,n=ndi.label(prediction==c,ndi.generate_binary_structure(3,1));sizes=np.bincount(cc.ravel())[1:]
        for component in np.argsort(sizes)[-10:]+1:
            ids=np.flatnonzero(cc.ravel()==component);pick=ids[np.linspace(0,len(ids)-1,min(300,len(ids))).astype(int)]
            xyz=np.column_stack(np.unravel_index(pick,prediction.shape))+lower
            points.append({'class':c,'component':int(component),'size':int(len(ids)),'xyz':xyz.tolist()})
    write(directory/'components.json',points);return metric


def chd_predict(volume,variant,checkpoint,lower,upper,directory,full=False):
    from heart3d.ml.infer_cardiac import load_recovered,prepare_volume
    from heart3d.ml.cardiac_data import restore_probability_plane
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    device='cuda' if torch.cuda.is_available() else 'cpu';model,saved,_,pre=load_recovered(variant,checkpoint,device=device)
    prepared,transform=prepare_volume(volume,pre);shape=tuple(map(int,upper-lower));prediction=np.empty(shape,np.uint8)
    probabilities=np.lib.format.open_memmap(directory/'probabilities.npy',mode='w+',dtype=np.float16,shape=(*shape,8))
    entropy=np.lib.format.open_memmap(directory/'entropy.npy',mode='w+',dtype=np.float32,shape=shape)
    sx,sy=slice(int(lower[0]),int(upper[0])),slice(int(lower[1]),int(upper[1]))
    for z in range(int(lower[2]),int(upper[2])):
        neighbors=np.clip(z+np.array(pre['context_offsets_indices']),0,volume.shape[2]-1)
        with torch.inference_mode():p=model(torch.from_numpy(prepared[neighbors].copy()).unsqueeze(0).to(device)).softmax(1).cpu().numpy()[0]
        restored=restore_probability_plane(p,transform)[:,sx,sy];restored/=np.maximum(restored.sum(0,keepdims=True),1e-12)
        q=np.moveaxis(restored,0,-1);prediction[:,:,z-lower[2]]=q.argmax(-1);probabilities[:,:,z-lower[2]]=q.astype(np.float16)
        entropy[:,:,z-lower[2]]=-np.sum(q*np.log(np.maximum(q,1e-12)),-1)
    probabilities.flush();entropy.flush();np.save(directory/'prediction.npy',prediction)
    del model,prepared;gc.collect()
    if device=='cuda':torch.cuda.empty_cache()
    return prediction


def run(chd_only=False):
    plan,records=freeze();seed_all(SEEDS[0]);summaries=[]
    for case in plan['dense_cases']:
        assert case in plan['train_cases']+plan['validation_cases'];source=nib.load(DEFAULT_DATA/f'{case}_image.nii.gz');volume=source.get_fdata(dtype=np.float32)
        target=cardiac_target(np.asanyarray(nib.load(DEFAULT_DATA/f'{case}_label.nii.gz').dataobj));lower,upper=roi_bounds(volume.shape,plan['dense_ROI'])
        for config in ([] if chd_only else plan['dense_models']):
            job={**config,'seed':plan['dense_seed']};name=slug(job);directory=OUT/'dense'/case/name;metric=directory/'metrics.json'
            if metric.exists():summaries.append(json.loads(metric.read_text()));continue
            model,saved=load_model(OUT/'models'/name/'best.pt');start=time.perf_counter()
            pred=cached_region(directory,model,saved['scaler'],saved['features'],lower,upper,volume.shape)
            if pred is None:pred=infer_region(volume,source.affine,model,saved['scaler'],saved['features'],lower,upper,directory,plan['dense_block_xyz'])
            summaries.append(evaluate_region(directory,pred,target,lower,upper,source,{'case':case,'model':name,'seconds':time.perf_counter()-start}))
            print('DENSE ROI COMPLETE',case,name,flush=True)
        for variant in ('v1','v2'):
            directory=OUT/'dense'/case/('CHD68_'+variant);directory.mkdir(parents=True,exist_ok=True);metric=directory/'metrics.json'
            if metric.exists():summaries.append(json.loads(metric.read_text()));continue
            checkpoint=Path(f'C:/Users/Andre/heart-model-recovery/chd68_{variant}_best.pt');start=time.perf_counter()
            pred=chd_predict(volume,variant,checkpoint,lower,upper,directory)
            summaries.append(evaluate_region(directory,pred,target,lower,upper,source,{'case':case,'model':'CHD68_'+variant,'seconds':time.perf_counter()-start,'checkpoint_sha256':fingerprint(checkpoint)}))
            print('CHD ROI COMPLETE',case,variant,flush=True)
        if case==plan['dense_full_volume_case']:
            lo=np.zeros(3,int);hi=np.array(volume.shape);job={**plan['dense_full_model'],'seed':plan['dense_seed']};name=slug(job)
            for key in (['CHD68_v1','CHD68_v2'] if chd_only else [name,'CHD68_v1','CHD68_v2']):
                directory=OUT/'dense_full'/case/key;metric=directory/'metrics.json'
                if metric.exists():summaries.append(json.loads(metric.read_text()));continue
                directory.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
                if key==name:
                    model,saved=load_model(OUT/'models'/name/'best.pt');pred=cached_region(directory,model,saved['scaler'],saved['features'],lo,hi,volume.shape)
                    if pred is None:pred=infer_region(volume,source.affine,model,saved['scaler'],saved['features'],lo,hi,directory,plan['dense_block_xyz'])
                else:
                    variant=key[-2:];pred=chd_predict(volume,variant,Path(f'C:/Users/Andre/heart-model-recovery/chd68_{variant}_best.pt'),lo,hi,directory,True)
                summaries.append(evaluate_region(directory,pred,target,lo,hi,source,{'case':case,'model':key,'seconds':time.perf_counter()-start}))
                print('DENSE FULL COMPLETE',key,flush=True)
        del volume,target;gc.collect()
    write(OUT/'dense_summary.json',summaries)


if __name__=='__main__':
    import sys
    run(chd_only='--chd-only' in sys.argv)
