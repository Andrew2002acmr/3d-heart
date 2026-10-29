"""GT-free native-grid prediction followed by common eight-class source evaluation."""
import gc
import hashlib
import json
import time
import numpy as np
import nibabel as nib
import torch
from ct_lab.roi_core import *
from ct_lab.roi_data import case_box,resize_stack
from ct_lab.roi_train import load_segmenter
from heart3d.ml.cardiac_data import confusion_counts,scores_from_confusion
from heart3d.ml.infer_cardiac import save_prediction


def core_volume(volume,affine,box,t,directory):
    """Exact native 3D-core with bounded Z halos, then aligned XY resize."""
    from filelock import FileLock
    directory.mkdir(parents=True,exist_ok=True)
    # Atomic completion + source/geometry key prevents accepting an interrupted cache.
    digest=hashlib.sha256()
    for z in range(0,volume.shape[2],8):digest.update(np.ascontiguousarray(volume[:,:,z:z+8]).view(np.uint8))
    expected={'format':2,'source_float32_SHA256':digest.hexdigest(),'shape':list(volume.shape),'affine':np.asarray(affine).tolist(),'box':np.asarray(box).tolist(),'transform':t}
    with FileLock(str(directory/'core_input.lock'),timeout=1800):
        path=directory/'core_input.npy';receipt=directory/'core_input.json'
        if path.exists() and receipt.exists() and json.loads(receipt.read_text())==expected:return np.load(path,mmap_mode='r')
        return _write_core_volume(volume,affine,box,t,directory,expected)


def _write_core_volume(volume,affine,box,t,directory,expected):
    from ct_lab.multiclass_dense import dense_features
    os.environ['NUMBA_CACHE_DIR']=str(OUT/'numba_cache')
    names=['3D Mean','3D Std','3D Gradient','3D LoG'];count=int(box[1,2]-box[0,2]);n=t['size'];directory.mkdir(parents=True,exist_ok=True)
    path=directory/'core_input.partial.npy';data=np.lib.format.open_memmap(path,mode='w+',dtype=np.float16,shape=(count,4,n,n));rx,ry=t['resize']['resized_xy'];px,py=t['resize']['pad_xy']
    data[:]=0;data[:,0]=-1
    for z in range(int(box[0,2]),int(box[1,2]),8):
        lo=np.array([box[0,0],box[0,1],z]);hi=np.array([box[1,0],box[1,1],min(z+8,box[1,2])]);shape=tuple(hi-lo)
        values=dense_features(volume,lo,hi,affine,names).reshape(*shape,4).transpose(2,3,1,0).copy()
        values[:,0]=values[:,0]/2015*2-1;values[:,1:]/=512
        resized=torch.nn.functional.interpolate(torch.from_numpy(values),size=(ry,rx),mode='bilinear',align_corners=False).numpy()
        data[z-box[0,2]:hi[2]-box[0,2],:,py:py+ry,px:px+rx]=resized.astype(np.float16)
    data.flush();del data
    complete=directory/'core_input.npy';os.replace(path,complete);write(directory/'core_input.json',expected)
    return np.load(complete,mmap_mode='r')


def predict_with_box(volume,affine,model,job,box,device='cpu',core_directory=None):
    """Only CT, affine, model and independently obtained box; no GT argument."""
    t=transform(volume.shape,box,job['size']);centers=np.arange(box[0,2],box[1,2]);neighbors=np.clip(centers[:,None]+np.array([-2,-1,0,1,2]),0,volume.shape[2]-1)
    unique=np.unique(neighbors);images=resize_stack(volume,unique,t).astype(np.float16);context=np.searchsorted(unique,neighbors)
    features=core_volume(volume,affine,box,t,core_directory) if job['branch'] else None
    prediction=np.zeros(volume.shape,np.uint8);model.eval();batch=2
    with torch.inference_mode():
        for start in range(0,len(centers),batch):
            take=slice(start,start+batch);x=torch.from_numpy(images[context[take]].astype(np.float32)).to(device)
            aux=torch.from_numpy(np.array(features[take],np.float32,copy=True)).to(device) if features is not None else None
            with torch.autocast(device_type=device,dtype=torch.float16,enabled=device=='cuda'):logits=model(x,aux)
            probability=logits.float().softmax(1)
            for i,z in enumerate(centers[take]):
                restored=restore_plane(probability[i],t,device);labels=restored.argmax(0).cpu().numpy().astype(np.uint8)
                prediction[slice(box[0,0],box[1,0]),slice(box[0,1],box[1,1]),z]=labels
    return prediction,t


def evaluate_model(result,cases,plan,topology=False,save_predictions=False):
    directory=OUT/'evaluation'/result['context']/result['job']['name'];receipt=directory/'summary.json'
    if receipt.exists():
        previous=json.loads(receipt.read_text())
        if previous.get('checkpoint_sha256',previous['records'][0]['checkpoint_sha256'])==result['checkpoint_sha256']:return previous
        write(directory/'pilot_previous_summary.json',previous)
    device='cuda' if torch.cuda.is_available() else 'cpu';model,saved=load_segmenter(OUT/result['checkpoint'],device);records=[]
    for case in cases:
        destination=directory/case;metric=destination/'scores.json'
        if metric.exists():
            previous=json.loads(metric.read_text())
            if previous['checkpoint_sha256']==result['checkpoint_sha256']:records.append(previous);continue
        source=nib.load(DEFAULT_DATA/f'{case}_image.nii.gz');volume=source.get_fdata(dtype=np.float32);job=result['job'];start=time.perf_counter()
        box=case_box(case,job['mode'],result['context'],oracle_diagnostic=job['mode']=='oracle')
        # Expert target is loaded for scoring after automatic prediction; only explicit oracle mode uses GT bbox.
        prediction,t=predict_with_box(volume,source.affine,model,job,box,device,OUT/'eval_core'/case/str(job['size']))
        target=cardiac_target(np.asanyarray(nib.load(DEFAULT_DATA/f'{case}_label.nii.gz').dataobj));cm=confusion_counts(prediction,target);score=scores_from_confusion(cm)
        info={'case':case,'job':job,'context':result['context'],'oracle_GT_ROI':job['mode']=='oracle','transform':t,'seconds':time.perf_counter()-start,
              'confusion_matrix':cm.tolist(),**score,'coverage':coverage(target,box),'checkpoint_sha256':result['checkpoint_sha256'],
              'source_image_sha256':fingerprint(DEFAULT_DATA/f'{case}_image.nii.gz'),'scope':'source native grid; reference-present foreground macro Dice'}
        destination.mkdir(parents=True,exist_ok=True)
        if save_predictions:
            np.save(destination/'prediction.npy',prediction);save_prediction(prediction,source,destination/'prediction.nii.gz')
        write(metric,info);records.append(info);print('SOURCE EVAL',result['context'],job['name'],case,info['macro_foreground_Dice'],flush=True)
        del volume,prediction,target;gc.collect()
    values=[r['macro_foreground_Dice'] for r in records];summary={'records':records,'mean_case_foreground_Dice':float(np.mean(values)),
              'case_SD':float(np.std(values)),'cases':list(cases),'oracle_GT_ROI':result['job']['mode']=='oracle','context':result['context'],'checkpoint_sha256':result['checkpoint_sha256']}
    write(receipt,summary);del model;gc.collect()
    if device=='cuda':torch.cuda.empty_cache()
    return summary
