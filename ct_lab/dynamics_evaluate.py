"""One8-class evaluation adapter: prepared and native grids from same probabilities."""
import gc,time
import nibabel as nib
from torch.nn import functional as F
from ct_lab.dynamics_core import *
from heart3d.ml.model import HeartUNet25D
from heart3d.ml.cardiac_data import confusion_counts,scores_from_confusion,index_neighbors,resize_plane,index_transform
from heart3d.ml.infer_cardiac import save_prediction,prepare_volume

def load_checkpoint(path,device='cpu'):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    split=saved.get('split_SHA256',saved.get('split_sha256'))
    if split!=fingerprint(REPO/'configs/splits/cardiac_chd68_v1.json'):raise ValueError('Checkpoint split mismatch')
    if saved.get('format')=='roi_segmenter05':
        if saved['job']['mode']!='full' or saved['job']['branch']:raise ValueError('Stage06 comparisons require plain Full FOV model')
        from ct_lab.roi_models import ROISegmenter
        os.environ['NUMBA_CACHE_DIR']=str(OUT/'numba_cache');model=ROISegmenter(False);pre=json.loads((REPO/'configs/cardiac_chd68_preprocessing_v1.json').read_text())
        pre['input_size']=saved['job']['size']
    else:
        model=HeartUNet25D(**saved['config']['model']);pre=saved.get('preprocessing') or json.loads((REPO/saved['config']['preprocessing']).read_text())
    model.load_state_dict(saved['model'],strict=True);model.to(device).eval();return model,saved,pre

def predict_batch_native(model,contexts,t,device='cpu'):
    """CT-only operation. Always FP32, inverse probabilities BEFORE argmax."""
    with torch.no_grad():prob=model(torch.from_numpy(contexts).to(device)).float().softmax(1)
    prepared=prob.argmax(1).cpu().numpy().astype(np.uint8)
    rx,ry=t['resized_xy'];px,py=t['pad_xy'];nx,ny,_=t['original_shape_xyz']
    native=F.interpolate(prob[:,:,py:py+ry,px:px+rx],size=(ny,nx),mode='bilinear',align_corners=False).argmax(1).cpu().numpy().astype(np.uint8)
    return prepared,native

def evaluate(path,cases,directory,batch_size=16,save_masks=True):
    plan=protocol();permitted=set(plan['train_cases']+plan['validation_cases'])
    if not set(cases)<=permitted:raise ValueError('Sealed test/unknown evaluation case')
    device='cuda' if torch.cuda.is_available() else 'cpu';model,saved,pre=load_checkpoint(path,device);sha=fingerprint(path);records=[]
    for case in cases:
        dest=directory/case;receipt=dest/'metrics.json'
        if receipt.exists():
            previous=json.loads(receipt.read_text())
            if previous['checkpoint_SHA256']==sha:records.append(previous);continue
        source=nib.load(DATA/f'{case}_image.nii.gz');target=cardiac_target(np.asanyarray(nib.load(DATA/f'{case}_label.nii.gz').dataobj));t=index_transform(source.shape,pre['input_size'])
        if pre['input_size']==256:
            image=np.load(OUT/'prepared256'/case/'image.npy',mmap_mode='r');prepared_target=np.load(OUT/'prepared256'/case/'mask.npy',mmap_mode='r')
        else:
            image,t=prepare_volume(source.get_fdata(dtype=np.float32),pre)
            prepared_target=np.stack([resize_plane(target[:,:,z],t,0,255).astype(np.uint8) for z in range(source.shape[2])])
        prediction=np.zeros(source.shape,np.uint8);cm=np.zeros((8,8),np.int64);pm=cm.copy();tick=time.perf_counter()
        for start in range(0,len(image),batch_size):
            centers=np.arange(start,min(start+batch_size,len(image)));neighbors=np.stack([index_neighbors(int(z),len(image),pre['context_offsets_indices'])[0] for z in centers])
            prepared,native=predict_batch_native(model,np.array(image[neighbors],np.float32),t,device)
            pm+=confusion_counts(prepared,prepared_target[centers]);cm+=confusion_counts(native,target[:,:,centers].transpose(2,1,0));prediction[:,:,centers]=native.transpose(2,1,0)
        role='development' if case in plan['validation_cases'] else 'train_monitor' if case in plan['config']['stage06']['train_monitor'] else 'train_example'
        r={'case':case,'role':role,'checkpoint_SHA256':sha,'prepared_grid':scores_from_confusion(pm),'prepared_confusion':pm.tolist(),'native_confusion':cm.tolist(),
           **scores_from_confusion(cm),'transform':t,'inference':'common FP32 from unquantized historical prepared inputs; no AMP; inverse probabilities then argmax',
           'foreground_recall':float(np.mean([v['recall'] for v in scores_from_confusion(cm)['classes'].values() if v['target_voxels']])),
           'BG_to_foreground_voxels':int(cm[0,1:].sum()),'seconds':time.perf_counter()-tick,'components_removed':False}
        dest.mkdir(parents=True,exist_ok=True)
        if save_masks:save_prediction(prediction,source,dest/'prediction.nii.gz')
        write(receipt,r);records.append(r);print('DYNAMICS NATIVE',directory.name,case,r['macro_foreground_Dice'],flush=True)
        del image,prepared_target,target,prediction;gc.collect()
    summary={'checkpoint_SHA256':sha,'records':records,'scope':'roles always separate; no11-case train+validation mean is a development metric',
       'by_role':{role:{'case_ids':[r['case'] for r in records if r['role']==role],
                       'native_Dice':float(np.mean([r['macro_foreground_Dice'] for r in records if r['role']==role])),
                       'prepared_Dice':float(np.mean([r['prepared_grid']['macro_foreground_Dice'] for r in records if r['role']==role]))}
                  for role in ('development','train_monitor','train_example') if any(r['role']==role for r in records)}}
    write(directory/'summary.json',summary);del model
    if device=='cuda':torch.cuda.empty_cache()
    return summary
