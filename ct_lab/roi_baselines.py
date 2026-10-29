"""Immutable recovered CHD68 source-grid references on the permitted cases."""
import gc
import json
import shutil
import time
import numpy as np
import nibabel as nib
import torch
from ct_lab.roi_core import *
from heart3d.ml.infer_cardiac import load_recovered,prepare_volume,save_prediction
from heart3d.ml.cardiac_data import confusion_counts,scores_from_confusion


def run():
    plan,records=freeze();device='cuda' if torch.cuda.is_available() else 'cpu';seed_runtime(SEED,1,device);results=[]
    for variant in ('v1','v2'):
        checkpoint=checkpoint_path(variant);model,_,_,pre=load_recovered(variant,checkpoint,device)
        for case in plan['evaluation_cases']:
            if case not in plan['train_cases']+plan['validation_cases']:raise ValueError('Sealed case')
            directory=OUT/'evaluation'/'all_train'/('CHD68_'+variant)/case;scorefile=directory/'scores.json'
            if scorefile.exists():results.append(json.loads(scorefile.read_text()));continue
            source_path=DEFAULT_DATA/f'{case}_image.nii.gz';assert fingerprint(source_path)==records[case]['image_SHA256'];source=nib.load(source_path);start=time.perf_counter()
            old=REPO/'outputs/multiclass_feature_04/dense_full'/case/('CHD68_'+variant)/'prediction.npy'
            if old.exists():pred=np.array(np.load(old,mmap_mode='r'),copy=True)
            else:
                volume=source.get_fdata(dtype=np.float32);prepared,resize=prepare_volume(volume,pre);t={'bbox_xyz':[[0,0,0],list(volume.shape)],'resize':resize}
                pred=np.empty(volume.shape,np.uint8);model.eval()
                with torch.inference_mode():
                    for z in range(0,volume.shape[2],2):
                        centers=list(range(z,min(z+2,volume.shape[2])));indices=np.stack([np.clip(c+np.array([-2,-1,0,1,2]),0,volume.shape[2]-1) for c in centers])
                        logits=model(torch.from_numpy(prepared[indices].copy()).to(device));p=logits.softmax(1)
                        for i,c in enumerate(centers):pred[:,:,c]=restore_plane(p[i],t,device).argmax(0).cpu().numpy().astype(np.uint8)
                del volume,prepared
            directory.mkdir(parents=True,exist_ok=True);np.save(directory/'prediction.npy',pred);save_prediction(pred,source,directory/'prediction.nii.gz')
            target=cardiac_target(np.asanyarray(nib.load(DEFAULT_DATA/f'{case}_label.nii.gz').dataobj));cm=confusion_counts(pred,target)
            score={'case':case,'method':'CHD68_'+variant,'oracle_GT_ROI':False,'checkpoint_sha256':fingerprint(checkpoint),'confusion_matrix':cm.tolist(),
                   **scores_from_confusion(cm),'seconds':time.perf_counter()-start,'source_image_sha256':records[case]['image_SHA256'],
                   'transform':transform(source.shape,[[0,0,0],source.shape],pre['input_size']),'coverage':coverage(target,[[0,0,0],source.shape]),
                   'precision':'historical FP32 inference; no weight or preprocessing changes','reused_exact_stage04_prediction':old.exists()}
            write(scorefile,score);results.append(score);print('BASELINE',variant,case,score['macro_foreground_Dice'],flush=True);del pred,target;gc.collect()
        del model;gc.collect()
        if device=='cuda':torch.cuda.empty_cache()
    write(OUT/'baseline_results.json',results);print('RECOVERED BASELINES COMPLETE',flush=True)


if __name__=='__main__':run()
