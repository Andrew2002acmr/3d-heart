"""Explicit real-development-case verification, CPU forward only, no fitting."""
import json
from pathlib import Path
import numpy as np
import nibabel as nib
from ct_lab.core import stats, fingerprint
from ct_lab.model_inspector import inspect, predict_slice


def main(full=False):
    repo=Path(__file__).resolve().parents[1]
    path=repo.parent/'laboratories/data/chd68/pairs/ct_1001_image.nii.gz'
    image=nib.load(path)
    y=np.asanyarray(nib.load(path.with_name('ct_1001_label.nii.gz')).dataobj)
    output=repo/'outputs/ct_preprocessing_lab';output.mkdir(parents=True,exist_ok=True)
    records=[]
    for variant in ('v1','v2'):
        result=inspect(image,variant,image.shape[2]//2)
        checkpoint=Path('C:/Users/Andre/heart-model-recovery')/f'chd68_{variant}_best.pt'
        probs,captured,entropy=predict_slice(result,variant,checkpoint,image.shape)
        from heart3d.ml.cardiac_data import confusion_counts,scores_from_confusion
        prediction=probs.argmax(0)
        record={'variant':variant,'case':'ct_1001','center_source_z':image.shape[2]//2,
                'source_sha256':fingerprint(path),'checkpoint_sha256':fingerprint(checkpoint),
                'tensor_shape':list(captured.shape),'tensor_stats':stats(captured.numpy()),
                'channels':result['channel_statistics'],'forward_hook_exact':True,
                'entropy_stats':stats(entropy),'scope':'one real public development slice; not patient or test Dice',
                'slice_metrics':scores_from_confusion(confusion_counts(prediction,y[:,:,image.shape[2]//2]))}
        records.append(record)
        print(variant,record['tensor_shape'],record['slice_metrics']['macro_foreground_Dice'],flush=True)
        if full:
            from ct_lab.model_inspector import diagnose_volume
            mask=nib.load(path.with_name('ct_1001_label.nii.gz'))
            _,report=diagnose_volume(image,mask,variant,checkpoint,components=True,progress=lambda message:print(variant,message,flush=True))
            (output/f'real_volume_{variant}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            print('FULL',variant,report['metrics']['macro_foreground_Dice'],flush=True)
    (output/'real_model_verification.json').write_text(json.dumps(records,indent=2),encoding='utf-8')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--full',action='store_true')
    main(parser.parse_args().full)
