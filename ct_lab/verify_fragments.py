"""Save real original-grid predictions and add localized fragment evidence."""
import json
from pathlib import Path
import nibabel as nib
import numpy as np
from ct_lab.core import fragment_comparison, fingerprint
from heart3d.ml.infer_cardiac import load_recovered, prepare_volume, infer_volume, save_prediction
from heart3d.ml.cardiac_data import confusion_counts, scores_from_confusion


def main():
    repo=Path(__file__).resolve().parents[1];root=repo/'outputs/ct_preprocessing_lab'
    data=repo.parent/'laboratories/data/chd68/pairs'
    image=nib.load(data/'ct_1001_image.nii.gz');mask=nib.load(data/'ct_1001_label.nii.gz');gt=np.asanyarray(mask.dataobj)
    for variant in ('v1','v2'):
        path=root/f'real_volume_{variant}.json'
        if not path.exists():raise ValueError('Complete verify_real --full first')
        report=json.loads(path.read_text())
        destination=root/f'ct_1001_{variant}_prediction.nii.gz'
        if destination.exists():prediction=np.asanyarray(nib.load(destination).dataobj)
        else:
            model,_,_,pre=load_recovered(variant,f'C:/Users/Andre/heart-model-recovery/chd68_{variant}_best.pt','cpu')
            prepared,transform=prepare_volume(image.get_fdata(dtype=np.float32),pre)
            prediction=infer_volume(model,prepared,transform,pre,1,'cpu')
            del prepared
            save_prediction(prediction,image,destination)
        metrics=scores_from_confusion(confusion_counts(prediction,gt))
        if abs(metrics['macro_foreground_Dice']-report['metrics']['macro_foreground_Dice'])>1e-12:
            raise ValueError('Prediction differs from completed full-volume run')
        report['fragment_gt_comparison']=fragment_comparison(prediction,gt,image.affine)
        report['prediction_sha256']=fingerprint(destination)
        path.write_text(json.dumps(report,indent=2),encoding='utf-8')
        wrong=[r for r in report['fragment_gt_comparison'] if not r['largest'] and r['gt_same_class_fraction']==0]
        print(variant,'satellites with zero same-class GT overlap:',len(wrong),flush=True)


if __name__=='__main__':main()
