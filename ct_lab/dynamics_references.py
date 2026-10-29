"""CommonFP32 audit of frozen models, keeping their original metrics untouched."""
import nibabel as nib
from ct_lab.dynamics_core import *
from ct_lab.dynamics_evaluate import evaluate
from heart3d.ml.cardiac_data import confusion_counts,scores_from_confusion

def run():
    seed_runtime(20261009,1,'cuda' if torch.cuda.is_available() else 'cpu');plan=protocol();cases=['ct_1001',*plan['validation_cases']];rows=[]
    models=[('CHD68_v1',WEIGHTS/'chd68_v1_best.pt'),('CHD68_v2',WEIGHTS/'chd68_v2_best.pt'),('ROI05_Full256',REPO/'outputs/heart_localization_roi_05/segmenters/all_train/B_full_256/best.pt')]
    for name,path in models:
        result=evaluate(path,cases,OUT/'references'/name)
        frozen_name='B_full_256' if name=='ROI05_Full256' else name;old=[]
        for case in cases:
            original=REPO/'outputs/heart_localization_roi_05/evaluation/all_train'/frozen_name/case
            expected=json.loads((original/'scores.json').read_text());pred=np.asanyarray(nib.load(original/'prediction.nii.gz').dataobj);gt=cardiac_target(np.asanyarray(nib.load(DATA/f'{case}_label.nii.gz').dataobj))
            measured=scores_from_confusion(confusion_counts(pred,gt));assert abs(expected['macro_foreground_Dice']-measured['macro_foreground_Dice'])<1e-12
            if case in plan['validation_cases']:old.append(measured['macro_foreground_Dice'])
        record={'method':name,'common_FP32':result['by_role']['development'],'frozen_original_native_Dice':float(np.mean(old)),
          'common_minus_frozen':result['by_role']['development']['native_Dice']-float(np.mean(old)),
          'frozen_metric_definition_verified_exact':True,'checkpoint_selection':'historical best prepared' if name.startswith('CHD') else 'ROI05 fixed last epoch'}
        rows.append(record);write(OUT/'reference_comparison.json',rows);print('COMMON ADAPTER REFERENCE',name,record,flush=True)
    write(OUT/'metric_audit_complete.json',{'models':3,'frozen_masks_scored_exactly':33,'test_read':False,'no_old_files_written':True})

if __name__=='__main__':run()
