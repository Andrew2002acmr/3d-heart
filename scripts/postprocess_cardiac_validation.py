"""Bounded validation-only postprocessing; original predictions/GT remain immutable."""
import argparse
from pathlib import Path
import time
import copy
import subprocess
import numpy as np
import nibabel as nib
from heart3d.ml.cardiac_data import read_json,write_json,load_protocol,confusion_counts,scores_from_confusion
from heart3d.ml.postprocessing import small_changes,apply_changes
from heart3d.storage import sha256_file,require_space
from heart3d.ml.bundle import safe_relative
from scripts.compare_cardiac_validation import compare,CLASSES


def gate(config,baseline):
    if config['partition'] != 'validation' or baseline['partition'] != 'validation':
        raise ValueError('Only validation authorized')
    rows,split = load_protocol(config)
    if baseline['cohort_SHA256'] != sha256_file(config['cohort']) or baseline['split_SHA256'] != sha256_file(config['split']):
        raise ValueError('Frozen protocol mismatch')
    cases=[r['case_id'] for r in baseline['records']]
    if len(cases)!=len(set(cases)) or set(cases)!=set(split['partitions']['validation']):
        raise ValueError('Must evaluate exactly frozen validation')
    if baseline['checkpoint_SHA256'] != config['baseline_checkpoint_SHA256']:
        raise ValueError('Wrong baseline weights')
    return rows,cases


def score(prediction,target):
    counts=np.zeros((8,8),dtype=np.int64)
    for z in range(0,target.shape[2],8):
        chunk=target[:,:,z:z+8]
        # Match original baseline evaluation: labels above 7 are ignored, not background.
        valid_target=np.where(chunk<=7,chunk.astype(np.uint8),255).astype(np.uint8)
        counts+=confusion_counts(prediction[:,:,z:z+8],valid_target)
    return scores_from_confusion(counts)


def run(config_path,data,baseline,output):
    start=time.perf_counter();config=read_json(config_path);base=read_json(baseline/'metrics.json')
    rows,cases=gate(config,base)  # Reject test before opening pixel data or creating outputs.
    if output.exists() or output.resolve().is_relative_to(baseline.resolve()):
        raise ValueError('Choose a new output outside source predictions')
    require_space(output,1_000_000_000,int(config['reserve_GB']*1e9))
    output.mkdir(parents=True)
    config_sha=sha256_file(config_path)
    write_json(output/'protocol.json',{'config':config,'config_SHA256':config_sha,'baseline_metrics_SHA256':sha256_file(baseline/'metrics.json'),
               'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'test_evaluated':False})
    results={name:[] for name in config['variants']};audits=[]
    before={r['case_id']:r for r in base['records']}
    for case in cases:
        row=rows[case];pred_path=baseline/case/'cardiac_prediction_original.nii.gz';prov=read_json(baseline/case/'provenance.json')
        gt_path=data/safe_relative(row['mask_relative_path'])
        if sha256_file(pred_path)!=prov['prediction_SHA256'] or sha256_file(gt_path)!=row['mask_SHA256'] or prov['source_mask_SHA256']!=row['mask_SHA256']:
            raise ValueError('Input SHA mismatch')
        pred_img,gt_img=nib.load(pred_path),nib.load(gt_path)
        if pred_img.shape!=gt_img.shape or not np.allclose(pred_img.affine,gt_img.affine,atol=1e-5,rtol=0):
            raise ValueError('Grid mismatch')
        p=np.asanyarray(pred_img.dataobj);gt=np.asanyarray(gt_img.dataobj)
        actual=score(p,gt)
        for cl in CLASSES:
            for key in ('target_voxels','predicted_voxels','Dice','IoU','precision','recall'):
                a,b=actual['classes'][cl][key],before[case]['classes'][cl][key]
                if a is None and b is None:continue
                if a is None or b is None or abs(a-b)>1e-12:raise ValueError('Baseline metric reproduction failed')
        removed,fill,component_info=small_changes(p,config)
        for variant in config['variants']:
            processed=apply_changes(p,removed,fill,variant)
            item={'case_id':case,**score(processed,gt)};results[variant].append(item)
            folder=output/variant/case;folder.mkdir(parents=True)
            image=nib.Nifti1Image(processed,pred_img.affine,pred_img.header.copy());image.set_data_dtype(np.uint8)
            dst=folder/'cardiac_prediction_original.nii.gz';nib.save(image,dst)
            provenance={**prov,'prediction_SHA256':sha256_file(dst),'input_prediction_SHA256':prov['prediction_SHA256'],
                        'postprocessing_config_SHA256':config_sha,'postprocessing_variant':variant,'GT_used_in_postprocessing':False}
            write_json(folder/'provenance.json',provenance)
            per_class={}
            for label,name in enumerate(CLASSES,1):
                deleted=(p==label)&(processed!=label);added=(p!=label)&(processed==label)
                per_class[name]={'removed_voxels':int(deleted.sum()),'removed_correct_GT_voxels':int(np.count_nonzero(deleted&(gt==label))),
                                'added_voxels':int(added.sum()),'added_correct_GT_voxels':int(np.count_nonzero(added&(gt==label)))}
            audits.append({'case_id':case,'variant':variant,'classes':per_class,'components':component_info})
            del processed
        if sha256_file(pred_path)!=prov['prediction_SHA256'] or sha256_file(gt_path)!=row['mask_SHA256']:
            raise ValueError('Source changed')
        write_json(output/'audit_progress.json',audits)
        print('POSTPROCESS_VALIDATION',case,flush=True)
        del p,gt,removed,fill
    comparisons={}
    for variant,records in results.items():
        j={**copy.deepcopy(base),'records':records,'postprocessing_config_SHA256':config_sha,'postprocessing_variant':variant,
           'mean_case_macro_foreground_Dice':float(np.mean([a['macro_foreground_Dice'] for a in records])),
           'per_class_mean_Dice':{name:float(np.mean([a['classes'][name]['Dice'] for a in records if a['classes'][name]['target_voxels']])) for name in CLASSES}}
        write_json(output/variant/'metrics.json',j)
        comparisons[variant]=compare(baseline/'metrics.json',output/variant/'metrics.json',output/f'{variant}_comparison.json')
    acceptable=[name for name,c in comparisons.items() if c['mean_paired_delta']>0 and c['cases_worsened']==0 and all(v['mean_delta']>=0 for v in c['per_class'].values())]
    summary={'partition':'validation','case_count':len(cases),'config_SHA256':config_sha,'comparisons':comparisons,
             'recommendation':acceptable[0] if acceptable else 'keep unmodified baseline',
             'strict_no_regression_rule_passed':acceptable,'seconds':time.perf_counter()-start,'test_evaluated':False,'GT_used_in_postprocessing':False,
             'source_hashes_rechecked_after_processing':True,'physical_thresholds_mm_claimed':False}
    write_json(output/'summary.json',summary)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','data','baseline','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.config,a.data,a.baseline,a.output)
