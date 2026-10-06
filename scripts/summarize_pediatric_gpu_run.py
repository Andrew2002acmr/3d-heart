"""Audit downloaded results and create small original-grid experiment summaries."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from heart3d.ml.bundle import verify
from heart3d.ml.metrics import aggregate
from heart3d.ml.splits import load_frozen_split
from heart3d.pediatric import write_json
from heart3d.storage import sha256_file


def summarize(results_root, run_name, output_json, figure_directory):
    root=Path(results_root);verify(root,root/'bundle_manifest.json')
    run=root/'experiments/heart_baseline_v1'/run_name
    training=json.loads((run/'training_summary.json').read_text())
    history=json.loads((run/'history.json').read_text())
    config=json.loads((run/'config.json').read_text())
    metrics=json.loads((run/'test/metrics.json').read_text())
    benchmark=json.loads((root/'experiments/heart_baseline_v1'/(run_name+'_gpu_benchmark')/'summary.json').read_text())
    split=load_frozen_split(config['split'],config['cohort'])
    if training['epochs']!=config['epochs'] or len(history)!=training['epochs']:raise ValueError('Incomplete baseline')
    for key in ('split_SHA256','preprocessing_SHA256','config_SHA256'):
        expected=sha256_file(config['split'] if key=='split_SHA256' else config['preprocessing'] if key=='preprocessing_SHA256' else 'configs/pediatric_heart_runpod_v1.json')
        if training[key]!=expected or benchmark[key]!=expected:raise ValueError('Frozen experiment provenance mismatch')
    expected_patients={r['patient_id']:r for r in split['partitions']['test']}
    if len(metrics['records'])!=len(expected_patients) or {r['patient_id'] for r in metrics['records']}!=set(expected_patients):
        raise ValueError('Incomplete or leaking test evaluation')
    records=[]
    for measured in metrics['records']:
        row=expected_patients[measured['patient_id']]
        record={**measured,'age':row['age'],'age_precision':row['age_precision'],
                'spacing_xyz_mm':row['geometry']['spacing_xyz_mm'],
                'native_z_spacing_mm':round(row['geometry']['spacing_xyz_mm'][2],6),
                'original_slices':row['geometry']['shape_xyz'][2],
                'CT_margins_below_above_OAR_slices':row['Heart']['uncontoured_CT_margin_z_slices'],
                'source_Heart_target':'original expert Heart OAR','healthy_status':'unknown'}
        record['relative_volume_error']=record['predicted_volume_ml']/record['GT_volume_ml']-1
        records.append(record)
    checkpoint_root=root/'checkpoints/heart_baseline_v1'/run_name
    for name,receipt in training['checkpoints'].items():
        if sha256_file(checkpoint_root/name)!=receipt['SHA256']:raise ValueError('Checkpoint SHA mismatch')
    best_checkpoint=training['checkpoints']['best.pt']['SHA256']
    if metrics['checkpoint_SHA256']!=best_checkpoint:raise ValueError('Test did not use the selected best checkpoint')
    summary={'run_name':run_name,'training_git_revision':training['git_revision'],
             'cohort_patients':sum(len(rows) for rows in split['partitions'].values()),
             'split_counts':{part:len(rows) for part,rows in split['partitions'].items()},
             'split_SHA256':training['split_SHA256'],'preprocessing_SHA256':training['preprocessing_SHA256'],
             'config_SHA256':training['config_SHA256'],'epochs':training['epochs'],
             'model':'project-owned 2.5D U-Net 5->16->32->64->128->1; 488993 parameters',
             'batch_size':config['batch_size'],'precision':'FP32','loss':config['loss'],
             'GPU_benchmark':benchmark,'training_seconds':training['seconds'],
             'mean_train_epoch_seconds':float(np.mean([h['train_seconds'] for h in history])),
             'mean_validation_epoch_seconds':float(np.mean([h['validation_seconds'] for h in history])),
             'training_resources':training['resources'],'best_epoch':training['best_epoch'],
             'best_validation_Dice_preprocessing_grid':training['best_validation_Dice_preprocessing_grid'],
             'test_grid':'original DICOM-derived XYZ; threshold after inverse linear probability transform',
             'test_aggregate':aggregate(records),'by_age_group':aggregate(records,'age_group'),
             'by_scanner':aggregate(records,'scanner'),'by_contrast':aggregate(records,'contrast'),
             'by_native_z_spacing':aggregate(records,'native_z_spacing_mm'),
             'test_records':records,'test_review_order_by_Dice':[r['patient_id'] for r in sorted(records,key=lambda r:r['Dice'])],
             'mean_absolute_relative_volume_error':float(np.mean([abs(r['relative_volume_error']) for r in records])),
             'checkpoints':training['checkpoints'],'all_downloaded_result_files_SHA256_verified':True,
             'full_training_completed':True,'test_evaluated_once_in_driver':True,
             'healthy_pathological_classifier_trained':False,'CT_MRI_joint_training':False,
             'limitations':['original OAR annotation extent, not full chamber/vessel anatomy','technical QA, no new clinician sign-off',
                            'test n=9; ages 2-16, no age17','held-out only LightSpeed/SOMATOM and contrast-agent-reported cases',
                            'unknown clinical status; public IDs do not prove all repeat identities']}
    write_json(output_json,summary)
    figures=Path(figure_directory);figures.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    epochs=[r['epoch'] for r in history]
    axes[0].plot(epochs,[r['train_loss'] for r in history]);axes[0].set(xlabel='Epoch',ylabel='BCE + Soft Dice loss',title='Balanced training slices')
    axes[1].plot(epochs,[r['validation_mean_patient_Dice'] for r in history]);axes[1].axvline(training['best_epoch'],color='gray',ls='--')
    axes[1].set(xlabel='Epoch',ylabel='Mean patient Dice',title='Validation: preprocessing grid; test unused')
    fig.savefig(figures/'training_history.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    ax.bar([r['patient_id'].removeprefix('Pediatric-CT-SEG-') for r in records],[r['Dice'] for r in records])
    ax.set(ylim=(0,1),ylabel='Dice',title='Frozen test, original DICOM grid (9 patients)')
    fig.savefig(figures/'test_patient_Dice.png',dpi=140);plt.close(fig)
    print('RESULT_SUMMARY',json.dumps(summary['test_aggregate']),flush=True)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-root',type=Path,required=True);p.add_argument('--run-name',required=True)
    p.add_argument('--output-json',type=Path,required=True);p.add_argument('--figure-dir',type=Path,required=True)
    a=p.parse_args();summarize(a.results_root,a.run_name,a.output_json,a.figure_dir)
