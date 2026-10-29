"""Frozen experiment 01: five development/train groups, ten validation groups, no test CT."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import nibabel as nib
import ct_lab.launch
from ct_lab.core import fingerprint,load_pair
from ct_lab.filtering import (CONFIGS,PAIRS,FEATURES,FEATURE_PROTOCOL,SEED,apply_filter,feature_maps,sample_voxels,
                              matrix,fixed_regions,measurements,knn_score)

REPO=Path(__file__).resolve().parents[1]
DEFAULT_DATA=REPO.parent/'laboratories/data/chd68/pairs'
DEFAULT_OUTPUT=REPO/'outputs/filtering_feature_01'


def protocol():
    split_path=REPO/'configs/splits/cardiac_chd68_v1.json'
    cohort_path=REPO/'metadata/pediatric/cardiac_chd68_cohort_v1.json'
    split=json.loads(split_path.read_text());cohort=json.loads(cohort_path.read_text())
    train=split['development_cases'];validation=split['partitions']['validation']
    records={r['case_id']:r for r in cohort['records']}
    assert split['frozen'] and set(train)<=set(split['partitions']['train'])
    assert not (set(train)|set(validation))&set(split['partitions']['test'])
    assert not {records[c]['group_id'] for c in train}&{records[c]['group_id'] for c in validation}
    return {'experiment':'filtering_feature_01','seed':SEED,'train_cases':train,'validation_cases':validation,
            'test_CT_read':False,'frozen_split_sha256':fingerprint(split_path),'cohort_sha256':fingerprint(cohort_path),
            'configs':CONFIGS,'features':FEATURES,'feature_protocol':FEATURE_PROTOCOL,
            'slice_fractions':[.35,.5,.65],'sampling_per_class_per_slice':400,
            'knn':{'k':15,'weights':'distance','algorithm':'kd_tree','feature_sets':{'Intensity only':[0],'Intensity + texture':[0,1,2,3,4]}},
            'selection':'All six configurations fixed before validation; no validation tuning or test selection',
            'validation_scope':'held-out released groups; no independently verified patient linkage',
            'PCA_UMAP_scope':'ct_1001, fixed stratified voxels, scaler/reducer fit only on training voxel fold'},records


def protected_hashes():
    paths=[REPO/'configs/splits/cardiac_chd68_v1.json',REPO/'configs/cardiac_chd68_preprocessing_v1.json',
           REPO/'configs/cardiac_chd68_preprocessing_resolution384_v2.json']
    paths += list(Path('C:/Users/Andre/heart-model-recovery').glob('chd68_*_best.pt'))
    return {str(p):fingerprint(p) for p in paths}


def extract(case,plan,record,data,output):
    cache=output/f'samples_{case}.npz';info_path=output/f'measurements_{case}.json'
    if cache.exists() and info_path.exists():
        with np.load(cache) as f:arrays={k:f[k] for k in f.files}
        info=json.loads(info_path.read_text())
        if info.get('image_sha256')!=record['image_SHA256'] or info.get('mask_sha256')!=record['mask_SHA256']:raise ValueError('Cached source differs')
        return arrays,info
    image_path=data/f'{case}_image.nii.gz';mask_path=data/f'{case}_label.nii.gz'
    hashes={str(p):fingerprint(p) for p in (image_path,mask_path)}
    if hashes[str(image_path)]!=record['image_SHA256'] or hashes[str(mask_path)]!=record['mask_SHA256']:raise ValueError('Source SHA differs from frozen cohort')
    image,mask_image=load_pair({'image':str(image_path),'mask':str(mask_path)})
    x=image.get_fdata(dtype=np.float32);mask=np.asanyarray(mask_image.dataobj)
    if nib.aff2axcodes(image.affine)!=('R','A','S') or not np.allclose(image.affine[:3,:3],np.diag(np.diag(image.affine[:3,:3]))):raise ValueError('Native axial RAS source required')
    spacing=nib.affines.voxel_sizes(image.affine)[[1,0]]
    pieces={};metrics=[];traces=[]
    def append(key,value):pieces.setdefault(key,[]).append(value)
    for z in sorted(set(int(round((image.shape[2]-1)*f)) for f in plan['slice_fractions'])):
        source=np.array(x[:,:,z].T,np.float32,copy=True);y=mask[:,:,z].T
        base=feature_maps(source,spacing);roi=(0,y.shape[0],0,y.shape[1])
        samples=[sample_voxels(y,pair,case,z,image.shape,roi,plan['sampling_per_class_per_slice']) for pair in PAIRS.values()]
        regions=[fixed_regions(base,y,pair,roi) for pair in PAIRS.values()]
        for p,sample in enumerate(samples):
            for name in ('labels','voxel_ids','xyz','train'):append(f'p{p}_{name}',sample[name])
        for i,config in enumerate(CONFIGS):
            after,provenance=apply_filter(source,config,spacing);maps=base if i==0 else feature_maps(after,spacing)
            for p,(sample,pair,region) in enumerate(zip(samples,PAIRS.values(),regions)):
                append(f'p{p}_c{i}',matrix(maps,sample))
                metrics.append(dict(case=case,z_source=z,pair=list(PAIRS)[p],config=config['name'],
                                    **measurements(maps,y,pair,region,provenance['clipped_fraction'])))
            traces.append({'z_source':z,**provenance})
    arrays={k:np.concatenate(v,axis=0) for k,v in pieces.items()}
    for path in (image_path,mask_path):
        if fingerprint(path)!=hashes[str(path)]:raise ValueError('Original source changed during analysis')
    info={'case':case,'group_id':record['group_id'],'image_sha256':hashes[str(image_path)],'mask_sha256':hashes[str(mask_path)],
          'source_unchanged':True,'shape_xyz':list(image.shape),'affine_ras':image.affine.tolist(),
          'measurements':metrics,'processing':traces}
    np.savez_compressed(cache,**arrays)
    info_path.write_text(json.dumps(info,indent=2,allow_nan=False),encoding='utf-8')
    return arrays,info


def run(data=DEFAULT_DATA,output=DEFAULT_OUTPUT):
    data,output=Path(data),Path(output);plan,records=protocol()
    if data.resolve()==output.resolve() or data.resolve() in output.resolve().parents:raise ValueError('Output must be outside source data')
    output.mkdir(parents=True,exist_ok=True)
    plan_path=output/'protocol.json'
    if plan_path.exists() and json.loads(plan_path.read_text())!=plan:raise ValueError('Frozen experiment protocol differs; use a new output directory')
    plan_path.write_text(json.dumps(plan,indent=2),encoding='utf-8')
    guards=protected_hashes();cohort={};measures=[]
    for role in ('train','validation'):
        for case in plan[role+'_cases']:
            arrays,info=extract(case,plan,records[case],data,output)
            cohort[case]=arrays;measures.extend(dict(role=role,**m) for m in info['measurements'])
            print('EXTRACTED',role,case,flush=True)
    scores=[]
    for p,pair in enumerate(PAIRS):
        for i,config in enumerate(CONFIGS):
            train=np.concatenate([cohort[c][f'p{p}_c{i}'] for c in plan['train_cases']])
            labels=np.concatenate([cohort[c][f'p{p}_labels'] for c in plan['train_cases']])
            valid=np.concatenate([cohort[c][f'p{p}_c{i}'] for c in plan['validation_cases']])
            truth=np.concatenate([cohort[c][f'p{p}_labels'] for c in plan['validation_cases']])
            groups=np.concatenate([np.repeat(c,len(cohort[c][f'p{p}_labels'])) for c in plan['validation_cases']])
            for feature_set,columns in plan['knn']['feature_sets'].items():
                result=knn_score(train,labels,valid,truth,columns,groups=groups)
                scores.append(dict(pair=pair,config=config['name'],feature_set=feature_set,role='held-out frozen validation',**result))
                dev=cohort['ct_1001'];values=dev[f'p{p}_c{i}'];fold=dev[f'p{p}_train'];dy=dev[f'p{p}_labels']
                scores.append(dict(pair=pair,config=config['name'],feature_set=feature_set,role='ct_1001 voxel holdout; optimistic',
                                   **knn_score(values[fold],dy[fold],values[~fold],dy[~fold],columns)))
            print('SCORED',pair,config['name'],flush=True)
    if protected_hashes()!=guards:raise ValueError('Frozen split, preprocessing or checkpoint changed')
    result={'protocol':plan,'scores':scores,'measurements':measures,'protected_files_unchanged':guards,
            'source_cases_unchanged':list(cohort),'test_data_accessed':False}
    (output/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    with (output/'measurements.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=sorted(set().union(*(m.keys() for m in measures))));writer.writeheader();writer.writerows(measures)
    with (output/'scores.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=['role','pair','config','feature_set','macro_f1','equal_case_macro_f1_mean','equal_case_macro_f1_std','train_samples','evaluation_samples'])
        writer.writeheader();writer.writerows({k:s.get(k) for k in writer.fieldnames} for s in scores)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=DEFAULT_DATA);parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args();run(args.data,args.output)
