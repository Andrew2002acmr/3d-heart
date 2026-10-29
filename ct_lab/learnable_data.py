"""Frozen ablation and 48-case sampling; validation never builds training pairs."""
import json
from pathlib import Path
import numpy as np
import nibabel as nib
from ct_lab.learnable_core import OUT,REPO,SEED,SETS,FEATURES,PAIRS,admissible_neighbors
from ct_lab.core import fingerprint,load_pair
from ct_lab.filtering import sample_voxels
from ct_lab.spatial_features import extract,columns
from ct_lab.filtering_run import DEFAULT_DATA,protected_hashes


def prior_hashes():
    paths=[p for folder in ('filtering_feature_01','spatial_feature_02') for p in (REPO/'outputs'/folder).rglob('*') if p.is_file()]
    paths += [REPO/n for n in ('DATA_PIPELINE_AUDIT.md','IMAGECHD68_DATA_ANALYSIS.md','FILTERING_FEATURE_EXPERIMENT_01.md','SPATIAL_FEATURE_EXPERIMENT_02.md')]
    return {str(p.relative_to(REPO)):fingerprint(p) for p in sorted(paths)}


def freeze():
    split_path=REPO/'configs/splits/cardiac_chd68_v1.json';split=json.loads(split_path.read_text())
    records={r['case_id']:r for r in json.loads((REPO/'metadata/pediatric/cardiac_chd68_cohort_v1.json').read_text())['records']}
    train=split['partitions']['train'];validation=split['partitions']['validation']
    if len(train)!=48 or len(validation)!=10 or set(train)&set(validation):raise ValueError('Expected frozen 48/10 split')
    if {records[c]['group_id'] for c in train}&{records[c]['group_id'] for c in validation}:raise ValueError('Release-group leakage')
    if (set(train)|set(validation))&set(split['partitions']['test']):raise ValueError('Sealed test included')
    plan={'experiment':'learnable_feature_03','seed':SEED,'split_sha256':fingerprint(split_path),
          'train_cases':train,'validation_cases':validation,'test_read':False,
          'feature_sets':SETS,'neural_inputs':['C 3D-core','G core+all'],'embedding_candidates':[8,16,32],
          'methods':['CE','CE+SupCon','CE+SupCon+Spatial'],
          'architecture':'input -> Linear64 -> ReLU -> Linear32 -> ReLU -> Linear embedding -> Linear classes',
          'training':{'optimizer':'AdamW','lr':.001,'weight_decay':.0001,'epochs':30,'patience':5,'min_delta':.0001,
                      'batch_size':512,'CPU_threads':1,'lambda_contrastive':.1,'lambda_spatial':.1,'temperature':.1,
                      'early_stopping':'maximum validation case-macro F1, fixed-label zero_division=0','class_case_balanced_batches':True},
          'sampling':{'Z_fractions':[.2,.3,.4,.5,.6,.7,.8],'per_class_per_Z':64,'seed':SEED,'max_anchor_voxels_per_pair_per_case':896},
          'neighbors':'one deterministic in-bounds 6-neighbor of same train GT class, distinct source coordinate; missing=-1 / invalid; never constructed for validation',
          'ablation':'legacy stage02 five train/ten validation, exact sampled IDs, plus new 48-train cohort; all A-G predeclared',
          'selection':'C/G neural inputs and all 36 candidates fixed BEFORE ablation/validation; no post-hoc feature selection',
          'scaler':'population mean/std fit only 48 train, separately per feature set; all methods share same train scaler',
          'robustness':{'space':'train-standardized features','seed':SEED,'Gaussian_std':.05,'intensity_shift_std_units':.1,
                        'single_feature_scale':1.1,'scale_feature':'Intensity','drop_feature':'3D LoG','drop_value':'train mean (standardized zero)'},
          'importance':'fixed stratified evaluation subset, permutation within each case; diagnostic only, not model selection',
          'projection':'same stratified 32/class/case source IDs; PCA/UMAP scaler/reducer fit only TRAIN; seed fixed; display e16, all model dimensions scored',
          'validation_status':'previously used validation; model selection/early stopping only; NOT independent final evaluation',
          'GT_input_features':False,'physical_units_verified':False,'source_intensity_HU':False}
    OUT.mkdir(parents=True,exist_ok=True);path=OUT/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('Frozen stage03 protocol differs')
    if not path.exists():
        path.write_text(json.dumps(plan,indent=2),encoding='utf-8')
        (OUT/'prior_files.json').write_text(json.dumps(prior_hashes(),indent=2),encoding='utf-8')
        (OUT/'protected_files.json').write_text(json.dumps(protected_hashes(),indent=2),encoding='utf-8')
    return plan,records


def extract_case(case,role,record,plan):
    if role not in ('train','validation') or case not in plan[role+'_cases']:raise ValueError('Case outside allowed partition')
    directory=OUT/'data';directory.mkdir(exist_ok=True);path=directory/f'{case}.npz';info_path=path.with_suffix('.json')
    if path.exists() and info_path.exists():
        info=json.loads(info_path.read_text())
        if info['role']!=role or info['image_sha256']!=record['image_SHA256']:raise ValueError('Stale case cache')
        with np.load(path) as f:return {k:f[k] for k in f.files},info
    image_path=DEFAULT_DATA/f'{case}_image.nii.gz';mask_path=DEFAULT_DATA/f'{case}_label.nii.gz'
    if fingerprint(image_path)!=record['image_SHA256'] or fingerprint(mask_path)!=record['mask_SHA256']:raise ValueError('Source hash differs')
    image,seg=load_pair({'image':str(image_path),'mask':str(mask_path)})
    volume=image.get_fdata(dtype=np.float32);mask=np.asanyarray(seg.dataobj)
    data={};case_seed=SEED+int(case.split('_')[1]);rng=np.random.default_rng(case_seed)
    for p,classes in enumerate(PAIRS.values()):
        pieces=[]
        for z in sorted(set(round((image.shape[2]-1)*f) for f in plan['sampling']['Z_fractions'])):
            sample=sample_voxels(mask[:,:,z].T,classes,case,z,image.shape,per_class=plan['sampling']['per_class_per_Z'])
            pieces.append(sample)
        for key in ('xyz','labels','voxel_ids'):data[f'p{p}_{key}']=np.concatenate([s[key] for s in pieces])
        xyz=data[f'p{p}_xyz'];labels=data[f'p{p}_labels']
        if not np.array_equal(mask[tuple(xyz.T)],labels):raise ValueError('Coordinate/class mismatch')
        bank=extract(volume,xyz,image.affine);data[f'p{p}_features']=columns(bank,FEATURES)
        if role=='train':
            neighbors,valid=admissible_neighbors(xyz,labels,mask,rng)
            neighbor_features=np.full((len(xyz),len(FEATURES)),np.nan,np.float32)
            if valid.any():neighbor_features[valid]=columns(extract(volume,neighbors[valid],image.affine),FEATURES)
            data.update({f'p{p}_neighbor_xyz':neighbors,f'p{p}_neighbor_valid':valid,f'p{p}_neighbor_features':neighbor_features})
    info={'case':case,'role':role,'group_id':record['group_id'],'shape_xyz':list(image.shape),'affine':image.affine.tolist(),
          'features':FEATURES,'image_sha256':record['image_SHA256'],'mask_sha256':record['mask_SHA256'],
          'training_neighbor_pairs_built':role=='train','source_unchanged':True}
    if fingerprint(image_path)!=record['image_SHA256'] or fingerprint(mask_path)!=record['mask_SHA256']:raise ValueError('Original changed')
    np.savez_compressed(path,**data);info_path.write_text(json.dumps(info,indent=2),encoding='utf-8')
    return data,info


def assemble(plan,role,pair,features):
    matrices=[];labels=[];cases=[];xyz=[];ids=[];neighbors=[];valid=[]
    indices=[FEATURES.index(n) for n in features]
    for case in plan[role+'_cases']:
        with np.load(OUT/'data'/f'{case}.npz') as data:
            matrices.append(data[f'p{pair}_features'][:,indices]);labels.append(data[f'p{pair}_labels'])
            xyz.append(data[f'p{pair}_xyz']);ids.append(data[f'p{pair}_voxel_ids'])
            cases.append(np.repeat(case,len(labels[-1])))
            if role=='train':neighbors.append(data[f'p{pair}_neighbor_features'][:,indices]);valid.append(data[f'p{pair}_neighbor_valid'])
    result={'features':np.concatenate(matrices),'labels':np.concatenate(labels),'cases':np.concatenate(cases),'xyz':np.concatenate(xyz),'ids':np.concatenate(ids)}
    if role=='train':result.update(neighbors=np.concatenate(neighbors),neighbor_valid=np.concatenate(valid))
    return result
