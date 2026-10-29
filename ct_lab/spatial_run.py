"""Frozen stage-02 extraction and diagnostic classification. No torch training."""
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from ct_lab.spatial_features import extract,columns,feature_sets,check_affine
from ct_lab.filtering import fit_scaler,standardize,SEED
from ct_lab.filtering_run import REPO,DEFAULT_DATA,protected_hashes
from ct_lab.core import fingerprint,load_pair

OUT=REPO/'outputs/spatial_feature_02'
PREVIOUS=REPO/'outputs/filtering_feature_01'
FILTERS=[{'name':'Baseline','method':'Baseline','dimensions':3},
         *[{'name':f'Gaussian {s:g} {d}D','method':'Gaussian','sigma':s,'dimensions':d} for s in (.5,1.,2.) for d in (2,3)],
         *[{'name':f'Bilateral {d}D','method':'Bilateral','dimensions':d,'spatial_sigma':1.,'intensity_sigma':100.} for d in (2,3)]]


def previous_hashes():
    paths=list(PREVIOUS.rglob('*'))+[REPO/n for n in ('DATA_PIPELINE_AUDIT.md','IMAGECHD68_DATA_ANALYSIS.md','FILTERING_FEATURE_EXPERIMENT_01.md')]
    return {str(p.relative_to(REPO)):fingerprint(p) for p in sorted(paths) if p.is_file()}


def freeze():
    old=json.loads((PREVIOUS/'protocol.json').read_text());split=json.loads((REPO/'configs/splits/cardiac_chd68_v1.json').read_text())
    records={r['case_id']:r for r in json.loads((REPO/'metadata/pediatric/cardiac_chd68_cohort_v1.json').read_text())['records']}
    train,val=old['train_cases'],old['validation_cases']
    if not set(train)<=set(split['partitions']['train']) or not set(val)<=set(split['partitions']['validation']):raise ValueError('Previous groups changed')
    if (set(train)|set(val))&set(split['partitions']['test']):raise ValueError('Sealed test included')
    if {records[c]['group_id'] for c in train}&{records[c]['group_id'] for c in val}:raise ValueError('Group leakage')
    plan={'experiment':'spatial_feature_02','seed':SEED,'train_cases':train,'validation_cases':val,
          'validation_status':'continuation of development on previously used validation; NOT independent final evaluation',
          'split_sha256':fingerprint(REPO/'configs/splits/cardiac_chd68_v1.json'),'previous_protocol_sha256':fingerprint(PREVIOUS/'protocol.json'),
          'sampling':'reuse previous p0/p1 xyz, labels, voxel_ids, train fold without resampling',
          'slice_fractions':old['slice_fractions'],'sample_per_class_per_slice':400,'feature_sets':feature_sets(),
          'filters':FILTERS,'filter_diagnostics':'separate fixed first 32 sampled IDs per class per source Z; B/E/F/3D-core/H; no Gaussian auxiliary channels in 3D-core',
          'context_padding':'nearest edge replicate; flags retained','texture_padding':'half-sample reflect',
          'texture_window':5,'LoG_sigma_grid':1.,'Gaussian_features_3D':[.5,1.,2.],
          'filter_2D':'independent XY Gaussian or exact finite-support bilateral on every Z plane',
          'filter_3D':'same coefficients over coherent XYZ neighborhoods; Gaussian truncate=4; bilateral truncate=3',
          'bilateral_comparison':'sigma_space=1, sigma_intensity=100 source units as stage 01; exact kernel in both 2D/3D, unlike stage01 bin-approximated skimage implementation',
          'profile_channels':'XYZ ±3 exported; raw XYZ profiles are not appended to F. Z channels appear only in C/D/E/H',
          'projection':'independent train-only PCA/UMAP; common-intersection shared projection is an explicit negative control',
          'knn':{'k':15,'weights':'distance','algorithm':'exact scipy cKDTree'},
          'scaler':'train groups only; population mean/std; no validation fit',
          'silhouette':'up to 128 deterministic samples per class per case; Euclidean train-standardized feature space',
          'Fisher':'squared distance between standardized class means / sum of within-class variance traces',
          'GT_distance':'distance in header grid units to nearest 6-connected class-boundary voxel CENTER; diagnostic only',
          'distance_bins':[0,1,3],'chunk_size':64,'filtered_chunk_size':16,'max_patch_radius':24,
          'physical_units_verified':False,'test_CT_read':False,'GT_input_features':False,'selection':'All parameters fixed before validation scoring; no iterative selection'}
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('Stage02 frozen protocol changed; use a new experiment version')
    if not path.exists():
        path.write_text(json.dumps(plan,indent=2),encoding='utf-8')
        (OUT/'preserved_stage01.json').write_text(json.dumps(previous_hashes(),indent=2),encoding='utf-8')
        (OUT/'protected_files.json').write_text(json.dumps(protected_hashes(),indent=2),encoding='utf-8')
    return plan,records


def subset(labels,xyz):
    indices=[]
    for z in np.unique(xyz[:,2]):
        for cls in np.unique(labels):indices.extend(np.flatnonzero((xyz[:,2]==z)&(labels==cls))[:32].tolist())
    return np.sort(indices)


def distances(mask,xyz,labels,spacing):
    result=np.zeros(len(xyz),np.float32)
    for cls in np.unique(labels):
        binary=np.ascontiguousarray(mask==cls)
        border=binary&~ndi.binary_erosion(binary,structure=ndi.generate_binary_structure(3,1),border_value=0)
        sites=np.argwhere(border).astype(np.float64)*spacing
        if not len(sites):raise ValueError('Missing expert class boundary')
        keep=labels==cls;result[keep]=cKDTree(sites).query(xyz[keep]*spacing,workers=1)[0]
    return result


def extract_case(case,record,diagnostics=True):
    cache=OUT/f'case_{case}.npz';info_path=cache.with_suffix('.json')
    if cache.exists() and info_path.exists():
        with np.load(cache) as f:arrays={k:f[k] for k in f.files}
        info=json.loads(info_path.read_text())
        if not diagnostics or info.get('filter_diagnostics_complete'):return arrays,info
    else:arrays={};info={}
    image_path=DEFAULT_DATA/f'{case}_image.nii.gz';mask_path=DEFAULT_DATA/f'{case}_label.nii.gz'
    if fingerprint(image_path)!=record['image_SHA256'] or fingerprint(mask_path)!=record['mask_SHA256']:raise ValueError('Source differs from frozen cohort')
    image,seg=load_pair({'image':str(image_path),'mask':str(mask_path)})
    spacing=check_affine(image.affine);volume=image.get_fdata(dtype=np.float32);mask=np.asanyarray(seg.dataobj)
    with np.load(PREVIOUS/f'samples_{case}.npz') as old:
        xyz=np.concatenate([old[f'p{p}_xyz'] for p in (0,1)]).astype(np.int64)
        sizes=[len(old[f'p{p}_xyz']) for p in (0,1)]
        for p in (0,1):
            for key in ('xyz','labels','voxel_ids','train'):arrays[f'p{p}_{key}']=old[f'p{p}_{key}'].copy()
            if not np.array_equal(np.ravel_multi_index(arrays[f'p{p}_xyz'].T,volume.shape),arrays[f'p{p}_voxel_ids']):raise ValueError('Voxel ID/XYZ mismatch')
            if not np.array_equal(mask[tuple(arrays[f'p{p}_xyz'].T)],arrays[f'p{p}_labels']):raise ValueError('Voxel class mismatch')
    if 'names' not in info:
        bank=extract(volume,xyz,image.affine,chunk_size=64);cut=0
        for p,n in enumerate(sizes):
            arrays[f'p{p}_values']=bank['values'][cut:cut+n];cut+=n
            arrays[f'p{p}_boundary_distance']=distances(mask,arrays[f'p{p}_xyz'],arrays[f'p{p}_labels'],spacing)
        info={'case':case,'group_id':record['group_id'],'names':bank['names'],'shape_xyz':list(volume.shape),'affine':image.affine.tolist(),
              'image_sha256':record['image_SHA256'],'mask_sha256':record['mask_SHA256'],'raw_extraction':{k:v for k,v in bank.items() if k not in ('values','xyz')}}
    if diagnostics and not info.get('filter_diagnostics_complete'):
        diagnostic_xyz=[];diagnostic_sizes=[]
        for p in (0,1):
            index=subset(arrays[f'p{p}_labels'],arrays[f'p{p}_xyz']);arrays[f'p{p}_diagnostic_indices']=index
            diagnostic_xyz.append(arrays[f'p{p}_xyz'][index]);diagnostic_sizes.append(len(index))
        points=np.concatenate(diagnostic_xyz)
        for i,config in enumerate(FILTERS):
            bank=extract(volume,points,image.affine,chunk_size=16,filter_config=config,include_gaussians=False)
            cut=0
            for p,n in enumerate(diagnostic_sizes):arrays[f'p{p}_filter{i}']=bank['values'][cut:cut+n];cut+=n
            info['diagnostic_names']=bank['names']
            print('FILTER',case,config['name'],flush=True)
        info['filter_diagnostics_complete']=True
    if fingerprint(image_path)!=record['image_SHA256'] or fingerprint(mask_path)!=record['mask_SHA256']:raise ValueError('Source modified')
    np.savez_compressed(cache,**arrays);info_path.write_text(json.dumps(info,indent=2),encoding='utf-8')
    return arrays,info


def predict(train,labels,query,scaler,k=15):
    tree=cKDTree(standardize(train,scaler));distance,index=tree.query(standardize(query,scaler),k=min(k,len(train)),workers=1)
    if distance.ndim==1:distance,index=distance[:,None],index[:,None]
    zero=distance==0;weight=np.where(zero.any(1)[:,None],zero.astype(float),1/np.maximum(distance,1e-300))
    classes=np.unique(labels)
    votes=np.stack([(weight*(labels[index]==c)).sum(1) for c in classes],1)
    return classes[votes.argmax(1)]


def metrics(truth,pred,values,classes,seed=SEED):
    from sklearn.metrics import silhouette_score
    cm=np.array([[np.count_nonzero((truth==a)&(pred==b)) for b in classes] for a in classes])
    denominator=cm.sum(0)+cm.sum(1)
    f1=np.divide(2*cm.diagonal(),denominator,out=np.zeros(len(classes),float),where=denominator>0).mean()
    count=cm.sum(1);recall=np.divide(cm.diagonal(),count,out=np.zeros(len(classes),float),where=count>0)
    groups=[values[truth==c] for c in classes];fisher=None
    if all(len(g)>1 for g in groups):
        denominator=sum(g.var(0).sum() for g in groups)
        fisher=float(np.sum((groups[0].mean(0)-groups[1].mean(0))**2)/denominator) if denominator else None
    rng=np.random.default_rng(seed);indices=np.concatenate([rng.choice(np.flatnonzero(truth==c),size=min(128,np.count_nonzero(truth==c)),replace=False) for c in classes])
    silhouette=float(silhouette_score(values[indices],truth[indices])) if len(np.unique(truth[indices]))==2 and len(indices)>2 else None
    return {'macro_f1':float(f1),'balanced_accuracy':float(recall[count>0].mean()),'confusion_matrix':cm.tolist(),
            'fisher_ratio':fisher,'silhouette':silhouette,'silhouette_samples':len(indices),'samples':len(truth)}


def score(train,labels,query,truth,groups,classes):
    scaler=fit_scaler(train);pred=predict(train,labels,query,scaler);scaled=standardize(query,scaler)
    percase={str(case):metrics(truth[groups==case],pred[groups==case],scaled[groups==case],classes) for case in np.unique(groups)}
    summary={key+'_mean':float(np.mean([m[key] for m in percase.values() if m[key] is not None])) for key in ('macro_f1','balanced_accuracy','fisher_ratio','silhouette')}
    summary.update({key+'_std':float(np.std([m[key] for m in percase.values() if m[key] is not None])) for key in ('macro_f1','balanced_accuracy')})
    return {'scaler':scaler,'per_case':percase,'case_macro':summary,
            'pooled_confusion_matrix':np.sum([np.array(m['confusion_matrix']) for m in percase.values()],axis=0).tolist()},pred


def run(diagnostics=True):
    plan,records=freeze();cases={};infos={}
    for case in plan['train_cases']:
        cases[case],infos[case]=extract_case(case,records[case],diagnostics)
        print('TRAIN READY',case,flush=True)
    # Preliminary ct_1001 development first, before reading validation volumes.
    dev=[]
    for p,pair in enumerate(('LV/MYO','RV/PA')):
        data=cases['ct_1001'];fold=data[f'p{p}_train'];labels=data[f'p{p}_labels']
        bank={'names':infos['ct_1001']['names'],'values':data[f'p{p}_values']}
        for name,names in feature_sets().items():
            values=columns(bank,names);scaler=fit_scaler(values[fold]);prediction=predict(values[fold],labels[fold],values[~fold],scaler)
            dev.append({'pair':pair,'configuration':name,'scope':'ct_1001 voxel holdout; optimistic',
                        **metrics(labels[~fold],prediction,standardize(values[~fold],scaler),np.unique(labels))})
    (OUT/'development.json').write_text(json.dumps(dev,indent=2),encoding='utf-8')
    for case in plan['validation_cases']:
        cases[case],infos[case]=extract_case(case,records[case],diagnostics)
        print('VALIDATION READY',case,flush=True)
    scores=[];predictions={};correlations={}
    diagnostic_sets={'B 2D':feature_sets()['B 2D'],'E Z±3':feature_sets()['E Z±3'],
                     'F Tri-planar':feature_sets()['F Tri-planar'],'G 3D-core':['Intensity','3D Gradient','3D Mean','3D Std','3D LoG'],
                     'H Tri-planar+Z':feature_sets()['H Tri-planar+Z']}
    for p,pair in enumerate(('LV/MYO','RV/PA')):
        def assemble(role,names,filter_index=None):
            values=[];labels=[];groups=[]
            for case in plan[role+'_cases']:
                d=cases[case];idx=d[f'p{p}_diagnostic_indices'] if filter_index is not None else np.arange(len(d[f'p{p}_labels']))
                bank={'names':infos[case]['diagnostic_names'] if filter_index is not None else infos[case]['names'],
                      'values':d[f'p{p}_filter{filter_index}'] if filter_index is not None else d[f'p{p}_values']}
                values.append(columns(bank,names));labels.append(d[f'p{p}_labels'][idx]);groups.append(np.repeat(case,len(idx)))
            return np.concatenate(values),np.concatenate(labels),np.concatenate(groups)
        for name,names in feature_sets().items():
            train,labels,_=assemble('train',names);query,truth,groups=assemble('validation',names)
            result,pred=score(train,labels,query,truth,groups,np.unique(labels))
            scores.append({'pair':pair,'configuration':name,'features':names,'feature_count':len(names),'scope':'main unfiltered',**result})
            predictions[f'p{p}_{name.split()[0]}']=pred
            print('MAIN SCORE',pair,name,result['case_macro']['macro_f1_mean'],flush=True)
        train,_,_=assemble('train',infos[plan['train_cases'][0]]['names'])
        corr=np.corrcoef(train.astype(float),rowvar=False);correlations[pair]={'names':infos[plan['train_cases'][0]]['names'],'correlation':np.nan_to_num(corr).tolist()}
        predictions[f'p{p}_truth']=truth;predictions[f'p{p}_cases']=groups
        predictions[f'p{p}_xyz']=np.concatenate([cases[c][f'p{p}_xyz'] for c in plan['validation_cases']])
        predictions[f'p{p}_voxel_ids']=np.concatenate([cases[c][f'p{p}_voxel_ids'] for c in plan['validation_cases']])
        predictions[f'p{p}_boundary_distance']=np.concatenate([cases[c][f'p{p}_boundary_distance'] for c in plan['validation_cases']])
        if diagnostics:
            for i,config in enumerate(FILTERS):
                for name,names in diagnostic_sets.items():
                    train,labels,_=assemble('train',names,i);query,truth,groups=assemble('validation',names,i)
                    result,_=score(train,labels,query,truth,groups,np.unique(labels))
                    scores.append({'pair':pair,'configuration':name,'filter':config['name'],'features':names,'feature_count':len(names),'scope':'fixed reduced filter diagnostic',**result})
                print('DIAGNOSTIC SCORE',pair,config['name'],flush=True)
    result={'protocol':plan,'scores':scores,'validation_status':plan['validation_status'],'test_CT_read':False,'GT_features':False}
    (OUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    (OUT/'correlations.json').write_text(json.dumps(correlations),encoding='utf-8')
    np.savez_compressed(OUT/'validation_predictions.npz',**predictions)
    with (OUT/'scores.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=['scope','pair','configuration','filter','feature_count','macro_f1_mean','macro_f1_std','balanced_accuracy_mean','balanced_accuracy_std','silhouette_mean','fisher_ratio_mean'])
        writer.writeheader();writer.writerows({k:s.get(k,s['case_macro'].get(k)) for k in writer.fieldnames} for s in scores)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--main-only',action='store_true');args=parser.parse_args();run(not args.main_only)
