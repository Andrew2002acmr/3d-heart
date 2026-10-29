"""Class/boundary train sampling and GT-independent natural source IDs."""
import json
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from ct_lab.multiclass_core import *
from ct_lab.core import load_pair


def natural_ids(shape,count,seed):
    return np.sort(np.random.default_rng(seed).choice(int(np.prod(shape)),min(count,int(np.prod(shape))),replace=False))


def stratified(pool,shape,count,rng,strata=16):
    pool=np.asarray(pool);count=min(count,len(pool))
    if not count:return np.empty(0,np.int64)
    z=(pool%shape[2]).astype(np.int32);bins=np.minimum(strata-1,z*strata//shape[2]).astype(np.uint8);chosen=[]
    for b in range(strata):
        available=pool[bins==b];take=min(len(available),count//strata)
        if take:chosen.extend(rng.choice(available,take,replace=False).tolist())
    if len(chosen)<count:
        remaining=np.setdiff1d(pool,chosen,assume_unique=True)
        chosen.extend(rng.choice(remaining,count-len(chosen),replace=False).tolist())
    return np.array(chosen,np.int64)


def training_ids(target,seed,config):
    """GT is permitted for training sampling only, never natural validation selection."""
    rng=np.random.default_rng(seed);fg=(target>0)&(target<8);structure=ndi.generate_binary_structure(3,1)
    near=ndi.binary_dilation(fg,structure,iterations=config['near_6connected_radius'])&~fg
    boundary=np.zeros(target.shape,bool)
    for a in range(3):
        s=[slice(None)]*3;t=s.copy();s[a]=slice(1,None);t[a]=slice(None,-1)
        diff=target[tuple(s)]!=target[tuple(t)];boundary[tuple(s)]|=diff;boundary[tuple(t)]|=diff
    pieces=[];flat=target.ravel();edge=boundary.ravel();near=near.ravel()
    for c in CLASSES:
        pool=np.flatnonzero(flat==c)
        if c==0:
            chosen=np.concatenate([stratified(pool[near[pool]],target.shape,config['background_near'],rng),
                                   stratified(pool[~near[pool]],target.shape,config['background_far'],rng)])
        else:chosen=stratified(pool,target.shape,config['class_uniform'],rng)
        extra=np.setdiff1d(pool[edge[pool]],chosen,assume_unique=True)
        chosen=np.concatenate([chosen,stratified(extra,target.shape,config['class_boundary_extra'],rng)])
        pieces.extend(chosen.tolist())
    return np.array(sorted(set(pieces)),np.int64)


def case_data(case,role,plan,record):
    path=OUT/'data'/f'{case}.npz';info_path=path.with_suffix('.json')
    if role not in ('train','validation') or case not in plan[role+'_cases']:raise ValueError('Forbidden partition')
    if path.exists():
        info=json.loads(info_path.read_text());assert info['role']==role and info['mask_sha256']==record['mask_SHA256']
        return
    image_path=DEFAULT_DATA/f'{case}_image.nii.gz';mask_path=DEFAULT_DATA/f'{case}_label.nii.gz'
    assert fingerprint(image_path)==record['image_SHA256'] and fingerprint(mask_path)==record['mask_SHA256']
    # Select natural IDs using image geometry BEFORE target loading.
    image=nib.load(image_path);check_affine(image.affine);seed=SEEDS[0]+int(case.split('_')[1])
    ids=natural_ids(image.shape,plan['sampling']['natural_'+role],seed);xyz=np.column_stack(np.unravel_index(ids,image.shape))
    image,seg=load_pair({'image':str(image_path),'mask':str(mask_path)})
    volume=image.get_fdata(dtype=np.float32);target=cardiac_target(np.asanyarray(seg.dataobj));data={}
    data.update(natural_ids=ids,natural_xyz=xyz,natural_labels=target[tuple(xyz.T)])
    data['natural_clean']=extract_points(volume,xyz,image.affine)
    if role=='train':
        train_ids=training_ids(target,seed+17,plan['sampling']);points=np.column_stack(np.unravel_index(train_ids,image.shape));labels=target[tuple(points.T)]
        assert set(np.unique(labels))<=set(CLASSES)
        data.update(train_ids=train_ids,train_xyz=points,train_labels=labels,train_clean=extract_points(volume,points,image.affine))
        neighbors,valid=admissible_neighbors(points,labels,target,np.random.default_rng(seed+31))
        values=np.zeros((len(points),len(FEATURES)),np.float32)
        if valid.any():values[valid]=extract_points(volume,neighbors[valid],image.affine)
        data.update(neighbor_xyz=neighbors,neighbor_valid=valid,neighbor_features=values)
        for i,(name,spec) in enumerate(TRAIN_AUG.items()):
            if name=='clean':continue
            altered=augment(volume,spec,seed+i*10007);mapped,keep=transformed_xyz(points,spec,image.shape)
            data['train_'+name]=extract_points(altered,mapped,image.affine)
            data['valid_'+name]=keep
            # Source-identity targets are valid at translated source XYZ, nearest integer interpolation.
            if 'translation' in spec:
                transformed_target=ndi.shift(target,spec['translation'],order=0,mode='constant',cval=255,prefilter=False)
                assert np.array_equal(transformed_target[tuple(mapped[keep].T)],labels[keep])
            del altered
    else:
        assert not any('neighbor' in key for key in data)
        for i,(name,spec) in enumerate(VAL_AUG.items()):
            if name=='clean':continue
            altered=augment(volume,spec,seed+900001+i*10007);mapped,keep=transformed_xyz(xyz,spec,image.shape)
            data['natural_'+name]=extract_points(altered,mapped,image.affine);data['valid_'+name]=keep
            del altered
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**data)
    write(info_path,{'case':case,'role':role,'group_id':record['group_id'],'shape_xyz':list(image.shape),'affine':image.affine.tolist(),
                     'features':FEATURES,'image_sha256':record['image_SHA256'],'mask_sha256':record['mask_SHA256'],
                     'natural_selected_without_GT':True,'counts_natural':{str(c):int(np.count_nonzero(data['natural_labels']==c)) for c in CLASSES+[255]},
                     'counts_train':{str(c):int(np.count_nonzero(data['train_labels']==c)) for c in CLASSES} if role=='train' else None})
    assert fingerprint(image_path)==record['image_SHA256'] and fingerprint(mask_path)==record['mask_SHA256']


def assemble(cases,kind='natural',features=None,augmentation='clean'):
    names=features or FEATURES;parts={k:[] for k in ('features','labels','cases','xyz','ids','valid')}
    for case in cases:
        with np.load(OUT/'data'/f'{case}.npz') as f:
            x=f[kind+'_'+augmentation];n=len(x);parts['features'].append(columns(x,names));parts['labels'].append(f[kind+'_labels'])
            parts['xyz'].append(f[kind+'_xyz']);parts['ids'].append(f[kind+'_ids']);parts['cases'].append(np.repeat(case,n))
            parts['valid'].append(f['valid_'+augmentation] if 'valid_'+augmentation in f else np.ones(n,bool))
    return {k:np.concatenate(v) for k,v in parts.items()}


def prepare():
    plan,records=freeze()
    from concurrent.futures import ThreadPoolExecutor,as_completed
    from threadpoolctl import threadpool_limits
    import psutil
    workers=2 if psutil.virtual_memory().available>4*1024**3 else 1
    with threadpool_limits(limits=1),ThreadPoolExecutor(max_workers=workers) as executor:
        pending={executor.submit(case_data,case,role,plan,records[case]):(role,case)
                 for role in ('train','validation') for case in plan[role+'_cases']}
        for future in as_completed(pending):
            future.result();role,case=pending[future];print('DATA',role,case,flush=True)


if __name__=='__main__':prepare()
