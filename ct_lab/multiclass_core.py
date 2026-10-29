"""Frozen stage-04 protocol; existing local encoder and source-grid feature engine."""
import json
import os
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage as ndi
from ct_lab.learnable_core import FeatureEncoder,seed_all,normalize,fit_scaler,infer,supcon_loss,spatial_loss,admissible_neighbors
from ct_lab.spatial_features import gather,stencil,kernels,check_affine
from ct_lab.core import cardiac_target,fingerprint
from ct_lab.filtering_run import protected_hashes,DEFAULT_DATA

REPO=Path(__file__).resolve().parents[1];OUT=REPO/'outputs/multiclass_feature_04'
os.environ['NUMBA_CACHE_DIR']=str(OUT/'numba_cache')
CLASSES=list(range(8));LABELS=['Background','LV','RV','LA','RA','MYO','AO','PA'];SEEDS=[20261029,20261030,20261031]
CORE=['Intensity','3D Mean','3D Std','3D Gradient','3D LoG']
GAUSS=[f'3D Gaussian {s:g}' for s in (.5,1.,2.)]
SETS={'A Intensity':['Intensity'],'B 2D':['Intensity','XY Gradient','XY Mean','XY Std','XY LoG'],
      'C 3D-core':CORE,'D core+Gaussian2':CORE+GAUSS[2:],'E core+all':CORE+GAUSS,
      'F core+Z':CORE+['Z I -1','Z I +1','Z1 Std']}
FEATURES=list(dict.fromkeys(n for cols in SETS.values() for n in cols))
TRAIN_AUG={'clean':{},'intensity':{'shift':32.,'scale':1.03,'gamma':1.04},'noise':{'noise':8.},
           'combined':{'noise':8.,'shift':-24.,'scale':.98,'gamma':.98,'smooth':.35,'translation':[1,0,-1]}}
VAL_AUG={'clean':{},'noise12':{'noise':12.},'shift-40':{'shift':-40.},'scale.96':{'scale':.96},
         'contrast.94':{'gamma':.94},'smooth.55':{'smooth':.55},'translateY+Z':{'translation':[0,1,1]},
         'heldout-combined':{'noise':12.,'shift':40.,'scale':1.04,'gamma':1.06,'smooth':.55,'translation':[-1,1,0]}}


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def prior_hashes():
    files=[p for d in ('filtering_feature_01','spatial_feature_02','learnable_feature_03') for p in (REPO/'outputs'/d).rglob('*') if p.is_file()]
    files += [REPO/n for n in ('DATA_PIPELINE_AUDIT.md','IMAGECHD68_DATA_ANALYSIS.md','FILTERING_FEATURE_EXPERIMENT_01.md','SPATIAL_FEATURE_EXPERIMENT_02.md','LEARNABLE_FEATURE_EXPERIMENT_03.md')]
    return {str(p.relative_to(REPO)):fingerprint(p) for p in sorted(files)}


def freeze():
    split_path=REPO/'configs/splits/cardiac_chd68_v1.json';split=json.loads(split_path.read_text())['partitions']
    records={r['case_id']:r for r in json.loads((REPO/'metadata/pediatric/cardiac_chd68_cohort_v1.json').read_text())['records']}
    train=split['train'];val=split['validation'];assert len(train)==48 and len(val)==10
    assert not set(train)&set(val) and not (set(train)|set(val))&set(split['test'])
    assert not {records[c]['group_id'] for c in train}&{records[c]['group_id'] for c in val}
    # Shared release-group assignment, never voxel-wise folds.
    groups=sorted({records[c]['group_id'] for c in train});np.random.default_rng(SEEDS[0]).shuffle(groups)
    folds=[[c for c in train if groups.index(records[c]['group_id'])%3==f] for f in range(3)]
    jobs=[{'features':f,'method':'CE','augmentation':'clean','seed':s} for f in SETS for s in SEEDS]
    jobs += [{'features':'E core+all','method':m,'augmentation':'clean','seed':s} for m in ('CE+SupCon','CE+Spatial') for s in SEEDS]
    jobs += [{'features':'E core+all','method':m,'augmentation':a,'seed':s} for m,a in [('CE','intensity'),('CE','noise'),('CE','combined'),('CE+SupCon','combined')] for s in SEEDS]
    plan={'experiment':'multiclass_feature_04','classes':CLASSES,'labels':LABELS,'ignore':255,'seeds':SEEDS,
          'train_cases':train,'validation_cases':val,'split_sha256':fingerprint(split_path),'sealed_test_read':False,
          'validation_status':'reused development validation; early stopping/model selection, NOT independent final evaluation',
          'feature_sets':SETS,'feature_schema':FEATURES,'embedding_dim':16,'jobs':jobs,'CV_folds':folds,
          'CV_jobs':[{'features':'C 3D-core','method':'CE','augmentation':'clean'},{'features':'E core+all','method':'CE','augmentation':'combined'}],
          'training':{'epochs':20,'patience':4,'min_delta':.0002,'batch_size':256,'optimizer':'AdamW','lr':.001,'weight_decay':.0001,
                      'lambda_contrastive':.1,'lambda_spatial':.1,'temperature':.1,'CPU_threads':1,'class_case_balanced':True},
          'sampling':{'class_uniform':128,'class_boundary_extra':32,'background_near':64,'background_far':64,'near_6connected_radius':6,
                      'Z_strata':16,'natural_train':4096,'natural_validation':8192,'natural_rule':'uniform without replacement from ALL source-grid IDs before reading GT; includes ignore, never foreground restricted'},
          'training_augmentations':TRAIN_AUG,'heldout_validation_transformations':VAL_AUG,
          'augment_rule':'source image then feature recomputation; integer translation maps source XYZ to XYZ+shift; out-of-bounds target ignore; no elastic/rotation required',
          'augmentation_units':'source counts, NOT HU; intensity scale around 1024, contrast gamma clipped source/4095; mild synthetic stress, not scanner realism',
          'early_stopping':'maximum fixed-eight-label case-macro F1 on natural validation points; scaler only fold TRAIN',
          'dense_cases':['ct_1001','ct_1027','ct_1028'],'dense_ROI':{'center_fraction':[.65,.45,.5],'size_xyz':[128,128,96],'GT_input':False},
          'dense_models':[{'features':'A Intensity','method':'CE','augmentation':'clean'},{'features':'C 3D-core','method':'CE','augmentation':'clean'},
                          {'features':'E core+all','method':'CE','augmentation':'combined'}],
          'dense_full_volume_case':'ct_1001','dense_full_model':{'features':'E core+all','method':'CE','augmentation':'combined'},
          'dense_seed':SEEDS[0],'dense_block_xyz':[64,64,16],'probability_dtype':'float16 disk, float32 softmax/metrics',
          'CHD68_comparison':'v1/v2 immutable recovered models; source grid probabilities before argmax; identical ROI/valid GT; ct_1001 full volume too',
          'geometry':'native RAS XYZ, unknown spatial units; HD95 in INDEX units; no mm interpretation',
          'statistics':'seed mean/std; case bootstrap 2000 resamples after seed averaging; paired case differences, never voxel bootstrap',
          'projection':'fixed 24/class/case training-stratified/GT-diagnostic selection; train-fit PCA/UMAP only; clean/noisy common basis learned clean TRAIN; no target labels in reducers',
          'ROI_GT_input':False,'GT_feature_input':False,'knn':{'k':15,'weights':'distance'},'parameters_fixed_before_validation':True}
    OUT.mkdir(parents=True,exist_ok=True);path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==plan,'Frozen protocol differs'
    else:
        write(path,plan);write(OUT/'prior_files.json',prior_hashes());write(OUT/'protected_files.json',protected_hashes())
    return plan,records


def extract_points(volume,xyz,affine,chunk_size=128):
    """Same tested stencil/kernels as stage02; only needed planes/channels."""
    spacing=check_affine(affine);xyz=np.asarray(xyz,int)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or np.any(xyz<0) or np.any(xyz>=volume.shape):raise ValueError('Invalid XYZ')
    radius=max(3,*[int(8/s+.5) for s in spacing]);pieces=[]
    if radius>24 or chunk_size<1:raise ValueError('Memory bound')
    for start in range(0,len(xyz),chunk_size):
        pts=xyz[start:start+chunk_size];patch=gather(volume,pts,radius);r=radius;center=patch[:,r,r,r]
        bank={'Intensity':center}
        for prefix,axes in [('XY',(0,1)),('3D',(0,1,2))]:
            values=stencil(patch,spacing,axes);g2=np.zeros(len(pts),float)
            for a in axes:
                plus=[slice(None),r,r,r];minus=plus.copy();plus[a+1]+=1;minus[a+1]-=1
                g=(patch[tuple(plus)].astype(float)-patch[tuple(minus)])/(2*spacing[a])
                lo=pts[:,a]==0;hi=pts[:,a]==volume.shape[a]-1
                g[lo]=(patch[tuple(plus)][lo]-center[lo])/spacing[a];g[hi]=(center[hi]-patch[tuple(minus)][hi])/spacing[a]
                g2+=g*g
            values[2]=np.sqrt(g2);bank.update({prefix+' '+key:v for key,v in zip(('Mean','Std','Gradient','LoG'),values)})
        for sigma in (.5,1.,2.):
            weights=[kernels(sigma/s)[0] for s in spacing];slices=tuple([slice(None)]+[slice(r-len(w)//2,r+len(w)//2+1) for w in weights])
            bank[f'3D Gaussian {sigma:g}']=np.einsum('nijk,i,j,k->n',patch[slices],*weights,optimize=True)
        zm=volume[pts[:,0],pts[:,1],np.maximum(0,pts[:,2]-1)];zp=volume[pts[:,0],pts[:,1],np.minimum(volume.shape[2]-1,pts[:,2]+1)]
        bank.update({'Z I -1':zm,'Z I +1':zp,'Z1 Std':np.stack([zm,center,zp],1).astype(float).std(1)})
        pieces.append(np.column_stack([bank[n] for n in FEATURES]).astype(np.float32))
    return np.concatenate(pieces) if pieces else np.empty((0,len(FEATURES)),np.float32)


def augment(volume,spec,seed):
    """No GT parameter: image-only transforms, spatial translation is exact integer."""
    x=np.array(volume,np.float32,copy=True);rng=np.random.default_rng(seed)
    if 'scale' in spec:x=(x-1024)*spec['scale']+1024
    if 'shift' in spec:x+=spec['shift']
    if 'gamma' in spec:x=(np.clip(x/4095,0,1)**spec['gamma'])*4095
    if 'noise' in spec:
        for z in range(x.shape[2]):x[:,:,z]+=rng.normal(0,spec['noise'],x.shape[:2]).astype(np.float32)
    if 'smooth' in spec:x=ndi.gaussian_filter(x,spec['smooth'],mode='reflect',truncate=4)
    if 'translation' in spec:x=ndi.shift(x,spec['translation'],order=0,mode='constant',cval=0,prefilter=False)
    return x.astype(np.float32,copy=False)


def transformed_xyz(xyz,spec,shape):
    mapped=np.asarray(xyz,int)+np.asarray(spec.get('translation',[0,0,0]),int)
    valid=np.all((mapped>=0)&(mapped<np.array(shape)),axis=1)
    return np.clip(mapped,0,np.array(shape)-1),valid


def columns(values,names):return values[:,[FEATURES.index(n) for n in names]]


def load_model(path):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved.get('format')!='multiclass_feature_04_v1' or saved['classes']!=CLASSES:raise ValueError('Invalid checkpoint')
    if saved['features']!=SETS[saved['feature_set']]:raise ValueError('Forbidden feature schema')
    if set(saved['train_cases'])&set(saved['validation_cases']):raise ValueError('Leakage')
    if saved['scaler']['fit_partition']!='train' or set(saved['scaler']['fit_cases'])!=set(saved['train_cases']):raise ValueError('Scaler provenance')
    if fingerprint(REPO/'configs/splits/cardiac_chd68_v1.json')!=saved['split_sha256']:raise ValueError('Split changed')
    frozen=json.loads((OUT/'protocol.json').read_text());allowed=set(frozen['train_cases'])|set(frozen['validation_cases'])
    if not set(saved['train_cases'])<=set(frozen['train_cases']) or not set(saved['validation_cases'])<=allowed:raise ValueError('Sealed case')
    model=FeatureEncoder(len(saved['features']),saved['embedding_dim'],8);model.load_state_dict(saved['state_dict'],strict=True);model.eval()
    return model,saved
