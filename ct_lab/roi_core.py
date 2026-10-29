"""Stage05 source-grid ROI geometry, frozen train-only design and provenance."""
import json
import os
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage as ndi
import ct_lab.launch
from ct_lab.core import cardiac_target,fingerprint,load_pair
from ct_lab.filtering_run import DEFAULT_DATA as DEFAULT_SOURCE
from heart3d.ml.cardiac_data import index_transform,resize_plane,restore_probability_plane
from heart3d.ml.runtime import seed_runtime

REPO=Path(__file__).resolve().parents[1];OUT=REPO/'outputs/heart_localization_roi_05';SEED=20261029
os.environ['NUMBA_CACHE_DIR']=str(OUT/'numba_cache')
DEFAULT_DATA=Path(os.environ.get('ROI05_DATA_ROOT',str(DEFAULT_SOURCE)))
WEIGHTS_ROOT=Path(os.environ.get('ROI05_WEIGHTS_ROOT','C:/Users/Andre/heart-model-recovery'))
LABELS=['Background','LV','RV','LA','RA','MYO','AO','PA']


def checkpoint_path(variant):return WEIGHTS_ROOT/f'chd68_{variant}_best.pt'


def protected_hashes():
    paths=[REPO/'configs/splits/cardiac_chd68_v1.json',REPO/'configs/cardiac_chd68_preprocessing_v1.json',REPO/'configs/cardiac_chd68_preprocessing_resolution384_v2.json']
    paths+=list(WEIGHTS_ROOT.glob('chd68_*_best.pt'));return {str(p):fingerprint(p) for p in paths}


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def prior_hashes():
    files=[p for name in ('filtering_feature_01','spatial_feature_02','learnable_feature_03','multiclass_feature_04') for p in (REPO/'outputs'/name).rglob('*') if p.is_file()]
    files += [REPO/name for name in ('DATA_PIPELINE_AUDIT.md','IMAGECHD68_DATA_ANALYSIS.md','FILTERING_FEATURE_EXPERIMENT_01.md','SPATIAL_FEATURE_EXPERIMENT_02.md','LEARNABLE_FEATURE_EXPERIMENT_03.md','MULTICLASS_FEATURE_EXPERIMENT_04.md')]
    return {str(p.relative_to(REPO)):fingerprint(p) for p in sorted(files)}


def cohort():
    split=json.loads((REPO/'configs/splits/cardiac_chd68_v1.json').read_text())['partitions']
    records={r['case_id']:r for r in json.loads((REPO/'metadata/pediatric/cardiac_chd68_cohort_v1.json').read_text())['records']}
    assert len(split['train'])==48 and len(split['validation'])==10
    assert not set(split['train'])&set(split['validation']) and not (set(split['train'])|set(split['validation']))&set(split['test'])
    assert not {records[c]['group_id'] for c in split['train']}&{records[c]['group_id'] for c in split['validation']}
    return split,records


def freeze():
    split,records=cohort();groups=sorted({records[c]['group_id'] for c in split['train']});np.random.default_rng(SEED).shuffle(groups)
    folds=[[c for c in split['train'] if groups.index(records[c]['group_id'])%3==f] for f in range(3)]
    plan={'experiment':'heart_localization_roi_05','seed':SEED,'train_cases':split['train'],'validation_cases':split['validation'],'CV_folds':folds,
          'split_sha256':fingerprint(REPO/'configs/splits/cardiac_chd68_v1.json'),'sealed_test_read':False,'ignore':255,
          'validation_status':'reused development validation, never independent final evaluation; no parameter choice from outer ten cases',
          'localizer':{'shape':[64,64,64],'channels':[4,8,16],'epochs':24,'batch_size':2,'lr':.001,'loss':'masked BCE+foreground soft Dice',
                       'threshold':.4,'components':'largest foreground component; empty/mean confidence<0.5 -> fixed train ROI',
                       'margin_candidates':[.1,.2,.3],'extra_margin_indices':8,'selection':'smallest margin with OOF mean foreground coverage>=.995 and minimum class mean coverage>=.98; else .3, report shortfalls',
                       'final_epochs':'median best CV epoch, fixed before outer validation',
                       'augmentation':{'axial_rotation_degrees':5,'scale_delta':.05,'translation_fraction':.03,'intensity_shift':.03,'noise_std':.01,'flips':False}},
          'fixed_ROI':'train-only normalized GT bbox 2nd/98th percentiles + .05 FOV; no validation GT input',
          'model':{'input_channels':5,'encoder_channels':[16,32,64,128],'norm_groups':8,'output_channels':8},
          'segmentation':{'epochs':20,'steps_per_epoch':192,'batch_size':2,'lr':.001,'optimizer':'AdamW','weight_decay':.0001,'loss':'CE+Dice, ignore255',
                          'CV_epochs':6,'CV_steps_per_epoch':48,'mixed_precision':'CUDA float16 GradScaler; CPU float32',
                          'context_offsets':[-2,-1,0,1,2],'source_clip':[0.,2015.],'normalization':'frozen CHD68 train-only clip -> [-1,1]',
                          'training_centers':'12 uniform Z positions inside train GT union bbox + four background positions outside if available, same centers B/C/D',
                          'augmentation':{'rotation_degrees':5,'scale_delta':.05,'shift_fraction':.03,'intensity_shift':.03,'noise_std':.01,'flips':False},
                          'selection':'fixed last epoch, no outer validation early stopping; CV B/D256 checks full pipeline before main runs'},
          'main_jobs':[{'name':'B_full_256','mode':'full','size':256,'branch':False},{'name':'C_oracle_256','mode':'oracle','size':256,'branch':False},
                       {'name':'D_predicted_256','mode':'predicted','size':256,'branch':False},{'name':'B_full_384','mode':'full','size':384,'branch':False},
                       {'name':'D_predicted_384','mode':'predicted','size':384,'branch':False},{'name':'D_core_256','mode':'predicted','size':256,'branch':True}],
          'branch':'optional native 3D Mean/Std/Gradient/LoG, no Gaussian; fixed source-count scaling [mean/2015*2-1,std/512,gradient/512,LoG/512], encoder1 additive fusion',
          'training_ROI':'final localizer predictions for train CT, no GT correction; CV localizer fitted only corresponding segmentation train fold',
          'evaluation_cases':['ct_1001',*split['validation']],'resize_diagnostic_cases':['ct_1001','ct_1002','ct_1004'],
          'comparison':'immutable CHD68 v1/v2; new matched B/C/D, all restored to native grid before metrics; no component removal',
          'geometry':'native RAS XYZ, half-open bboxes, index units; no verified HU or millimeters',
          'oracle':'explicit diagnostic upper-bound mechanism only; not automatic, includes GT at development inference, reported separately',
          'resource_gate':'CPU real-data smoke + bounded RTX2060 benchmark must pass before training; local only, no paid cloud launch',
          'budget_note':'bounded scratch research pilot with equal B/C/D sample/step budget; recovered CHD68 historical training budget differs'}
    OUT.mkdir(parents=True,exist_ok=True);path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==plan,'Frozen stage05 protocol differs'
    else:write(path,plan);write(OUT/'prior_files.json',prior_hashes());write(OUT/'protected_files.json',protected_hashes())
    return plan,records


def bbox(binary):
    if not np.any(binary):return None
    lo=[];hi=[]
    for a in range(3):
        active=np.flatnonzero(np.any(binary,axis=tuple(i for i in range(3) if i!=a)));lo.append(int(active[0]));hi.append(int(active[-1])+1)
    return np.array([lo,hi],int)


def clip_box(box,shape):
    b=np.array(box,int,copy=True);b[0]=np.clip(b[0],0,np.array(shape)-1);b[1]=np.clip(b[1],b[0]+1,shape);return b


def expand_box(box,shape,fraction=.2,extra=8):
    b=np.asarray(box,int);margin=np.ceil((b[1]-b[0])*fraction).astype(int)+extra
    return clip_box(np.stack([b[0]-margin,b[1]+margin]),shape)


def slices(box):return tuple(slice(int(a),int(b)) for a,b in zip(*box))


def transform(shape,box,size):
    b=clip_box(box,shape);t=index_transform([*(b[1,:2]-b[0,:2]),int(shape[2])],size)
    return {'source_shape_xyz':list(shape),'bbox_xyz':b.tolist(),'resize':t,'size':size,'voxel_mapping':'pixel centers; crop offset then production XY resize/pad; original Z'}


def prepare_plane(plane,t,order=1,fill=0):
    b=np.array(t['bbox_xyz']);return resize_plane(plane[slice(b[0,0],b[1,0]),slice(b[0,1],b[1,1])],t['resize'],order,fill)


def restore_plane(prob,t,device='cpu'):
    """Production pixel-center probability restoration, optionally CUDA for throughput."""
    if isinstance(prob,np.ndarray):p=torch.from_numpy(prob).to(device)
    else:p=prob
    rx,ry=t['resize']['resized_xy'];px,py=t['resize']['pad_xy'];b=np.array(t['bbox_xyz']);nx,ny=b[1,:2]-b[0,:2]
    return torch.nn.functional.interpolate(p[None,:,py:py+ry,px:px+rx].float(),size=(int(ny),int(nx)),mode='bilinear',align_corners=False)[0].transpose(-1,-2)


def coverage(target,box):
    valid=target<8;fg=(target>0)&valid;b=clip_box(box,target.shape);inside=target[slices(b)]
    gt=bbox(fg);volume=int(np.prod(b[1]-b[0]));counts={str(c):int(np.count_nonzero(target==c)) for c in range(1,8)}
    kept={str(c):int(np.count_nonzero(inside==c)) for c in range(1,8)}
    inter=np.prod(np.maximum(0,np.minimum(b[1],gt[1])-np.maximum(b[0],gt[0]))) if gt is not None else 0
    union=volume+int(np.prod(gt[1]-gt[0]))-inter if gt is not None else volume
    return {'bbox':b.tolist(),'foreground_coverage':float(sum(kept.values())/max(1,sum(counts.values()))),
            'foreground_outside_fraction':float(1-sum(kept.values())/max(1,sum(counts.values()))),
            'per_class_coverage':{str(c):float(kept[str(c)]/counts[str(c)]) if counts[str(c)] else None for c in range(1,8)},
            'bbox_IoU':float(inter/max(1,union)),'center_distance_index':float(np.linalg.norm((b.mean(0)-gt.mean(0)))) if gt is not None else None,
            'ROI_to_GT_bbox_volume_ratio':float(volume/max(1,np.prod(gt[1]-gt[0]))) if gt is not None else None,
            'ROI_to_full_FOV':float(volume/np.prod(target.shape)),'all_structures_fully_covered':bool(all(kept[k]==counts[k] for k in counts))}
