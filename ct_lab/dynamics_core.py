"""Stage06 isolation, immutable study protocol and release-group guards."""
import hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch
import ct_lab.launch
from ct_lab.core import cardiac_target,fingerprint
from heart3d.ml.cardiac_data import load_protocol
from heart3d.ml.runtime import seed_runtime

REPO=Path(__file__).resolve().parents[1]
OUT=Path(os.environ.get('DYNAMICS06_OUTPUT_ROOT',str(REPO/'outputs/training_dynamics_06')))
DATA=Path(os.environ.get('DYNAMICS06_DATA_ROOT',str(REPO.parent/'laboratories/data/chd68/pairs')))
WEIGHTS=Path(os.environ.get('DYNAMICS06_WEIGHTS_ROOT','C:/Users/Andre/heart-model-recovery'))
CONFIG=REPO/'configs/experiments/training_dynamics_06_baseline.json'
LABELS=['Background','LV','RV','LA','RA','MYO','AO','PA']
os.environ['NUMBA_CACHE_DIR']=str(OUT/'numba_cache')

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.partial')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');os.replace(temporary,path)

def write_if_changed(path,value):
    """Preserve frozen JSON bytes, including original platform line endings."""
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))==value:return False
    write(path,value);return True

def write_text_if_changed(path,value):
    """An unchanged exported source must retain its manifest SHA across OSes."""
    if path.exists() and path.read_text(encoding='utf-8')==value:return False
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value,encoding='utf-8');return True

def atomic_checkpoint(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.partial.pt')
    torch.save(value,temporary);os.replace(temporary,path)

def cohort():
    base=json.loads((REPO/'configs/cardiac_chd68_runpod_v1.json').read_text());rows,split=load_protocol(base)
    assert [len(split['partitions'][r]) for r in ('train','validation','test')]==[48,10,10]
    return split['partitions'],rows

def allowed(case):
    split,_=cohort()
    if case not in split['train']+split['validation']:raise ValueError('Sealed test/unknown case must not be opened')

def protected_files():
    paths=[REPO/n for n in ('configs/splits/cardiac_chd68_v1.json','configs/cardiac_chd68_runpod_v1.json','configs/cardiac_chd68_resolution384_v2.json','configs/cardiac_chd68_preprocessing_v1.json','configs/cardiac_chd68_preprocessing_resolution384_v2.json')]
    paths += [WEIGHTS/f'chd68_{v}_best.pt' for v in ('v1','v2')]
    return {str(p):fingerprint(p) for p in paths}

def previous_files():
    directories=['filtering_feature_01','spatial_feature_02','learnable_feature_03','multiclass_feature_04','heart_localization_roi_05']
    paths=[p for name in directories for p in (REPO/'outputs'/name).rglob('*') if p.is_file()]
    paths += [p for p in REPO.glob('*.md') if p.name in ('DATA_PIPELINE_AUDIT.md','IMAGECHD68_DATA_ANALYSIS.md','FILTERING_FEATURE_EXPERIMENT_01.md','SPATIAL_FEATURE_EXPERIMENT_02.md','LEARNABLE_FEATURE_EXPERIMENT_03.md','MULTICLASS_FEATURE_EXPERIMENT_04.md','HEART_LOCALIZATION_ROI_SEGMENTATION_05.md')]
    return {str(p.relative_to(REPO)):fingerprint(p) for p in sorted(paths)}

def protocol():return json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))

def training_cases(n=48,subset_seed=20261029):
    split,rows=cohort();groups=sorted({rows[c]['group_id'] for c in split['train']});np.random.default_rng(subset_seed).shuffle(groups)
    selected=[c for group in groups[:n] for c in split['train'] if rows[c]['group_id']==group]
    if len(selected)!=n:raise ValueError('Subset size/group cardinality mismatch')
    return selected

def initialize():
    OUT.mkdir(parents=True,exist_ok=True)
    if not (OUT/'protected_files.json').exists():write(OUT/'protected_files.json',protected_files())
    if not (OUT/'previous_files.json').exists():
        print('SNAPSHOT previous stages (read-only SHA256)',flush=True);write(OUT/'previous_files.json',previous_files())
