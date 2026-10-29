"""Allowlisted, self-verifying manual RunPod kit. No data upload or cloud API."""
import argparse,hashlib,zipfile
from ct_lab.dynamics_core import *

def verify(root):
    root=Path(root).resolve();manifest=json.loads((root/'MANIFEST_DYNAMICS06.json').read_text(encoding='utf-8'))
    for row in manifest['files']:
        path=(root/row['path']).resolve()
        assert path.is_relative_to(root) and path.is_file(),row['path']
        assert fingerprint(path)==row['SHA256'],row['path']
    print('BUNDLE SHA256 VERIFIED',len(manifest['files']),flush=True)

def verify_data():
    split,records=cohort();count=0
    for role in ['train','validation']:
        for case in split[role]:
            for suffix,key in [('image','image_SHA256'),('label','mask_SHA256')]:
                path=DATA/f'{case}_{suffix}.nii.gz';assert fingerprint(path)==records[case][key],str(path);count+=1
    print('PUBLIC TRAIN/DEVELOPMENT DATA VERIFIED',count,'files; test not read',flush=True)

def build():
    files={}
    for pattern in ['ct_lab/*.py','heart3d/*.py','heart3d/ml/*.py']:
        for path in REPO.glob(pattern):files[path.relative_to(REPO).as_posix()]=path
    for name in ['ct_lab/TRAINING_DYNAMICS_README.md','scripts/run_training_dynamics06_runpod.sh','requirements-dynamics06.txt',
      'tests/test_training_dynamics.py','configs/cardiac_chd68_runpod_v1.json','configs/cardiac_chd68_resolution384_v2.json',
      'configs/cardiac_chd68_preprocessing_v1.json','configs/cardiac_chd68_preprocessing_resolution384_v2.json',
      'configs/splits/cardiac_chd68_v1.json','configs/experiments/training_dynamics_06_baseline.json','configs/experiments/training_dynamics_06_runpod.json',
      'metadata/pediatric/cardiac_chd68_cohort_v1.json','metadata/pediatric/cardiac_chd68_train_statistics_v1.json']:
        files[name]=REPO/name
    for name in ['inputs/ROI05_protocol.json','data_transfer_manifest.json','manual_runpod_plan.json','environment.json','report_versions.json']:
        files['outputs/training_dynamics_06/'+name]=OUT/name
    for path in (OUT/'historical_source').rglob('*.py'):files['outputs/training_dynamics_06/'+path.relative_to(OUT).as_posix()]=path
    for v in ['v1','v2']:files[f'inputs/checkpoints/chd68_{v}_best.pt']=WEIGHTS/f'chd68_{v}_best.pt'
    manifest={'purpose':'manual existing RTX4090 execution; no provisioning','raw_CT_GT_included':False,'sealed_test_opened':False,
      'files':[{'path':name,'bytes':path.stat().st_size,'SHA256':fingerprint(path)} for name,path in sorted(files.items())]}
    archive=OUT/'runpod_manual_bundle.zip';temporary=archive.with_suffix('.partial.zip')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,path in sorted(files.items()):z.write(path,name)
        z.writestr('MANIFEST_DYNAMICS06.json',json.dumps(manifest,indent=2,ensure_ascii=False))
    with zipfile.ZipFile(temporary) as z:
        assert z.testzip() is None
        assert not any(n.endswith(('.nii','.nii.gz','.dcm','.DCM','.npy')) or '/.git/' in n for n in z.namelist())
    os.replace(temporary,archive)
    write(OUT/'bundle_receipt.json',{'path':str(archive),'SHA256':fingerprint(archive),'bytes':archive.stat().st_size,'file_count':len(files),'CRC_verified':True,'raw_data_included':False})
    print('MANUAL RUNPOD BUNDLE READY',archive,len(files),'files; not uploaded or executed',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify',type=Path);p.add_argument('--verify-data',action='store_true');a=p.parse_args()
    if a.verify:verify(a.verify)
    elif a.verify_data:verify_data()
    else:build()
