import {execFileSync} from 'node:child_process';
import path from 'node:path';
export default function setup(){
  const code=`import sys
from heart3d.web.app import data_root
from heart3d.web.db import database
from heart3d.web.cli import create_user,seed_project
root=data_root();e,f=database(root);pw=sys.stdin.readline().strip()
u=create_user(f,'resident',pw); seed_project(root,f,u)
v=create_user(f,'foreign-resident',pw); seed_project(root,f,v,'002')
u=create_user(f,'anatomy-resident',pw); seed_project(root,f,u,'003')
u=create_user(f,'ai-resident',pw)
from pathlib import Path
import numpy as np,nibabel as nib
from heart3d.web.imaging import phantom,save_mask
from heart3d.web.predictions import import_predictions
from heart3d.storage import sha256_file
ct=root/'fixtures'/'ct.nii.gz';phantom(ct);predictions=[]
for variant in ['v1','v2']:
 mask=np.zeros((96,96,64),np.uint8);mask[35:55,35:55,25:40]=1
 if variant=='v2':mask[60:70,35:45,25:40]=2
 p=root/'fixtures'/(variant+'.nii.gz');digest=save_mask(mask,ct,p)
 predictions.append((p,{'variant':variant,'expert_approved':False,'exploratory_only':True,'geometry_verified':True,'ct_sha256':sha256_file(ct),'prediction_sha256':digest,'checkpoint_sha256':'0'*64,'synthetic_fixture':True}))
project=import_predictions(root,f,'ai-resident',ct,predictions,'AI-TEST-001',synthetic=True)
create_user(f,'navigation-resident',pw)
import_predictions(root,f,'navigation-resident',ct,predictions,'NAV-TEST-001',synthetic=True)
surgeon=create_user(f,'ai-surgeon',pw,'surgeon')
from heart3d.web.db import Assignment
with f.begin() as db:db.add(Assignment(project_id=project,user_id=surgeon))
u=create_user(f,'slice-resident',pw)
slice_project=import_predictions(root,f,'slice-resident',ct,predictions,'SLICE-TEST-001',synthetic=True)
from heart3d.web.predictions import append_slice_draft
single=np.zeros((96,96,64),np.uint8);single[35:55,35:55,28]=1
p=root/'fixtures'/'single.nii.gz';digest=save_mask(single,ct,p)
append_slice_draft(root,f,'slice-resident',slice_project,p,{'variant':'slice_draft','scope':{'axis':2,'index':28,'complete_volume':False},'method':'synthetic one-plane fixture','warning':'One slice only; no expert review','review_required_classes':['LV'],'expert_approved':False,'exploratory_only':True,'geometry_verified':True,'ct_sha256':sha256_file(ct),'prediction_sha256':digest,'source_prediction_sha256':[sha256_file(x[0]) for x in predictions],'predicted_voxels':int(np.count_nonzero(single))})
e.dispose()`;
  execFileSync(path.resolve('../.venv/Scripts/python.exe'),['-c',code],{cwd:path.resolve('..'),env:process.env,
    input:process.env.HEART3D_E2E_PASSWORD+'\n',stdio:['pipe','pipe','pipe']});
  execFileSync(path.resolve('../.venv/Scripts/python.exe'),['web/e2e/seed_anatomy_fixture.py'],{cwd:path.resolve('..'),env:{...process.env,PYTHONPATH:path.resolve('..')},
    input:process.env.HEART3D_E2E_PASSWORD+'\n',stdio:['pipe','pipe','pipe']});
}
