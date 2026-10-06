"""Content fingerprints, not DICOM IDs, to screen exact image duplicates before split."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import nibabel as nib
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from heart3d.pediatric import write_json
from heart3d.storage import sha256_file


def fingerprint(path, destination):
    source_hash=sha256_file(path)
    if destination.exists():
        result=json.loads(destination.read_text())
        if result['source_NIfTI_SHA256']!=source_hash:raise ValueError('Image changed since fingerprinting')
        return result
    image=nib.load(path);array=np.asanyarray(image.dataobj)
    digest=hashlib.sha256()
    for z in range(array.shape[2]):
        digest.update(np.asarray(array[:,:,z],dtype='<f4',order='C').tobytes())
    result={'source_NIfTI_SHA256':source_hash,'voxel_SHA256':digest.hexdigest(),
        'voxel_hash_order':'ascending z, each XY plane C-order, little-endian float32 HU',
        'shape_xyz':list(array.shape),'affine_ras':image.affine.tolist()}
    write_json(destination,result);return result


def audit(root,out):
    groups=defaultdict(list);records=[]
    for path in sorted(root.glob('Pediatric-CT-SEG-*/prepared/ct_original.nii.gz')):
        # Some failed conversions may leave an image without a passed technical gate.
        report_path=path.parent.parent/'pilot_qa.json'
        if not report_path.exists() or not json.loads(report_path.read_text()).get('mask_rasterized'):continue
        result=fingerprint(path,path.parent/'ct_fingerprint.json')
        pid=path.parent.parent.name
        key=(tuple(result['shape_xyz']),result['voxel_SHA256']);groups[key].append(pid)
        records.append({'patient_id':pid,**result});print('FINGERPRINT',pid,flush=True)
    duplicates=[ids for ids in groups.values() if len(ids)>1]
    summary={'version':1,'method':'exact decoded HU voxel content, independent of released PatientID/UID',
        'limitation':'cannot establish clinical identity or exclude repeat acquisitions with different pixels',
        'patients_screened':len(records),'exact_duplicate_groups':duplicates,'records':records}
    write_json(out,summary);print('EXACT DUPLICATE GROUPS',duplicates,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();audit(a.data,a.out)
