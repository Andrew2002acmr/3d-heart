"""Record an explicit technical visual decision with hashed QA evidence."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import pydicom
from pydicom.uid import RTStructureSetStorage

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from heart3d.storage import sha256_file
from heart3d.pediatric import write_json

STATUSES={'approved','coverage_incomplete','coverage_review','annotation_scope_review',
          'reference_failure','empty_roi','geometry_failure','identity_review'}


def record_review(root, patient, status, note, destination):
    report=json.loads((root/patient/'pilot_qa.json').read_text())
    picture=root/patient/'prepared'/'qa_review.png'
    if status=='approved':
        if not report.get('mask_rasterized') or report.get('qa_sampling_version')!=2:
            raise ValueError('Cannot approve without full technical QA and sampling v2')
        if not report.get('prepared_CT_sha256') or not report.get('prepared_mask_sha256'):
            raise ValueError('Wait for final prepared-data hashes before review')
        reader=report['independent_reader']; alternate=report['independent_rasterizer']; heart=report['Heart']
        if not (reader['geometry_match'] and reader['array_match'] and alternate['nonboundary_differences']==0
                and alternate['Dice_between_rasterizers']>=.99):
            raise ValueError('Independent checks failed')
        if any(heart['touches_grid_faces']) or heart['near_scan_z_boundary'] or heart['internal_uncontoured_slices'] or heart['components_26']!=1:
            raise ValueError('Unresolved boundary/gap/topology gate cannot be approved')
        rtfiles=list((root/patient/'rtstruct').rglob('*.dcm'))
        if len(rtfiles)!=1:raise ValueError('Need original RTSTRUCT')
        ds=pydicom.dcmread(rtfiles[0],specific_tags=['Modality','SOPClassUID','SeriesInstanceUID','PatientID'])
        if str(ds.SeriesInstanceUID)!=report['rt_series_uid'] or str(ds.PatientID)!=patient or ds.SOPClassUID!=RTStructureSetStorage:
            raise ValueError('Original RTSTRUCT identity mismatch')
    if not picture.exists() and status not in {'geometry_failure','reference_failure','empty_roi'}:
        raise ValueError('Visual evidence missing')
    frozen=Path(__file__).resolve().parents[1]/'configs'/'splits'/'pediatric_ct_heart_v1.json'
    if frozen.exists():raise ValueError('Baseline v1 split is frozen; reviews must use a new cohort version')
    data=json.loads(destination.read_text()) if destination.exists() else {'version':1,'records':[]}
    row={'patient_id':patient,'status':status,'reason':note,
         'reviewer':'Codex technical visual QA; no new clinician annotation sign-off',
         'target':'original expert Heart OAR; full chamber/vessel completeness not asserted',
         'healthy_status':'unknown','date_utc':datetime.now(timezone.utc).isoformat(),
         'sampling_version':2,'QA_relative_path':f'{patient}/prepared/qa_review.png',
         'QA_SHA256':sha256_file(picture) if picture.exists() else None,
         'report_SHA256':sha256_file(root/patient/'pilot_qa.json'),
         'mask_SHA256':report.get('prepared_mask_sha256'),
         'CT_SHA256':report.get('prepared_CT_sha256'),
         'source_annotation_modified':False}
    data['records']=[r for r in data['records'] if r['patient_id']!=patient]+[row]
    data['records'].sort(key=lambda r:r['patient_id'])
    write_json(destination,data)
    print(patient,status,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--patient',required=True)
    p.add_argument('--status',choices=sorted(STATUSES),required=True);p.add_argument('--note',required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    record_review(a.data,a.patient,a.status,a.note,a.out)
