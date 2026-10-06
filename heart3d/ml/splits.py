import json
from pathlib import Path
from heart3d.storage import sha256_file


def load_frozen_split(path, cohort_path):
    data=json.loads(Path(path).read_text())
    if not data.get('frozen') or data['split_unit']!='patient':
        raise ValueError('A frozen patient-level split is required')
    if data['cohort_manifest_SHA256']!=sha256_file(cohort_path):
        raise ValueError('Cohort manifest differs from the frozen split')
    cohort=json.loads(Path(cohort_path).read_text())['records']
    expected={r['patient_id']:r for r in cohort}
    ids=[r['patient_id'] for rows in data['partitions'].values() for r in rows]
    if len(ids)!=len(set(ids)) or set(ids)!=set(expected):
        raise ValueError('Patient leakage or incomplete cohort assignment')
    hashes=[r['image_voxel_SHA256'] for r in cohort if r.get('image_voxel_SHA256')]
    if len(hashes)!=len(set(hashes)) or any(set(r.get('related_patient_ids',[]))&set(ids) for r in cohort):
        raise ValueError('Image duplicate or unresolved identity leakage')
    if set(data['partitions'])!={'train','validation','test'}:
        raise ValueError('Three explicit partitions required')
    for rows in data['partitions'].values():
        for row in rows:
            if row!=expected[row['patient_id']] or row['review_status']!='approved':
                raise ValueError('Partition metadata is not the approved cohort record')
            if not row['geometry'].get('physical_geometry_confirmed_from_DICOM'):
                raise ValueError('Physical DICOM geometry not confirmed')
    dev=set(data['development_patients_excluded_from_test'])
    if any(r['patient_id'] in dev for r in data['partitions']['test']):
        raise ValueError('Engineering development patient leaked into test')
    return data
