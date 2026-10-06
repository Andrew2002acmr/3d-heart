"""Export actual QA evidence, then freeze an approved cohort and patient split."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from heart3d.cohort import freeze_patient_split
from heart3d.pediatric import write_json
from heart3d.storage import sha256_file


def export_records(root, registry, reviews):
    records=[]
    for review in reviews:
        pid=review['patient_id']; report_path=root/pid/'pilot_qa.json'
        if sha256_file(report_path)!=review['report_SHA256']:
            raise ValueError(f'QA evidence changed after review: {pid}')
        report=json.loads(report_path.read_text()); row=registry[pid]
        ct=report.get('CT',{}); heart=report.get('Heart',{})
        passed=bool(report.get('mask_rasterized'))
        record={k:row.get(k) for k in ('patient_id','age','age_group','age_precision','scanner',
            'contrast_status','contrast_agent','contrast_route','convolution_kernel','ct_series_uid','rt_series_uid')}
        receipt=(root/pid/'full_download_receipt.json').exists()
        record.update(source='Pediatric-CT-SEG / TCIA',healthy_status='unknown',review_status=review['status'],
            review=review, full_CT_downloaded=receipt,
            original_RTSTRUCT_downloaded=receipt,
            DICOM_gate_passed=bool(report.get('DICOM_gate_passed',passed)),
            RTSTRUCT_gate_passed=bool(report.get('RTSTRUCT_gate_passed',passed)),
            geometry=ct,Heart=heart,independent_reader=report.get('independent_reader'),
            independent_rasterizer=report.get('independent_rasterizer'),
            ct_relative_path=f'{pid}/prepared/ct_original.nii.gz',
            mask_relative_path=f'{pid}/prepared/heart_gt_original.nii.gz')
        record['related_patient_ids']=review.get('related_patient_ids',[])
        if review['status']=='approved':
            for name,key in [('ct_original.nii.gz','CT_SHA256'),('heart_gt_original.nii.gz','mask_SHA256')]:
                if sha256_file(root/pid/'prepared'/name)!=review[key]:
                    raise ValueError(f'Prepared data changed: {pid}')
            fingerprint=json.loads((root/pid/'prepared'/'ct_fingerprint.json').read_text())
            if fingerprint['source_NIfTI_SHA256']!=review['CT_SHA256']:
                raise ValueError('Content fingerprint does not match reviewed CT')
            record['image_voxel_SHA256']=fingerprint['voxel_SHA256']
        records.append(record)
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','registry','reviews','queue','data','cohort','split','audit'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--per-age',type=int,default=20)
    a=p.parse_args()
    if a.cohort.exists() or a.split.exists():
        raise ValueError('Cohort/split already frozen; create a new version to change membership')
    config=json.loads(a.config.read_text()); queue=json.loads(a.queue.read_text())
    registry={r['patient_id']:r for r in json.loads(a.registry.read_text())['records']}
    reviews=json.loads(a.reviews.read_text())['records']
    reviewed={r['patient_id'] for r in reviews}
    unreviewed=[path.parent.name for path in a.data.glob('Pediatric-CT-SEG-*/pilot_qa.json')
                if path.parent.name not in reviewed]
    if unreviewed:
        raise ValueError(f'Complete explicit reviews before freezing: {unreviewed}')
    records=export_records(a.data,registry,reviews); by={r['patient_id']:r for r in records}
    priority=config['development_pilot_patient_ids']+[r['patient_id'] for r in queue['records']]
    selected=[]; counts=Counter()
    for pid in priority:
        row=by.get(pid)
        if row and row['review_status']=='approved' and counts[row['age_group']]<a.per_age:
            selected.append(row); counts[row['age_group']]+=1
    if any(counts[g]<3 for g in ('2-5','6-11','12-17')):
        raise ValueError('Insufficient approved representation for a three-way age-stratified split')
    split=freeze_patient_split(selected,config['seed'],set(config['development_pilot_patient_ids']))
    cohort={'version':1,'target':'original expert Heart OAR; anatomical chamber/vessel completeness not asserted',
        'healthy_status':'unknown','source_registry_SHA256':sha256_file(a.registry),
        'reviews_SHA256':sha256_file(a.reviews),'queue_SHA256':sha256_file(a.queue),
        'selection':'first approved patients per age in declared metadata queue, development patients first',
        'clinical_signoff':'original dataset expert labels; current approval is technical/visual only',
        'age_counts':dict(counts),'records':selected}
    write_json(a.audit,{'version':1,'counts':dict(Counter(r['review_status'] for r in records)),
        'DICOM_passed':sum(r['DICOM_gate_passed'] for r in records),
        'RTSTRUCT_passed':sum(r['RTSTRUCT_gate_passed'] for r in records),'records':records})
    write_json(a.cohort,cohort)
    write_json(a.split,{'version':1,'frozen':True,'seed':config['seed'],
        'cohort_manifest_SHA256':sha256_file(a.cohort),'split_unit':'patient',
        'development_patients_excluded_from_test':config['development_pilot_patient_ids'],
        'algorithm':'largest-remainder age quotas; scanner/contrast evidence balanced deterministic greedy selection',
        'counts':{k:len(v) for k,v in split.items()},'partitions':split})
    print('COHORT',dict(counts),'SPLIT',{k:len(v) for k,v in split.items()})


if __name__=='__main__':main()
