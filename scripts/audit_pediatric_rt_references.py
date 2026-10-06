"""Verify census reference failures against complete original RTSTRUCTs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import pydicom

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from heart3d.dicom.tcia import fetch_full_series
from heart3d.dicom.rtstruct import referenced_series, heart_item
from heart3d.pediatric import write_json


def audit(row, data):
    pid=row['patient_id']; directory=data/pid
    receipt=fetch_full_series(row['rt_series_uid'],directory/'reference_review_rt')
    files=list((directory/'reference_review_rt').rglob('*.dcm'))
    if len(files)!=1:raise ValueError('Original RTSTRUCT instance count ambiguous')
    original=pydicom.dcmread(files[0])
    if str(original.SeriesInstanceUID)!=row['rt_series_uid'] or str(original.PatientID)!=pid:
        raise ValueError('Original RT identity differs from census')
    derived=pydicom.dcmread(directory/'census'/'heart_extract.dcm')
    _,full_heart=heart_item(original); _,extracted_heart=heart_item(derived)
    if full_heart!=extracted_heart or referenced_series(original)!=referenced_series(derived):
        raise ValueError('Full original RT does not match extracted Heart/references')
    ct={s['SOPInstanceUID'] for s in json.loads((directory/'census'/'ct_sops.json').read_text())}
    global_refs={s for _,_,uids in referenced_series(original) for s in uids}
    heart_refs={str(r.ReferencedSOPInstanceUID) for c in full_heart.ContourSequence for r in c.ContourImageSequence}
    result={'patient_id':pid,'source_RT_series_uid':row['rt_series_uid'],
            'original_RT_SOP_uid':str(original.SOPInstanceUID),'original_full_RTSTRUCT_downloaded':True,
            'original_archive_receipt':receipt,'full_original_Heart_matches_extract':True,
            'full_original_global_references_match_extract':True,
            'published_CT_instances':len(ct),'original_global_RT_referenced_instances':len(global_refs),
            'missing_global_RT_SOP_reference_count':len(global_refs-ct),
            'original_Heart_referenced_instances':len(heart_refs),
            'missing_Heart_SOP_reference_count':len(heart_refs-ct),
            'contours':len(full_heart.ContourSequence),
            'full_CT_downloaded':False,'status':'requires_source_reference_review',
            'reference_remapping_performed':False,'healthy_status':'unknown'}
    write_json(directory/'reference_review.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    census=json.loads((args.data/'census.json').read_text())
    rows=[r for r in census['records'] if r['status']=='requires_review'
          and any('SOP references absent' in s for s in r['exclusion_reasons'])]
    with ThreadPoolExecutor(max_workers=2) as pool:
        reports=list(pool.map(lambda r:audit(r,args.data),rows))
    write_json(args.out,{'date':'2026-10-06','method':'original full RTSTRUCT vs extracted complete Heart item and published CT SOP inventory',
                        'records':reports})
    for r in reports:print(r['patient_id'],r['missing_global_RT_SOP_reference_count'],r['missing_Heart_SOP_reference_count'],flush=True)


if __name__=='__main__':main()
