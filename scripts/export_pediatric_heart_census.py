"""Export deidentified evidence, preserving every pending/full-stack/visual gate."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from heart3d.dicom.tcia import fetch_heart_extract
from heart3d.dicom.rtstruct import referenced_series, heart_item, validate_references
from heart3d.pediatric import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--reviews', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    census = json.loads((args.data/'census.json').read_text())
    reviews = json.loads(args.reviews.read_text())['records']
    by_review = {r['patient_id']: r for r in reviews}
    if not census['census_complete'] or len(census['records']) != census['expected_patients']:
        raise ValueError('Cannot export an incomplete 327-patient census')
    pilot_summaries = []
    for row in census['records']:
        pid = row['patient_id']; directory = args.data/pid
        extract_path = directory/'census'/'heart_extract.dcm'
        if extract_path.exists():
            # Cache-only migration to distinct derived UIDs; original RT remains unchanged.
            rt, receipt = fetch_heart_extract(row['rt_series_uid'], directory/'census')
            row['RT_heart_extract_sha256'] = receipt['derived_sha256']
            row['RT_source_sop_uid'] = receipt['source_rt_sop_uid']
            row['RT_derived_extract_sop_uid'] = receipt['derived_sop_uid']
            inventory = json.loads((directory/'census'/'ct_sops.json').read_text())
            sops = {s['SOPInstanceUID'] for s in inventory}
            row['missing_global_RT_SOP_reference_count'] = len({s for _,_,uids in referenced_series(rt) for s in uids}-sops)
            if row['heart_roi_present']:
                try:
                    _, item = heart_item(rt)
                    refs = {str(r.ReferencedSOPInstanceUID) for c in item.ContourSequence for r in c.ContourImageSequence}
                    row['missing_Heart_SOP_reference_count'] = len(refs-sops)
                    try:
                        validate_references(rt,row['ct_series_uid'],row['frame_uid'],sops)
                    except ValueError as error:
                        row['status']='requires_review'
                        row['exclusion_reasons']=list(dict.fromkeys(row['exclusion_reasons']+[f'ValueError: {error}']))
                except ValueError:
                    row['missing_Heart_SOP_reference_count'] = None
        report_path = directory/'pilot_qa.json'
        if not report_path.exists(): continue
        report = json.loads(report_path.read_text())
        report.pop('output_directory', None)
        review = by_review.get(pid, {'patient_id':pid, 'alignment':'pending', 'status':'requires_review',
                                    'anatomical_annotation_scope':'pending', 'notes':[]})
        report['visual_review'] = review
        report['status'] = review['status']
        row['full_stack_QA'] = report
        row['full_original_rtstruct_downloaded'] = report.get('original_full_RTSTRUCT_downloaded', False)
        row['full_ct_geometry_verified'] = report.get('CT',{}).get('physical_geometry_confirmed_from_DICOM', False)
        row['can_rasterize_on_full_CT'] = report.get('mask_rasterized', False)
        row['actual_slice_spacing_mm'] = report.get('CT',{}).get('spacing_xyz_mm', [None,None,None])[2]
        row['scan_coverage'] = review.get('scan_coverage', 'requires_review')
        row['status'] = review['status']
        row['technical_DICOM_gate_passed'] = report.get('mask_rasterized',False) and review['alignment']=='pass'
        row['suitable_for_training'] = review['status']=='approved_for_Heart_OAR_baseline'
        row['pending_checks'] = [] if row['suitable_for_training'] else review.get('pending_checks',['annotation_extent_review'])
        row['exclusion_reasons'] = review.get('exclusion_reasons', [])
        row['original_source_archive_receipts'] = json.loads((directory/'full_download_receipt.json').read_text())
        pilot_summaries.append(report)
    records = census['records']
    summary = {'patients':len(records), 'by_age_group':dict(Counter(r['age_group'] for r in records)),
               'reported_age_min_max':[min(r['age'] for r in records), max(r['age'] for r in records)],
               'by_scanner':dict(Counter(r['scanner'] for r in records)),
               'by_contrast_evidence':dict(Counter(r.get('contrast_status','unknown') for r in records)),
               'by_status':dict(Counter(r['status'] for r in records)),
               'Heart_present':sum(r['heart_roi_present'] is True for r in records),
               'no_ROI_availability_unknown':sum(r['heart_roi_present'] is None for r in records)==0,
               'Heart_missing_or_empty':sum(r['status']=='excluded' and 'missing_or_empty_Heart_ROI' in r['exclusion_reasons'] for r in records),
               'metadata_references_passed':sum(r.get('Heart_SOP_references_verified',False) for r in records),
               'full_studies_downloaded':len(pilot_summaries),
               'full_studies_rasterized':sum(p.get('mask_rasterized',False) for p in pilot_summaries),
               'technical_DICOM_gate_passed':sum(r.get('technical_DICOM_gate_passed',False) for r in records),
               'pilot_incomplete_coverage':sum(r['status']=='excluded_incomplete_heart_coverage' for r in records),
               'fully_approved_for_training':sum(r['suitable_for_training'] for r in records),
               'CT_published_bytes':sum(r.get('CT_published_bytes',0) for r in records),
               'RT_published_bytes':sum(r.get('RT_published_bytes',0) for r in records)}
    output = dict(census, records=records, summary=summary,
                  preparation_complete_for_entire_cohort=False,
                  source='https://www.cancerimagingarchive.net/collection/pediatric-ct-seg/',
                  clinical_labels_available=False, date='2026-10-06',
                  evidence_levels=['public inventory + CT probe', 'complete Heart ROI extract',
                                   'full original studies + original-grid QA for pilot only'])
    write_json(args.out, output)
    write_json(args.out.with_name('heart_segmentation_pilot_qa.json'), {'date':'2026-10-06', 'records':pilot_summaries})
    print(json.dumps(summary, indent=2))


if __name__=='__main__': main()
