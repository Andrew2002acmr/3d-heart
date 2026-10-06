"""Verify published source ZIPs against download receipts without modifying DICOM."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from heart3d.pediatric import write_json
from heart3d.storage import exclusive_file_lock, sha256_file


def verify(root, output, incident_path=None):
    records = []
    for receipt_path in sorted(root.glob('Pediatric-CT-SEG-*/full_download_receipt.json')):
        receipt = json.loads(receipt_path.read_text())
        patient = receipt_path.parent.name
        if receipt['patient_id'] != patient:
            raise ValueError('Receipt patient differs from directory')
        row = {'patient_id': patient, 'archives': {}}
        for key, name in [('CT', 'ct.zip'), ('RTSTRUCT', 'rtstruct.zip')]:
            path = receipt_path.parent / name
            digest = sha256_file(path)
            if digest != receipt[key]['sha256']:
                raise ValueError(f'Source archive hash mismatch: {patient}/{name}')
            row['archives'][key] = {'SHA256': digest, 'bytes': path.stat().st_size}
        records.append(row)
        if len(records) % 20 == 0:
            print('VERIFIED SOURCE PAIRS', len(records), flush=True)
    if not records:
        raise ValueError('No complete source receipts found')
    summary = {'version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'verified_patient_pairs': len(records), 'verified_archives': 2 * len(records),
        'archive_bytes': sum(a['bytes'] for r in records for a in r['archives'].values()),
        'hash_mismatches': [], 'raw_DICOM_modified': False,
        'verification': 'SHA256 equals receipt produced after downloader ZIP CRC/extraction verification',
        'records': records}
    write_json(output, summary)
    if incident_path:
        incident = json.loads(incident_path.read_text())
        verified = {r['patient_id'] for r in records}
        retry = incident['missing_receipts_to_retry']
        if not set(retry).issubset(verified):
            raise ValueError('Incident recovery has incomplete patient receipts')
        # Preserve the historical report and failed cache; add explicit resolution.
        incident['resolution'] = {'recovered_patient_ids': retry,
            'all_current_source_archives_match_receipts': True,
            'source_verification_summary_SHA256': sha256_file(output),
            'verified_patient_pairs': len(records), 'verified_archives': 2 * len(records),
            'existing_DICOM_modified_or_deleted': False,
            'prevention': 'OS-exclusive cohort preparation lock for this data root'}
        incident['pending_retry_patient_ids'] = []
        write_json(incident_path, incident)
        for patient in verified:
            failure = root / patient / 'cohort_failure.json'
            if failure.exists():
                write_json(root / patient / 'download_failure_resolution.json', {
                    'historical_failure_SHA256': sha256_file(failure),
                    'download_resolved': True, 'CT_RT_archives_SHA256_verified': True,
                    'technical_suitability': 'see subsequent pilot_qa.json and visual review',
                    'historical_failure_deleted': False})
    print('SOURCE INTEGRITY', len(records), 'pairs;', 2 * len(records), 'archives', flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--incident', type=Path)
    args = parser.parse_args()
    with exclusive_file_lock(args.data / 'cache' / 'cohort_preparation.lock'):
        verify(args.data, args.out, args.incident)
