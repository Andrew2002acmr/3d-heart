"""Convert full pilot studies, validate independently, keep original geometry."""
import argparse
import json
from pathlib import Path
import sys
import time
import pydicom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from heart3d.dicom.ct import inspect_series, load_volume, save_nifti
from heart3d.dicom.rtstruct import rasterize_heart
from heart3d.dicom.qa import check_simpleitk, check_vtk, make_mosaic
from heart3d.dicom.tcia import fetch_ct_probe
from heart3d.pediatric import write_json


def prepare_case(row, data):
    directory = data / row['patient_id']
    report = dict(row, status='requires_review', visual_review='pending', healthy_status='unknown')
    start = time.perf_counter()
    try:
        _, sops, _ = fetch_ct_probe(row['ct_series_uid'], directory / 'census')
        geometry = inspect_series(directory / 'ct', sops, row['ct_series_uid'])
        rt_files = [p for p in (directory / 'rtstruct').rglob('*') if p.suffix.lower() == '.dcm']
        if len(rt_files) != 1:
            raise ValueError('Need one original full RTSTRUCT file')
        rt = pydicom.dcmread(rt_files[0])
        if str(rt.PatientID) != row['patient_id'] or geometry.report['patient_id'] != row['patient_id']:
            raise ValueError('Patient identity mismatch')
        mask, mask_report, polygons = rasterize_heart(rt, geometry)
        volume = load_volume(geometry)
        independent = check_simpleitk(geometry, volume)
        independent_mask = check_vtk(mask, polygons, mask_report['combination'] == 'XOR')
        prepared = directory / 'prepared'; prepared.mkdir(exist_ok=True)
        save_nifti(volume, geometry.affine_ras, prepared / 'ct_original.nii.gz')
        save_nifti(mask, geometry.affine_ras, prepared / 'heart_gt_original.nii.gz')
        make_mosaic(volume, mask, geometry, polygons, prepared / 'qa_native.png',
                    row['patient_id']+f" | age {row['age']} | {row['scanner']}")
        report.update(CT=geometry.report, Heart=mask_report, independent_reader=independent,
                      independent_rasterizer=independent_mask, full_CT_downloaded=True,
                      original_full_RTSTRUCT_downloaded=True, mask_rasterized=True,
                      status=mask_report['initial_status'], output_directory=str(prepared),
                      CT_array_MiB=volume.nbytes / 2**20, mask_array_MiB=mask.nbytes / 2**20)
    except Exception as error:
        report['reason'] = f'{type(error).__name__}: {error}'
    report['conversion_and_QA_seconds'] = time.perf_counter()-start
    write_json(directory / 'pilot_qa.json', report)
    return report


def main():
    p = argparse.ArgumentParser(); p.add_argument('--data', type=Path, required=True)
    args = p.parse_args()
    cases = json.loads((args.data/'pilot_selection.json').read_text())['cases']
    for row in cases:
        if not (args.data / row['patient_id'] / 'full_download_receipt.json').exists():
            print('WAITING full original studies', row['patient_id'], flush=True); continue
        existing = args.data / row['patient_id'] / 'pilot_qa.json'
        if existing.exists() and json.loads(existing.read_text()).get('mask_rasterized'):
            continue
        result = prepare_case(row, args.data)
        print('QA', row['patient_id'], result['status'], result.get('reason', ''), flush=True)


if __name__ == '__main__': main()
