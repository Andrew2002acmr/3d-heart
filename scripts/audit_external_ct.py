"""Read-only clinical DICOM inventory; outputs stay outside Git.

Group by Study/Series UID, never by filename/InstanceNumber alone. This is a
technical import audit, not diagnosis, de-identification or clinical approval.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
import shutil
import sys

import numpy as np
import pydicom
from pydicom.uid import CTImageStorage
from heart3d.dicom.ct import inspect_headers, load_volume, save_nifti

TAGS = [
    'StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID', 'SOPClassUID',
    'PatientID', 'PatientAge', 'PatientName', 'PatientBirthDate',
    'PatientIdentityRemoved', 'AccessionNumber', 'InstitutionName',
    'Modality', 'Manufacturer', 'ManufacturerModelName', 'SoftwareVersions',
    'FrameOfReferenceUID', 'Rows', 'Columns', 'NumberOfFrames', 'PixelSpacing',
    'ImageOrientationPatient', 'ImagePositionPatient', 'ImageType',
    'SliceThickness', 'SpacingBetweenSlices', 'RescaleSlope', 'RescaleIntercept',
    'RescaleType', 'InstanceNumber', 'AcquisitionNumber',
    'TemporalPositionIdentifier', 'TriggerTime', 'HeartRate', 'PatientPosition',
    'ContrastBolusAgent', 'ContrastBolusRoute', 'ConvolutionKernel',
    'BodyPartExamined', 'SeriesDescription', 'ProtocolName',
    'StructureSetROISequence', 'SegmentSequence', 'ReferencedSeriesSequence', 'SegmentationType',
    'BitsAllocated', 'BitsStored', 'PixelRepresentation',
]
IDENTITY_FIELDS = ['PatientName', 'PatientID', 'PatientBirthDate', 'AccessionNumber']


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def scalar(header, name):
    value = getattr(header, name, None)
    if value is None or str(value).strip() == '':
        return None
    return str(value)


def numbers(value):
    if value is None:
        return None
    try:
        result = np.atleast_1d(np.asarray(value, dtype=float))
        return result.tolist() if np.isfinite(result).all() else None
    except (TypeError, ValueError):
        return None


def unique(headers, name, numeric=False):
    values = [numbers(getattr(h, name, None)) if numeric else scalar(h, name)
              for h in headers]
    return [json.loads(s) for s in sorted({json.dumps(v) for v in values}, key=str)]


def classify_series(headers):
    classes = {str(getattr(h, 'SOPClassUID', '')) for h in headers}
    types = {str(x).upper() for h in headers for x in getattr(h, 'ImageType', [])}
    modalities = {str(getattr(h, 'Modality', '')) for h in headers}
    if modalities == {'CT'} and classes == {str(CTImageStorage)}:
        if 'LOCALIZER' in types or 'SCOUT' in types:
            return 'localizer'
        if any(int(getattr(h, 'NumberOfFrames', 1) or 1) != 1 for h in headers):
            return 'multiframe_requires_adapter'
        return 'classic_ct_candidate'
    if modalities & {'RTSTRUCT', 'SEG'}:
        return 'segmentation_object'
    return 'other_dicom_object'


def safe_geometry(geometry):
    return {k: v for k, v in geometry.report.items()
            if k not in {'patient_id', 'series_uid', 'frame_of_reference_uid'}}


def cardiac_phase_tokens(headers):
    # Extract only percentages, never export free-text descriptions/protocols.
    return sorted({int(token) for h in headers for name in ('SeriesDescription', 'ProtocolName')
                   for token in re.findall(r'(?<!\d)(\d{1,3})%', scalar(h, name) or '')
                   if int(token) <= 100})


def position_repetitions(headers):
    positions = [numbers(getattr(h, 'ImagePositionPatient', None)) for h in headers]
    positions = [tuple(p) for p in positions if p is not None and len(p) == 3]
    counts = Counter(positions)
    return {'objects_with_IPP': len(positions), 'distinct_IPP': len(counts),
            'multiplicity_histogram': dict(sorted(Counter(counts.values()).items())),
            'phase_separation_performed': False}


def sr_structure(header):
    # Summarize coded concept names and ValueTypes only; no TEXT/PNAME/UID/date values.
    types, concepts = Counter(), Counter()
    def visit(node):
        value_type = scalar(node, 'ValueType')
        if value_type:
            types[value_type] += 1
        for concept in getattr(node, 'ConceptNameCodeSequence', []):
            concepts[(scalar(concept, 'CodingSchemeDesignator'),
                      scalar(concept, 'CodeValue'), scalar(concept, 'CodeMeaning'))] += 1
        for child in getattr(node, 'ContentSequence', []):
            visit(child)
    visit(header)
    return {'standard_content_sequence_present': bool(getattr(header, 'ContentSequence', [])),
            'value_type_counts': dict(types),
            'coded_concepts': [{'scheme': scheme, 'code': code, 'meaning': meaning, 'count': count}
                               for (scheme, code, meaning), count in sorted(concepts.items(), key=str)],
            'free_text_and_person_values_exported': False}


def scan(input_root, output_root, reserve_gb=80):
    input_root = Path(input_root).resolve()
    output_root = Path(output_root).resolve()
    if output_root == input_root or input_root in output_root.parents:
        raise ValueError('Output must be outside the immutable incoming directory')
    if shutil.disk_usage(output_root.parent).free < reserve_gb * 10**9:
        raise ValueError('Insufficient free-space reserve')
    paths = sorted(p for p in input_root.rglob('*') if p.is_file())
    groups = defaultdict(list)
    records, invalid = [], []
    identity_counts = Counter()
    for i, path in enumerate(paths, 1):
        before = path.stat()
        try:
            h = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=TAGS)
        except pydicom.errors.InvalidDicomError:
            invalid.append({'relative_path': path.relative_to(input_root).as_posix(),
                            'bytes': before.st_size, 'sha256': sha256(path)})
            continue
        study, series = scalar(h, 'StudyInstanceUID'), scalar(h, 'SeriesInstanceUID')
        for key in IDENTITY_FIELDS:
            identity_counts[key] += bool(scalar(h, key))
        digest = sha256(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('Input changed during audit')
        record = {'relative_path': path.relative_to(input_root).as_posix(),
                  'bytes': before.st_size, 'sha256': digest,
                  'sop_uid': scalar(h, 'SOPInstanceUID')}
        records.append(record)
        groups[(study, series)].append((path, h, record))
        if i % 250 == 0:
            print(f'Headers and SHA-256: {i}/{len(paths)}', file=sys.stderr, flush=True)
    study_uids = sorted({study for study, _ in groups if study is not None})
    aliases = {uid: f'case_{i:03d}' for i, uid in enumerate(study_uids, 1)}
    summary = {'created_utc': datetime.now(timezone.utc).isoformat(),
               'purpose': 'technical_import_audit_not_clinical_approval',
               'files': len(paths), 'dicom_files': len(records),
               'input_bytes': sum(p.stat().st_size for p in paths),
               'non_dicom_files': len(invalid), 'study_count': len(study_uids),
               'distinct_nonempty_patient_id_values': len({scalar(h, 'PatientID')
                    for entries in groups.values() for _, h, _ in entries
                    if scalar(h, 'PatientID')}),
               'patient_identity_verified': False,
               'identity_fields_nonempty_counts': dict(identity_counts),
               'source_deidentification_not_certified': True,
               'clinical_status': 'unknown', 'studies': [], 'unassigned_objects': 0}
    locators = {'input_root': str(input_root), 'contains_source_UIDs_and_paths': True,
                'never_commit_or_upload': True, 'series': {}, 'files': records,
                'non_dicom_files': invalid}
    for study in study_uids:
        case = aliases[study]
        group_keys = sorted((k for k in groups if k[0] == study), key=lambda k: str(k[1]))
        case_report = {'case': case, 'clinical_status': 'unknown', 'series': []}
        for index, key in enumerate(group_keys, 1):
            series_alias = f'series_{index:03d}'
            entries = groups[key]
            hs = [h for _, h, _ in entries]
            paths = [p for p, _, _ in entries]
            candidate = classify_series(hs)
            report = {'series': series_alias, 'files': len(hs), 'category': candidate,
                'modalities': dict(Counter(str(getattr(h, 'Modality', '')) for h in hs)),
                'sop_classes': unique(hs, 'SOPClassUID'),
                'patient_age_tags': unique(hs, 'PatientAge'),
                'scanner_models': unique(hs, 'ManufacturerModelName'),
                'manufacturers': unique(hs, 'Manufacturer'),
                'dimensions_rows_columns_frames': sorted({
                    (int(getattr(h, 'Rows', 0) or 0), int(getattr(h, 'Columns', 0) or 0),
                     int(getattr(h, 'NumberOfFrames', 1) or 1)) for h in hs}),
                'image_types': sorted({tuple(str(x) for x in getattr(h, 'ImageType', [])) for h in hs}),
                'transfer_syntaxes': sorted({str(getattr(h.file_meta, 'TransferSyntaxUID', '')) for h in hs}),
                'pixel_spacing_values': unique(hs, 'PixelSpacing', True),
                'slice_thickness_values': unique(hs, 'SliceThickness', True),
                'acquisition_numbers': unique(hs, 'AcquisitionNumber'),
                'temporal_position_ids': unique(hs, 'TemporalPositionIdentifier'),
                'trigger_time_values_ms': unique(hs, 'TriggerTime', True),
                'cardiac_phase_percentage_tokens': cardiac_phase_tokens(hs),
                'position_repetitions': position_repetitions(hs),
                'convolution_kernels': unique(hs, 'ConvolutionKernel'),
                'contrast_agent_tag_nonempty_count': sum(bool(scalar(h, 'ContrastBolusAgent')) for h in hs),
                'contrast_status': 'tag_reported' if any(scalar(h, 'ContrastBolusAgent') for h in hs) else 'unknown',
                'unique_sop_count': len({scalar(h, 'SOPInstanceUID') for h in hs}),
                'geometry_status': 'not_volume_candidate', 'anatomical_coverage': 'not_reviewed',
                'annotation_objects_present': candidate == 'segmentation_object'}
            sops = [scalar(h, 'SOPInstanceUID') for h in hs]
            report['duplicate_sop_instances'] = len(hs) - len(set(sops))
            if candidate == 'classic_ct_candidate':
                try:
                    geometry = inspect_headers(hs, paths, require_demographic_metadata=False)
                    report['geometry_status'] = 'passed_header_geometry'
                    report['geometry'] = safe_geometry(geometry)
                    report['demographic_strict_pediatric_loader_compatible'] = all(
                        geometry.report.get(k) is not None for k in ('patient_id', 'patient_age'))
                except (ValueError, AttributeError, TypeError) as exc:
                    report['geometry_status'] = 'requires_review'
                    # Existing gate messages contain no patient names, dates or UID values.
                    report['geometry_error'] = str(exc)
            locator = {'study_uid': study, 'series_uid': key[1],
                       'relative_paths': [p.relative_to(input_root).as_posix() for p in paths],
                       'source_hashes': [r['sha256'] for _, _, r in entries]}
            locators['series'][f'{case}/{series_alias}'] = locator
            case_report['series'].append(report)
        summary['studies'].append(case_report)
    summary['unassigned_objects'] = sum(len(v) for (study, _), v in groups.items() if study is None)
    summary['modalities'] = dict(Counter(str(getattr(h, 'Modality', ''))
                                 for v in groups.values() for _, h, _ in v))
    segmentation_inventory = []
    uid_to_alias = {(v['study_uid'], v['series_uid']): k for k, v in locators['series'].items()}
    for case in summary['studies']:
        for series in case['series']:
            if series['category'] != 'segmentation_object':
                continue
            alias = f"{case['case']}/{series['series']}"
            location = locators['series'][alias]
            entries = groups[(location['study_uid'], location['series_uid'])]
            for index, (_, h, _) in enumerate(entries, 1):
                segmentation_inventory.append({
                    'object': f'{alias}/object_{index:03d}',
                    'sop_class': str(h.SOPClassUID),
                    'sop_class_name': h.SOPClassUID.name,
                    'referenced_series': [uid_to_alias.get((location['study_uid'], str(v.SeriesInstanceUID)),
                        'outside_supplied_inventory') for v in getattr(h, 'ReferencedSeriesSequence', [])],
                    'frames': int(getattr(h, 'NumberOfFrames', 0) or 0),
                    'segmentation_type': str(getattr(h, 'SegmentationType', '')),
                    'segments': [{'number': int(v.SegmentNumber),
                        'label': str(getattr(v, 'SegmentLabel', '')),
                        'algorithm_type': str(getattr(v, 'SegmentAlgorithmType', '')),
                        'coded_meanings': [str(e.value) for e in v.iterall() if e.keyword == 'CodeMeaning']}
                        for v in getattr(h, 'SegmentSequence', [])]})
    sr_inventory = []
    for case in summary['studies']:
        for series in case['series']:
            if 'SR' not in series['modalities']:
                continue
            alias = f"{case['case']}/{series['series']}"
            location = locators['series'][alias]
            entries = groups[(location['study_uid'], location['series_uid'])]
            for index, (path, _, _) in enumerate(entries, 1):
                header = pydicom.dcmread(path, stop_before_pixels=True)
                sr_inventory.append({'object': f'{alias}/object_{index:03d}',
                    'sop_class_name': header.SOPClassUID.name, **sr_structure(header)})
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / 'sr_structure_inventory.json').write_text(
        json.dumps(sr_inventory, ensure_ascii=False, indent=2), encoding='utf-8')
    (output_root / 'segmentation_inventory.json').write_text(
        json.dumps(segmentation_inventory, ensure_ascii=False, indent=2), encoding='utf-8')
    (output_root / 'source_locators.private.json').write_text(
        json.dumps(locators, ensure_ascii=False, indent=2), encoding='utf-8')
    (output_root / 'audit_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


def preview(volume, geometry, destination, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    spacing = geometry.report['spacing_xyz_mm']
    for ax, fraction in zip(axes[0], (.25, .5, .75)):
        k = min(volume.shape[2]-1, int(fraction*(volume.shape[2]-1)))
        ax.imshow(volume[:, :, k].T, cmap='gray', vmin=-200, vmax=600,
                  origin='lower', aspect=spacing[1]/spacing[0])
        ax.set_title(f'Native axial {fraction:.0%}, slice {k}')
    for ax, axis, name in zip(axes[1, :2], (1, 0), ('native coronal', 'native sagittal')):
        k = volume.shape[axis]//2
        remaining = [a for a in range(3) if a != axis]
        ax.imshow(np.take(volume, k, axis=axis).T, cmap='gray', vmin=-200, vmax=600,
                  origin='lower', aspect=spacing[remaining[1]]/spacing[remaining[0]])
        ax.set_title(f'{name}, index {k}')
    axes[1, 2].axis('off')
    axes[1, 2].text(0, .8, 'Source CT only; no segmentation\nNative grid, physical aspect\nClinical interpretation not performed\nNo patient identity text rendered', fontsize=11)
    fig.suptitle(title)
    fig.savefig(destination, dpi=110)
    plt.close(fig)


def prepare_series(output_root, selection, reserve_gb=80):
    root = Path(output_root).resolve()
    locators = json.loads((root/'source_locators.private.json').read_text(encoding='utf-8'))
    selected = locators['series'][selection]
    input_root = Path(locators['input_root']).resolve()
    paths = [(input_root/p).resolve() for p in selected['relative_paths']]
    if any(input_root not in p.parents for p in paths):
        raise ValueError('Source locator escapes incoming directory')
    for p, expected in zip(paths, selected['source_hashes']):
        if sha256(p) != expected:
            raise ValueError('Source SHA-256 differs from frozen inventory')
    headers = [pydicom.dcmread(p, stop_before_pixels=True, specific_tags=TAGS) for p in paths]
    geometry = inspect_headers(headers, paths, expected_series=selected['series_uid'],
                               require_demographic_metadata=False)
    estimated_bytes = int(np.prod(geometry.shape))*4
    if shutil.disk_usage(root).free - estimated_bytes*3 < reserve_gb*10**9:
        raise ValueError('Preparation would consume required free-space reserve')
    volume = load_volume(geometry)
    destination = root/'prepared'/selection
    destination.mkdir(parents=True, exist_ok=True)
    report = {'series': selection, 'geometry': safe_geometry(geometry),
              'intensity_min_max': [float(volume.min()), float(volume.max())],
              'clinical_status': 'unknown', 'ground_truth_certified': False,
              'source_segmentation_objects': [o['object'] for o in json.loads(
                  (root/'segmentation_inventory.json').read_text(encoding='utf-8'))
                  if selection in o['referenced_series']],
              'source_hashes_verified': True, 'no_resampling': True}
    from heart3d.dicom.qa import check_simpleitk
    report['independent_reader'] = check_simpleitk(geometry, volume)
    save_nifti(volume, geometry.affine_ras, destination/'ct.nii.gz')
    preview(volume, geometry, destination/'ct_preview.png', selection)
    report['nifti_sha256'] = sha256(destination/'ct.nii.gz')
    (destination/'conversion.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--prepare-series', action='append', default=[])
    parser.add_argument('--reserve-gb', type=float, default=80)
    args = parser.parse_args()
    if args.input:
        report = scan(args.input, args.output, args.reserve_gb)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    for selection in args.prepare_series:
        print(json.dumps(prepare_series(args.output, selection, args.reserve_gb), indent=2))
    if not args.input and not args.prepare_series:
        parser.error('Specify --input and/or --prepare-series')

if __name__ == '__main__':
    main()
