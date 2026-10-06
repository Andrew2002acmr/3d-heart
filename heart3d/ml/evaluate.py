"""One held-out evaluation on original confirmed DICOM geometry, no parameter fitting."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import nibabel as nib
import numpy as np
from heart3d.ml.geometry import restore_probability
from heart3d.ml.inference import predict_patient
from heart3d.ml.metrics import physical_metrics, aggregate
from heart3d.ml.runtime import load_model, seed_runtime
from heart3d.ml.splits import load_frozen_split
from heart3d.pediatric import write_json
from heart3d.storage import sha256_file, require_space


def evaluate(config_path, checkpoint_path, output, partition='test', data_root=None, device='cuda'):
    config = json.loads(Path(config_path).read_text())
    root = Path(data_root or config['data_root'])
    output = Path(output)
    if output.exists():
        raise ValueError('Evaluation output already exists; do not overwrite predictions/results')
    split = load_frozen_split(config['split'], config['cohort'])
    seed_runtime(config['seed'], config['cpu_threads'], device)
    model, checkpoint = load_model(checkpoint_path, config, device)
    if checkpoint.get('training_scope') != 'full_scratch_baseline':
        raise ValueError('A completed full baseline checkpoint is required')
    pre = json.loads(Path(config['preprocessing']).read_text())
    require_space(root, 1_000_000_000, int(config['reserve_GB']*1e9))
    output.mkdir(parents=True)
    records = []
    for row in split['partitions'][partition]:
        start = time.perf_counter()
        directory = root/config['cache_relative_path']/row['patient_id']
        probability, provenance = predict_patient(model, directory, pre['context_offsets_mm'], config['batch_size'], device)
        gt_path = root/row['mask_relative_path']
        if sha256_file(gt_path) != row['review']['mask_SHA256']:
            raise ValueError('Original reviewed GT integrity failure')
        gt_image = nib.load(gt_path)
        transform = provenance['transform']
        if gt_image.shape != tuple(transform['original_shape_xyz']) or not np.allclose(gt_image.affine, transform['original_affine_ras'], atol=1e-5):
            raise ValueError('Original GT and prediction transform geometry differ')
        restored = restore_probability(probability, transform)
        prediction = (restored >= .5).astype(np.uint8)
        del restored, probability
        target = np.asanyarray(gt_image.dataobj) > 0
        record = {'patient_id': row['patient_id'], 'age_group': row.get('age_group'),
                  'scanner': row.get('scanner'), 'contrast': row.get('contrast_status'),
                  **physical_metrics(prediction, target, transform['original_spacing_xyz_mm'],
                                     row['geometry']['physical_geometry_confirmed_from_DICOM'])}
        destination = output/row['patient_id']; destination.mkdir()
        prediction_path = destination/'heart_prediction_original.nii.gz'
        header = gt_image.header.copy(); header.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(prediction, gt_image.affine, header), prediction_path)
        write_json(destination/'provenance.json', {
            'patient_id': row['patient_id'], 'source_series_uid': provenance['source_series_uid'],
            'checkpoint_SHA256': sha256_file(checkpoint_path), 'model_version': checkpoint['model_version'],
            'git_revision': checkpoint['git_revision'], 'split_SHA256': checkpoint['split_SHA256'],
            'preprocessing_SHA256': checkpoint['preprocessing_SHA256'], 'original_geometry': transform,
            'source_GT_SHA256': row['review']['mask_SHA256'], 'threshold': .5,
            'postprocessing': 'none', 'prediction_SHA256': sha256_file(prediction_path),
            'original_GT_modified': False})
        record['seconds'] = time.perf_counter()-start
        records.append(record)
        write_json(output/'patients_partial.json', records)
        print('EVALUATION', partition, row['patient_id'], json.dumps(record), flush=True)
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'partition': partition,
              'checkpoint_SHA256': sha256_file(checkpoint_path), 'split_SHA256': checkpoint['split_SHA256'],
              'threshold': .5, 'grid': 'original DICOM-derived XYZ',
              'HD95_definition': 'max of 95th percentiles of both directed surface distances',
              'ASSD_definition': 'mean of pooled directed surface voxel distances; 6-neighbour boundary',
              'records': records, 'aggregate': aggregate(records),
              'by_age_group': aggregate(records, 'age_group'), 'by_scanner': aggregate(records, 'scanner'),
              'by_contrast': aggregate(records, 'contrast')}
    write_json(output/'metrics.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True); p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--data', type=Path)
    p.add_argument('--partition', choices=['validation','test'], default='test')
    p.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    a = p.parse_args(); evaluate(a.config, a.checkpoint, a.output, a.partition, a.data, a.device)
