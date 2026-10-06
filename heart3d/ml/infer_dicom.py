"""Image-only DICOM CT -> fixed preprocessing -> Heart mask on the original grid."""
import argparse
import json
from pathlib import Path
import nibabel as nib
import numpy as np
from heart3d.dicom.ct import inspect_series, load_volume, save_nifti
from heart3d.ml.geometry import make_transform, resample_xyz, restore_probability
from heart3d.ml.inference import predict_patient
from heart3d.ml.runtime import load_model, revision, seed_runtime
from heart3d.ml.resources import environment
from heart3d.ml.splits import load_frozen_split
from heart3d.pediatric import dicom_age, write_json
from heart3d.storage import require_space, sha256_file


def infer_dicom(config_path, checkpoint_path, ct_directory, output, device='cpu',
                expected_series=None, expected_sop_hash=None, reserve_GB=80):
    config=json.loads(Path(config_path).read_text());output=Path(output)
    if output.exists():raise ValueError('Inference output exists; do not overwrite')
    load_frozen_split(config['split'],config['cohort'])
    preprocessing=json.loads(Path(config['preprocessing']).read_text())
    if preprocessing['fitted_partition']!='train' or preprocessing['split_SHA256']!=sha256_file(config['split']):
        raise ValueError('Preprocessing provenance differs from the frozen training split')
    geometry=inspect_series(ct_directory,expected_series=expected_series)
    if expected_sop_hash and geometry.report['sop_set_sha256']!=expected_sop_hash:
        raise ValueError('DICOM SOP inventory differs from expected source fingerprint')
    if not np.allclose(geometry.report['orientation_iop'],preprocessing['orientation_iop'],atol=1e-5):
        raise ValueError('Native orientation differs from train; explicit reorientation required')
    transform=make_transform(geometry.shape,geometry.affine_ras,preprocessing['input_size'],preprocessing['z_spacing_mm'])
    needed=len(transform['z_positions_mm'])*transform['size']**2*4+int(np.prod(geometry.shape))+100_000_000
    require_space(output,needed,int(reserve_GB*1e9))
    seed_runtime(config['seed'],config['cpu_threads'],device)
    model,checkpoint=load_model(checkpoint_path,config,device)
    if checkpoint.get('training_scope')!='full_scratch_baseline':raise ValueError('Full baseline checkpoint required')
    volume=load_volume(geometry)
    low,high=preprocessing['HU_clip']
    image=resample_xyz(volume,transform,1,low);del volume
    np.clip(image,low,high,out=image);image-=low;image*=2/(high-low);image-=1
    cache=output/'preprocessed';cache.mkdir(parents=True)
    np.save(cache/'image.npy',image);del image
    cache_provenance={'patient_id':geometry.report['patient_id'],'source_series_uid':geometry.report['series_uid'],
                      'transform':transform,'image_SHA256':sha256_file(cache/'image.npy'),
                      'preprocessing_SHA256':sha256_file(config['preprocessing']),
                      'target_mask_used':False,'RTSTRUCT_used':False}
    write_json(cache/'provenance.json',cache_provenance)
    probability,_=predict_patient(model,cache,preprocessing['context_offsets_mm'],config['batch_size'],device)
    restored=restore_probability(probability,transform);del probability
    prediction=(restored>=.5).astype(np.uint8);del restored
    destination=output/'heart_prediction_original.nii.gz'
    save_nifti(prediction,geometry.affine_ras,destination)
    result={'patient_id':geometry.report['patient_id'],**dicom_age(geometry.report['patient_age']),
            'healthy_status':'unknown','source_series_uid':geometry.report['series_uid'],
            'source_frame_uid':geometry.report['frame_of_reference_uid'],
            'source_geometry':geometry.report,'original_geometry':transform,
            'expected_SOP_set_SHA256':expected_sop_hash,'threshold':.5,'postprocessing':'none',
            'predicted_voxels':int(prediction.sum()),'empty_prediction':not bool(prediction.any()),
            'touches_scan_faces':[bool(np.take(prediction,index,axis=axis).any()) for axis in range(3) for index in (0,-1)],
            'model_version':checkpoint['model_version'],'checkpoint_SHA256':sha256_file(checkpoint_path),
            'training_git_revision':checkpoint['git_revision'],'inference_git_revision':revision(),
            'config_SHA256':sha256_file(config_path),'split_SHA256':checkpoint['split_SHA256'],
            'preprocessing_SHA256':checkpoint['preprocessing_SHA256'],
            'prediction_SHA256':sha256_file(destination),'environment':environment(device),
            'original_DICOM_modified':False,'RTSTRUCT_used':False,'GT_used':False,
            'interpretation':'binary source Heart OAR prediction; no clinical diagnosis or healthy classification'}
    write_json(output/'provenance.json',result)
    print('DICOM_INFERENCE',geometry.report['patient_id'],'shape',prediction.shape,'voxels',result['predicted_voxels'],flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--ct-series',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--expected-series');p.add_argument('--expected-sop-hash');p.add_argument('--reserve-GB',type=float,default=80)
    a=p.parse_args();infer_dicom(a.config,a.checkpoint,a.ct_series,a.output,a.device,a.expected_series,a.expected_sop_hash,a.reserve_GB)
