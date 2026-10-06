import json
import numpy as np
import nibabel as nib
import torch
from pydicom.dataset import FileDataset,FileMetaDataset
from pydicom.uid import CTImageStorage,ExplicitVRLittleEndian
from test_pediatric_dicom import ct_headers
from test_ml_dataset import toy_experiment
from heart3d.dicom.ct import inspect_headers
from heart3d.ml.infer_dicom import infer_dicom
from heart3d.ml.model import HeartUNet25D
from heart3d.pediatric import write_json
from heart3d.storage import sha256_file


def test_dicom_image_only_inference_restores_physical_grid_without_gt(toy_experiment):
    config_path,_,_=toy_experiment;config=json.loads(config_path.read_text())
    from pathlib import Path
    root=Path(config['data_root']);ct_directory=root/'raw_CT';ct_directory.mkdir()
    headers=ct_headers();geometry=inspect_headers(headers)
    for header in headers:
        meta=FileMetaDataset();meta.TransferSyntaxUID=ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID=CTImageStorage;meta.MediaStorageSOPInstanceUID=header.SOPInstanceUID
        image=FileDataset(None,header,file_meta=meta,preamble=b'\0'*128)
        image.SamplesPerPixel=1;image.PhotometricInterpretation='MONOCHROME2'
        image.BitsAllocated=16;image.BitsStored=16;image.HighBit=15;image.PixelRepresentation=1
        image.PixelData=np.arange(90,dtype=np.int16).reshape(9,10).tobytes()
        image.save_as(ct_directory/(str(header.SOPInstanceUID)+'.dcm'),enforce_file_format=True)
    hashes={path.name:sha256_file(path) for path in ct_directory.iterdir()}
    pre_path=Path(config['preprocessing']);pre=json.loads(pre_path.read_text())
    pre.update(input_size=16,z_spacing_mm=2,HU_clip=[-1000,969],orientation_iop=geometry.report['orientation_iop'])
    write_json(pre_path,pre)
    config.update(model={'input_channels':5,'encoder_channels':[4,8,16,32],'norm_groups':4},
                  seed=42,cpu_threads=1,batch_size=2);write_json(config_path,config)
    model=HeartUNet25D(**config['model'])
    checkpoint=root/'synthetic.pt'
    torch.save({'model':model.state_dict(),'config':config,'split_SHA256':sha256_file(config['split']),
                'preprocessing_SHA256':sha256_file(pre_path),'training_scope':'full_scratch_baseline',
                'model_version':'synthetic','git_revision':'synthetic'},checkpoint)
    output=root/'inference';result=infer_dicom(config_path,checkpoint,ct_directory,output,
        expected_series='1.2.4',expected_sop_hash=geometry.report['sop_set_sha256'],reserve_GB=0)
    prediction=nib.load(output/'heart_prediction_original.nii.gz')
    assert prediction.shape==geometry.shape
    assert np.allclose(prediction.affine,geometry.affine_ras)
    assert prediction.header.get_xyzt_units()[0]=='mm'
    assert result['source_series_uid']=='1.2.4' and result['source_frame_uid']=='1.2.5'
    assert not result['GT_used'] and not result['RTSTRUCT_used']
    assert not (output/'preprocessed/mask.npy').exists()
    assert {p.name:sha256_file(p) for p in ct_directory.iterdir()}==hashes
