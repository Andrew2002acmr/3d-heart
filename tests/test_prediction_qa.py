import nibabel as nib
import numpy as np
from scripts.qa_pediatric_predictions import compare_case
from heart3d.storage import sha256_file


def test_binary_heart_mesh_metadata_units_and_components_are_preserved(tmp_path):
    patient='SYNTHETIC';root=tmp_path/'data';prepared=root/patient/'prepared';prepared.mkdir(parents=True)
    prediction_directory=tmp_path/'predictions'/patient;prediction_directory.mkdir(parents=True)
    shape=(32,32,32);affine=np.diag([-1.,-1.,2.,1.])
    target=np.zeros(shape,np.uint8);target[8:16,8:16,8:16]=1
    prediction=np.zeros_like(target);prediction[9:17,8:16,8:16]=1;prediction[25,25,25]=1
    ct=target.astype(np.int16)*150
    paths=[prepared/'ct_original.nii.gz',prepared/'heart_gt_original.nii.gz',prediction_directory/'heart_prediction_original.nii.gz']
    for path,values in zip(paths,[ct,target,prediction]):
        image=nib.Nifti1Image(values,affine);image.header.set_xyzt_units('mm');nib.save(image,path)
    row={'patient_id':patient,'age':6,'scanner':'synthetic',
         'ct_relative_path':paths[0].relative_to(root).as_posix(),'mask_relative_path':paths[1].relative_to(root).as_posix(),
         'review':{'CT_SHA256':sha256_file(paths[0]),'mask_SHA256':sha256_file(paths[1])},
         'geometry':{'physical_geometry_confirmed_from_DICOM':True}}
    result=compare_case(row,root,prediction_directory.parent,tmp_path/'qa')
    assert result['meshes']['GT']['surface']['short_name']=='Heart'
    assert result['meshes']['GT']['surface']['voxel_volume_in_coordinate_units_cubed']==1024
    assert result['meshes']['Prediction']['surface']['components_26_connected']==2
    assert result['meshes']['Prediction']['surface']['components_removed'] is False
    assert result['mesh_distances']['HD95_vertex_nearest_mm']>0
    assert (tmp_path/'qa'/patient/'comparison.png').is_file()
    assert sha256_file(paths[1])==row['review']['mask_SHA256']
