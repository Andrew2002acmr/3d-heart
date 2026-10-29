import json
import nibabel as nib
import numpy as np
import pytest
from heart3d.ml.clinical_experiment import canonical_input, admit
from heart3d.storage import sha256_file


def test_lossless_RAS_input_and_inverse_preserve_original_physical_grid():
    values = np.arange(5*6*7, dtype=np.float32).reshape(5,6,7)
    # Permutation, reflection, anisotropic spacing and translated origin.
    affine = np.array([[0,-.8,0,20],[0,0,1.3,-40],[-2.1,0,0,55],[0,0,0,1.]])
    source = nib.Nifti1Image(values, affine)
    ras, inverse = canonical_input(source)
    restored = nib.orientations.apply_orientation(ras, inverse)
    np.testing.assert_array_equal(restored, values)
    assert nib.aff2axcodes(nib.as_closest_canonical(source).affine) == ('R','A','S')


def test_clinical_gates_reject_unreviewed_pixels_and_wrong_metadata_before_inference(tmp_path):
    path = tmp_path/'ct.nii.gz'
    image = nib.Nifti1Image(np.zeros((5,6,7),np.float32),np.diag([.8,1.1,2.,1.]))
    image.header.set_xyzt_units('mm');nib.save(image,path)
    adapter = {'gain':1,'offset':1024,'exploratory_only':True,'source_intensity_convention_confirmed':False}
    fields = ('user_attested_permission_and_basis','local_only','geometry_verified','independent_reader_verified','metadata_deidentified','pixel_identifier_screening_passed')
    admission = {k:True for k in fields};admission['ct_sha256']=sha256_file(path)
    assert admit(path,admission,adapter).shape == image.shape
    for field in fields:
        with pytest.raises(ValueError,match='admission gate'):
            admit(path,dict(admission,**{field:False}),adapter)
    with pytest.raises(ValueError,match='adapter'):
        admit(path,admission,dict(adapter,offset=0))
    image.header['descrip']=b'Identifying text must never enter a clinical import'
    nib.save(image,path);admission['ct_sha256']=sha256_file(path)
    with pytest.raises(ValueError,match='metadata'):
        admit(path,admission,adapter)
