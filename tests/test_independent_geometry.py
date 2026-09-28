import nibabel as nib
import numpy as np
from heart3d.geometry_audit import audit_header


def test_independent_lps_ras_agreement(tmp_path, monkeypatch):
    affine = np.array([[0., -2., 0., 30.], [3., 0., 0., -40.], [0., 0., 5., 60.], [0., 0., 0., 1.]])
    img = nib.Nifti1Image(np.zeros((7, 8, 9), np.int16), affine)
    img.header.set_xyzt_units("mm")
    path = tmp_path / "test.nii.gz"
    nib.save(img, path)
    # ITK's Windows file reader cannot open some Unicode absolute paths.
    # Relative ASCII filename tests the reader without changing the input file.
    monkeypatch.chdir(tmp_path)
    result = audit_header(path.relative_to(tmp_path))
    assert result["readers_agree_affine_ras"] and result["readers_agree_shape"]
    assert not result["anatomical_orientation_confirmed"]
