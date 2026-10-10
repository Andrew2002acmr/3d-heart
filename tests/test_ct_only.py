from pathlib import Path
import nibabel as nib
import numpy as np
from heart3d.volume import sha256
from heart3d.interactive.data import catalog_ct, load_source
from heart3d.interactive.state import ViewState, slice_rgb


def test_ct_only_has_no_ground_truth_and_never_writes_inputs(tmp_path):
    path = tmp_path/"ct.nii.gz"
    values = np.arange(6*7*8, dtype=np.float32).reshape(6,7,8)
    affine = np.diag([-0.5, 0.6, 1.2, 1])
    image = nib.Nifti1Image(values, affine)
    image.header.set_xyzt_units("mm")
    nib.save(image,path)
    before = sha256(path)
    sources, _ = catalog_ct([path])
    assert sources[0].mask is None
    loaded = load_source(sources[0])
    assert loaded.volume.report["mask"] is None
    assert not loaded.volume.report["segmentation_available"]
    assert loaded.meshes == {} and loaded.report_path is None
    np.testing.assert_array_equal(loaded.volume.ct, values[::-1,:,:])
    state = ViewState(loaded)
    for axis in range(3): assert slice_rgb(state,axis).ndim == 3
    assert list(tmp_path.iterdir()) == [path]
    assert sha256(path) == before
