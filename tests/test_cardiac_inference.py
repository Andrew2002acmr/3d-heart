"""Recovery gates and image-only original-grid inference, without clinical data."""
import copy

import nibabel as nib
import numpy as np
import pytest

from heart3d.ml import infer_cardiac as recovery


def test_wrong_checkpoint_rejected_before_deserialization(tmp_path, monkeypatch):
    wrong = tmp_path / "wrong.pt"
    wrong.write_bytes(b"not the historical checkpoint")
    monkeypatch.setattr(recovery, "load_checkpoint", lambda *args: pytest.fail("Must verify SHA before torch.load"))
    with pytest.raises(ValueError, match="size/SHA256"):
        recovery.load_recovered("v1", wrong, "cpu")


def test_test_case_rejected_before_ct_io(tmp_path):
    config, _, _, _ = recovery.protocol("v2")
    with pytest.raises(ValueError, match="Only declared public development"):
        recovery.public_development_ct(tmp_path / "does_not_exist.nii.gz", "ct_1032", config)
    with pytest.raises(ValueError, match="Only declared public development"):
        recovery.public_development_ct(tmp_path / "clinical.nii.gz", "clinical", config)


def test_image_only_preparation_keeps_source_and_frozen_preprocessing():
    _, _, pre, _ = recovery.protocol("v1")
    before_pre = copy.deepcopy(pre)
    ct = np.full((16, 8, 3), 2015.0, dtype=np.float32)
    before = ct.copy()
    prepared, transform = recovery.prepare_volume(ct, pre)
    assert prepared.shape == (3, 256, 256)
    assert prepared[:, 128, 128].tolist() == [1, 1, 1]
    assert prepared[:, 0, 0].tolist() == [-1, -1, -1]
    assert transform["z_resampling"] is False
    np.testing.assert_array_equal(ct, before)
    assert pre == before_pre
    ct[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Finite 3D"):
        recovery.prepare_volume(ct, pre)


def test_prediction_restored_before_argmax_and_saved_on_source_grid(tmp_path, monkeypatch):
    _, _, pre, _ = recovery.protocol("v1")
    source = np.zeros((16, 8, 3), dtype=np.float32)
    image, transform = recovery.prepare_volume(source, pre)

    def probabilities(*args):
        p = np.zeros((3, 8, 256, 256), dtype=np.float32)
        p[:, 6] = 1
        yield [0, 1, 2], p

    monkeypatch.setattr(recovery, "probability_batches", probabilities)
    prediction = recovery.infer_volume(None, image, transform, pre, 2, "cpu")
    assert prediction.shape == source.shape and np.all(prediction == 6)
    affine = np.array([[.5, 0, 0, -30], [0, .8, 0, 10], [0, 0, 2, 45], [0, 0, 0, 1.]])
    ct = nib.Nifti1Image(source, affine)
    ct.header.set_xyzt_units("mm")
    ct.set_qform(affine, 1)
    ct.set_sform(affine, 1)
    output = tmp_path / "prediction.nii.gz"
    recovery.save_prediction(prediction, ct, output)
    saved = nib.load(output)
    np.testing.assert_allclose(saved.affine, ct.affine)
    assert saved.shape == ct.shape and saved.header.get_xyzt_units()[0] == "mm"
    assert saved.header.get_data_dtype() == np.dtype("uint8")
    assert ct.header.get_data_dtype() == np.dtype("float32")
    np.testing.assert_array_equal(np.asanyarray(ct.dataobj), source)
