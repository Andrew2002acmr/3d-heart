import nibabel as nib
import numpy as np
import pytest

from heart3d.volume import load_case, discover
from heart3d.surfaces import build_surface


def save_pair(tmp_path, ct, mask, affine, unit="mm", mask_affine=None):
    paths = tmp_path / "ct_0001_image.nii.gz", tmp_path / "ct_0001_label.nii.gz"
    for data, matrix, path in zip((ct, mask), (affine, affine if mask_affine is None else mask_affine), paths):
        image = nib.Nifti1Image(data, matrix)
        image.header.set_xyzt_units(unit)
        nib.save(image, path)
    return paths


def test_affine_rotation_reflection_and_anisotropic_spacing():
    mask = np.zeros((12, 15, 18), dtype=np.uint8)
    mask[2:7, 3:9, 4:12] = 1
    affine = np.array([[0, -2, 0, 30], [-3, 0, 0, 40], [0, 0, 5, -60], [0, 0, 0, 1]])
    mesh, info = build_surface(mask, 1, affine)
    # Voxel-center boxes have boundaries at half indices. The affine swaps and
    # reflects X/Y, scales all axes differently, and translates the entire box.
    np.testing.assert_allclose(mesh.bounds, [13, 25, 20.5, 35.5, -42.5, -2.5])
    assert mesh.n_open_edges == 0
    assert info["boundary_edges"] == info["non_manifold_edges"] == 0
    assert info["voxels"] == 5 * 6 * 8
    assert info["voxel_volume_in_coordinate_units_cubed"] == pytest.approx(5 * 6 * 8 * 30)
    assert not info["touches_scan_boundary"]
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    triangle = mesh.points[faces]
    signed_volume = np.sum(triangle[:, 0] * np.cross(triangle[:, 1], triangle[:, 2])) / 6
    assert signed_volume > 0  # outward triangle winding even with a reflection


def test_no_silent_geometry_mismatch(tmp_path):
    data = np.ones((6, 7, 8), dtype=np.uint8)
    shifted = np.eye(4)
    shifted[0, 3] = 4
    paths = save_pair(tmp_path, data, data, np.eye(4), mask_affine=shifted)
    with pytest.raises(ValueError, match="affines differ"):
        load_case(*paths)


def test_shape_mismatch(tmp_path):
    paths = save_pair(tmp_path, np.ones((6, 7, 8), np.uint8), np.ones((7, 7, 8), np.uint8), np.eye(4))
    with pytest.raises(ValueError, match="shapes differ"):
        load_case(*paths)


def test_lossless_reorientation_aligns_equivalent_grids(tmp_path):
    data = np.zeros((6, 7, 8), dtype=np.uint8)
    data[1:3, 2:5, 3:7] = 1
    flipped = np.diag([-1., 1., 1., 1.])
    flipped[0, 3] = data.shape[0] - 1
    paths = save_pair(tmp_path, data, data[::-1], np.eye(4), mask_affine=flipped)
    case = load_case(*paths)
    np.testing.assert_array_equal(case.ct, case.mask)
    assert not case.report["geometry"]["resampled"]


def test_missing_units_are_not_called_millimetres(tmp_path):
    data = np.ones((6, 7, 8), dtype=np.uint8)
    case = load_case(*save_pair(tmp_path, data, data, np.eye(4), unit="unknown"))
    assert case.unit == "unknown"
    assert any("NOT confirmed millimetres" in w for w in case.report["warnings"])


def test_metres_are_converted_to_mm_once(tmp_path):
    data = np.ones((6, 7, 8), dtype=np.uint8)
    affine = np.diag([0.002, 0.003, 0.004, 1])
    affine[0, 3] = 0.01
    case = load_case(*save_pair(tmp_path, data, data, affine, unit="meter"))
    np.testing.assert_allclose(case.affine, [[2, 0, 0, 10], [0, 3, 0, 0], [0, 0, 4, 0], [0, 0, 0, 1]], atol=1e-6)
    assert case.unit == "mm"


@pytest.mark.parametrize("value", [0.5, -1, np.nan])
def test_invalid_mask_rejected(tmp_path, value):
    ct = np.ones((6, 7, 8), dtype=np.float32)
    mask = ct.copy()
    mask[2, 3, 4] = value
    with pytest.raises(ValueError):
        load_case(*save_pair(tmp_path, ct, mask, np.eye(4)))


def test_border_caps_and_components_are_reported():
    mask = np.zeros((8, 8, 8), np.uint8)
    mask[0:2, 0:2, 0:2] = 1
    mask[5:7, 5:7, 5:7] = 1
    mesh, info = build_surface(mask, 1, np.eye(4))
    assert info["artificial_boundary_caps"]
    assert info["components_26_connected"] == 2
    assert info["voxels"] == 16
    assert mesh.n_open_edges == 0


def test_unknown_labels_are_preserved_and_reported(tmp_path):
    data = np.ones((6, 7, 8), dtype=np.uint8)
    mask = data.copy()
    mask[0, 0, 0] = 14
    paths = save_pair(tmp_path, data, mask, np.eye(4))
    case = load_case(*paths)
    assert case.mask[0, 0, 0] == 14
    assert case.report["ignored_labels"] == [14]
    assert len(discover(tmp_path)["pairs"]) == 1
