import numpy as np
import pyvista as pv
import pytest
from heart3d.postprocess import variants, deviation
from heart3d.mesh_quality import quality


def test_variants_preserve_baseline_and_closed_sphere():
    original = pv.Sphere(theta_resolution=32, phi_resolution=32).triangulate()
    points, faces = original.points.copy(), original.faces.copy()
    processed = variants(original)
    np.testing.assert_array_equal(original.points, points)
    np.testing.assert_array_equal(original.faces, faces)
    assert processed["taubin"].n_cells == original.n_cells
    assert processed["decimate50"].n_cells < original.n_cells * 0.6
    for surface in processed.values():
        assert quality(surface)["enclosed_volume"] > 0
        assert quality(surface)["boundary_edges"] == 0


def test_distance_on_identical_surface_and_translated_planes():
    plane = pv.Plane(i_size=10, j_size=10).triangulate()
    identical = deviation(plane, plane, count=100)
    assert identical["sampled_max"] < 1e-10
    moved = plane.translate((0, 0, 2), inplace=False)
    offset = deviation(plane, moved, count=100)
    assert offset["symmetric_mean"] == pytest.approx(2.)
    assert offset["sampled_max"] == pytest.approx(2.)
    assert not offset["is_exact_hausdorff"]
