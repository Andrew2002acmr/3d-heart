import numpy as np
import pyvista as pv
import pytest

from heart3d.mesh_quality import quality
from heart3d.protected_smoothing import PROFILES, protected_points, smooth_primary


@pytest.mark.parametrize("profile", PROFILES)
def test_small_shell_is_preserved_even_when_first_in_input(profile):
    small = pv.PlatonicSolid("octahedron").translate((4, 0, 0))
    large = pv.Sphere(theta_resolution=24, phi_resolution=24)
    mesh = small.merge(large, merge_points=False)
    before = mesh.points.copy()
    locked = protected_points(mesh)
    assert locked.sum() == small.n_points
    assert (mesh.points[locked, 0] > 2).all()
    result, _ = smooth_primary(mesh, profile)
    np.testing.assert_array_equal(mesh.points, before)
    np.testing.assert_array_equal(result.points[locked], before[locked])
    np.testing.assert_array_equal(result.faces, mesh.faces)
    assert not np.array_equal(result.points[~locked], before[~locked])
    q = quality(result)
    assert q["surface_components_vertex_connected"] == 2
    assert q["zero_area_triangles"] == q["boundary_edges"] == q["non_manifold_edges"] == 0
