import numpy as np
import pyvista as pv
import pytest

from heart3d.mesh_quality import quality
from heart3d.smoothing import MU, LAMBDA, PROFILES, smooth_surface, taubin_coordinates, umbrella_operator


def test_taubin_matches_two_simultaneous_steps_and_preserves_translation():
    mesh = pv.Sphere(theta_resolution=12, phi_resolution=12)
    weights, locked = umbrella_operator(mesh)
    points = np.asarray(mesh.points, dtype=float)
    # Independent dense neighbour average, avoiding the implementation's CSR multiplication.
    neighbours = [weights.getrow(i).indices for i in range(len(points))]
    first = points + LAMBDA*(np.array([points[n].mean(axis=0) for n in neighbours])-points)
    expected = first + MU*(np.array([first[n].mean(axis=0) for n in neighbours])-first)
    actual = taubin_coordinates(points, weights, locked, 1)
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    translated = taubin_coordinates(points + [10, -20, 30], weights, locked, 1)
    np.testing.assert_allclose(translated, actual + [10, -20, 30], atol=1e-13)


@pytest.mark.parametrize("method", PROFILES)
def test_profiles_preserve_baseline_connectivity_and_closed_sphere(method):
    mesh = pv.Sphere(theta_resolution=24, phi_resolution=24)
    points, faces = mesh.points.copy(), mesh.faces.copy()
    q0 = quality(mesh)
    for level in PROFILES[method]:
        output = smooth_surface(mesh, method, level)
        q = quality(output)
        np.testing.assert_array_equal(output.faces, faces)
        assert q["vertices"] == q0["vertices"]
        assert q["boundary_edges"] == q["non_manifold_edges"] == q["zero_area_triangles"] == 0
        assert q["surface_components_vertex_connected"] == 1
        # Non-shrinking does not mean volume-preserving: lambda/mu can expand
        # coarse spheres. Quantitative acceptance is evaluated by the experiment.
        assert np.isfinite(q["enclosed_volume"]) and q["enclosed_volume"] > 0
        assert not np.array_equal(output.points, points)
    np.testing.assert_array_equal(mesh.points, points)
    np.testing.assert_array_equal(mesh.faces, faces)


def test_fixed_boundary_and_nonmanifold_endpoints_and_retained_components():
    # Three faces share edge 0--1; every vertex is on an open boundary here.
    mesh = pv.PolyData(np.array([[0,0,0], [1,0,0], [0,1,0], [0,-1,0], [0,0,1.]]),
                       [3,0,1,2, 3,1,0,3, 3,0,1,4])
    weights, locked = umbrella_operator(mesh)
    assert locked.all()
    output = smooth_surface(mesh, "taubin", "strong", (weights, locked))
    np.testing.assert_array_equal(output.points, mesh.points)
    assert quality(output)["non_manifold_edges"] == 1
    two = pv.Sphere().merge(pv.Sphere(center=(4,0,0)), merge_points=False)
    for method in PROFILES:
        assert quality(smooth_surface(two, method, "medium"))["surface_components_vertex_connected"] == 2


def test_windowed_sinc_matches_named_pyvista_taubin_filter_but_not_lambda_mu():
    mesh = pv.Sphere(theta_resolution=12, phi_resolution=12)
    mesh.points = mesh.points.astype(float)
    expected = mesh.smooth_taubin(n_iter=20, pass_band=.1, window_function="nuttall",
                                  boundary_smoothing=False, feature_smoothing=False,
                                  non_manifold_smoothing=False, normalize_coordinates=True)
    actual = smooth_surface(mesh, "windowed_sinc", "medium")
    np.testing.assert_array_equal(actual.points, expected.points)
    assert not np.allclose(actual.points, smooth_surface(mesh, "taubin", "medium").points, atol=1e-7)
