import numpy as np
import pyvista as pv
import pytest
from heart3d.mesh_quality import quality
from heart3d.surfaces import build_surface


def test_cavity_volume_subtracted_not_added():
    mask = np.zeros((14, 14, 14), np.uint8)
    mask[2:12, 2:12, 2:12] = 1
    solid, _ = build_surface(mask, 1, np.eye(4))
    cavity_mask = np.zeros_like(mask)
    cavity_mask[5:9, 5:9, 5:9] = 1
    cavity, _ = build_surface(cavity_mask, 1, np.eye(4))
    mask[cavity_mask == 1] = 0
    hollow, _ = build_surface(mask, 1, np.eye(4))
    q = quality(hollow)
    assert q["surface_components_vertex_connected"] == 2
    assert q["enclosed_volume"] == pytest.approx(quality(solid)["enclosed_volume"] - quality(cavity)["enclosed_volume"])
    assert q["enclosed_volume"] == pytest.approx(927.)


def test_open_surface_volume_not_reported_as_valid():
    q = quality(pv.Plane().triangulate())
    assert q["boundary_edges"] > 0
    assert q["enclosed_volume"] is None


def test_non_manifold_edge_and_winding_detected():
    points = np.array([[0,0,0],[1,0,0],[0,1,0],[0,-1,0],[0,0,1]], dtype=float)
    mesh = pv.PolyData(points, [3,0,1,2, 3,1,0,3, 3,0,1,4])
    q = quality(mesh)
    assert q["non_manifold_edges"] == 1
    assert q["enclosed_volume"] is None
