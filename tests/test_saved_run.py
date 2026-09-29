import numpy as np
import pyvista as pv

from scripts.verify_saved_run import checked_quality, differences
from heart3d.mesh_quality import quality


def test_comparison_rejects_missing_results_null_zero_and_changed_counts():
    assert differences({"cases": [1, 2]}, {"cases": [1, 2, 3]})
    assert differences({"volume": 0}, {"volume": None})
    assert differences({"triangles": 2001}, {"triangles": 2000})
    assert not differences({"area": 10.0 + 1e-10}, {"area": 10.0})
    assert differences({"area": float("nan")}, {"area": 10.0})


def test_reloading_mesh_detects_geometry_changed_after_report(tmp_path):
    mesh = pv.Sphere(theta_resolution=12, phi_resolution=12).triangulate()
    reported = quality(mesh)
    path = tmp_path / "mesh.vtp"
    mesh.save(path)
    assert not checked_quality(pv.read(path), reported)
    damaged = mesh.scale(2, inplace=False)
    damaged.save(path)
    assert checked_quality(pv.read(path), reported)
    damaged.points[0] = np.nan
    assert checked_quality(damaged, reported)
