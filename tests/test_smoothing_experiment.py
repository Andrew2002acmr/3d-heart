import json

import numpy as np
import pyvista as pv
import pytest

from heart3d.mesh_quality import quality
from heart3d.smoothing_experiment import assess, compare, run
from heart3d.volume import sha256


def test_surface_distance_and_continuous_upper_bound():
    a = pv.Plane(i_size=10, j_size=10).triangulate()
    b = a.translate((0, 0, 2), inplace=False)
    q = compare(a, b, quality(a), count=100)
    d = q["deviation_from_baseline"]
    assert d["symmetric_mean"] == pytest.approx(2)
    assert d["sampled_max"] == pytest.approx(2)
    assert d["hausdorff_upper_bound"] == pytest.approx(2)
    assert not d["is_exact_hausdorff"]
    assert q["volume_change_percent"] is None
    with pytest.raises(ValueError, match="connectivity"):
        compare(a, b.subdivide(1), quality(a), count=100)


def test_screening_rejects_new_defects_and_large_volume_drift():
    mesh = pv.Sphere()
    baseline = quality(mesh)
    metrics = compare(mesh, mesh, baseline, count=100)
    assert assess(baseline, metrics)["status"] == "candidate"
    for key in ("non_manifold_edges", "boundary_edges", "zero_area_triangles", "surface_components_vertex_connected"):
        bad = {**metrics, key: metrics[key]+1}
        assert assess(baseline, bad)["status"] == "rejected"
    assert assess(baseline, {**metrics, "volume_change_percent": -3})["status"] == "rejected"
    assert assess({**baseline, "enclosed_volume": None}, {**metrics, "enclosed_volume": None,
                  "volume_change_percent": None})["status"] == "review_invalid_baseline"


def test_run_reads_only_saved_baseline_skips_absent_myo_and_never_overwrites(tmp_path):
    audit = tmp_path / "audit"
    folder = audit / "example" / "meshes"
    folder.mkdir(parents=True)
    mesh = pv.Sphere(theta_resolution=12, phi_resolution=12)
    source = folder / "01_LV.vtp"
    mesh.save(source)
    digest = sha256(source)
    report = {"geometry": {"unit": "unknown"}, "structures": [{"label": 1,
              "points": mesh.n_points, "triangles": mesh.n_cells, "file": "meshes/01_LV.vtp",
              "smoothing": False, "decimation": False, "normal_policy": "preserve winding"}]}
    (folder.parent / "report.json").write_text(json.dumps(report))
    out = tmp_path / "experiment"
    result = run(audit, ["example"], out, count=100)
    assert result["completed"] and result["all_source_hashes_unchanged"]
    assert "MYO" in result["missing_labels"]["example"]
    assert len(result["structures"][0]["variants"]) == 6
    assert sha256(source) == sha256(out / "example/LV/baseline.vtp") == digest
    for variant in result["structures"][0]["variants"].values():
        assert (out / variant["file"]).is_file()
        np.testing.assert_array_equal(pv.read(out / variant["file"]).faces, mesh.faces)
    with pytest.raises(ValueError, match="empty"):
        run(audit, ["example"], out, count=100)
    with pytest.raises(ValueError, match="overlap"):
        run(audit, ["example"], audit / "nested", count=100)
