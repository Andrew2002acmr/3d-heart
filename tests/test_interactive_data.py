import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from heart3d.interactive.data import catalog, load_source, variant_allowed
from heart3d.interactive.state import ViewState, plane_corners, slice_rgb
from heart3d.mesh_quality import quality
from heart3d.surfaces import export_surfaces
from heart3d.volume import load_case


@pytest.fixture
def saved_case(tmp_path):
    data, results = tmp_path / "data", tmp_path / "results"
    data.mkdir()
    folder = results / "audit" / "example_42"
    folder.mkdir(parents=True)
    mask = np.zeros((12, 10, 8), dtype=np.uint8)
    mask[2:5, 2:5, 2:5] = 1
    mask[7:10, 5:8, 3:6] = 7
    ct = mask.astype(np.float32) * 20
    for kind, values in (("image", ct), ("label", mask)):
        nib.save(nib.Nifti1Image(values, np.eye(4)), data / f"example_42_{kind}.nii.gz")
    case = load_case(data / "example_42_image.nii.gz", data / "example_42_label.nii.gz")
    meshes = export_surfaces(case, folder / "meshes")
    (folder / "report.json").write_text(json.dumps(case.report), encoding="utf-8")
    return data, results, folder, case, meshes


def test_discover_arbitrary_id_load_vtp_and_missing_myo(saved_case):
    data, results, _, case, meshes = saved_case
    sources, _ = catalog(data, results)
    assert [s.case_id for s in sources] == ["example_42"]
    loaded = load_source(sources[0])
    assert loaded.volume.unit == "unknown"
    assert set(loaded.meshes) == {1, 7}
    assert 5 not in loaded.meshes
    assert loaded.report_path is not None
    np.testing.assert_array_equal(loaded.meshes[1]["Original"].points, meshes[1].points)
    np.testing.assert_array_equal(loaded.volume.affine, case.affine)
    assert any("unspecified" in w for w in loaded.warnings)


def test_stale_report_cannot_assign_mesh_to_different_mask(saved_case):
    data, results, folder, *_ = saved_case
    report = json.loads((folder / "report.json").read_text())
    report["mask"]["sha256"] = "different_input"
    (folder / "report.json").write_text(json.dumps(report))
    loaded = load_source(catalog(data, results)[0][0])
    assert loaded.report_path is None
    assert loaded.meshes == {1: {}, 7: {}}


def test_missing_optional_metadata_and_missing_vtp_do_not_crash(saved_case):
    data, results, folder, *_ = saved_case
    (folder / "meshes/01_LV.vtp").unlink()
    report = json.loads((folder / "report.json").read_text())
    report.pop("slices", None)
    (folder / "report.json").write_text(json.dumps(report))
    loaded = load_source(catalog(data, results)[0][0])
    assert not loaded.meshes[1]
    assert "Original" in loaded.meshes[7]
    assert loaded.window[0] < loaded.window[1]


def test_variants_require_matching_baseline_and_recorded_topology(saved_case):
    data, results, _, case, meshes = saved_case
    experiment = results / "postprocess"
    folder = experiment / "example_42" / "LV"
    folder.mkdir(parents=True)
    for name in ("original", "taubin", "decimate50"):
        meshes[1].save(folder / f"{name}.vtp")
    baseline = quality(meshes[1])
    bad = {**baseline, "non_manifold_edges": 1}
    record = {"all_input_hashes_unchanged": True, "structures": [{
        "case": "example_42", "label": 1, "unit": "unknown",
        "input_sha256": case.report["mask"]["sha256"], "original": baseline,
        "variants": {"taubin": baseline, "decimate50": bad}}]}
    (experiment / "experiment.json").write_text(json.dumps(record))
    loaded = load_source(catalog(data, results)[0][0])
    assert set(loaded.meshes[1]) == {"Original", "Smoothed"}
    assert any("blocked" in w for w in loaded.warnings)
    assert not variant_allowed(baseline, {})


def test_visibility_overlay_palette_and_unavailable_variant(saved_case):
    data, results, *_ = saved_case
    state = ViewState(load_source(catalog(data, results)[0][0]))
    state.indices[2] = 3
    state.overlay_opacity = 1
    rgb = slice_rgb(state, 2)
    # Display rows are reversed so voxel j increases upwards.
    np.testing.assert_array_equal(rgb[-1-3, 3], [0xe8, 0x5d, 0x75])
    state.set_visible(1, False)
    hidden = slice_rgb(state, 2)
    assert not np.array_equal(rgb, hidden)
    state.set_visible(5, True)
    assert 5 not in state.visible
    state.overlay = False
    gray = slice_rgb(state, 2)
    np.testing.assert_array_equal(gray[:, :, 0], gray[:, :, 1])
    with pytest.raises(ValueError, match="Unavailable"):
        state.select_variant(7, "Smoothed")


def test_slice_planes_use_full_affine_exactly_once(saved_case):
    _, _, _, case, _ = saved_case
    case.affine = np.array([[0, -2, 0, 10], [3, 0, 0, 20], [0, 0, 4, -5], [0, 0, 0, 1.]])
    corners = plane_corners(case, 2, 3)
    expected = nib.affines.apply_affine(case.affine, [[-.5, -.5, 3], [11.5, -.5, 3], [11.5, 9.5, 3], [-.5, 9.5, 3]])
    np.testing.assert_array_equal(corners, expected)
