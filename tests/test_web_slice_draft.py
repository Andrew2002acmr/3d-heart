"""Synthetic-only checks for immutable, explicitly partial axial annotations."""
import copy
import gzip
import json
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from heart3d.dicom.ct import LPS_TO_RAS
from heart3d.storage import sha256_file
from heart3d.web.app import create_app
from heart3d.web.cli import create_user, main
from heart3d.web.db import AnnotationProject, AnnotationVersion, Approval, Assignment
from heart3d.web.imaging import save_mask
from heart3d.web.predictions import append_slice_draft, import_predictions


@pytest.fixture
def slice_app(tmp_path):
    app = create_app(tmp_path / "private")
    password = "Synthetic-slice-password!"
    create_user(app.state.factory, "resident", password)
    create_user(app.state.factory, "outsider", password)
    surgeon = create_user(app.state.factory, "surgeon", password, "surgeon")
    ct = tmp_path / "synthetic-ct.nii.gz"
    affine = np.diag([.8, 1.1, 1.6, 1.])
    affine[:3, 3] = [-6.4, -9.9, -6.4]
    image = nib.Nifti1Image(np.zeros((16, 18, 8), np.float32), affine)
    image.header.set_xyzt_units("mm")
    image.set_qform(affine, 1)
    image.set_sform(affine, 1)
    nib.save(image, ct)
    originals = []
    for variant, label in (("v1", 1), ("v2", 2)):
        array = np.zeros(image.shape, np.uint8)
        array[3:8, 4:9, 2:6] = label
        path = tmp_path / f"{variant}.nii.gz"
        digest = save_mask(array, ct, path)
        originals.append((path, {"variant": variant, "ct_sha256": sha256_file(ct),
            "prediction_sha256": digest, "expert_approved": False, "exploratory_only": True,
            "geometry_verified": True}))
    project = import_predictions(app.state.root, app.state.factory, "resident", ct, originals,
                                 "TEST-SLICE", synthetic=True)
    with app.state.factory.begin() as db:
        db.add(Assignment(project_id=project, user_id=surgeon))
    array = np.zeros(image.shape, np.uint8)
    array[3:8, 4:9, 3] = 1
    array[9:12, 5:9, 3] = 7
    out = tmp_path / "slice-result"
    out.mkdir()
    mask = out / "prediction.nii.gz"
    digest = save_mask(array, ct, mask)
    names = ("background", "LV", "RV", "LA", "RA", "MYO", "AO", "PA")
    provenance = {"variant": "slice_draft", "ct_sha256": sha256_file(ct), "prediction_sha256": digest,
        "scope": {"axis": 2, "index": 3, "complete_volume": False}, "expert_approved": False,
        "exploratory_only": True, "geometry_verified": True, "method": "synthetic equal ensemble",
        "warning": "Single-slice exploratory draft; no expert review; outside-plane zeros are unannotated.",
        "review_required_classes": list(names[1:]),
        "source_prediction_sha256": [p[1]["prediction_sha256"] for p in originals],
        "predicted_voxels": {name: int(np.count_nonzero(array == label)) for label, name in enumerate(names)},
        "unsafe_source_filename": "must not be copied into the database"}
    with TestClient(app, base_url="http://127.0.0.1") as client:
        yield SimpleNamespace(app=app, client=client, project=project, password=password, ct=ct,
                              mask=mask, array=array, provenance=provenance, out=out)
    app.state.engine.dispose()


def login(fixture, username="resident"):
    response = fixture.client.post("/api/auth/login", json={"username": username, "password": fixture.password})
    assert response.status_code == 200
    fixture.client.headers["X-CSRF-Token"] = response.json()["csrf"]


def append(fixture, **overrides):
    kwargs = {"root": fixture.app.state.root, "factory": fixture.app.state.factory, "username": "resident",
              "project_id": fixture.project, "mask": fixture.mask, "provenance": fixture.provenance}
    kwargs.update(overrides)
    return append_slice_draft(**kwargs)


def adopt(fixture, version):
    before = fixture.client.get(f"/api/projects/{fixture.project}").json()
    response = fixture.client.post(f"/api/projects/{fixture.project}/adopt-prediction",
        json={"version_id": version, "base_version_id": before["version_id"]})
    assert response.status_code == 200, response.text
    return response.json()


def test_append_keeps_existing_head_state_and_original_AI_immutable(slice_app, tmp_path):
    f = slice_app
    login(f)
    original_view = f.client.get(f"/api/projects/{f.project}").json()
    original_ids = [p["id"] for p in original_view["predictions"]]
    original_exports = {v: f.client.get(f"/api/projects/{f.project}/export/{v}").content for v in original_ids}
    head = adopt(f, original_ids[0])
    version = append(f)
    view = f.client.get(f"/api/projects/{f.project}").json()
    assert (view["version_id"], view["status"]) == (head["version_id"], head["status"])
    assert view["revision"] == head["revision"] + 1 and view["annotation_scope"] is None
    assert len(view["predictions"]) == 3
    row = view["predictions"][-1]
    assert row["id"] == version and row["number"] == view["revision"]
    assert row["source"] == "AI-assisted slice draft" and row["approved"] is False
    assert row["provenance"]["scope"] == f.provenance["scope"]
    assert row["provenance"]["expert_approved"] is False
    assert "unsafe_source_filename" not in row["provenance"]
    assert row["provenance"]["predicted_voxels"] == f.provenance["predicted_voxels"]
    response = f.client.get(f"/api/projects/{f.project}/mask?version_id={version}")
    assert response.headers["X-Annotation-Scope"] == "single-slice"
    assert response.headers["X-Annotation-Slice-Index"] == "3"
    np.testing.assert_array_equal(np.frombuffer(response.content, np.uint8).reshape(f.array.shape, order="F"), f.array)
    with pytest.raises(ValueError, match="already exists"):
        append(f)
    # Recompressing the same voxels changes the file digest, not its identity as a plane.
    other_mask = tmp_path / "same-voxels.nii.gz"
    other_mask.write_bytes(gzip.compress(gzip.decompress(f.mask.read_bytes()), mtime=1))
    other_provenance = dict(f.provenance, prediction_sha256=sha256_file(other_mask))
    assert sha256_file(other_mask) != sha256_file(f.mask)
    with pytest.raises(ValueError, match="already exists"):
        append(f, mask=other_mask, provenance=other_provenance)
    for original, content in original_exports.items():
        assert f.client.get(f"/api/projects/{f.project}/export/{original}").content == content
    assert f.client.get(f"/api/projects/{f.project}").json()["revision"] == view["revision"]


def test_adopt_edit_review_preserves_partial_scope_and_rejects_3D(slice_app):
    f = slice_app
    version = append(f)
    login(f)
    view = adopt(f, version)
    human_id = view["version_id"]
    assert human_id != version and view["annotation_scope"] == f.provenance["scope"]
    assert "other slices are unannotated" in view["annotation_warning"]
    assert view["document"]["base_prediction_id"] == version
    document = view["document"]
    brush_index = int(np.ravel_multi_index((4, 5, 3), f.array.shape, order="F"))
    document["operations"].append({"kind": "brush", "indices": [brush_index], "values": [6]})
    document["flags"].append({"id": "review-1", "axis": 2, "slice": 3, "label": 1, "kind": "uncertain"})
    preview = f.client.post(f"/api/projects/{f.project}/preview", json=document)
    assert preview.status_code == 200, preview.text
    assert preview.headers["X-Annotation-Scope"] == "single-slice"
    traces = f.client.post(f"/api/projects/{f.project}/trace-contours",
        json={"document": document, "axis": 2, "slice": 3, "label": 7})
    assert traces.status_code == 200 and traces.json()
    document["contours"] = traces.json()
    document["operations"] += [{"kind": "contour", "contour_id": c["id"]} for c in traces.json()]
    saved = f.client.post(f"/api/projects/{f.project}/drafts", json={"base_version_id": human_id,
        "ct_sha256": view["ct_sha256"], "document": document})
    assert saved.status_code == 200, saved.text
    saved_id = saved.json()["version_id"]
    assert saved.json()["annotation_scope"] == f.provenance["scope"]
    assert f.client.post(f"/api/projects/{f.project}/submit", json={"version_id": saved_id}).status_code == 200
    login(f, "surgeon")
    approved = f.client.post(f"/api/projects/{f.project}/reviews", json={"version_id": saved_id, "decision": "approved"})
    assert approved.status_code == 200, approved.text
    reference_id = approved.json()["immutable_version_id"]
    assert reference_id != saved_id
    current = f.client.get(f"/api/projects/{f.project}").json()
    assert current["version_id"] == reference_id and current["annotation_scope"] == f.provenance["scope"]
    assert current["document"]["base_prediction_id"] == version
    assert current["document"]["annotation_scope"] == f.provenance["scope"]
    with f.app.state.factory() as db:
        assert db.get(AnnotationVersion, version).source == "AI-assisted slice draft"
        assert db.get(AnnotationVersion, saved_id).source == "human-edited draft"
        reference = db.get(AnnotationVersion, reference_id)
        assert reference.source == "surgeon-approved reference" and reference.parent_id == saved_id
        assert db.scalar(select(Approval).where(Approval.version_id == reference_id))
        assert not db.scalar(select(Approval).where(Approval.version_id == version))
    for item in (version, human_id, saved_id, reference_id):
        for path in (f"/reconstruction/{item}", f"/mesh/{item}/1"):
            result = f.client.get(f"/api/projects/{f.project}{path}")
            assert result.status_code == 422, result.text
        exported = f.client.get(f"/api/projects/{f.project}/export/{item}")
        assert exported.status_code == 200 and exported.headers["X-Annotation-Scope"] == "single-slice"
        assert exported.headers["X-Annotation-Slice-Index"] == "3"
        array = np.asanyarray(nib.Nifti1Image.from_bytes(gzip.decompress(exported.content)).dataobj)
        assert not np.any(array[:, :, :3]) and not np.any(array[:, :, 4:])
    assert not (f.app.state.root / "derived").exists()


def test_partial_scope_rejects_contours_brush_flags_trace_and_base_removal_outside_plane(slice_app):
    f = slice_app
    version = append(f)
    login(f)
    view = adopt(f, version)
    document = view["document"]
    affine = LPS_TO_RAS @ np.asarray(view["geometry"]["affine_ras"])
    points = nib.affines.apply_affine(affine, [[3, 4, 4], [7, 4, 4], [7, 8, 4]])
    wrong_contour = {"id": "outside", "label": 1, "axis": 2, "slice": 4, "points": points.tolist(),
                     "closed": True, "applied": True}
    candidates = []
    bad = copy.deepcopy(document)
    bad["contours"].append(wrong_contour)
    bad["operations"].append({"kind": "contour", "contour_id": "outside"})
    candidates.append(bad)
    bad = copy.deepcopy(document)
    bad["operations"].append({"kind": "brush", "indices": [0], "values": [0]})
    candidates.append(bad)
    bad = copy.deepcopy(document)
    bad["flags"].append({"id": "outside", "axis": 2, "slice": 4, "label": 1, "kind": "boundary"})
    candidates.append(bad)
    bad = copy.deepcopy(document)
    bad["flags"].append({"id": "wrong-axis", "axis": 0, "slice": 3, "label": 1, "kind": "uncertain"})
    candidates.append(bad)
    bad = copy.deepcopy(document)
    bad.pop("base_prediction_id")
    candidates.append(bad)
    bad = copy.deepcopy(document)
    bad["base_prediction_id"] = view["predictions"][0]["id"]
    candidates.append(bad)
    for bad in candidates:
        assert f.client.post(f"/api/projects/{f.project}/preview", json=bad).status_code == 422
        saved = f.client.post(f"/api/projects/{f.project}/drafts", json={"base_version_id": view["version_id"],
            "ct_sha256": view["ct_sha256"], "document": bad})
        assert saved.status_code == 422, saved.text
    for axis, index in ((2, 4), (0, 3), (1, 3)):
        assert f.client.post(f"/api/projects/{f.project}/trace-contours",
            json={"document": document, "axis": axis, "slice": index, "label": 1}).status_code == 422
    assert f.client.get(f"/api/projects/{f.project}").json()["version_id"] == view["version_id"]


def test_import_validates_hashes_geometry_plane_provenance_and_role_before_writes(slice_app, tmp_path):
    f = slice_app
    invalid = [
        {"ct_sha256": "0" * 64}, {"prediction_sha256": "0" * 64}, {"expert_approved": True},
        {"exploratory_only": False}, {"geometry_verified": False}, {"variant": "v1"},
        {"scope": {"axis": 1, "index": 3, "complete_volume": False}},
        {"scope": {"axis": 2, "index": 3, "complete_volume": True}},
        {"scope": {"axis": 2, "index": 99, "complete_volume": False}},
        {"method": ""}, {"warning": ""}, {"review_required_classes": [7]},
        {"source_prediction_sha256": ["0" * 64, "1" * 64]}, {"predicted_voxels": {"LV": 999}},
    ]
    for change in invalid:
        with pytest.raises(ValueError):
            append(f, provenance=dict(f.provenance, **change))
    for username in ("outsider", "surgeon", "unknown"):
        with pytest.raises(ValueError, match="assigned resident/admin"):
            append(f, username=username)
    for kind in ("shape", "affine", "qform", "sform", "unit", "label", "nonfinite", "outside", "metadata"):
        image = nib.load(f.mask)
        array = np.asanyarray(image.dataobj).copy()
        if kind == "shape":
            image = nib.Nifti1Image(np.zeros((2, 3, 4), np.uint8), image.affine, image.header.copy())
        elif kind in ("affine", "qform", "sform"):
            matrix = image.affine.copy()
            matrix[0, 3] += 5
            if kind in ("affine", "qform"):
                image.set_qform(matrix, 1)
            if kind in ("affine", "sform"):
                image.set_sform(matrix, 1)
        elif kind == "unit":
            image.header.set_xyzt_units("meter")
        elif kind in ("label", "nonfinite", "outside"):
            if kind == "nonfinite":
                array = array.astype(np.float32)
                array[0, 0, 3] = np.nan
            else:
                array[0, 0, 4 if kind == "outside" else 3] = 1 if kind == "outside" else 8
            image = nib.Nifti1Image(array, image.affine, image.header.copy())
            if kind == "nonfinite":
                image.set_data_dtype(np.float32)
        else:
            image.header["descrip"] = b"untrusted text"
        path = tmp_path / f"bad-{kind}.nii.gz"
        nib.save(image, path)
        with pytest.raises(ValueError):
            append(f, mask=path, provenance=dict(f.provenance, prediction_sha256=sha256_file(path)))
    with f.app.state.factory() as db:
        project = db.get(AnnotationProject, f.project)
        assert project.revision == 2 and project.head_version_id is None and project.status == "not_started"
        assert len(list(db.scalars(select(AnnotationVersion).where(AnnotationVersion.project_id == f.project)))) == 2
    assert len(list((f.app.state.root / "annotations" / f.project).glob("*/mask.nii.gz"))) == 2
    stored_ct = f.app.state.root / "raw" / f"{f.project}.nii.gz"
    stored_ct.write_bytes(stored_ct.read_bytes() + b"tampered-synthetic-data")
    with pytest.raises(ValueError, match="Source CT integrity"):
        append(f)


def test_owner_CLI_imports_existing_project_without_adoption(slice_app, monkeypatch, capsys):
    f = slice_app
    (f.out / "provenance.json").write_text(json.dumps(f.provenance), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["heart3d.web.cli", "--root", str(f.app.state.root), "import-slice-draft",
        "--username", "resident", "--project", f.project, "--out", str(f.out)])
    main()
    assert "head unchanged" in capsys.readouterr().out
    login(f)
    view = f.client.get(f"/api/projects/{f.project}").json()
    assert view["version_id"] is None and view["status"] == "not_started" and view["revision"] == 3
    assert len(view["predictions"]) == 3
