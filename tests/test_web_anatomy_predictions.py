"""Synthetic-only checks for anatomy evidence-linked immutable slice import."""
import copy
import json
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from heart3d.anatomy_bundle import assemble, _digest
from heart3d.storage import sha256_file
from heart3d.web.anatomy_predictions import append_anatomy_bundle
from heart3d.web.anatomy_review import canonical_json
from heart3d.web.app import create_app
from heart3d.web.cli import create_user, main, seed_project
from heart3d.web.db import AnnotationProject, AnnotationVersion, Series
from heart3d.web.imaging import save_mask


@pytest.fixture
def anatomy_candidate(tmp_path):
    app = create_app(tmp_path / "synthetic-web")
    password = "Synthetic-anatomy-password!"
    resident = create_user(app.state.factory, "resident", password)
    create_user(app.state.factory, "outsider", password)
    create_user(app.state.factory, "surgeon", password, "surgeon")
    project = seed_project(app.state.root, app.state.factory, resident)
    with app.state.factory() as db:
        source = db.get(Series, db.get(AnnotationProject, project).series_id)
        ct, grid, ct_digest = app.state.root / source.ct_path, source.geometry, source.ct_sha256
    affine_lps = np.diag([-1., -1., 1., 1.]) @ np.asarray(grid["affine_ras"])
    observations = []
    for name, view, axis, index in (("ax-3", "AX", 2, 3), ("ax-4", "AX", 2, 4), ("cor", "COR", 1, 4)):
        anchor = [4., 4., 3.]
        anchor[axis] = float(index)
        normal = affine_lps[:3, axis] / np.linalg.norm(affine_lps[:3, axis])
        observations.append({"id": name, "kind": "observation", "structure": 1,
            "statement": "Synthetic visible edge only.", "features": ["synthetic edge"],
            "confidence": "low", "uncertainty_reason": "Synthetic, not clinical.", "evidence_refs": [],
            "source": "cloud_model", "locator": {"view": view, "anchor_ijk": anchor,
                "focal_point_lps_mm": nib.affines.apply_affine(affine_lps, anchor).tolist(), "normal_lps": normal.tolist(),
                "native_axis": axis, "native_index": index, "window_width": 400., "window_level": 40., "overlay_visible": False}})
    document = {"schema_version": 1, "target": {"axis": 2, "index": 3}, "observations": observations,
        "tracks": [{"id": "track-lv", "structure": 1, "observation_ids": [o["id"] for o in observations],
                    "relations": [{"from_id": "ax-3", "to_id": "ax-4", "kind": "continuation", "gap_reason": ""}]}],
        "decisions": [{"label": label, "state": "delineated" if label == 1 else "not_assessable",
            "supporting_observation_ids": [o["id"] for o in observations] if label == 1 else [],
            "counterevidence_ids": [], "track_ids": ["track-lv"] if label == 1 else [],
            "boundary_rule": "Synthetic boundary." if label == 1 else "",
            "rationale": "Synthetic AX adjacency and COR support." if label == 1 else "",
            "unresolved_questions": ["Synthetic test, no clinical accuracy."], "confidence": "low"} for label in range(1, 8)]}
    with TestClient(app, base_url="http://127.0.0.1") as client:
        auth = client.post("/api/auth/login", json={"username": "resident", "password": password})
        client.headers["X-CSRF-Token"] = auth.json()["csrf"]
        saved = client.post(f"/api/projects/{project}/anatomy-review", json={"base_version_id": None,
            "ct_sha256": ct_digest, "document": document})
        assert saved.status_code == 200, saved.text
        export = client.get(f"/api/projects/{project}/anatomy-review/export")
        notebook = tmp_path / "notebook.json"
        notebook.write_bytes(export.content)
        admission = tmp_path / "admission.json"
        admission.write_text(json.dumps({**{key: True for key in ("user_attested_permission_and_basis", "local_only",
            "geometry_verified", "independent_reader_verified", "metadata_deidentified", "pixel_identifier_screening_passed")},
            "ct_sha256": ct_digest}), encoding="utf-8")
        data = np.zeros(grid["shape"], np.uint8)
        data[2:7, 2:8, 3] = 1
        mask = tmp_path / "mask.nii.gz"
        save_mask(data, ct, mask)
        bundle = tmp_path / "bundle"
        assemble(ct, admission, notebook, mask, bundle)
        yield SimpleNamespace(app=app, client=client, project=project, bundle=bundle, ct=ct, data=data,
                              saved=saved.json(), password=password)
    app.state.engine.dispose()


def append(f, **changes):
    args = dict(root=f.app.state.root, factory=f.app.state.factory, username="resident", project_id=f.project, bundle=f.bundle)
    args.update(changes)
    return append_anatomy_bundle(**args)


def test_prepare_editable_boundaries_preserves_candidate_and_mask(anatomy_candidate):
    f = anatomy_candidate
    candidate = append(f)
    endpoint = f"/api/projects/{f.project}"
    before = f.client.get(endpoint).json()
    original = f.client.get(endpoint + f"/export/{candidate}").content
    adopted = f.client.post(endpoint + "/adopt-prediction", json={
        "version_id": candidate, "base_version_id": before["version_id"], "prepare_boundaries": True})
    assert adopted.status_code == 200, adopted.text
    view = adopted.json()
    assert view["version_id"] != candidate
    assert view["document"]["base_prediction_id"] == candidate
    assert len(view["document"]["contours"]) == 1
    assert len(view["document"]["operations"]) == 1
    unchanged = np.frombuffer(f.client.get(endpoint + "/mask").content, np.uint8).reshape(f.data.shape, order="F")
    assert np.array_equal(unchanged, f.data)
    preview = f.client.post(endpoint + "/preview", json=view["document"])
    assert preview.status_code == 200
    assert np.array_equal(np.frombuffer(preview.content, np.uint8).reshape(f.data.shape, order="F"), f.data)
    assert f.client.get(endpoint + f"/export/{candidate}").content == original
    assert f.client.get(endpoint + "/anatomy-review").json()["version_id"] == f.saved["version_id"]


def test_prepare_existing_draft_keeps_brush_edits_and_requires_no_existing_contours(anatomy_candidate):
    f = anatomy_candidate
    candidate = append(f)
    endpoint = f"/api/projects/{f.project}"
    before = f.client.get(endpoint).json()
    view = f.client.post(endpoint + "/adopt-prediction", json={"version_id": candidate, "base_version_id": before["version_id"]}).json()
    document = view["document"]
    index = int(np.ravel_multi_index((2, 3, 3), f.data.shape, order="F"))
    document["operations"].append({"kind": "brush", "indices": [index], "values": [0]})
    original = f.client.post(endpoint + "/preview", json=document).content
    response = f.client.post(endpoint + "/prepare-boundaries", json=document)
    assert response.status_code == 200, response.text
    prepared = response.json()
    assert all(prepared["operations"][0][key] == value for key, value in document["operations"][0].items())
    assert f.client.post(endpoint + "/preview", json=prepared).content == original
    assert f.client.post(endpoint + "/prepare-boundaries", json=prepared).status_code == 422
    assert f.client.get(endpoint).json()["version_id"] == view["version_id"]
    hole = copy.deepcopy(document)
    hole["operations"].append({"kind": "brush", "indices": [int(np.ravel_multi_index((4, 4, 3), f.data.shape, order="F"))], "values": [0]})
    assert f.client.post(endpoint + "/prepare-boundaries", json=hole).status_code == 422
    assert f.client.get(endpoint).json()["version_id"] == view["version_id"]


def rewrite_manifest(bundle, mutate):
    manifest = json.loads((bundle / "manifest.json").read_text())
    mutate(manifest)
    manifest["manifest_sha256"] = _digest({key: value for key, value in manifest.items() if key != "manifest_sha256"})
    (bundle / "manifest.json").write_bytes(canonical_json(manifest))


def test_import_links_saved_cloud_notebook_without_CNN_or_adoption(anatomy_candidate):
    f = anatomy_candidate
    before = f.client.get(f"/api/projects/{f.project}").json()
    version = append(f)
    after = f.client.get(f"/api/projects/{f.project}").json()
    assert (after["version_id"], after["status"]) == (before["version_id"], before["status"])
    assert after["revision"] == before["revision"] + 1
    assert len(after["predictions"]) == 1  # No recovered predictions were required.
    candidate = after["predictions"][0]
    assert candidate["id"] == version and candidate["source"] == "AI anatomy slice draft"
    assert not candidate["approved"] and not candidate["provenance"]["expert_approved"]
    provenance = candidate["provenance"]
    assert provenance["variant"] == "anatomy_slice_draft" and provenance["observer_sources"] == ["cloud_model"]
    assert provenance["notebook_version_id"] == f.saved["version_id"]
    assert provenance["image_cloud_transfer"] and provenance["external_validation_required"]
    assert "source_prediction_sha256" not in provenance and not provenance["medical_correctness_verified"]
    assert provenance["absent_classes"] == ["RV", "LA", "RA", "MYO", "AO", "PA"]
    with f.app.state.factory() as db:
        row = db.get(AnnotationVersion, version)
        folder = (f.app.state.root / row.mask_path).parent
        assert (folder / "anatomy-notebook.json").read_bytes() == (f.bundle / "notebook.json").read_bytes()
        assert (folder / "anatomy-bundle-manifest.json").read_bytes() == (f.bundle / "manifest.json").read_bytes()
    mask = f.client.get(f"/api/projects/{f.project}/mask?version_id={version}")
    assert mask.status_code == 200 and mask.headers["X-Annotation-Scope"] == "single-slice"
    np.testing.assert_array_equal(np.frombuffer(mask.content, np.uint8).reshape(f.data.shape, order="F"), f.data)
    with pytest.raises(ValueError, match="already exists"):
        append(f)
    assert f.client.get(f"/api/projects/{f.project}").json()["revision"] == after["revision"]


def test_adoption_retains_scope_and_protects_unannotated_planes(anatomy_candidate):
    f = anatomy_candidate
    version = append(f)
    head = f.client.get(f"/api/projects/{f.project}").json()
    adopted = f.client.post(f"/api/projects/{f.project}/adopt-prediction", json={"version_id": version, "base_version_id": head["version_id"]})
    assert adopted.status_code == 200, adopted.text
    view = adopted.json()
    assert view["version_id"] != version and view["annotation_scope"] == {"axis": 2, "index": 3, "complete_volume": False}
    assert view["document"]["base_prediction_id"] == version
    document = copy.deepcopy(view["document"])
    document["operations"].append({"kind": "brush", "indices": [0], "values": [0]})
    assert f.client.post(f"/api/projects/{f.project}/preview", json=document).status_code == 422
    for item in (version, view["version_id"]):
        assert f.client.get(f"/api/projects/{f.project}/reconstruction/{item}").status_code == 422
        assert f.client.get(f"/api/projects/{f.project}/export/{item}").headers["X-Annotation-Scope"] == "single-slice"
    assert f.client.get(f"/api/projects/{f.project}/anatomy-review").json()["version_id"] == f.saved["version_id"]


@pytest.mark.parametrize("change", ["manifest_hash", "approved", "medical", "ct", "scope", "cloud", "mask_path", "notebook_hash", "positive_labels", "readiness"])
def test_invalid_bundle_rejected_without_files_or_revision(anatomy_candidate, change):
    f = anatomy_candidate
    if change == "manifest_hash":
        path = f.bundle / "manifest.json"
        payload = json.loads(path.read_text()); payload["expert_approved"] = True
        path.write_text(json.dumps(payload))
    else:
        def mutate(m):
            if change == "approved": m["expert_approved"] = True
            elif change == "medical": m["medical_correctness_verified"] = True
            elif change == "ct": m["ct_sha256"] = "0" * 64
            elif change == "scope": m["scope"]["index"] = 4
            elif change == "cloud": m["image_cloud_transfer"] = False
            elif change == "mask_path": m["mask"]["file"] = "../mask.nii.gz"
            elif change == "notebook_hash": m["notebook"]["sha256"] = "0" * 64
            elif change == "positive_labels": m["positive_labels"] = [1, 7]
            elif change == "readiness": m["readiness"]["expert_validated"] = True
        rewrite_manifest(f.bundle, mutate)
    before = f.client.get(f"/api/projects/{f.project}").json()
    with pytest.raises(ValueError):
        append(f)
    after = f.client.get(f"/api/projects/{f.project}").json()
    assert (after["revision"], after["version_id"], after["status"]) == (before["revision"], before["version_id"], before["status"])
    assert not list((f.app.state.root / "annotations").glob("**/mask.nii.gz"))


@pytest.mark.parametrize("change", ["unsaved_version", "different_document", "foreign_project"])
def test_rehashed_export_must_match_immutable_DB_notebook(anatomy_candidate, change):
    f = anatomy_candidate
    path = f.bundle / "notebook.json"
    export = json.loads(path.read_text())
    if change == "unsaved_version": export["version_id"] = "not-a-saved-version"
    elif change == "different_document": export["document"]["observations"][0]["statement"] = "Altered synthetic statement."
    elif change == "foreign_project": export["project_id"] = "another-project"
    export["document_sha256"] = _digest(export["document"])
    export["export_sha256"] = _digest({key: value for key, value in export.items() if key != "export_sha256"})
    path.write_bytes(canonical_json(export))
    def mutate(m):
        m["notebook"].update({key: export[key] for key in ("version_id", "project_id", "document_sha256", "export_sha256")})
        m["notebook"]["sha256"] = sha256_file(path)
    rewrite_manifest(f.bundle, mutate)
    with pytest.raises(ValueError, match="immutable saved version"):
        append(f)


@pytest.mark.parametrize("username", ["outsider", "surgeon", "unknown"])
def test_assigned_owner_role_required(anatomy_candidate, username):
    with pytest.raises(ValueError, match="assigned resident/admin"):
        append(anatomy_candidate, username=username)


def test_owner_cli_imports_bundle_without_adoption(anatomy_candidate, monkeypatch, capsys):
    f = anatomy_candidate
    monkeypatch.setattr("sys.argv", ["heart3d.web.cli", "--root", str(f.app.state.root), "import-anatomy-bundle",
        "--username", "resident", "--project", f.project, "--bundle", str(f.bundle)])
    main()
    assert "head unchanged" in capsys.readouterr().out
    view = f.client.get(f"/api/projects/{f.project}").json()
    assert view["version_id"] is None and view["status"] == "not_started"
    assert view["predictions"][0]["source"] == "AI anatomy slice draft"
