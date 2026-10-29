"""All fixtures are synthetic; clinical files never enter the test suite."""
from pathlib import Path
import gzip
import copy
import nibabel as nib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from heart3d.web.app import create_app
from heart3d.web.cli import create_user
from heart3d.web.db import AnnotationVersion, Assignment
from heart3d.web.imaging import phantom, save_mask, rasterize, trace_mask_contours
from heart3d.web.predictions import import_predictions
from heart3d.storage import sha256_file


@pytest.fixture
def prediction_app(tmp_path):
    app = create_app(tmp_path / "private")
    password = "Synthetic-AI-fixture-password!"
    create_user(app.state.factory, "resident", password)
    surgeon = create_user(app.state.factory, "surgeon", password, "surgeon")
    create_user(app.state.factory, "outsider", password)
    ct = tmp_path / "ct.nii.gz"
    phantom(ct)
    masks = []
    for variant in ("v1", "v2"):
        mask = np.zeros((96, 96, 64), np.uint8)
        mask[35:55, 35:55, 30:35] = 1
        if variant == "v2": mask[60:70, 35:45, 30:35] = 2
        path = tmp_path / (variant+".nii.gz")
        digest = save_mask(mask, ct, path)
        masks.append((path, {"variant": variant, "expert_approved": False, "exploratory_only": True,
                             "geometry_verified": True, "ct_sha256": sha256_file(ct),
                             "prediction_sha256": digest, "checkpoint_sha256": "0"*64}))
    project = import_predictions(app.state.root, app.state.factory, "resident", ct, masks, "TEST-001", synthetic=True)
    with app.state.factory.begin() as db:
        db.add(Assignment(project_id=project, user_id=surgeon))
    with TestClient(app, base_url="http://127.0.0.1") as client:
        yield app, client, project, password, masks, ct
    app.state.engine.dispose()


def login(client, password, user="resident"):
    r = client.post("/api/auth/login", json={"username": user, "password": password})
    assert r.status_code == 200
    client.headers["X-CSRF-Token"] = r.json()["csrf"]


def test_AI_versions_stay_immutable_while_human_draft_preserves_base_and_can_be_approved(prediction_app):
    app, client, project, password, _, _ = prediction_app
    assert client.get(f"/api/projects/{project}/mask?version_id=unknown").status_code == 401
    login(client, password)
    view = client.get(f"/api/projects/{project}").json()
    assert view["version_id"] is None and view["status"] == "not_started"
    assert len(view["predictions"]) == 2
    v1, v2 = [r["id"] for r in view["predictions"]]
    original = client.get(f"/api/projects/{project}/export/{v1}").content
    ai = np.frombuffer(client.get(f"/api/projects/{project}/mask?version_id={v1}").content, np.uint8)
    assert np.count_nonzero(ai) == 2000
    adopted = client.post(f"/api/projects/{project}/adopt-prediction", json={"version_id": v1, "base_version_id": None})
    assert adopted.status_code == 200, adopted.text
    view = adopted.json()
    doc = view["document"]
    doc["operations"].append({"kind": "brush", "indices": [0], "values": [7]})
    doc["flags"].append({"id": "suspicious-1", "axis": 2, "slice": 32, "label": 1, "kind": "uncertain"})
    body = {"base_version_id": view["version_id"], "ct_sha256": view["ct_sha256"], "document": doc}
    result = client.post(f"/api/projects/{project}/drafts", json=body)
    assert result.status_code == 200, result.text
    version = result.json()["version_id"]
    mask = np.frombuffer(client.get(f"/api/projects/{project}/mask").content, np.uint8)
    assert np.count_nonzero(mask == 1) == 2000 and mask[0] == 7
    assert client.get(f"/api/projects/{project}/export/{v1}").content == original
    assert client.post(f"/api/projects/{project}/adopt-prediction", json={"version_id": v2, "base_version_id": None}).status_code == 409
    assert client.post(f"/api/projects/{project}/submit", json={"version_id": version}).status_code == 200
    login(client, password, "surgeon")
    approved = client.post(f"/api/projects/{project}/reviews", json={"version_id": version, "decision": "approved"})
    assert approved.status_code == 200
    reference_id = approved.json()["immutable_version_id"]
    assert reference_id != version
    with app.state.factory() as db:
        assert db.get(AnnotationVersion, v1).source == "AI prediction"
        assert db.get(AnnotationVersion, version).source == "human-edited draft"
        reference = db.get(AnnotationVersion, reference_id)
        assert reference.source == "surgeon-approved reference" and reference.parent_id == version
        assert reference.mask_sha256 == db.get(AnnotationVersion, version).mask_sha256
    assert client.get(f"/api/projects/{project}/export/{v1}").content == original
    login(client, password, "resident")
    assert client.post(f"/api/projects/{project}/drafts", json=dict(body, base_version_id=version)).status_code == 409


def test_prediction_trace_3D_authorization_and_absent_labels(prediction_app):
    app, client, project, password, _, _ = prediction_app
    login(client, password)
    view = client.get(f"/api/projects/{project}").json()
    v1 = view["predictions"][0]["id"]
    view = client.post(f"/api/projects/{project}/adopt-prediction", json={"version_id": v1}).json()
    traces = client.post(f"/api/projects/{project}/trace-contours", json={"document": view["document"], "axis": 2, "slice": 32, "label": 1})
    assert traces.status_code == 200
    assert len(traces.json()) == 1 and traces.json()[0]["replace_label_slice"]
    doc = view["document"]
    doc["contours"] = traces.json()
    doc["operations"] = [{"kind": "contour", "contour_id": c["id"]} for c in doc["contours"]]
    preview = client.post(f"/api/projects/{project}/preview", json=doc)
    assert preview.status_code == 200
    assert np.count_nonzero(np.frombuffer(preview.content, np.uint8)) == 2000
    metadata = client.get(f"/api/projects/{project}/reconstruction/{v1}")
    assert metadata.status_code == 200, metadata.text
    assert metadata.json()["absent_labels"] == [2,3,4,5,6,7]
    assert client.get(f"/api/projects/{project}/mesh/{v1}/1").status_code == 200
    assert client.get(f"/api/projects/{project}/mesh/{v1}/2").status_code == 404
    login(client, password, "outsider")
    for suffix in (f"/mask?version_id={v1}", f"/reconstruction/{v1}", f"/mesh/{v1}/1"):
        assert client.get(f"/api/projects/{project}{suffix}").status_code == 404
    assert client.post(f"/api/projects/{project}/adopt-prediction", json={"version_id": v1}).status_code == 404


def test_import_refuses_misaligned_mask_and_preserves_existing_study(prediction_app, tmp_path):
    app, _, _, _, masks, ct = prediction_app
    with pytest.raises(ValueError, match="exists"):
        import_predictions(app.state.root, app.state.factory, "resident", ct, masks, "TEST-001", synthetic=True)
    bad = tmp_path / "bad.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((2,2,2), np.uint8), np.eye(4)), bad)
    with pytest.raises(ValueError, match="grid/labels"):
        import_predictions(app.state.root, app.state.factory, "resident", ct, [(bad,masks[0][1]),masks[1]], "TEST-002", synthetic=True)


def test_trace_does_not_fill_holes_or_clip_scan_boundary():
    mask = np.zeros((20,20,4), np.uint8)
    mask[3:15,3:15,2] = 1
    mask[7:9,7:9,2] = 0
    grid = {"shape": list(mask.shape), "affine_ras": np.eye(4).tolist()}
    with pytest.raises(ValueError, match="holes"):
        trace_mask_contours(mask, grid, 1, 2, 2)
    mask[7:9,7:9,2] = 1
    mask[0,5,2] = 1
    with pytest.raises(ValueError, match="edge"):
        trace_mask_contours(mask, grid, 1, 2, 2)
