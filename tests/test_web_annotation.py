import gzip
from io import BytesIO
from pathlib import Path
import numpy as np
import nibabel as nib
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from heart3d.web.app import create_app
from heart3d.web.cli import create_user, seed_project, backup
from heart3d.web.db import Assignment, AnnotationVersion, Approval, AuditEvent, User
from heart3d.web.imaging import rasterize
from heart3d.dicom.ct import LPS_TO_RAS
from heart3d.storage import sha256_file


@pytest.fixture
def setup_web(tmp_path):
    app = create_app(tmp_path / "private")
    factory = app.state.factory
    password = "Synthetic-test-password!"
    resident = create_user(factory, "resident", password)
    other = create_user(factory, "other-resident", password)
    surgeon = create_user(factory, "surgeon", password, "surgeon")
    project = seed_project(app.state.root, factory, resident)
    foreign = seed_project(app.state.root, factory, other, "002")
    with factory.begin() as db:
        db.add(Assignment(project_id=project, user_id=surgeon))
    with TestClient(app, base_url="http://127.0.0.1") as client:
        yield app, client, project, foreign, password
    app.state.engine.dispose()


def login(client, password, username="resident"):
    response = client.post("/api/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200
    client.headers["X-CSRF-Token"] = response.json()["csrf"]
    return response


def draft(client, project):
    metadata = client.get(f"/api/projects/{project}").json()
    affine = LPS_TO_RAS @ np.asarray(metadata["geometry"]["affine_ras"])
    points = nib.affines.apply_affine(affine, [[40, 40, 32], [55, 40, 32], [55, 55, 32], [40, 55, 32]])
    return {"base_version_id": metadata["version_id"], "ct_sha256": metadata["ct_sha256"],
            "label_schema": metadata["label_schema"], "document": {
                "contours": [{"id": "polygon-1", "label": 1, "axis": 2, "slice": 32,
                              "points": points.tolist(), "closed": True, "applied": True}],
                "operations": [{"kind": "contour", "contour_id": "polygon-1"}]}}


def test_auth_session_logout_csrf_and_study_authorization(setup_web):
    app, client, project, foreign, password = setup_web
    for suffix in ("", "/volume", "/mask", "/versions", "/export/unknown"):
        assert client.get(f"/api/projects/{project}{suffix}").status_code == 401
    assert client.get("/api/projects").status_code == 401
    assert client.post("/api/auth/login", json={"username": "resident", "password": "bad"}).status_code == 401
    response = login(client, password)
    assert "httponly" in response.headers["set-cookie"].lower()
    assert "samesite=strict" in response.headers["set-cookie"].lower()
    assert [p["id"] for p in client.get("/api/projects").json()] == [project]
    for suffix in ("", "/volume", "/mask", "/versions", "/export/unknown"):
        assert client.get(f"/api/projects/{foreign}{suffix}").status_code == 404
    payload = draft(client, project)
    for suffix, body in (("/preview", payload["document"]), ("/drafts", payload),
                         ("/submit", {"version_id": "foreign"}),
                         ("/reviews", {"version_id": "foreign", "decision": "approved"})):
        assert client.post(f"/api/projects/{foreign}{suffix}", json=body).status_code == 404
    token = client.headers.pop("X-CSRF-Token")
    assert client.post(f"/api/projects/{project}/drafts", json=payload).status_code == 403
    client.headers["X-CSRF-Token"] = token
    assert client.post(f"/api/projects/{project}/drafts", json=payload, headers={"Origin": "https://untrusted.invalid"}).status_code == 403
    assert client.post("/api/auth/logout").status_code == 200
    assert client.get("/api/auth/me").status_code == 401
    with app.state.factory() as db:
        user = db.scalar(select(User).where(User.username == "resident"))
        assert user.password_hash.startswith("$argon2id$") and password not in user.password_hash


def test_save_reload_geometry_export_immutable_versions_and_conflict(setup_web):
    app, client, project, _, password = setup_web
    login(client, password)
    payload = draft(client, project)
    response = client.post(f"/api/projects/{project}/drafts", json=payload)
    assert response.status_code == 200, response.text
    version = response.json()["version_id"]
    assert client.post(f"/api/projects/{project}/drafts", json=payload).status_code == 409
    exported = client.get(f"/api/projects/{project}/export/{version}")
    mask = nib.Nifti1Image.from_bytes(gzip.decompress(exported.content))
    grid = client.get(f"/api/projects/{project}").json()["geometry"]
    assert mask.shape == tuple(grid["shape"])
    np.testing.assert_allclose(mask.affine, grid["affine_ras"], atol=1e-6)
    array = np.asanyarray(mask.dataobj)
    assert array.sum() == 256 and np.all(array[40:56, 40:56, 32] == 1)
    assert mask.header.get_xyzt_units()[0] == "mm"
    assert client.post("/api/auth/logout").status_code == 200
    login(client, password)
    reloaded = client.get(f"/api/projects/{project}").json()
    assert reloaded["document"]["contours"][0]["points"] == payload["document"]["contours"][0]["points"]
    assert reloaded["version_id"] == version
    payload["base_version_id"] = version
    payload["document"]["contours"][0]["label"] = 2
    new = client.post(f"/api/projects/{project}/drafts", json=payload).json()["version_id"]
    assert new != version
    assert client.get(f"/api/projects/{project}/export/{version}").content == exported.content
    assert len(client.get(f"/api/projects/{project}/versions").json()) == 2
    with app.state.factory() as db:
        assert db.scalar(select(AuditEvent).where(AuditEvent.action == "save_draft"))


def test_review_transitions_and_approved_snapshot_cannot_be_edited(setup_web):
    app, client, project, _, password = setup_web
    login(client, password)
    payload = draft(client, project)
    version = client.post(f"/api/projects/{project}/drafts", json=payload).json()["version_id"]
    assert client.post(f"/api/projects/{project}/reviews", json={"version_id": version, "decision": "approved"}).status_code == 403
    assert client.post(f"/api/projects/{project}/submit", json={"version_id": version}).status_code == 200
    assert client.post(f"/api/projects/{project}/drafts", json=dict(payload, base_version_id=version)).status_code == 409
    login(client, password, "surgeon")
    assert client.post(f"/api/projects/{project}/reviews", json={"version_id": version, "decision": "needs_revision"}).status_code == 200
    login(client, password)
    payload["base_version_id"] = version
    next_version = client.post(f"/api/projects/{project}/drafts", json=payload).json()["version_id"]
    client.post(f"/api/projects/{project}/submit", json={"version_id": next_version})
    login(client, password, "surgeon")
    decision = client.post(f"/api/projects/{project}/reviews", json={"version_id": next_version, "decision": "approved"})
    assert decision.status_code == 200
    reference_id = decision.json()["immutable_version_id"]
    assert reference_id != next_version
    login(client, password)
    assert client.post(f"/api/projects/{project}/drafts", json=dict(payload, base_version_id=next_version)).status_code == 409
    with app.state.factory() as db:
        assert db.scalar(select(Approval).where(Approval.version_id == reference_id))
        assert not db.scalar(select(Approval).where(Approval.version_id == next_version))


def test_invalid_plane_brush_and_changed_source_are_rejected(setup_web):
    app, client, project, _, password = setup_web
    login(client, password)
    payload = draft(client, project)
    payload["document"]["contours"][0]["slice"] = 31
    assert client.post(f"/api/projects/{project}/drafts", json=payload).status_code == 422
    payload = draft(client, project)
    payload["document"]["operations"].append({"kind": "brush", "indices": [-1], "values": [1]})
    assert client.post(f"/api/projects/{project}/drafts", json=payload).status_code == 422
    source = next((app.state.root / "raw").glob("*.nii.gz"))
    source.write_bytes(source.read_bytes()+b"tampered")
    assert client.get(f"/api/projects/{project}/volume").status_code == 409


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_world_polygon_and_native_x_fastest_patch_on_all_planes(axis):
    affine = np.diag([.7, 1.2, 2.3, 1.])
    affine[:3, 3] = [22, -13, 48]
    shape = [12, 13, 14]
    other = [i for i in range(3) if i != axis]
    points = np.zeros((4, 3))
    points[:, axis] = 5
    points[:, other] = [[2, 2], [6, 2], [6, 6], [2, 6]]
    world = nib.affines.apply_affine(LPS_TO_RAS @ affine, points)
    document = {"contours": [{"id": "a", "axis": axis, "slice": 5, "label": 3, "points": world.tolist(), "applied": True, "closed": True}],
                "operations": [{"kind": "contour", "contour_id": "a"}, {"kind": "brush", "indices": [0], "values": [7]}]}
    mask = rasterize(document, {"shape": shape, "affine_ras": affine.tolist()})
    assert np.count_nonzero(mask == 3) == 25
    assert mask[0, 0, 0] == 7


def test_backup_restores_saved_work_and_relative_storage_paths(setup_web, tmp_path):
    app, client, project, _, password = setup_web
    login(client, password)
    payload = draft(client, project)
    version = client.post(f"/api/projects/{project}/drafts", json=payload).json()["version_id"]
    destination = tmp_path / "restored"
    backup(app.state.root, destination)
    restored = create_app(destination)
    with TestClient(restored, base_url="http://127.0.0.1") as fresh:
        login(fresh, password)
        assert fresh.get(f"/api/projects/{project}").json()["version_id"] == version
        assert fresh.get(f"/api/projects/{project}/export/{version}").status_code == 200
    restored.state.engine.dispose()


def test_secure_cookie_configuration_and_server_session_expiration(tmp_path):
    app = create_app(tmp_path / "https", secure_cookie=True)
    create_user(app.state.factory, "resident", "Secure-test-password!")
    with TestClient(app, base_url="https://127.0.0.1") as client:
        response = login(client, "Secure-test-password!")
        assert "; secure" in response.headers["set-cookie"].lower()
        from heart3d.web.db import Session
        with app.state.factory.begin() as db:
            for session in db.scalars(select(Session)):
                session.expires_at = "2000-01-01T00:00:00+00:00"
        assert client.get("/api/auth/me").status_code == 401
    app.state.engine.dispose()


def test_rotated_affine_and_fractional_vertices_keep_original_grid():
    angle = .3
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    affine = np.eye(4)
    affine[:3, :3] = rotation @ np.diag([.8, 1.4, 2.1])
    affine[:3, 3] = [-54, 12, 87]
    voxels = [[2.2, 2.2, 4], [6.8, 2.2, 4], [6.8, 6.8, 4], [2.2, 6.8, 4]]
    world = nib.affines.apply_affine(LPS_TO_RAS @ affine, voxels)
    doc = {"contours": [{"id": "r", "axis": 2, "slice": 4, "label": 2,
                         "points": world.tolist(), "applied": True, "closed": True}],
           "operations": [{"kind": "contour", "contour_id": "r"}]}
    mask = rasterize(doc, {"shape": [10, 10, 10], "affine_ras": affine.tolist()})
    assert np.count_nonzero(mask) == 16
    assert np.all(mask[3:7, 3:7, 4] == 2)
