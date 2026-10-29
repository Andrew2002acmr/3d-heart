"""Synthetic-only tests for CT observation notebooks (no clinical input)."""
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import threading

import nibabel as nib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import create_engine, delete, inspect, select, update
from sqlalchemy.exc import IntegrityError

from heart3d.dicom.ct import LPS_TO_RAS
from heart3d.web.anatomy_review import AnatomyDocument, canonical_json, readiness, validate_document
from heart3d.web.app import create_app
from heart3d.web.cli import create_user, seed_project
from heart3d.web.db import AnatomyNotebookVersion, AnnotationProject, AnnotationVersion, Assignment, Base, Role, Series, User, database


def synthetic_grid():
    angle = .27
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    affine = np.eye(4)
    affine[:3, :3] = rotation @ np.diag([.8, 1.4, 2.1])
    affine[:3, 3] = [-54, 12, 87]
    return {"shape": [12, 14, 16], "affine_ras": affine.tolist()}


def observation(grid, identifier="obs-ax", view="AX", axis=2, source="human", index=None):
    affine = LPS_TO_RAS @ np.asarray(grid["affine_ras"])
    anchor = [4., 5., 6.]
    if index is not None:
        anchor[axis] = float(index)
    normal = affine[:3, axis] / np.linalg.norm(affine[:3, axis])
    return {"id": identifier, "kind": "observation", "structure": 1,
            "statement": "Synthetic CT region has a visible edge.", "features": ["visible edge"],
            "confidence": "low", "uncertainty_reason": "Synthetic test observation only.",
            "evidence_refs": [], "source": source,
            "locator": {"view": view, "anchor_ijk": anchor,
                        "focal_point_lps_mm": nib.affines.apply_affine(affine, anchor).tolist(),
                        "normal_lps": normal.tolist(), "native_axis": axis, "native_index": int(anchor[axis]),
                        "window_width": 400., "window_level": 40., "overlay_visible": False}}


def document(grid=None, source="human"):
    grid = grid or synthetic_grid()
    return {"schema_version": 1, "target": {"axis": 2, "index": 6},
            "observations": [observation(grid, source=source), observation(grid, "obs-ax-next", source=source, index=7),
                             observation(grid, "obs-cor", "COR", 1, source)],
            "tracks": [{"id": "track-1", "structure": 1, "observation_ids": ["obs-ax", "obs-ax-next", "obs-cor"],
                        "relations": [{"from_id": "obs-ax", "to_id": "obs-ax-next", "kind": "continuation", "gap_reason": ""}]}],
            "decisions": [{"label": label, "state": "delineated" if label == 1 else "not_assessable",
                           "supporting_observation_ids": ["obs-ax", "obs-ax-next", "obs-cor"] if label == 1 else [],
                           "counterevidence_ids": [], "track_ids": ["track-1"] if label == 1 else [],
                           "boundary_rule": "Follow the described CT edge." if label == 1 else "",
                           "rationale": "Synthetic observations only." if label == 1 else "",
                           "unresolved_questions": [], "confidence": "low"} for label in range(1, 8)]}


@pytest.fixture
def notebook_web(tmp_path):
    app = create_app(tmp_path / "synthetic-notebook")
    password = "Synthetic-notebook-password!"
    resident = create_user(app.state.factory, "resident", password)
    outsider = create_user(app.state.factory, "outsider", password)
    surgeon = create_user(app.state.factory, "surgeon", password, "surgeon")
    create_user(app.state.factory, "admin", password, "admin")
    project = seed_project(app.state.root, app.state.factory, resident)
    foreign = seed_project(app.state.root, app.state.factory, outsider, "002")
    with app.state.factory.begin() as db:
        db.add(Assignment(project_id=project, user_id=surgeon))
    with TestClient(app, base_url="http://127.0.0.1") as client:
        yield app, client, project, foreign, password
    app.state.engine.dispose()


def login(client, password, username="resident"):
    result = client.post("/api/auth/login", json={"username": username, "password": password})
    assert result.status_code == 200
    client.headers["X-CSRF-Token"] = result.json()["csrf"]


def payload(client, project, doc=None):
    current = client.get(f"/api/projects/{project}/anatomy-review").json()
    return {"base_version_id": current["version_id"], "ct_sha256": current["ct_sha256"],
            "document": doc if doc is not None else current["document"]}


def test_empty_get_auth_assignment_csrf_and_roles(notebook_web):
    _, client, project, foreign, password = notebook_web
    route = f"/api/projects/{project}/anatomy-review"
    assert client.get(route).status_code == 401
    assert client.get(route + "/export").status_code == 401
    login(client, password)
    current = client.get(route)
    assert current.status_code == 200
    body = current.json()
    assert body["version_id"] is None and body["number"] == 0
    assert body["document"] == {"schema_version": 1, "target": None, "observations": [], "tracks": [], "decisions": []}
    assert body["readiness"]["ready"] is False
    assert body["readiness"]["expert_validated"] is False
    assert all(isinstance(issue, str) for issue in body["readiness"]["issues"])
    assert client.get(route + "/export").status_code == 404
    for suffix in ("", "/export?version_id=unknown"):
        assert client.get(f"/api/projects/{foreign}/anatomy-review{suffix}").status_code == 404
    saved_body = payload(client, project)
    assert client.post(f"/api/projects/{foreign}/anatomy-review", json=saved_body).status_code == 404
    csrf = client.headers.pop("X-CSRF-Token")
    assert client.post(route, json=saved_body).status_code == 403
    client.headers["X-CSRF-Token"] = csrf
    assert client.post(route, json=saved_body, headers={"Origin": "https://untrusted.invalid"}).status_code == 403
    login(client, password, "surgeon")
    assert client.get(route).status_code == 200
    assert client.post(route, json=saved_body).status_code == 403
    login(client, password, "admin")
    assert client.post(route, json=saved_body).status_code == 200


def test_append_only_conflict_export_hashes_and_no_mask_mutation(notebook_web, monkeypatch):
    app, client, project, _, password = notebook_web
    login(client, password)
    route = f"/api/projects/{project}/anatomy-review"
    before = client.get(f"/api/projects/{project}").json()
    mask_before = client.get(f"/api/projects/{project}/mask").content
    with app.state.factory() as db:
        series = db.get(Series, db.get(AnnotationProject, project).series_id)
        grid = series.geometry
    def forbidden(*args, **kwargs):
        raise AssertionError("Notebook endpoint must not rasterize or write masks")
    monkeypatch.setattr("heart3d.web.app.rasterize", forbidden)
    monkeypatch.setattr("heart3d.web.app.save_mask", forbidden)
    body = payload(client, project, document(grid))
    invalid = copy.deepcopy(body)
    invalid["document"]["observations"][0]["evidence_refs"] = ["prediction:forbidden"]
    assert client.post(route, json=invalid).status_code == 422
    assert client.get(route).json()["number"] == 0
    saved = client.post(route, json=body)
    assert saved.status_code == 200, saved.text
    first = saved.json()
    assert first["number"] == 1 and first["readiness"]["ready"] is True
    assert client.post(route, json=body).status_code == 409
    original_export = client.get(route + "/export", params={"version_id": first["version_id"]})
    assert original_export.status_code == 200
    exported = original_export.json()
    export_hash = exported.pop("export_sha256")
    assert hashlib.sha256(canonical_json(exported)).hexdigest() == export_hash
    assert hashlib.sha256(canonical_json(exported["document"])).hexdigest() == exported["document_sha256"]
    assert exported["ct_sha256"] == body["ct_sha256"]
    assert exported["author_id"] and exported["created_at"]
    assert exported["provenance"]["coordinate_system"] == "LPS"
    assert exported["provenance"]["expert_validated"] is False
    body["base_version_id"] = first["version_id"]
    body["document"]["observations"][0]["statement"] = "Revised synthetic edge description."
    second = client.post(route, json=body)
    assert second.status_code == 200, second.text
    assert second.json()["number"] == 2
    assert second.json()["version_id"] != first["version_id"]
    assert client.get(route + "/export", params={"version_id": first["version_id"]}).content == original_export.content
    assert client.get(route).json()["document"] == second.json()["document"]
    after = client.get(f"/api/projects/{project}").json()
    assert after == before
    assert client.get(f"/api/projects/{project}/mask").content == mask_before
    assert not (app.state.root / "annotations").exists()
    assert not (app.state.root / "derived").exists()
    with app.state.factory() as db:
        assert len(list(db.scalars(select(AnatomyNotebookVersion)))) == 2
        assert not list(db.scalars(select(AnnotationVersion)))
    # Database guards enforce append-only versions even outside the HTTP API.
    for mutation in (update(AnatomyNotebookVersion).values(document={}), delete(AnatomyNotebookVersion)):
        with pytest.raises(IntegrityError, match="immutable"):
            with app.state.factory.begin() as db:
                db.execute(mutation)


def test_existing_mask_annotation_head_and_revision_unchanged(notebook_web):
    app, client, project, _, password = notebook_web
    login(client, password)
    view = client.get(f"/api/projects/{project}").json()
    saved_mask = client.post(f"/api/projects/{project}/drafts", json={"base_version_id": None,
        "ct_sha256": view["ct_sha256"], "document": {"contours": [], "operations": [
            {"kind": "brush", "indices": [0, 1, 2], "values": [1, 2, 3]}]}})
    assert saved_mask.status_code == 200
    before = client.get(f"/api/projects/{project}").json()
    version_id = before["version_id"]
    export_before = client.get(f"/api/projects/{project}/export/{version_id}").content
    route = f"/api/projects/{project}/anatomy-review"
    assert client.post(route, json=payload(client, project)).status_code == 200
    assert client.get(f"/api/projects/{project}").json() == before
    assert client.get(f"/api/projects/{project}/export/{version_id}").content == export_before
    with app.state.factory() as db:
        assert len(list(db.scalars(select(AnnotationVersion)))) == 1


def test_concurrent_initial_save_has_exactly_one_winner(notebook_web):
    app, client, project, _, password = notebook_web
    login(client, password)
    route = f"/api/projects/{project}/anatomy-review"
    body = payload(client, project)
    barrier = threading.Barrier(2)
    cookies, headers = dict(client.cookies), {"X-CSRF-Token": client.headers["X-CSRF-Token"]}
    def save():
        with TestClient(app, base_url="http://127.0.0.1", cookies=cookies, headers=headers) as concurrent:
            barrier.wait(timeout=10)
            return concurrent.post(route, json=body).status_code
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(lambda _: save(), range(2))) == [200, 409]
    assert client.get(route).json()["number"] == 1


@pytest.mark.parametrize("status", ["submitted", "approved"])
def test_submitted_and_approved_notebooks_are_read_only(notebook_web, status):
    app, client, project, _, password = notebook_web
    login(client, password)
    route = f"/api/projects/{project}/anatomy-review"
    body = payload(client, project)
    with app.state.factory.begin() as db:
        db.get(AnnotationProject, project).status = status
    assert client.get(route).status_code == 200
    assert client.post(route, json=body).status_code == 409


def test_source_hash_checks_and_cross_project_version_export(notebook_web):
    app, client, project, foreign, password = notebook_web
    login(client, password)
    route = f"/api/projects/{project}/anatomy-review"
    body = payload(client, project)
    bad = dict(body, ct_sha256="0" * 64)
    assert client.post(route, json=bad).status_code == 409
    first = client.post(route, json=body).json()["version_id"]
    login(client, password, "admin")
    assert client.get(f"/api/projects/{foreign}/anatomy-review/export", params={"version_id": first}).status_code == 404
    with app.state.factory() as db:
        series = db.get(Series, db.get(AnnotationProject, project).series_id)
        source = app.state.root / series.ct_path
    source.write_bytes(source.read_bytes() + b"synthetic-integrity-test")
    assert client.get(route).status_code == 409
    assert client.get(route + "/export", params={"version_id": first}).status_code == 409
    assert client.post(route, json=dict(body, base_version_id=first)).status_code == 409


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_rotated_anisotropic_locator_roundtrip_and_unoriented_plane(axis):
    grid = synthetic_grid()
    doc = document(grid)
    doc["observations"] = [observation(grid, axis=axis)]
    doc["tracks"], doc["decisions"] = [], []
    validate_document(doc, grid)
    doc["observations"][0]["locator"]["normal_lps"] = [-x for x in doc["observations"][0]["locator"]["normal_lps"]]
    validate_document(doc, grid)
    doc["observations"][0]["locator"]["anchor_ijk"][(axis + 1) % 3] += .5
    affine = LPS_TO_RAS @ np.asarray(grid["affine_ras"])
    doc["observations"][0]["locator"]["focal_point_lps_mm"] = nib.affines.apply_affine(affine, doc["observations"][0]["locator"]["anchor_ijk"]).tolist()
    validate_document(doc, grid)


@pytest.mark.parametrize("kind", ["outside", "mismatched_point", "nonunit", "wrong_normal", "wrong_index", "missing_index", "nan", "infinity", "target"])
def test_invalid_locator_geometry_rejected(kind):
    grid, doc = synthetic_grid(), document()
    loc = doc["observations"][0]["locator"]
    if kind == "outside":
        loc["anchor_ijk"][0] = -1.
    elif kind == "mismatched_point":
        loc["focal_point_lps_mm"][0] += .051
    elif kind == "nonunit":
        loc["normal_lps"] = [0., 0., 2.]
    elif kind == "wrong_normal":
        loc["normal_lps"] = [1., 0., 0.]
    elif kind == "wrong_index":
        loc["native_index"] += 1
    elif kind == "missing_index":
        del loc["native_index"]
    elif kind == "nan":
        loc["anchor_ijk"][0] = float("nan")
    elif kind == "infinity":
        loc["window_level"] = float("inf")
    else:
        doc["target"]["index"] = 16
    with pytest.raises((ValueError, ValidationError)):
        validate_document(doc, grid)


@pytest.mark.parametrize("kind", ["duplicate", "dangling", "hypothesis", "prediction", "mask", "probability", "relation", "label", "extra", "overlay", "strict", "unresolved"])
def test_invalid_references_and_ai_or_mask_evidence_rejected(kind):
    grid, doc = synthetic_grid(), document()
    if kind == "duplicate":
        doc["observations"].append(copy.deepcopy(doc["observations"][0]))
    elif kind == "dangling":
        doc["tracks"][0]["observation_ids"].append("other-study-observation")
    elif kind == "hypothesis":
        doc["observations"][0]["kind"] = "hypothesis"
    elif kind in ("prediction", "mask", "probability"):
        doc["observations"][0]["evidence_refs"] = [kind + ":some-id"]
    elif kind == "relation":
        doc["tracks"][0]["relations"][0]["to_id"] = "other-study-observation"
    elif kind == "label":
        doc["decisions"][0]["track_ids"] = ["missing-track"]
    elif kind == "extra":
        doc["base_prediction_id"] = "prediction-id"
    elif kind == "overlay":
        doc["observations"][0]["locator"]["overlay_visible"] = True
    elif kind == "strict":
        doc["observations"][0]["structure"] = "1"
    else:
        doc["tracks"][0]["relations"][0]["kind"] = "unresolved"
    with pytest.raises((ValueError, ValidationError)):
        validate_document(doc, grid)


def test_incomplete_saves_hypotheses_and_multiview_workflow_completeness():
    grid = synthetic_grid()
    empty = validate_document(AnatomyDocument(), grid)
    assert empty["target"] is None
    assert readiness(empty)["ready"] is False
    doc = document(grid)
    assert readiness(validate_document(doc, grid))["ready"] is True
    for change in ("one_view", "blank_observation", "no_continuation", "no_rule", "no_rationale", "unassessed", "no_target"):
        incomplete = copy.deepcopy(doc)
        if change == "one_view":
            for item in incomplete["observations"]:
                item["locator"]["view"] = "AX"
        elif change == "blank_observation":
            incomplete["observations"][1]["statement"] = " "
        elif change == "no_continuation":
            incomplete["tracks"][0]["relations"][0]["kind"] = "branch"
        elif change == "no_rule":
            incomplete["decisions"][0]["boundary_rule"] = " "
        elif change == "no_rationale":
            incomplete["decisions"][0]["rationale"] = ""
        elif change == "unassessed":
            incomplete["decisions"][6]["state"] = "unassessed"
        else:
            incomplete["target"] = None
        assert readiness(validate_document(incomplete, grid))["ready"] is False
    hypothesis = observation(grid, "hypothesis-1")
    hypothesis.update(kind="hypothesis", evidence_refs=["obs-ax"])
    doc["observations"].append(hypothesis)
    validate_document(doc, grid)
    doc["decisions"][0]["supporting_observation_ids"] = ["hypothesis-1"]
    with pytest.raises(ValueError, match="CT observations"):
        validate_document(doc, grid)


@pytest.mark.parametrize("case", ["cross_view_only", "same_plane", "skipped_plane", "no_native_indices", "different_native_axes"])
def test_delineation_requires_sequential_adjacent_native_plane_tracing(case):
    grid, doc = synthetic_grid(), document()
    if case == "cross_view_only":
        doc["tracks"][0]["relations"][0]["to_id"] = "obs-cor"
    elif case == "same_plane":
        doc["observations"][1] = observation(grid, "obs-ax-next", index=6)
    elif case == "skipped_plane":
        doc["observations"][1] = observation(grid, "obs-ax-next", index=8)
    elif case == "no_native_indices":
        doc["observations"][1]["locator"].pop("native_axis")
        doc["observations"][1]["locator"].pop("native_index")
    else:
        doc["observations"][1] = observation(grid, "obs-ax-next", axis=1, index=7)
    report = readiness(validate_document(doc, grid))
    assert report["ready"] is False
    assert any("two neighbouring native CT planes" in issue for issue in report["issues"])


def test_different_view_names_on_parallel_planes_do_not_complete_multiview_support():
    grid, doc = synthetic_grid(), document()
    # Keep the independent COR view name while duplicating the actual AX plane.
    doc["observations"][2] = observation(grid, "obs-cor", view="COR", axis=2)
    report = readiness(validate_document(doc, grid))
    assert report["ready"] is False
    assert any("physically nonparallel CT planes" in issue for issue in report["issues"])
    assert readiness(validate_document(document(grid), grid))["ready"] is True


@pytest.mark.parametrize("source", ["local_model", "cloud_model"])
def test_model_proposals_preserve_source_and_require_external_validation(source):
    grid = synthetic_grid()
    doc = document(grid, source=source)
    proposed = observation(grid, "model-hypothesis", source=source)
    proposed.update(kind="hypothesis", evidence_refs=["obs-ax", "obs-cor"])
    doc["observations"].append(proposed)
    valid = validate_document(doc, grid)
    report = readiness(valid)
    assert report["ready"] is True
    assert report["external_validation_required"] is True
    assert report["ready_factual"] is False and report["expert_validated"] is False
    assert all(o["source"] == source for o in valid["observations"])
    doc["observations"][-1]["source"] = "human"
    with pytest.raises(ValueError, match="Human hypotheses"):
        validate_document(doc, grid)


def test_create_all_upgrade_preserves_original_schema_and_rows(tmp_path):
    root = tmp_path / "synthetic-upgrade"
    root.mkdir()
    original_tables = [table for table in Base.metadata.sorted_tables if table.name != "anatomy_notebook_versions"]
    original_engine = create_engine("sqlite:///" + str(root / "annotation.sqlite3"))
    Base.metadata.create_all(original_engine, tables=original_tables)
    with original_engine.begin() as connection:
        connection.execute(Role.__table__.insert(), [{"name": name} for name in ("resident", "admin", "surgeon")])
        connection.execute(User.__table__.insert(), {"id": "synthetic-user", "username": "synthetic-user",
            "password_hash": "synthetic-unused-hash", "role": "resident", "active": True})
    with original_engine.connect() as connection:
        before_rows = {table.name: list(connection.execute(select(table)).mappings()) for table in original_tables}
    def column_snapshot(engine):
        return {table.name: [(c["name"], str(c["type"]), c["nullable"], c["default"], c["primary_key"])
                             for c in inspect(engine).get_columns(table.name)] for table in original_tables}
    before_columns = column_snapshot(original_engine)
    original_engine.dispose()
    upgraded, _ = database(root)
    try:
        assert set(inspect(upgraded).get_table_names()) == set(before_rows) | {"anatomy_notebook_versions"}
        with upgraded.connect() as connection:
            assert {table.name: list(connection.execute(select(table)).mappings()) for table in original_tables} == before_rows
        assert column_snapshot(upgraded) == before_columns
    finally:
        upgraded.dispose()
