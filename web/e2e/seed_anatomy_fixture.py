"""Synthetic-only browser fixture with old and anatomy-linked slice versions."""
import json
import sys

import nibabel as nib
import numpy as np
from fastapi.testclient import TestClient

from heart3d.anatomy_bundle import assemble
from heart3d.storage import sha256_file
from heart3d.web.anatomy_predictions import append_anatomy_bundle
from heart3d.web.app import create_app, data_root
from heart3d.web.cli import create_user
from heart3d.web.db import AnnotationProject, Series
from heart3d.web.imaging import save_mask
from heart3d.web.predictions import append_slice_draft, import_predictions


root = data_root()
app = create_app(root)
password = sys.stdin.readline().strip()
username = "cloud-slice-resident"
create_user(app.state.factory, username, password)
ct = root / "fixtures" / "ct.nii.gz"
predictions = []
for variant in ("v1", "v2"):
    path = root / "fixtures" / (variant + ".nii.gz")
    predictions.append((path, {"variant": variant, "expert_approved": False, "exploratory_only": True,
        "geometry_verified": True, "ct_sha256": sha256_file(ct), "prediction_sha256": sha256_file(path),
        "checkpoint_sha256": "0" * 64, "synthetic_fixture": True}))
project = import_predictions(root, app.state.factory, username, ct, predictions, "ANATOMY-TEST-001", synthetic=True)
old_mask = root / "fixtures" / "single.nii.gz"
append_slice_draft(root, app.state.factory, username, project, old_mask, {
    "variant": "slice_draft", "scope": {"axis": 2, "index": 28, "complete_volume": False},
    "method": "Synthetic old one-plane fixture", "warning": "Synthetic only, not expert reviewed",
    "review_required_classes": ["LV"], "expert_approved": False, "exploratory_only": True,
    "geometry_verified": True, "ct_sha256": sha256_file(ct), "prediction_sha256": sha256_file(old_mask),
    "source_prediction_sha256": [sha256_file(p[0]) for p in predictions], "predicted_voxels": 400})
with app.state.factory() as db:
    series = db.get(Series, db.get(AnnotationProject, project).series_id)
    ct, grid, ct_digest = root / series.ct_path, series.geometry, series.ct_sha256
affine_lps = np.diag([-1., -1., 1., 1.]) @ np.asarray(grid["affine_ras"])
observations = []
for identifier, view, axis, index in (("ax-31", "AX", 2, 31), ("ax-32", "AX", 2, 32), ("cor-44", "COR", 1, 44)):
    anchor = [47.5, 47.5, 31.]
    anchor[axis] = float(index)
    normal = affine_lps[:3, axis] / np.linalg.norm(affine_lps[:3, axis])
    observations.append({"id": identifier, "kind": "observation", "structure": 1,
        "statement": "Synthetic cloud CT observation " + identifier, "features": ["synthetic feature"],
        "confidence": "low", "uncertainty_reason": "Synthetic, no medical interpretation.", "evidence_refs": [],
        "source": "cloud_model", "locator": {"view": view, "anchor_ijk": anchor,
            "focal_point_lps_mm": nib.affines.apply_affine(affine_lps, anchor).tolist(), "normal_lps": normal.tolist(),
            "native_axis": axis, "native_index": index, "window_width": 400., "window_level": 40., "overlay_visible": False}})
document = {"schema_version": 1, "target": {"axis": 2, "index": 31}, "observations": observations,
    "tracks": [{"id": "track-lv", "structure": 1, "observation_ids": [row["id"] for row in observations],
        "relations": [{"from_id": "ax-31", "to_id": "ax-32", "kind": "continuation", "gap_reason": ""}]}],
    "decisions": [{"label": label, "state": "delineated" if label == 1 else "not_assessable",
        "supporting_observation_ids": [row["id"] for row in observations] if label == 1 else [],
        "counterevidence_ids": [], "track_ids": ["track-lv"] if label == 1 else [],
        "boundary_rule": "Synthetic boundary." if label == 1 else "",
        "rationale": "Synthetic multiview continuation." if label == 1 else "",
        "unresolved_questions": ["Synthetic, not clinical."], "confidence": "low"} for label in range(1, 8)]}
folder = root / "fixtures" / "anatomy"
folder.mkdir()
with TestClient(app, base_url="http://127.0.0.1") as client:
    auth = client.post("/api/auth/login", json={"username": username, "password": password})
    client.headers["X-CSRF-Token"] = auth.json()["csrf"]
    saved = client.post(f"/api/projects/{project}/anatomy-review", json={"base_version_id": None,
        "ct_sha256": ct_digest, "document": document})
    saved.raise_for_status()
    exported = client.get(f"/api/projects/{project}/anatomy-review/export")
    exported.raise_for_status()
    notebook = folder / "notebook.json"
    notebook.write_bytes(exported.content)
    admission = folder / "admission.json"
    admission.write_text(json.dumps({**{key: True for key in ("user_attested_permission_and_basis", "local_only",
        "geometry_verified", "independent_reader_verified", "metadata_deidentified", "pixel_identifier_screening_passed")},
        "ct_sha256": ct_digest}), encoding="utf-8")
    data = np.zeros(grid["shape"], np.uint8)
    data[35:55, 35:55, 31] = 1
    mask = folder / "mask.nii.gz"
    save_mask(data, ct, mask)
    bundle = folder / "bundle"
    assemble(ct, admission, notebook, mask, bundle)
    append_anatomy_bundle(root, app.state.factory, username, project, bundle)
app.state.engine.dispose()
