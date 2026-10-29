"""Synthetic-only checks for binding an existing mask to CT notebook evidence."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import nibabel as nib
import numpy as np
import pytest

from heart3d.anatomy_bundle import assemble
from heart3d.storage import sha256_file
from heart3d.web.anatomy_review import canonical_json, readiness, validate_document
from heart3d.web.imaging import geometry


def digest(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def write_export(path, payload):
    payload["document_sha256"] = digest(payload["document"])
    payload["export_sha256"] = digest({key: value for key, value in payload.items() if key != "export_sha256"})
    path.write_bytes(canonical_json(payload))


@pytest.fixture
def bundle_inputs(tmp_path):
    affine = np.diag([-.7, -1.1, 2., 1.])
    affine[:3, 3] = [20., 30., -40.]
    image = nib.Nifti1Image(np.arange(6 * 8 * 10, dtype=np.float32).reshape(6, 8, 10), affine)
    image.header.set_xyzt_units("mm")
    image.set_qform(affine, 1)
    image.set_sform(affine, 1)
    ct = tmp_path / "synthetic-ct.nii.gz"
    nib.save(image, ct)
    admission = {key: True for key in ("user_attested_permission_and_basis", "local_only", "geometry_verified",
                 "independent_reader_verified", "metadata_deidentified", "pixel_identifier_screening_passed")}
    admission["ct_sha256"] = sha256_file(ct)
    ap = tmp_path / "synthetic-admission.json"
    ap.write_text(json.dumps(admission), encoding="utf-8")
    lps = np.diag([-1., -1., 1., 1.]) @ affine
    observations = []
    for identifier, view, axis, z in (("obs-ax", "AX", 2, 4.), ("obs-ax-next", "AX", 2, 5.),
                                      ("obs-cor", "COR", 1, 4.)):
        anchor = [2., 3., z]
        normal = lps[:3, axis] / np.linalg.norm(lps[:3, axis])
        observations.append({"id": identifier, "kind": "observation", "structure": 1,
            "statement": "A synthetic region has a visible edge.", "features": ["synthetic edge"],
            "confidence": "low", "uncertainty_reason": "Synthetic only.", "evidence_refs": [], "source": "human",
            "locator": {"view": view, "anchor_ijk": anchor,
                "focal_point_lps_mm": nib.affines.apply_affine(lps, anchor).tolist(), "normal_lps": normal.tolist(),
                "native_axis": axis, "native_index": int(anchor[axis]), "window_width": 400.,
                "window_level": 40., "overlay_visible": False}})
    document = {"schema_version": 1, "target": {"axis": 2, "index": 4}, "observations": observations,
        "tracks": [{"id": "track-1", "structure": 1, "observation_ids": ["obs-ax", "obs-ax-next", "obs-cor"],
                    "relations": [{"from_id": "obs-ax", "to_id": "obs-ax-next", "kind": "continuation", "gap_reason": ""}]}],
        "decisions": [{"label": label, "state": "delineated" if label == 1 else "not_visible",
            "supporting_observation_ids": ["obs-ax", "obs-ax-next", "obs-cor"] if label == 1 else [], "counterevidence_ids": [],
            "track_ids": ["track-1"] if label == 1 else [], "boundary_rule": "Synthetic edge." if label == 1 else "",
            "rationale": "Synthetic multiview observations." if label == 1 else "", "unresolved_questions": [],
            "confidence": "low"} for label in range(1, 8)]}
    document = validate_document(document, geometry(image))
    payload = {"export_schema_version": 1, "kind": "anatomy_review_notebook", "project_id": "synthetic-project",
        "version_id": "synthetic-notebook-version", "number": 1, "author_id": "synthetic-author",
        "created_at": "2026-10-25T00:00:00+00:00", "ct_sha256": admission["ct_sha256"],
        "document": document, "readiness": readiness(document),
        "provenance": {"coordinate_system": "LPS", "unit": "mm", "expert_validated": False}}
    notebook = tmp_path / "synthetic-notebook.json"
    write_export(notebook, payload)
    data = np.zeros(image.shape, dtype=np.uint8)
    data[1:4, 2:5, 4] = 1
    mask = tmp_path / "synthetic-mask.nii.gz"
    mask_image = nib.Nifti1Image(data, affine, image.header.copy())
    mask_image.set_data_dtype(np.uint8)
    nib.save(mask_image, mask)
    return {"ct": ct, "admission": ap, "notebook": notebook, "mask": mask,
            "out": tmp_path / "candidate"}, payload


def change_mask(inputs, mutate):
    current = nib.load(inputs["mask"])
    data = np.asanyarray(current.dataobj).copy()
    updated = mutate(current, data)
    nib.save(updated if updated is not None else nib.Nifti1Image(data, current.affine, current.header.copy()), inputs["mask"])


def test_valid_binding_copies_existing_files_and_preserves_missing_labels(bundle_inputs):
    inputs, payload = bundle_inputs
    before = {name: path.read_bytes() for name, path in inputs.items() if name != "out"}
    result = assemble(**inputs)
    assert result["source"] == "anatomy-linked research candidate"
    assert result["scope"] == {"axis": 2, "index": 4, "complete_volume": False}
    assert result["positive_labels"] == [1]
    assert result["readiness"]["ready"] is True
    assert not result["expert_approved"] and not result["medical_correctness_verified"]
    assert not result["pixels_created"] and not result["inference_performed"] and not result["image_cloud_transfer"]
    assert {path.name for path in inputs["out"].iterdir()} == {"manifest.json", "mask.nii.gz", "notebook.json"}
    assert (inputs["out"] / "mask.nii.gz").read_bytes() == before["mask"]
    assert (inputs["out"] / "notebook.json").read_bytes() == before["notebook"]
    assert result["notebook"]["export_sha256"] == payload["export_sha256"]
    assert result["notebook"]["document_sha256"] == payload["document_sha256"]
    assert result["mask"]["sha256"] == sha256_file(inputs["mask"])
    assert result["ct_sha256"] == sha256_file(inputs["ct"])
    assert all(len(value) == 64 for value in result["code_sha256"].values())
    manifest = json.loads((inputs["out"] / "manifest.json").read_text())
    claimed = manifest.pop("manifest_sha256")
    assert digest(manifest) == claimed
    assert all(path.read_bytes() == before[name] for name, path in inputs.items() if name != "out")
    with pytest.raises(ValueError, match="new output"):
        assemble(**inputs)
    assert (inputs["out"] / "mask.nii.gz").read_bytes() == before["mask"]


@pytest.mark.parametrize("kind", ["tampered_export", "stale_document", "stale_readiness", "not_ready", "wrong_ct", "wrong_kind", "unsaved", "wrong_axis"])
def test_notebook_integrity_and_readiness_rejected_before_output(bundle_inputs, kind):
    inputs, payload = bundle_inputs
    if kind == "tampered_export":
        payload["document"]["observations"][0]["statement"] = "Changed after export."
        inputs["notebook"].write_bytes(canonical_json(payload))
    elif kind == "stale_document":
        payload["document_sha256"] = "0" * 64
        payload["export_sha256"] = digest({key: value for key, value in payload.items() if key != "export_sha256"})
        inputs["notebook"].write_bytes(canonical_json(payload))
    else:
        if kind == "stale_readiness":
            payload["readiness"]["expert_validated"] = True
        elif kind == "not_ready":
            payload["document"]["decisions"][0]["boundary_rule"] = ""
            payload["readiness"] = readiness(payload["document"])
        elif kind == "wrong_ct":
            payload["ct_sha256"] = "0" * 64
        elif kind == "wrong_kind":
            payload["kind"] = "prediction"
        elif kind == "unsaved":
            payload["version_id"] = ""
        else:
            payload["document"]["target"]["axis"] = 1
        write_export(inputs["notebook"], payload)
    with pytest.raises(ValueError):
        assemble(**inputs)
    assert not inputs["out"].exists()


@pytest.mark.parametrize("kind", ["outside_plane", "unsupported_label", "empty", "dtype", "labels", "affine", "qform", "units", "text", "extension", "scaling"])
def test_mask_guards_reject_before_output(bundle_inputs, kind):
    inputs, _ = bundle_inputs
    def mutate(current, data):
        if kind == "outside_plane":
            data[1, 2, 5] = 1
        elif kind == "unsupported_label":
            data[1, 2, 4] = 2
        elif kind == "empty":
            data[:] = 0
        elif kind == "labels":
            data[1, 2, 4] = 8
        updated = nib.Nifti1Image(data, current.affine, current.header.copy())
        if kind == "dtype":
            updated.set_data_dtype(np.uint16)
        elif kind == "affine":
            affine = current.affine.copy()
            affine[0, 3] += 1
            updated = nib.Nifti1Image(data, affine, current.header.copy())
        elif kind == "qform":
            updated.set_qform(current.affine, 2)
        elif kind == "units":
            updated.header.set_xyzt_units("meter")
        elif kind == "text":
            updated.header["descrip"] = b"Unapproved free text"
        elif kind == "extension":
            updated.header.extensions.append(nib.nifti1.Nifti1Extension(6, b"free text"))
        elif kind == "scaling":
            updated.header.set_slope_inter(2, 0)
        return updated
    change_mask(inputs, mutate)
    with pytest.raises(ValueError):
        assemble(**inputs)
    assert not inputs["out"].exists()


def test_missing_tracked_multiview_evidence_rejected(bundle_inputs):
    inputs, payload = bundle_inputs
    payload["document"]["tracks"][0]["relations"] = []
    payload["readiness"] = readiness(payload["document"])
    write_export(inputs["notebook"], payload)
    with pytest.raises(ValueError, match="not ready"):
        assemble(**inputs)
    assert not inputs["out"].exists()


def test_cross_plane_relation_without_adjacent_native_slices_is_not_ready(bundle_inputs):
    inputs, payload = bundle_inputs
    document = payload["document"]
    document["observations"] = [row for row in document["observations"] if row["id"] != "obs-ax-next"]
    document["decisions"][0]["supporting_observation_ids"] = ["obs-ax", "obs-cor"]
    document["tracks"][0]["observation_ids"] = ["obs-ax", "obs-cor"]
    document["tracks"][0]["relations"] = [{"from_id": "obs-ax", "to_id": "obs-cor", "kind": "continuation", "gap_reason": ""}]
    payload["readiness"] = readiness(document)
    assert payload["readiness"]["ready"] is False
    write_export(inputs["notebook"], payload)
    with pytest.raises(ValueError, match="not ready"):
        assemble(**inputs)
    assert not inputs["out"].exists()


def test_missing_admission_does_not_create_bundle(bundle_inputs):
    inputs, _ = bundle_inputs
    record = json.loads(inputs["admission"].read_text())
    record["local_only"] = False
    inputs["admission"].write_text(json.dumps(record))
    with pytest.raises(ValueError, match="admitted"):
        assemble(**inputs)
    assert not inputs["out"].exists()


def test_changed_source_ct_rejected_before_output(bundle_inputs):
    inputs, _ = bundle_inputs
    image = nib.load(inputs["ct"])
    nib.save(nib.Nifti1Image(image.get_fdata().astype(np.float32) + 1, image.affine, image.header.copy()), inputs["ct"])
    with pytest.raises(ValueError, match="admitted"):
        assemble(**inputs)
    assert not inputs["out"].exists()


def test_uncompressed_mask_keeps_format_and_exact_bytes(bundle_inputs):
    inputs, _ = bundle_inputs
    original = nib.load(inputs["mask"])
    uncompressed = inputs["mask"].with_name("synthetic-mask.nii")
    nib.save(original, uncompressed)
    inputs["mask"] = uncompressed
    before = uncompressed.read_bytes()
    manifest = assemble(**inputs)
    assert manifest["mask"]["file"] == "mask.nii"
    assert (inputs["out"] / "mask.nii").read_bytes() == before
    assert nib.load(inputs["out"] / "mask.nii").shape == original.shape


def test_cli_assemble_synthetic_bundle(bundle_inputs):
    inputs, _ = bundle_inputs
    command = [sys.executable, "-m", "heart3d.anatomy_bundle", "assemble"]
    for name, path in inputs.items():
        command += ["--" + name, str(path)]
    result = subprocess.run(command, cwd=Path(__file__).parents[1], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["bundle_created"] is True
