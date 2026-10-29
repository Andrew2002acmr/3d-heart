"""Bind an existing single-slice mask to an immutable CT evidence notebook.

This command creates no pixels, performs no inference, and has no network client.
Notebook completeness is a documentation check, never medical correctness.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import nibabel as nib
import numpy as np

from heart3d.anatomy_harness import load_admitted_ct
from heart3d.storage import sha256_file
from heart3d.web.anatomy_review import canonical_json, readiness, validate_document
from heart3d.web.imaging import SCHEMA, geometry


def _digest(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate notebook JSON key")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError("Nonfinite notebook JSON value")


def _load_notebook(path, ct_digest, grid):
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"),
                         object_pairs_hook=_unique_object, parse_constant=_nonfinite)
    if not isinstance(payload, dict):
        raise ValueError("Notebook export must be an object")
    claimed_export = payload.get("export_sha256")
    content = {key: value for key, value in payload.items() if key != "export_sha256"}
    if not isinstance(claimed_export, str) or _digest(content) != claimed_export:
        raise ValueError("Notebook export_sha256 integrity check failed")
    if payload.get("kind") != "anatomy_review_notebook" or type(payload.get("export_schema_version")) is not int or payload["export_schema_version"] != 1:
        raise ValueError("Expected anatomy_review_notebook export schema 1")
    for field in ("project_id", "version_id", "author_id", "created_at"):
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip() or len(value) > 200:
            raise ValueError(f"Notebook requires a saved {field} string")
    if type(payload.get("number")) is not int or payload["number"] < 1:
        raise ValueError("Notebook requires a saved version number")
    if payload.get("ct_sha256") != ct_digest:
        raise ValueError("Notebook CT hash differs from admitted source CT")
    if "document" not in payload or payload.get("document_sha256") != _digest(payload["document"]):
        raise ValueError("Notebook document_sha256 integrity check failed")
    document = validate_document(payload["document"], grid)
    report = readiness(document)
    if payload.get("readiness") != report:
        raise ValueError("Notebook readiness snapshot is stale; export the validated notebook again")
    if not report["ready"]:
        raise ValueError("Notebook is not ready: " + "; ".join(report["issues"]))
    target = document.get("target")
    if not target or target["axis"] != 2:
        raise ValueError("A single original-grid axial target is required")
    return payload, document, report


def _check_mask(path, source, target_index):
    mask = nib.load(path)
    if mask.header.get_data_dtype() != np.dtype("uint8"):
        raise ValueError("Mask storage dtype must be uint8")
    if mask.shape != source.shape or not np.allclose(mask.affine, source.affine, atol=1e-5, rtol=0):
        raise ValueError("Mask must use the original CT shape and affine")
    if mask.header.get_xyzt_units()[0] != "mm":
        raise ValueError("Mask must use original millimetre units")
    if not np.allclose(mask.header.get_zooms(), source.header.get_zooms(), atol=1e-5, rtol=0):
        raise ValueError("Mask spacing differs from original CT")
    for name in ("qform", "sform"):
        original = getattr(source, "get_" + name)()
        actual = getattr(mask, "get_" + name)()
        if (int(source.header[name + "_code"]) != int(mask.header[name + "_code"]) or
                not np.allclose(original, actual, atol=1e-5, rtol=0)):
            raise ValueError(f"Mask {name}/code differs from original CT")
    text_fields = ("data_type", "db_name", "descrip", "aux_file", "intent_name")
    if mask.header.extensions or any(bytes(mask.header[key]).strip(b"\0 ") for key in text_fields if key in mask.header):
        raise ValueError("Mask must have no extensions or free text metadata")
    if getattr(mask.dataobj, "slope", 1) != 1 or getattr(mask.dataobj, "inter", 0) != 0:
        raise ValueError("Mask label scaling must be identity")
    data = np.asanyarray(mask.dataobj)
    if data.dtype != np.dtype("uint8") or not np.isin(data, range(8)).all():
        raise ValueError("Mask must contain only uint8 label IDs 0..7")
    if np.any(data[:, :, :target_index]) or np.any(data[:, :, target_index + 1:]):
        raise ValueError("Mask has foreground outside the single axial target slice")
    labels = sorted(int(label) for label in np.unique(data) if label)
    if not labels:
        raise ValueError("Mask has no foreground")
    return labels


def _check_label_evidence(document, labels):
    observations = {row["id"]: row for row in document["observations"]}
    tracks = {row["id"]: row for row in document["tracks"]}
    decisions = {row["label"]: row for row in document["decisions"]}
    for label in labels:
        decision = decisions.get(label)
        if decision is None or decision["state"] != "delineated":
            raise ValueError(f"Mask label {label} requires a delineated notebook decision")
        support = {reference: observations[reference] for reference in decision["supporting_observation_ids"]
                   if observations[reference]["kind"] == "observation"
                   and observations[reference]["structure"] in (0, label)
                   and observations[reference]["statement"].strip()}
        linked = False
        for reference in decision["track_ids"]:
            track = tracks[reference]
            members = set(track["observation_ids"]) & support.keys()
            if len({support[key]["locator"]["view"] for key in members}) < 2:
                continue
            if any(relation["kind"] == "continuation" and relation["from_id"] in members
                   and relation["to_id"] in members for relation in track["relations"]):
                linked = True
        if not linked:
            raise ValueError(f"Mask label {label} requires multiview tracked CT evidence")


def assemble(ct, admission, notebook, mask, out):
    """Validate before creating NEWDIR, then copy both artifacts byte-for-byte.

    The manifest is written last. A concurrent input mutation can leave an
    incomplete new directory without a manifest; existing files are never edited.
    """
    ct, admission, notebook, mask, out = map(Path, (ct, admission, notebook, mask, out))
    if out.exists() or out.is_symlink():
        raise ValueError("Choose a new output directory; existing files are preserved")
    if mask.name.lower().endswith(".nii.gz"):
        mask_name = "mask.nii.gz"
    elif mask.suffix.lower() == ".nii":
        mask_name = "mask.nii"
    else:
        raise ValueError("Mask must be a single-file .nii or .nii.gz NIfTI")
    hashes = {name: sha256_file(path) for name, path in
              (("ct", ct), ("admission", admission), ("notebook", notebook), ("mask", mask))}
    source = load_admitted_ct(ct, admission)
    grid = geometry(source)
    exported, document, report = _load_notebook(notebook, hashes["ct"], grid)
    target_index = document["target"]["index"]
    labels = _check_mask(mask, source, target_index)
    _check_label_evidence(document, labels)
    paths = {"ct": ct, "admission": admission, "notebook": notebook, "mask": mask}
    if any(sha256_file(path) != hashes[name] for name, path in paths.items()):
        raise ValueError("Input changed during bundle validation")
    code = {"heart3d/anatomy_bundle.py": Path(__file__),
            "heart3d/anatomy_harness.py": Path(load_admitted_ct.__code__.co_filename),
            "heart3d/web/anatomy_review.py": Path(validate_document.__code__.co_filename),
            "heart3d/web/imaging.py": Path(geometry.__code__.co_filename)}
    manifest = {"bundle_schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "anatomy-linked research candidate", "expert_approved": False,
        "medical_correctness_verified": False,
        "meaning": "Notebook completeness never establishes medical correctness or expert approval.",
        "ct_sha256": hashes["ct"], "admission_sha256": hashes["admission"],
        "label_schema": SCHEMA, "scope": {"axis": 2, "index": target_index, "complete_volume": False},
        "shape_ijk": list(source.shape), "affine_ras": source.affine.tolist(), "unit": "mm",
        "positive_labels": labels, "readiness": report,
        "notebook": {"file": "notebook.json", "sha256": hashes["notebook"],
                     "export_sha256": exported["export_sha256"], "document_sha256": exported["document_sha256"],
                     "project_id": exported["project_id"], "version_id": exported["version_id"],
                     "number": exported["number"]},
        "mask": {"file": mask_name, "sha256": hashes["mask"], "copied_without_modification": True},
        "code_sha256": {name: sha256_file(path) for name, path in code.items()},
        "inference_performed": False, "assembler_image_cloud_transfer": False,
        "observations_include_cloud_model": any(row["source"] == "cloud_model" for row in document["observations"]),
        "image_cloud_transfer": any(row["source"] == "cloud_model" for row in document["observations"]),
        "pixels_created": False}
    out.mkdir(parents=True, exist_ok=False)
    # Use exclusive destinations even inside the newly created directory.
    for original, destination, name in ((mask, out / mask_name, "mask"),
                                         (notebook, out / "notebook.json", "notebook")):
        with original.open("rb") as incoming, destination.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing)
        if sha256_file(destination) != hashes[name]:
            raise ValueError("Input changed while copying; bundle is incomplete")
    if any(sha256_file(path) != hashes[name] for name, path in paths.items()):
        raise ValueError("Input changed while assembling; bundle is incomplete")
    manifest["manifest_sha256"] = _digest(manifest)
    with (out / "manifest.json").open("xb") as handle:
        handle.write(canonical_json(manifest) + b"\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("assemble")
    for name in ("ct", "admission", "notebook", "mask", "out"):
        command.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        result = assemble(args.ct, args.admission, args.notebook, args.mask, args.out)
    except (ValueError, OSError, KeyError, TypeError, np.linalg.LinAlgError, nib.filebasedimages.ImageFileError) as error:
        parser.exit(2, f"Bundle rejected: {error}\n")
    print(json.dumps({"bundle_created": True, "scope": result["scope"], "positive_labels": result["positive_labels"],
                      "expert_approved": False, "medical_correctness_verified": False}))


if __name__ == "__main__":
    main()
