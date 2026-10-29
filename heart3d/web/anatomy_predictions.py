"""Owner-only import of an evidence-linked, unapproved single-slice candidate."""
import hashlib
import json
from pathlib import Path
import shutil

import nibabel as nib
import numpy as np
from sqlalchemy import select, update

from heart3d.anatomy_bundle import (_check_label_evidence, _check_mask, _digest,
                                   _load_notebook, _nonfinite, _unique_object)
from heart3d.storage import sha256_file
from .anatomy_review import canonical_json
from .db import (AnatomyNotebookVersion, AnnotationProject, AnnotationVersion,
                 Assignment, AuditEvent, Series, User, uid)
from .imaging import SCHEMA, geometry
from .predictions import ANATOMY_SLICE_SOURCE, slice_scope


def append_anatomy_bundle(root, factory, username, project_id, bundle):
    """Append a checked immutable mask plus notebook snapshot, preserving head/status.

    An export hash alone is not authority: its document and identity must match an
    immutable notebook version already saved in this project's database. The
    command performs no network requests, model inference, or mask alterations.
    """
    root, bundle = Path(root).resolve(), Path(bundle).resolve()
    manifest_path = bundle / "manifest.json"
    if not bundle.is_dir() or not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("Existing anatomy bundle manifest required")
    if manifest_path.stat().st_size > 2_000_000:
        raise ValueError("Anatomy manifest is oversized")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"),
                          object_pairs_hook=_unique_object, parse_constant=_nonfinite)
    if not isinstance(manifest, dict):
        raise ValueError("Anatomy manifest must be an object")
    claimed = manifest.get("manifest_sha256")
    if not isinstance(claimed, str) or _digest({k: v for k, v in manifest.items() if k != "manifest_sha256"}) != claimed:
        raise ValueError("Anatomy manifest integrity check failed")
    if (type(manifest.get("bundle_schema_version")) is not int or manifest["bundle_schema_version"] != 1
            or manifest.get("source") != "anatomy-linked research candidate"
            or manifest.get("expert_approved") is not False
            or manifest.get("medical_correctness_verified") is not False
            or manifest.get("label_schema") != SCHEMA):
        raise ValueError("Only explicitly unapproved cardiac7 anatomy candidates accepted")
    # The existing export API serves gzip NIfTI; do not rename an uncompressed file.
    if (not isinstance(manifest.get("mask"), dict) or not isinstance(manifest.get("notebook"), dict)
            or manifest["mask"].get("file") != "mask.nii.gz" or manifest["notebook"].get("file") != "notebook.json"):
        raise ValueError("Web import requires mask.nii.gz and notebook.json inside the bundle")
    mask, notebook = bundle / "mask.nii.gz", bundle / "notebook.json"
    for path in (mask, notebook):
        if not path.is_file() or path.is_symlink() or path.resolve().parent != bundle:
            raise ValueError("Bundle artifacts must be regular files inside the bundle")
    if notebook.stat().st_size > 16_000_000:
        raise ValueError("Anatomy notebook is oversized")
    hashes = {"mask": sha256_file(mask), "notebook": sha256_file(notebook),
              "manifest": sha256_file(manifest_path)}
    if any(manifest[name].get("sha256") != hashes[name] for name in ("mask", "notebook")):
        raise ValueError("Anatomy bundle artifact hash differs")
    staged = None
    staged_created = False
    try:
        with factory.begin() as db:
            user = db.scalar(select(User).where(User.username == username, User.active.is_(True)))
            project = db.get(AnnotationProject, project_id)
            assigned = user and db.scalar(select(Assignment).where(
                Assignment.project_id == project_id, Assignment.user_id == user.id))
            if not user or user.role not in ("resident", "admin") or not assigned or not project:
                raise ValueError("Existing assigned resident/admin and project required")
            if project.label_schema != SCHEMA:
                raise ValueError("Project label schema differs from cardiac7-v1")
            series = db.get(Series, project.series_id)
            ct = (root / series.ct_path).resolve()
            if not ct.is_relative_to(root) or not ct.is_file() or sha256_file(ct) != series.ct_sha256:
                raise ValueError("Source CT integrity check failed")
            if manifest.get("ct_sha256") != series.ct_sha256:
                raise ValueError("Anatomy bundle belongs to a different CT")
            image = nib.load(ct)
            grid = geometry(image)
            if (tuple(grid["shape"]) != tuple(series.geometry["shape"])
                    or not np.allclose(grid["affine_ras"], series.geometry["affine_ras"], atol=1e-6, rtol=0)
                    or manifest.get("shape_ijk") != list(image.shape) or manifest.get("unit") != "mm"
                    or not np.allclose(manifest.get("affine_ras"), image.affine, atol=1e-6, rtol=0)):
                raise ValueError("Anatomy bundle or stored CT geometry differs")
            exported, document, report = _load_notebook(notebook, series.ct_sha256, grid)
            saved = db.get(AnatomyNotebookVersion, exported["version_id"])
            if (not saved or saved.project_id != project.id or saved.ct_sha256 != series.ct_sha256
                    or exported["project_id"] != project.id or exported["author_id"] != saved.author_id
                    or exported["created_at"] != saved.created_at or exported["number"] != saved.number
                    or canonical_json(saved.document) != canonical_json(document)):
                raise ValueError("Notebook export must match an immutable saved version for this project")
            notebook_meta = manifest["notebook"]
            if any(notebook_meta.get(k) != exported[k] for k in
                   ("version_id", "project_id", "number", "document_sha256", "export_sha256")):
                raise ValueError("Manifest notebook identity or hashes differ")
            scope = slice_scope(manifest, image.shape)
            if scope["index"] != document["target"]["index"] or manifest.get("readiness") != report:
                raise ValueError("Anatomy bundle scope or completeness differs from notebook")
            labels = _check_mask(mask, image, scope["index"])
            _check_label_evidence(document, labels)
            if manifest.get("positive_labels") != labels:
                raise ValueError("Anatomy bundle positive labels differ from mask")
            observer_sources = sorted({row["source"] for row in document["observations"]})
            cloud = "cloud_model" in observer_sources
            if (manifest.get("observations_include_cloud_model", False) is not cloud
                    or manifest.get("image_cloud_transfer", False) is not cloud):
                raise ValueError("Anatomy bundle cloud-observer provenance differs from notebook")
            values = np.asanyarray(nib.load(mask).dataobj)
            voxel_digest = hashlib.sha256(values.tobytes(order="F")).hexdigest()
            for previous in db.scalars(select(AnnotationVersion).where(
                    AnnotationVersion.project_id == project.id, AnnotationVersion.source == ANATOMY_SLICE_SOURCE)):
                prior = previous.document.get("ai_provenance", {})
                if prior.get("scope") == scope and prior.get("mask_voxel_sha256") == voxel_digest:
                    raise ValueError("Identical anatomy slice draft already exists; versions cannot be replaced")
            names = ("background", "LV", "RV", "LA", "RA", "MYO", "AO", "PA")
            safe = {"variant": "anatomy_slice_draft", "scope": scope,
                "method": "Anatomy-linked single-slice candidate; see immutable CT notebook evidence and boundary decisions.",
                "warning": "Unvalidated model/anatomy research candidate. Outside-plane zeros are unannotated, not background reference. Requires human and surgeon review.",
                "review_required_classes": list(names[1:]), "expert_approved": False,
                "medical_correctness_verified": False, "exploratory_only": True, "geometry_verified": True,
                "ct_sha256": series.ct_sha256, "prediction_sha256": hashes["mask"],
                "predicted_voxels": {name: int(np.count_nonzero(values == label)) for label, name in enumerate(names)},
                "absent_classes": [names[label] for label in range(1, 8) if label not in labels],
                "mask_voxel_sha256": voxel_digest, "notebook_version_id": saved.id,
                "notebook_document_sha256": exported["document_sha256"],
                "notebook_export_sha256": exported["export_sha256"], "anatomy_bundle_sha256": claimed,
                "observer_sources": observer_sources, "image_cloud_transfer": cloud,
                "external_validation_required": report["external_validation_required"]}
            version_id, revision = uid(), project.revision
            relative = Path("annotations") / project.id / version_id / "mask.nii.gz"
            staged = (root / relative).resolve()
            if not staged.is_relative_to(root):
                raise ValueError("Invalid annotation storage path")
            staged.parent.mkdir(parents=True, exist_ok=False)
            staged_created = True
            for source, destination, name in ((mask, staged, "mask"),
                    (notebook, staged.parent / "anatomy-notebook.json", "notebook"),
                    (manifest_path, staged.parent / "anatomy-bundle-manifest.json", "manifest")):
                with source.open("rb") as incoming, destination.open("xb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
                if sha256_file(destination) != hashes[name] or sha256_file(source) != hashes[name]:
                    raise ValueError("Anatomy bundle changed during staging")
            changed = db.execute(update(AnnotationProject).where(
                AnnotationProject.id == project.id, AnnotationProject.revision == revision).values(revision=revision+1))
            if changed.rowcount != 1:
                raise ValueError("Concurrent project change; retry import")
            db.add(AnnotationVersion(id=version_id, project_id=project.id, number=revision+1, author_id=user.id,
                ct_sha256=series.ct_sha256, label_schema=SCHEMA, source=ANATOMY_SLICE_SOURCE,
                document={"contours": [], "operations": [], "flags": [], "ai_provenance": safe},
                mask_path=relative.as_posix(), mask_sha256=hashes["mask"]))
            db.add(AuditEvent(actor_id=user.id, action="import_unapproved_anatomy_slice_draft", project_id=project.id))
        return version_id
    except Exception:
        if staged_created and staged is not None and staged.is_relative_to(root):
            for name in ("mask.nii.gz", "anatomy-notebook.json", "anatomy-bundle-manifest.json"):
                target = staged.parent / name
                if target.is_file():
                    target.unlink()
            staged.parent.rmdir()
        raise
