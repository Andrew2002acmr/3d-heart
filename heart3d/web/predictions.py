"""Owner-only local import of verified AI masks; no public upload endpoint."""
import hashlib
import re
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
from sqlalchemy import select, update

from heart3d.storage import sha256_file
from .db import User, Patient, Study, Series, AnnotationProject, AnnotationVersion, Assignment, AuditEvent, uid
from .imaging import geometry, SCHEMA

AI_SOURCES = ("AI prediction", "AI-assisted slice draft", "AI anatomy slice draft")
SLICE_DRAFT_SOURCE = AI_SOURCES[1]
ANATOMY_SLICE_SOURCE = AI_SOURCES[2]
SLICE_DRAFT_SOURCES = (SLICE_DRAFT_SOURCE, ANATOMY_SLICE_SOURCE)


def slice_scope(provenance, shape):
    """Return a strict native axial plane; a missing plane is never a full volume."""
    scope = provenance.get("scope")
    if (not isinstance(scope, dict) or type(scope.get("axis")) is not int or scope["axis"] != 2
            or type(scope.get("index")) is not int or not 0 <= scope["index"] < shape[2]
            or scope.get("complete_volume") is not False):
        raise ValueError("Slice draft requires one in-range axial scope with complete_volume=false")
    return {"axis": 2, "index": scope["index"], "complete_volume": False}


def append_slice_draft(root, factory, username, project_id, mask, provenance):
    """Owner CLI only: append an immutable exploratory plane to an assigned project.

    The existing CT and model versions are checked in place. Import advances only
    the revision counter; it never adopts the result or changes the review state.
    """
    root, mask = Path(root).resolve(), Path(mask).resolve()
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
            if not isinstance(provenance, dict):
                raise ValueError("Slice draft provenance is required")
            digest = sha256_file(mask)
            if provenance.get("ct_sha256") != series.ct_sha256 or provenance.get("prediction_sha256") != digest:
                raise ValueError("Slice draft provenance/hash differs")
            if (provenance.get("variant") != "slice_draft" or provenance.get("expert_approved") is not False
                    or provenance.get("exploratory_only") is not True or provenance.get("geometry_verified") is not True):
                raise ValueError("Only explicitly unapproved exploratory geometry-verified slice drafts accepted")
            image, prediction = nib.load(ct), nib.load(mask)
            grid = geometry(image)
            if (image.header.get_xyzt_units()[0] != "mm" or tuple(grid["shape"]) != tuple(series.geometry["shape"])
                    or not np.allclose(grid["affine_ras"], series.geometry["affine_ras"], atol=1e-6, rtol=0)):
                raise ValueError("Stored CT grid differs from original source")
            values = np.asanyarray(prediction.dataobj)
            if (prediction.shape != image.shape or prediction.header.get_xyzt_units()[0] != "mm"
                    or prediction.get_data_dtype() != np.dtype("uint8")
                    or not np.allclose(prediction.affine, image.affine, atol=1e-6, rtol=0)
                    or not np.isfinite(values).all() or not np.isin(values, range(8)).all()):
                raise ValueError("Slice draft differs from original CT grid/labels or uint8 type")
            for form in ("qform", "sform"):
                expected, code = getattr(image, "get_" + form)(coded=True)
                actual, actual_code = getattr(prediction, "get_" + form)(coded=True)
                if (int(actual_code) != int(code) or (code and not np.allclose(actual, expected, atol=1e-6, rtol=0))):
                    raise ValueError("Slice draft qform/sform differs from source CT")
            if (not np.allclose(prediction.header.get_zooms(), image.header.get_zooms(), atol=1e-6, rtol=0)
                    or prediction.header.extensions
                    or any(bytes(prediction.header[k]).strip(b"\0 ") for k in ("descrip", "aux_file", "intent_name", "db_name"))):
                raise ValueError("Slice draft spacing or metadata differs from the safe source grid")
            scope = slice_scope(provenance, image.shape)
            index = scope["index"]
            if np.any(values[:, :, :index]) or np.any(values[:, :, index+1:]):
                raise ValueError("Slice draft contains labels outside its target plane")
            for key, limit in (("method", 1000), ("warning", 4000)):
                if not isinstance(provenance.get(key), str) or not provenance[key].strip() or len(provenance[key]) > limit:
                    raise ValueError("Slice draft requires bounded method and warning text")
            review = provenance.get("review_required_classes")
            if (not isinstance(review, list) or len(review) > 32
                    or any(not isinstance(item, str) or not item.strip() or len(item) > 100 for item in review)):
                raise ValueError("review_required_classes must be a list of class strings")
            originals = list(db.scalars(select(AnnotationVersion).where(
                AnnotationVersion.project_id == project.id, AnnotationVersion.source == "AI prediction")))
            source_hashes = provenance.get("source_prediction_sha256")
            if ({v.document.get("ai_provenance", {}).get("variant") for v in originals} != {"v1", "v2"}
                    or len(originals) != 2 or not isinstance(source_hashes, list) or len(source_hashes) != 2
                    or any(not isinstance(item, str) or not re.fullmatch(r"[0-9a-f]{64}", item) for item in source_hashes)
                    or sorted(source_hashes) != sorted(v.mask_sha256 for v in originals)):
                raise ValueError("Source prediction hashes must identify this project's immutable v1/v2")
            # Label names match existing inference reports; numeric IDs are also unambiguous.
            names = ("background", "LV", "RV", "LA", "RA", "MYO", "AO", "PA")
            counts = {name: int(np.count_nonzero(values == label)) for label, name in enumerate(names)}
            claimed = provenance.get("predicted_voxels")
            if isinstance(claimed, dict):
                expected_counts = {**counts, **{str(i): counts[name] for i, name in enumerate(names)}}
                if (not claimed or any(key not in expected_counts or type(count) is not int or count != expected_counts[key]
                                       for key, count in claimed.items())):
                    raise ValueError("predicted_voxels differs from slice draft labels")
            elif type(claimed) is not int or claimed != int(np.count_nonzero(values)):
                raise ValueError("predicted_voxels differs from slice draft labels")
            voxel_digest = hashlib.sha256(values.astype(np.uint8).tobytes(order="F")).hexdigest()
            for previous in db.scalars(select(AnnotationVersion).where(
                    AnnotationVersion.project_id == project.id, AnnotationVersion.source == SLICE_DRAFT_SOURCE)):
                safe = previous.document.get("ai_provenance", {})
                if safe.get("scope") == scope and safe.get("mask_voxel_sha256") == voxel_digest:
                    raise ValueError("Identical slice draft already exists; immutable versions cannot be replaced")
            safe = {key: provenance[key] for key in (
                "variant", "method", "warning", "review_required_classes", "expert_approved", "exploratory_only",
                "geometry_verified", "ct_sha256", "prediction_sha256", "source_prediction_sha256")}
            safe.update(scope=scope, predicted_voxels=counts, mask_voxel_sha256=voxel_digest,
                        source_intensity_convention_confirmed=False)
            version_id, revision = uid(), project.revision
            relative = Path("annotations") / project.id / version_id / "mask.nii.gz"
            staged = (root / relative).resolve()
            if not staged.is_relative_to(root):
                raise ValueError("Invalid annotation storage path")
            staged.parent.mkdir(parents=True, exist_ok=False)
            staged_created = True
            shutil.copyfile(mask, staged)
            if sha256_file(staged) != digest:
                raise ValueError("Slice draft staging integrity failed")
            changed = db.execute(update(AnnotationProject).where(
                AnnotationProject.id == project.id, AnnotationProject.revision == revision).values(revision=revision+1))
            if changed.rowcount != 1:
                raise ValueError("Concurrent project change; retry import")
            db.add(AnnotationVersion(id=version_id, project_id=project.id, number=revision+1, author_id=user.id,
                ct_sha256=series.ct_sha256, label_schema=SCHEMA, source=SLICE_DRAFT_SOURCE,
                document={"contours": [], "operations": [], "flags": [], "ai_provenance": safe},
                mask_path=relative.as_posix(), mask_sha256=digest))
            db.add(AuditEvent(actor_id=user.id, action="import_unapproved_slice_draft", project_id=project.id))
        return version_id
    except Exception:
        if staged_created and staged is not None and staged.is_relative_to(root):
            if staged.is_file():
                staged.unlink()
            staged.parent.rmdir()
        raise


def import_predictions(root, factory, username, ct, predictions, alias, synthetic=False):
    """Each prediction is (mask path, provenance dict). All validation precedes DB writes."""
    if not re.fullmatch(r"[A-Z0-9-]{3,40}", alias):
        raise ValueError("A pseudonymous ASCII study alias is required")
    root, ct = Path(root), Path(ct)
    image = nib.load(ct)
    grid = geometry(image)
    if image.header.extensions or any(bytes(image.header[k]).strip(b"\0 ") for k in ("descrip", "aux_file", "intent_name", "db_name")):
        raise ValueError("CT metadata text/extensions must be removed before import")
    digest = sha256_file(ct)
    staged = []
    for mask, provenance in predictions:
        mask = Path(mask)
        im = nib.load(mask)
        values = np.asanyarray(im.dataobj)
        if (im.shape != image.shape or not np.allclose(im.affine, image.affine, atol=1e-6)
                or im.header.get_xyzt_units()[0] != "mm" or not np.isfinite(values).all()
                or not np.isin(values, range(8)).all()):
            raise ValueError("Prediction differs from original CT grid/labels")
        if provenance.get("ct_sha256") != digest or provenance.get("prediction_sha256") != sha256_file(mask):
            raise ValueError("Prediction provenance/hash differs")
        if provenance.get("variant") not in ("v1", "v2") or provenance.get("expert_approved") is not False:
            raise ValueError("Only explicitly unapproved v1/v2 predictions accepted")
        if not synthetic and not (provenance.get("geometry_verified") is True and provenance.get("exploratory_only") is True):
            raise ValueError("Clinical import requires explicit exploratory/geometry provenance")
        safe = {key: provenance[key] for key in ("variant", "exploratory_only", "checkpoint_sha256", "adapter_sha256",
                                                "predicted_voxels", "absent_classes") if key in provenance}
        safe["source_intensity_convention_confirmed"] = False
        staged.append((mask, sha256_file(mask), safe))
    if {p[2]["variant"] for p in staged} != {"v1", "v2"} or len(staged) != 2:
        raise ValueError("Exactly one prediction per recovered model required")
    with factory.begin() as db:
        user = db.scalar(select(User).where(User.username == username, User.active.is_(True)))
        if not user or user.role not in ("resident", "admin"):
            raise ValueError("Existing resident/admin required")
        if db.scalar(select(Study).where(Study.pseudonym == alias)):
            raise ValueError("Study alias exists; prior versions cannot be replaced")
        project_id = uid()
        ct_relative = Path("raw") / f"{project_id}.nii.gz"
        (root / ct_relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ct, root / ct_relative)
        if sha256_file(root / ct_relative) != digest:
            raise ValueError("CT staging integrity failed")
        patient = Patient(pseudonym="P-UNLINKED-"+alias)
        db.add(patient); db.flush()
        study = Study(patient_id=patient.id, pseudonym=alias, synthetic=synthetic)
        db.add(study); db.flush()
        series = Series(study_id=study.id, ct_path=ct_relative.as_posix(), ct_sha256=digest, geometry=grid)
        db.add(series); db.flush()
        project = AnnotationProject(id=project_id, series_id=series.id, revision=2)
        db.add(project); db.flush()
        db.add(Assignment(project_id=project.id, user_id=user.id))
        for number, (path, mask_digest, safe) in enumerate(staged, 1):
            version_id = uid()
            relative = Path("annotations") / project.id / version_id / "mask.nii.gz"
            (root / relative).parent.mkdir(parents=True)
            shutil.copyfile(path, root / relative)
            if sha256_file(root / relative) != mask_digest:
                raise ValueError("Mask staging integrity failed")
            version = AnnotationVersion(id=version_id, project_id=project.id, number=number, author_id=user.id,
                                        ct_sha256=digest, label_schema=SCHEMA, source="AI prediction",
                                        document={"contours": [], "operations": [], "ai_provenance": safe},
                                        mask_path=relative.as_posix(), mask_sha256=mask_digest)
            db.add(version)
        db.add(AuditEvent(actor_id=user.id, action="import_unapproved_AI_predictions", project_id=project.id))
        return project.id
