"""Local CT observation notebooks; this module never creates segmentation masks.

Locators are native-grid anchors with LPS millimetre camera metadata. Readiness
describes documentation completeness, never correctness or expert approval.
"""
import hashlib
import json
from typing import Annotated, Literal

import nibabel as nib
import numpy as np
from fastapi import Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError, OperationalError

from heart3d.dicom.ct import LPS_TO_RAS
from .db import AnatomyNotebookVersion, uid


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


Identifier = Annotated[str, Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")]
Structure = Annotated[int, Field(ge=1, le=7)]
Confidence = Literal["low", "medium", "high"]
Vector3 = Annotated[list[float], Field(min_length=3, max_length=3)]
References = Annotated[list[Identifier], Field(max_length=2048)]


class Target(StrictModel):
    axis: int = Field(ge=2, le=2)
    index: int = Field(ge=0)


class Locator(StrictModel):
    view: Literal["AX", "COR", "SAG"]
    anchor_ijk: Vector3
    focal_point_lps_mm: Vector3
    normal_lps: Vector3
    native_axis: int | None = Field(default=None, ge=0, le=2)
    native_index: int | None = Field(default=None, ge=0)
    window_width: float = Field(gt=0)
    window_level: float
    overlay_visible: Literal[False]

    @field_validator("overlay_visible", mode="before")
    @classmethod
    def CT_only(cls, value):
        if value is not False:
            raise ValueError("CT observations require overlay_visible=false")
        return value


class Observation(StrictModel):
    id: Identifier
    kind: Literal["observation", "hypothesis"]
    structure: int = Field(ge=0, le=7)
    statement: str = Field(max_length=3000)
    features: list[Annotated[str, Field(max_length=300)]] = Field(max_length=20)
    confidence: Confidence
    uncertainty_reason: str = Field(max_length=1500)
    evidence_refs: References
    source: Literal["human", "local_model", "cloud_model"]
    locator: Locator


class Relation(StrictModel):
    from_id: Identifier
    to_id: Identifier
    kind: Literal["continuation", "branch", "unresolved"]
    gap_reason: str = Field(max_length=1500)


class Track(StrictModel):
    id: Identifier
    structure: Structure
    observation_ids: References
    relations: list[Relation] = Field(max_length=2048)


class StructureDecision(StrictModel):
    label: Structure
    state: Literal["unassessed", "delineated", "uncertain", "not_visible", "not_assessable"]
    supporting_observation_ids: References
    counterevidence_ids: References
    track_ids: References
    boundary_rule: str = Field(max_length=3000)
    rationale: str = Field(max_length=3000)
    unresolved_questions: list[Annotated[str, Field(max_length=1500)]] = Field(max_length=20)
    confidence: Confidence


class AnatomyDocument(StrictModel):
    schema_version: int = Field(default=1, ge=1, le=1)
    target: Target | None = None
    observations: list[Observation] = Field(default_factory=list, max_length=2048)
    tracks: list[Track] = Field(default_factory=list, max_length=512)
    decisions: list[StructureDecision] = Field(default_factory=list, max_length=7)


class SaveAnatomyNotebook(StrictModel):
    base_version_id: Identifier | None = None
    ct_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    document: AnatomyDocument


def _unique(values, name):
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate {name}")


def validate_document(document, grid):
    """Validate geometry and document-local references without reading CT pixels."""
    document = document if isinstance(document, AnatomyDocument) else AnatomyDocument.model_validate(document)
    shape = np.asarray(grid["shape"], dtype=float)
    affine = LPS_TO_RAS @ np.asarray(grid["affine_ras"], dtype=float)
    if shape.shape != (3,) or affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError("Invalid source CT geometry")
    inverse = np.linalg.inv(affine)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    if np.any(spacing <= 0):
        raise ValueError("Invalid source CT spacing")
    if document.target is not None and document.target.index >= shape[2]:
        raise ValueError("Target slice outside CT")
    _unique([o.id for o in document.observations] + [t.id for t in document.tracks], "notebook IDs")
    _unique([d.label for d in document.decisions], "decision labels")
    observations = {o.id: o for o in document.observations}
    tracks = {t.id: t for t in document.tracks}

    def ct_observations(refs, context):
        _unique(refs, f"{context} references")
        for reference in refs:
            found = observations.get(reference)
            if found is None or found.kind != "observation":
                raise ValueError(f"{context} must reference same-document CT observations, never predictions, masks, probabilities or hypotheses")
        return [observations[reference] for reference in refs]

    for observation in document.observations:
        locator = observation.locator
        anchor = np.asarray(locator.anchor_ijk, dtype=float)
        point = np.asarray(locator.focal_point_lps_mm, dtype=float)
        normal = np.asarray(locator.normal_lps, dtype=float)
        if not np.isfinite(np.concatenate((anchor, point, normal))).all():
            raise ValueError("Locator coordinates must be finite")
        if np.any(anchor < 0) or np.any(anchor > shape - 1):
            raise ValueError("Observation anchor outside CT")
        expected = nib.affines.apply_affine(affine, anchor)
        if not np.allclose(expected, point, atol=.05, rtol=0):
            raise ValueError("Observation anchor and LPS focal point differ by more than 0.05 mm")
        # Verify both directions in physical millimetres, including anisotropic grids.
        roundtrip = nib.affines.apply_affine(inverse, point)
        if not np.allclose((roundtrip - anchor) * spacing, 0, atol=.05, rtol=0):
            raise ValueError("Observation LPS/native-grid round trip failed")
        if not np.isclose(np.linalg.norm(normal), 1, atol=1e-4, rtol=0):
            raise ValueError("Observation plane normal must be a unit vector")
        if (locator.native_axis is None) != (locator.native_index is None):
            raise ValueError("native_axis and native_index must be provided together")
        if locator.native_axis is not None:
            axis, index = locator.native_axis, locator.native_index
            if index >= shape[axis] or not np.isclose(anchor[axis], index, atol=1e-3, rtol=0):
                raise ValueError("Native slice index does not match the observation anchor")
            direction = affine[:3, axis] / spacing[axis]
            if not np.isclose(abs(float(normal @ direction)), 1, atol=1e-4, rtol=0):
                raise ValueError("Native plane normal does not align with the CT axis")
        if observation.kind == "observation" and observation.evidence_refs:
            raise ValueError("Direct CT observations use their CT locator, not prediction/mask/probability evidence_refs")
        if observation.kind == "hypothesis":
            evidence = ct_observations(observation.evidence_refs, "Hypothesis evidence")
            if observation.source == "human" and any(o.source != "human" for o in evidence):
                raise ValueError("Human hypotheses require human CT observations; model observations remain proposals")

    for track in document.tracks:
        linked = ct_observations(track.observation_ids, "Track")
        if any(o.structure not in (0, track.structure) for o in linked):
            raise ValueError("Track CT observations must have the track structure or unknown structure")
        member_ids = set(track.observation_ids)
        _unique([(r.from_id, r.to_id, r.kind) for r in track.relations], "track relations")
        for relation in track.relations:
            if relation.from_id not in member_ids or relation.to_id not in member_ids or relation.from_id == relation.to_id:
                raise ValueError("Track relations need two different CT observations belonging to that track")
            if relation.kind == "unresolved" and not relation.gap_reason.strip():
                raise ValueError("Unresolved track relations require a gap reason")
    for decision in document.decisions:
        ct_observations(decision.supporting_observation_ids, "Decision support")
        ct_observations(decision.counterevidence_ids, "Decision counterevidence")
        _unique(decision.track_ids, "decision track references")
        for reference in decision.track_ids:
            track = tracks.get(reference)
            if track is None or track.structure != decision.label:
                raise ValueError("Decision tracks must belong to the same document and structure")
    result = document.model_dump(exclude_none=True)
    result["target"] = document.target.model_dump() if document.target is not None else None
    return result


def readiness(document):
    """Report workflow completeness only, including unvalidated model proposals."""
    issues = []
    if document.get("target") is None:
        issues.append("Select an axial target slice.")
    observations = {o["id"]: o for o in document["observations"]}
    tracks = {t["id"]: t for t in document["tracks"]}
    decisions = {d["label"]: d for d in document["decisions"]}
    for label in range(1, 8):
        decision = decisions.get(label)
        if not decision or decision["state"] == "unassessed":
            issues.append(f"Structure {label}: record an assessment state.")
            continue
        if decision["state"] != "delineated":
            continue
        support = [observations[i] for i in decision["supporting_observation_ids"]
                   if i in observations and observations[i]["kind"] == "observation" and observations[i]["statement"].strip()]
        if len(support) < 2 or len({o["locator"]["view"] for o in support}) < 2:
            issues.append(f"Structure {label}: delineation requires CT observations in at least two views.")
        normals = [np.asarray(o["locator"]["normal_lps"], dtype=float) for o in support]
        if not any(abs(float(first @ second)) < .95 for index, first in enumerate(normals) for second in normals[index + 1:]):
            issues.append(f"Structure {label}: multiview support requires physically nonparallel CT planes; different view names alone are insufficient.")
        supporting_ids = {o["id"] for o in support}
        def neighbouring_native_planes(relation):
            if relation["kind"] != "continuation" or relation["from_id"] not in supporting_ids or relation["to_id"] not in supporting_ids:
                return False
            first = observations[relation["from_id"]]["locator"]
            second = observations[relation["to_id"]]["locator"]
            return (first["view"] == second["view"] and first.get("native_axis") is not None
                    and first.get("native_axis") == second.get("native_axis")
                    and first.get("native_index") is not None and second.get("native_index") is not None
                    and abs(first["native_index"] - second["native_index"]) == 1)
        continuation = any(neighbouring_native_planes(r)
                           for reference in decision["track_ids"] if reference in tracks
                           for r in tracks[reference]["relations"])
        if not continuation:
            issues.append(f"Structure {label}: document a tracked continuation on two neighbouring native CT planes in the same view; cross-view links and skipped planes alone do not complete sequential tracing.")
        if not decision["boundary_rule"].strip() or not decision["rationale"].strip():
            issues.append(f"Structure {label}: delineation requires a boundary rule and rationale.")
    return {"ready": not issues, "issues": issues, "expert_validated": False, "ready_factual": False,
            "external_validation_required": any(o["source"] != "human" for o in document["observations"]),
            "meaning": "Documentation workflow completeness only; no medical correctness or expert approval is asserted."}


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def register_anatomy_routes(app, *, db_session, signed_in, project_for, source_for, editable, audit):
    """Reuse the application's authorization, CSRF and source integrity checks."""
    def latest(db, project_id):
        return db.scalar(select(AnatomyNotebookVersion).where(AnatomyNotebookVersion.project_id == project_id)
                         .order_by(AnatomyNotebookVersion.number.desc()).limit(1))

    def response_for(series, version):
        document = version.document if version else AnatomyDocument().model_dump()
        return {"version_id": version.id if version else None, "number": version.number if version else 0,
                "ct_sha256": version.ct_sha256 if version else series.ct_sha256,
                "document": document, "readiness": readiness(document)}

    @app.get("/api/projects/{project_id}/anatomy-review")
    def get_notebook(project_id: str, auth=Depends(signed_in), db=Depends(db_session)):
        project = project_for(db, auth[0], project_id)
        series, _ = source_for(db, project)
        version = latest(db, project.id)
        if version and version.ct_sha256 != series.ct_sha256:
            raise HTTPException(409, "Notebook belongs to a different source CT")
        audit(db, "read_anatomy_notebook", auth[0], project.id)
        result = response_for(series, version)
        db.commit()
        return result

    @app.post("/api/projects/{project_id}/anatomy-review")
    def save_notebook(project_id: str, body: SaveAnatomyNotebook, auth=Depends(signed_in), db=Depends(db_session)):
        user = auth[0]
        # Serialize SQLite append/check transactions without touching project revision,
        # annotation head or status. A waiting writer rechecks the latest notebook.
        db.rollback()
        try:
            db.execute(text("BEGIN IMMEDIATE"))
            project = project_for(db, user, project_id)
            editable(project, user)
            series, _ = source_for(db, project)
            if body.ct_sha256 != series.ct_sha256:
                raise HTTPException(409, "CT changed; reload before editing")
            previous = latest(db, project.id)
            if body.base_version_id != (previous.id if previous else None):
                raise HTTPException(409, "Newer notebook version exists; reload before editing")
            try:
                document = validate_document(body.document, series.geometry)
            except (ValueError, TypeError, KeyError, np.linalg.LinAlgError) as error:
                raise HTTPException(422, str(error))
            version = AnatomyNotebookVersion(id=uid(), project_id=project.id, number=previous.number + 1 if previous else 1,
                                            author_id=user.id, ct_sha256=series.ct_sha256, document=document)
            db.add(version)
            audit(db, "save_anatomy_notebook", user, project.id)
            db.commit()
        except (IntegrityError, OperationalError):
            db.rollback()
            raise HTTPException(409, "Concurrent notebook change; reload before editing")
        return response_for(series, version)

    @app.get("/api/projects/{project_id}/anatomy-review/export")
    def export_notebook(project_id: str, version_id: str | None = None, auth=Depends(signed_in), db=Depends(db_session)):
        project = project_for(db, auth[0], project_id)
        series, _ = source_for(db, project)
        version = db.get(AnatomyNotebookVersion, version_id) if version_id else latest(db, project.id)
        if version is None or version.project_id != project.id:
            raise HTTPException(404, "Notebook version not found")
        if version.ct_sha256 != series.ct_sha256:
            raise HTTPException(409, "Notebook belongs to a different source CT")
        document_sha256 = hashlib.sha256(canonical_json(version.document)).hexdigest()
        payload = {"export_schema_version": 1, "kind": "anatomy_review_notebook", "project_id": project.id,
                   "version_id": version.id, "number": version.number, "author_id": version.author_id,
                   "created_at": version.created_at, "ct_sha256": version.ct_sha256,
                   "document_sha256": document_sha256, "document": version.document,
                   "readiness": readiness(version.document),
                   "provenance": {"coordinate_system": "LPS", "unit": "mm", "source": "CT_observation_notebook",
                                  "evidence_policy": "CT locators only; no prediction, mask or probability evidence",
                                  "expert_validated": False,
                                  "observation_sources": {o["id"]: o["source"] for o in version.document["observations"]}}}
        # Hash all immutable exported content except the hash field itself.
        payload["export_sha256"] = hashlib.sha256(canonical_json(payload)).hexdigest()
        audit(db, "export_anatomy_notebook", auth[0], project.id)
        db.commit()
        return Response(canonical_json(payload), media_type="application/json",
                        headers={"Content-Disposition": f'attachment; filename="anatomy_notebook_v{version.number}.json"'})
