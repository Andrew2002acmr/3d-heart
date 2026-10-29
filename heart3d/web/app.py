"""Authenticated local annotation API; storage is never mounted as static files."""
from datetime import datetime, timedelta, timezone
import hashlib
import os
from pathlib import Path
import secrets
import threading
import copy
import shutil

from argon2 import PasswordHasher
from argon2.exceptions import VerificationError, InvalidHashError
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import select, update, func
from sqlalchemy.exc import IntegrityError
import nibabel as nib
import numpy as np

from heart3d.storage import sha256_file
from .db import (database, User, Patient, Study, Series, Assignment, AnnotationProject,
                 AnnotationVersion, AnnotationOperation, Review, Approval, Session,
                 AuditEvent, uid, now)
from .imaging import rasterize, save_mask, labels, SCHEMA, trace_mask_contours
from .predictions import AI_SOURCES, SLICE_DRAFT_SOURCES, slice_scope
from .anatomy_review import register_anatomy_routes

ROOT = Path(__file__).resolve().parents[2]
PASSWORDS = PasswordHasher()
COOKIE = "heart3d_session"
MESH_LOCK = threading.Lock()


class Login(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class Contour(BaseModel):
    id: str = Field(max_length=80)
    label: int = Field(ge=1, le=7)
    axis: int = Field(ge=0, le=2)
    slice: int = Field(ge=0)
    points: list[list[float]] = Field(min_length=1, max_length=2048)
    closed: bool = False
    applied: bool = False
    replace_label_slice: bool = False


class Operation(BaseModel):
    kind: str = Field(pattern="^(contour|brush)$")
    contour_id: str | None = Field(default=None, max_length=80)
    indices: list[int] = Field(default_factory=list, max_length=600000)
    values: list[int] = Field(default_factory=list, max_length=600000)


class ReviewFlag(BaseModel):
    id: str = Field(max_length=80)
    axis: int = Field(ge=0, le=2)
    slice: int = Field(ge=0)
    label: int = Field(ge=1, le=7)
    kind: str = Field(pattern="^(boundary|missing_structure|artifact|uncertain)$")


class Document(BaseModel):
    contours: list[Contour] = Field(default_factory=list, max_length=1024)
    operations: list[Operation] = Field(default_factory=list, max_length=2048)
    base_prediction_id: str | None = Field(default=None, max_length=80)
    flags: list[ReviewFlag] = Field(default_factory=list, max_length=2048)


class SaveDraft(BaseModel):
    base_version_id: str | None = None
    ct_sha256: str
    label_schema: str = SCHEMA
    document: Document


class Head(BaseModel):
    version_id: str


class Decision(Head):
    decision: str = Field(pattern="^(approved|needs_revision)$")
    note: str = Field(default="", max_length=2000)


class Adopt(Head):
    base_version_id: str | None = None
    prepare_boundaries: bool = False


class Trace(BaseModel):
    document: Document
    label: int = Field(ge=1, le=7)
    axis: int = Field(ge=0, le=2)
    slice: int = Field(ge=0)


def data_root():
    return Path(os.environ.get("HEART3D_WEB_DATA", str(ROOT / ".web-data"))).resolve()


def create_app(root=None, secure_cookie=None):
    root = Path(root or data_root()).resolve()
    engine, factory = database(root)
    secure_cookie = secure_cookie if secure_cookie is not None else os.environ.get("HEART3D_COOKIE_SECURE") == "1"
    allowed = os.environ.get("HEART3D_ALLOWED_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173,http://127.0.0.1:8000,http://localhost:8000").split(",")
    app = FastAPI(title="Heart3D Annotation", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.root, app.state.factory, app.state.engine = root, factory, engine
    dummy_password_hash = PASSWORDS.hash(secrets.token_urlsafe(32))
    app.add_middleware(CORSMiddleware, allow_origins=allowed, allow_credentials=True,
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type", "X-CSRF-Token"],
                       expose_headers=["X-Annotation-Scope", "X-Annotation-Slice-Index", "X-Annotation-Warning"])

    @app.middleware("http")
    async def protections(request, call_next):
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin")
            if origin and origin not in allowed:
                return JSONResponse({"detail": "Origin not allowed"}, 403)
            if len(await request.body()) > 8_000_000:
                return JSONResponse({"detail": "Request too large"}, 413)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    def db_session():
        with factory() as db:
            yield db

    def audit(db, action, user=None, project=None):
        db.add(AuditEvent(actor_id=user.id if user else None, action=action, project_id=project))

    def signed_in(request: Request, db=Depends(db_session)):
        raw = request.cookies.get(COOKIE, "")
        session = db.get(Session, hashlib.sha256(raw.encode()).hexdigest()) if raw else None
        if not session or session.expires_at <= now():
            raise HTTPException(401, "Authentication required")
        user = db.get(User, session.user_id)
        if not user or not user.active:
            raise HTTPException(401, "Authentication required")
        if request.method not in ("GET", "HEAD") and not secrets.compare_digest(request.headers.get("X-CSRF-Token", ""), session.csrf):
            raise HTTPException(403, "CSRF check failed")
        return user, session

    def project_for(db, user, project_id):
        project = db.get(AnnotationProject, project_id)
        assignment = db.scalar(select(Assignment).where(Assignment.project_id == project_id, Assignment.user_id == user.id))
        if not project or (user.role != "admin" and not assignment):
            raise HTTPException(404, "Study not found")
        return project

    def source_for(db, project):
        series = db.get(Series, project.series_id)
        path = (root / series.ct_path).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha256_file(path) != series.ct_sha256:
            raise HTTPException(409, "Source CT integrity check failed")
        return series, path

    def editable(project, user):
        if user.role not in ("resident", "admin"):
            raise HTTPException(403, "Resident editing role required")
        if project.status in ("submitted", "approved"):
            raise HTTPException(409, "This version is read-only")

    def version_for(db, project, version_id=None):
        version = db.get(AnnotationVersion, version_id or project.head_version_id) if (version_id or project.head_version_id) else None
        if version and version.project_id != project.id:
            raise HTTPException(404, "Version not found")
        if version_id and not version:
            raise HTTPException(404, "Version not found")
        return version

    def mask_for(version, series):
        path = (root / version.mask_path).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha256_file(path) != version.mask_sha256:
            raise HTTPException(409, "Mask integrity check failed")
        image = nib.load(path)
        data = np.asanyarray(image.dataobj)
        if (image.shape != tuple(series.geometry["shape"]) or not np.allclose(image.affine, series.geometry["affine_ras"], atol=1e-5)
                or not np.isfinite(data).all() or not np.isin(data, range(8)).all()):
            raise HTTPException(409, "Mask grid/labels differ from CT")
        return data.astype(np.uint8)

    def annotation_scope(db, project, version=None, document=None):
        """Scope follows the immutable base, including after expert snapshotting."""
        if document is None:
            document = version.document if version else {}
        provenance = None
        if version and version.source in SLICE_DRAFT_SOURCES:
            provenance = document.get("ai_provenance", {})
        elif document.get("base_prediction_id"):
            base = version_for(db, project, document["base_prediction_id"])
            if base.source in SLICE_DRAFT_SOURCES:
                provenance = base.document.get("ai_provenance", {})
        if provenance is not None:
            try:
                return slice_scope(provenance, db.get(Series, project.series_id).geometry["shape"])
            except ValueError as error:
                raise HTTPException(409, str(error))
        return None

    def scope_warning(scope):
        return (f"Single-slice annotation: only axial slice {scope['index']} (zero-based) is annotated; "
                "other slices are unannotated. This is not a complete-volume reference.") if scope else None

    def scope_headers(scope):
        return {"X-Annotation-Scope": "single-slice", "X-Annotation-Slice-Index": str(scope["index"]),
                "X-Annotation-Warning": scope_warning(scope)} if scope else {}

    def within_scope(scope, axis, index):
        if scope and (axis != scope["axis"] or index != scope["index"]):
            raise HTTPException(422, "Annotation is restricted to its single axial target slice")

    def document_mask(db, project, series, document):
        base = None
        if document.get("base_prediction_id"):
            version = version_for(db, project, document["base_prediction_id"])
            if version.source not in AI_SOURCES or version.ct_sha256 != series.ct_sha256:
                raise HTTPException(422, "Base must be an immutable AI prediction for this CT")
            base = mask_for(version, series)
        scope = annotation_scope(db, project, document=document)
        head_scope = annotation_scope(db, project, version_for(db, project))
        if head_scope and head_scope != scope:
            raise HTTPException(422, "A single-slice draft must preserve its immutable slice base")
        for contour in document.get("contours", []):
            within_scope(scope, contour["axis"], contour["slice"])
        for flag in document.get("flags", []):
            within_scope(scope, flag["axis"], flag["slice"])
            if not 0 <= flag["slice"] < series.geometry["shape"][flag["axis"]]:
                raise HTTPException(422, "Flag outside CT")
        if scope:
            shape = tuple(series.geometry["shape"])
            plane_size = shape[0] * shape[1]
            for operation in document.get("operations", []):
                if operation["kind"] == "brush":
                    indices = np.asarray(operation.get("indices", []), dtype=np.int64)
                    if np.any(indices < 0) or np.any(indices >= np.prod(shape)) or np.any(indices // plane_size != scope["index"]):
                        raise HTTPException(422, "Brush patch is outside the single axial target slice")
            # This metadata is server-derived and is copied into expert references.
            document["annotation_scope"] = scope
            document["scope_warning"] = scope_warning(scope)
        array = rasterize(document, series.geometry, base)
        if scope and (np.any(array[:, :, :scope["index"]]) or np.any(array[:, :, scope["index"]+1:])):
            raise HTTPException(422, "Annotation contains labels outside its single axial target slice")
        return array

    def prediction_rows(db, project):
        return [{"id": v.id, "number": v.number, "source": v.source,
                 "provenance": v.document.get("ai_provenance", {}), "approved": False}
                for v in db.scalars(select(AnnotationVersion).where(
                    AnnotationVersion.project_id == project.id, AnnotationVersion.source.in_(AI_SOURCES)).order_by(AnnotationVersion.number))]

    def view(db, project):
        series = db.get(Series, project.series_id)
        study = db.get(Study, series.study_id)
        patient = db.get(Patient, study.patient_id)
        version = version_for(db, project)
        document = version.document if version else Document().model_dump()
        progress = len({c["label"] for c in document["contours"] if c["applied"]})
        scope = annotation_scope(db, project, version)
        return {"id": project.id, "patient_id": patient.pseudonym, "study_id": study.pseudonym,
                "status": project.status, "revision": project.revision, "updated_at": project.updated_at,
                "version_id": version.id if version else None, "version_number": version.number if version else 0,
                "progress_classes": progress, "synthetic": study.synthetic, "ct_sha256": series.ct_sha256,
                "label_schema": project.label_schema, "geometry": series.geometry, "labels": labels(), "document": document,
                "predictions": prediction_rows(db, project), "annotation_scope": scope,
                "annotation_warning": scope_warning(scope)}

    @app.get("/api/health")
    def health():
        return {"status": "ok", "mode": "local_research_only"}

    @app.post("/api/auth/login")
    def login(body: Login, response: Response, db=Depends(db_session)):
        since = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        if db.scalar(select(func.count()).select_from(AuditEvent).where(AuditEvent.action == "login_failed", AuditEvent.created_at > since)) >= 20:
            raise HTTPException(429, "Too many attempts; retry in five minutes")
        user = db.scalar(select(User).where(User.username == body.username, User.active.is_(True)))
        try:
            valid = PASSWORDS.verify(user.password_hash if user else dummy_password_hash, body.password) and bool(user)
        except (VerificationError, InvalidHashError):
            valid = False
        if not valid:
            audit(db, "login_failed")
            db.commit()
            raise HTTPException(401, "Invalid login or password")
        if PASSWORDS.check_needs_rehash(user.password_hash):
            user.password_hash = PASSWORDS.hash(body.password)
        raw = secrets.token_urlsafe(32)
        session = Session(token_hash=hashlib.sha256(raw.encode()).hexdigest(), user_id=user.id,
                          csrf=secrets.token_urlsafe(32), expires_at=(datetime.now(timezone.utc)+timedelta(hours=8)).isoformat())
        db.add(session)
        audit(db, "login", user)
        db.commit()
        response.set_cookie(COOKIE, raw, httponly=True, secure=secure_cookie, samesite="strict", max_age=28800, path="/api")
        return {"username": user.username, "role": user.role, "csrf": session.csrf}

    @app.get("/api/auth/me")
    def me(auth=Depends(signed_in)):
        user, session = auth
        return {"username": user.username, "role": user.role, "csrf": session.csrf}

    @app.post("/api/auth/logout")
    def logout(response: Response, auth=Depends(signed_in), db=Depends(db_session)):
        user, session = auth
        db.delete(session)
        audit(db, "logout", user)
        db.commit()
        response.delete_cookie(COOKIE, path="/api", secure=secure_cookie, httponly=True, samesite="strict")
        return {"ok": True}

    @app.get("/api/projects")
    def projects(auth=Depends(signed_in), db=Depends(db_session)):
        user, _ = auth
        query = select(AnnotationProject)
        if user.role != "admin":
            query = query.join(Assignment).where(Assignment.user_id == user.id)
        return [view(db, p) for p in db.scalars(query.order_by(AnnotationProject.updated_at.desc()))]

    @app.get("/api/projects/{project_id}")
    def project(project_id: str, auth=Depends(signed_in), db=Depends(db_session)):
        user, _ = auth
        p = project_for(db, user, project_id)
        source_for(db, p)
        audit(db, "open_study", user, p.id)
        db.commit()
        return view(db, p)

    @app.get("/api/projects/{project_id}/volume")
    def volume(project_id: str, auth=Depends(signed_in), db=Depends(db_session)):
        user, _ = auth
        p = project_for(db, user, project_id)
        _, path = source_for(db, p)
        data = nib.load(path).get_fdata(dtype=np.float32)
        audit(db, "read_CT", user, p.id)
        db.commit()
        return Response(data.astype("<f4").tobytes(order="F"), media_type="application/octet-stream")

    @app.get("/api/projects/{project_id}/mask")
    def mask(project_id: str, version_id: str | None = None, auth=Depends(signed_in), db=Depends(db_session)):
        p = project_for(db, auth[0], project_id)
        series, _ = source_for(db, p)
        version = version_for(db, p, version_id)
        if version:
            array = mask_for(version, series)
        else:
            array = np.zeros(series.geometry["shape"], np.uint8)
        audit(db, "read_mask", auth[0], p.id)
        db.commit()
        return Response(array.tobytes(order="F"), media_type="application/octet-stream",
                        headers=scope_headers(annotation_scope(db, p, version)))

    @app.post("/api/projects/{project_id}/preview")
    def preview(project_id: str, body: Document, auth=Depends(signed_in), db=Depends(db_session)):
        p = project_for(db, auth[0], project_id)
        editable(p, auth[0])
        series, _ = source_for(db, p)
        try:
            array = document_mask(db, p, series, body.model_dump())
        except (ValueError, KeyError, TypeError) as error:
            raise HTTPException(422, str(error))
        return Response(array.tobytes(order="F"), media_type="application/octet-stream",
                        headers=scope_headers(annotation_scope(db, p, document=body.model_dump())))

    @app.post("/api/projects/{project_id}/drafts")
    def save(project_id: str, body: SaveDraft, auth=Depends(signed_in), db=Depends(db_session)):
        user, _ = auth
        p = project_for(db, user, project_id)
        editable(p, user)
        series, source = source_for(db, p)
        if body.ct_sha256 != series.ct_sha256 or body.label_schema != p.label_schema:
            raise HTTPException(409, "CT/label schema changed")
        if body.base_version_id != p.head_version_id:
            raise HTTPException(409, "Newer saved version exists; reload before editing")
        document = body.document.model_dump()
        try:
            array = document_mask(db, p, series, document)
        except (ValueError, KeyError, TypeError) as error:
            raise HTTPException(422, str(error))
        version_id, revision = uid(), p.revision
        relative = Path("annotations") / p.id / version_id / "mask.nii.gz"
        (root / relative).parent.mkdir(parents=True)
        digest = save_mask(array, source, root / relative)
        version = AnnotationVersion(id=version_id, project_id=p.id, parent_id=p.head_version_id,
                                    number=revision+1, author_id=user.id, ct_sha256=series.ct_sha256,
                                    label_schema=p.label_schema, document=document,
                                    mask_path=relative.as_posix(), mask_sha256=digest,
                                    source=("human-edited draft" if document["operations"] else "human draft") if document.get("base_prediction_id") else "manual")
        try:
            changed = db.execute(update(AnnotationProject).where(AnnotationProject.id == p.id, AnnotationProject.revision == revision,
                                 AnnotationProject.status.in_(("not_started", "in_progress", "needs_revision")))
                                 .values(head_version_id=version_id, revision=revision+1, status="in_progress", updated_at=now()))
            if changed.rowcount != 1:
                raise HTTPException(409, "Concurrent save detected")
            db.add(version)
            db.flush()
            for index, operation in enumerate(document["operations"]):
                db.add(AnnotationOperation(version_id=version_id, position=index, kind=operation["kind"], payload=operation))
            audit(db, "save_draft", user, p.id)
            db.commit()
        except IntegrityError:
            db.rollback()
            raise HTTPException(409, "Concurrent save detected")
        scope = annotation_scope(db, p, version)
        return {"version_id": version_id, "version_number": revision+1, "mask_sha256": digest, "saved_at": version.created_at,
                "annotation_scope": scope, "annotation_warning": scope_warning(scope)}

    @app.get("/api/projects/{project_id}/versions")
    def versions(project_id: str, auth=Depends(signed_in), db=Depends(db_session)):
        p = project_for(db, auth[0], project_id)
        return [{"id": v.id, "number": v.number, "author_id": v.author_id, "created_at": v.created_at,
                 "ct_sha256": v.ct_sha256, "label_schema": v.label_schema, "source": v.source,
                 "annotation_scope": annotation_scope(db, p, v),
                 "mask_sha256": v.mask_sha256, "approved": bool(db.scalar(select(Approval).where(Approval.version_id == v.id)))}
                for v in db.scalars(select(AnnotationVersion).where(AnnotationVersion.project_id == p.id).order_by(AnnotationVersion.number.desc()))]

    @app.post("/api/projects/{project_id}/adopt-prediction")
    def adopt_prediction(project_id: str, body: Adopt, auth=Depends(signed_in), db=Depends(db_session)):
        user, _ = auth
        p = project_for(db, user, project_id)
        editable(p, user)
        series, source = source_for(db, p)
        selected = version_for(db, p, body.version_id)
        if selected.source not in AI_SOURCES or selected.ct_sha256 != series.ct_sha256:
            raise HTTPException(422, "Choose an AI prediction for this CT")
        if body.base_version_id != p.head_version_id:
            raise HTTPException(409, "A newer draft exists")
        array = mask_for(selected, series)
        document = Document(base_prediction_id=selected.id).model_dump()
        scope = annotation_scope(db, p, selected)
        if scope:
            document.update(annotation_scope=scope, scope_warning=scope_warning(scope))
        if body.prepare_boundaries:
            if not scope:
                raise HTTPException(422, "Boundary preparation requires a single-slice candidate")
            try:
                for label in np.unique(np.take(array, scope["index"], axis=scope["axis"])):
                    if label:
                        document["contours"].extend(trace_mask_contours(array, series.geometry, int(label), scope["axis"], scope["index"]))
                document["operations"] = [{"kind": "contour", "contour_id": c["id"]} for c in document["contours"]]
                if not np.array_equal(rasterize(document, series.geometry, base=array), array):
                    raise ValueError("Editable contours would change the mask; adopt without contours and use brush/eraser")
            except ValueError as error:
                raise HTTPException(422, str(error))
        version_id, revision = uid(), p.revision
        relative = Path("annotations") / p.id / version_id / "mask.nii.gz"
        (root / relative).parent.mkdir(parents=True)
        digest = save_mask(array, source, root / relative)
        version = AnnotationVersion(id=version_id, project_id=p.id, parent_id=selected.id, number=revision+1,
                                    author_id=user.id, ct_sha256=series.ct_sha256, label_schema=p.label_schema,
                                    source="human draft", document=document, mask_path=relative.as_posix(), mask_sha256=digest)
        changed = db.execute(update(AnnotationProject).where(AnnotationProject.id == p.id, AnnotationProject.revision == revision,
                             AnnotationProject.status.in_(("not_started", "in_progress", "needs_revision")))
                             .values(head_version_id=version_id, revision=revision+1, status="in_progress", updated_at=now()))
        if changed.rowcount != 1:
            raise HTTPException(409, "Concurrent draft change")
        db.add(version)
        audit(db, "adopt_AI_as_draft", user, p.id)
        db.commit()
        return view(db, p)

    @app.post("/api/projects/{project_id}/trace-contours")
    def trace(project_id: str, body: Trace, auth=Depends(signed_in), db=Depends(db_session)):
        p = project_for(db, auth[0], project_id)
        editable(p, auth[0])
        series, _ = source_for(db, p)
        try:
            document = body.document.model_dump()
            within_scope(annotation_scope(db, p, document=document), body.axis, body.slice)
            mask = document_mask(db, p, series, document)
            return trace_mask_contours(mask, series.geometry, body.label, body.axis, body.slice)
        except (ValueError, KeyError, TypeError) as error:
            raise HTTPException(422, str(error))

    @app.post("/api/projects/{project_id}/prepare-boundaries")
    def prepare_boundaries(project_id: str, body: Document, auth=Depends(signed_in), db=Depends(db_session)):
        p = project_for(db, auth[0], project_id)
        editable(p, auth[0])
        series, _ = source_for(db, p)
        document = body.model_dump()
        scope = annotation_scope(db, p, document=document)
        if not scope or document["contours"]:
            raise HTTPException(422, "Choose a single-slice draft without existing editable contours")
        try:
            original = document_mask(db, p, series, document)
            for label in np.unique(np.take(original, scope["index"], axis=scope["axis"])):
                if label:
                    document["contours"].extend(trace_mask_contours(original, series.geometry, int(label), scope["axis"], scope["index"]))
            document["operations"].extend({"kind": "contour", "contour_id": c["id"]} for c in document["contours"])
            if not np.array_equal(document_mask(db, p, series, document), original):
                raise ValueError("Editable contours would change the mask; use brush/eraser")
            return document
        except (ValueError, KeyError, TypeError) as error:
            raise HTTPException(422, str(error))

    def meshes_for(version, series):
        from heart3d.surfaces import build_surface
        import json
        folder = root / "derived" / version.mask_sha256
        manifest = folder / "meshes.json"
        # Single process local server. No concurrent writes of a partially built mesh.
        with MESH_LOCK:
            if not manifest.exists():
                mask = mask_for(version, series)
                folder.mkdir(parents=True, exist_ok=True)
                rows = []
                for label in range(1, 8):
                    mesh, metrics = build_surface(mask, label, np.asarray(series.geometry["affine_ras"]))
                    if mesh is None:
                        continue
                    path = folder / f"{label}.vtp"
                    mesh.save(path)
                    rows.append({**metrics, "sha256": sha256_file(path)})
                    del mesh
                manifest.write_text(json.dumps(rows), encoding="utf-8")
            return folder, json.loads(manifest.read_text(encoding="utf-8"))

    @app.get("/api/projects/{project_id}/reconstruction/{version_id}")
    def reconstruction(project_id: str, version_id: str, auth=Depends(signed_in), db=Depends(db_session)):
        p = project_for(db, auth[0], project_id)
        series, _ = source_for(db, p)
        version = version_for(db, p, version_id)
        if annotation_scope(db, p, version):
            raise HTTPException(422, "3D reconstruction is unavailable for a single-slice annotation")
        mask_for(version, series)
        _, rows = meshes_for(version, series)
        audit(db, "read_3D", auth[0], p.id)
        db.commit()
        return {"version_id": version.id, "coordinate_system": "RAS", "unit": "mm", "structures": rows,
                "absent_labels": [i for i in range(1, 8) if i not in {r['label'] for r in rows}]}

    @app.get("/api/projects/{project_id}/mesh/{version_id}/{label}")
    def mesh_file(project_id: str, version_id: str, label: int, auth=Depends(signed_in), db=Depends(db_session)):
        p = project_for(db, auth[0], project_id)
        series, _ = source_for(db, p)
        version = version_for(db, p, version_id)
        if annotation_scope(db, p, version):
            raise HTTPException(422, "3D meshes are unavailable for a single-slice annotation")
        mask_for(version, series)
        folder, rows = meshes_for(version, series)
        entry = next((r for r in rows if r["label"] == label), None)
        if not entry:
            raise HTTPException(404, "Structure absent")
        path = folder / f"{label}.vtp"
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise HTTPException(409, "Mesh integrity failed")
        audit(db, "read_3D_mesh", auth[0], p.id)
        db.commit()
        return FileResponse(path, media_type="application/octet-stream")

    @app.get("/api/projects/{project_id}/export/{version_id}")
    def export(project_id: str, version_id: str, auth=Depends(signed_in), db=Depends(db_session)):
        p = project_for(db, auth[0], project_id)
        source_for(db, p)
        version = version_for(db, p, version_id)
        path = (root / version.mask_path).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha256_file(path) != version.mask_sha256:
            raise HTTPException(409, "Mask integrity check failed")
        audit(db, "export_NIfTI", auth[0], p.id)
        db.commit()
        return FileResponse(path, media_type="application/gzip", filename=f"cardiac7_v{version.number}.nii.gz",
                            headers=scope_headers(annotation_scope(db, p, version)))

    @app.post("/api/projects/{project_id}/submit")
    def submit(project_id: str, body: Head, auth=Depends(signed_in), db=Depends(db_session)):
        p = project_for(db, auth[0], project_id)
        editable(p, auth[0])
        source_for(db, p)
        if body.version_id != p.head_version_id or not p.head_version_id:
            raise HTTPException(409, "Save the current draft first")
        p.status, p.updated_at = "submitted", now()
        audit(db, "submit", auth[0], p.id)
        db.commit()
        return {"status": p.status}

    @app.post("/api/projects/{project_id}/reviews")
    def review(project_id: str, body: Decision, auth=Depends(signed_in), db=Depends(db_session)):
        user, _ = auth
        p = project_for(db, user, project_id)
        if user.role != "surgeon":
            raise HTTPException(403, "Assigned surgeon role required")
        source_for(db, p)
        if p.status != "submitted" or body.version_id != p.head_version_id:
            raise HTTPException(409, "Review only the submitted head version")
        human = version_for(db, p, body.version_id)
        if human.source in AI_SOURCES:
            raise HTTPException(422, "AI predictions cannot become expert references directly")
        values = {"status": body.decision, "updated_at": now()}
        reference_id = body.version_id
        reference = None
        if body.decision == "approved":
            mask_for(human, db.get(Series, p.series_id))
            reference_id = uid()
            relative = Path("annotations") / p.id / reference_id / "mask.nii.gz"
            (root / relative).parent.mkdir(parents=True)
            shutil.copyfile(root / human.mask_path, root / relative)
            if sha256_file(root / relative) != human.mask_sha256:
                raise HTTPException(409, "Reference snapshot integrity failed")
            reference = AnnotationVersion(id=reference_id, project_id=p.id, parent_id=human.id,
                number=p.revision+1, author_id=user.id, ct_sha256=human.ct_sha256,
                label_schema=human.label_schema, source="surgeon-approved reference",
                document=copy.deepcopy(human.document), mask_path=relative.as_posix(), mask_sha256=human.mask_sha256)
            values.update(head_version_id=reference_id, revision=p.revision+1)
        changed = db.execute(update(AnnotationProject).where(AnnotationProject.id == p.id,
            AnnotationProject.status == "submitted", AnnotationProject.head_version_id == body.version_id,
            AnnotationProject.revision == p.revision).values(**values))
        if changed.rowcount != 1:
            raise HTTPException(409, "Concurrent review detected")
        db.add(Review(version_id=body.version_id, surgeon_id=user.id, decision=body.decision, note=body.note))
        if reference is not None:
            db.add(reference)
            db.flush()
            db.add(Approval(version_id=reference_id, surgeon_id=user.id))
        audit(db, "review_"+body.decision, user, p.id)
        db.commit()
        return {"status": body.decision, "immutable_version_id": reference_id, "reviewed_draft_version_id": body.version_id}

    register_anatomy_routes(app, db_session=db_session, signed_in=signed_in,
                            project_for=project_for, source_for=source_for,
                            editable=editable, audit=audit)
    return app
