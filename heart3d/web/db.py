from datetime import datetime, timezone
from pathlib import Path
import uuid

from sqlalchemy import JSON, Boolean, Column, DDL, ForeignKey, Integer, String, Text, UniqueConstraint, create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker


def uid():
    return str(uuid.uuid4())


def now():
    return datetime.now(timezone.utc).isoformat()


class Base(DeclarativeBase):
    pass


class Role(Base):
    __tablename__ = "roles"
    name = Column(String, primary_key=True)


class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, default=uid)
    username = Column(String, unique=True, nullable=False)
    password_hash = Column(Text, nullable=False)
    role = Column(ForeignKey("roles.name"), nullable=False)
    active = Column(Boolean, default=True, nullable=False)


class Patient(Base):
    __tablename__ = "patients"
    id = Column(String, primary_key=True, default=uid)
    pseudonym = Column(String, unique=True, nullable=False)


class Study(Base):
    __tablename__ = "studies"
    id = Column(String, primary_key=True, default=uid)
    patient_id = Column(ForeignKey("patients.id"), nullable=False)
    pseudonym = Column(String, unique=True, nullable=False)
    synthetic = Column(Boolean, default=True, nullable=False)


class Series(Base):
    __tablename__ = "series"
    id = Column(String, primary_key=True, default=uid)
    study_id = Column(ForeignKey("studies.id"), nullable=False)
    ct_path = Column(Text, nullable=False)
    ct_sha256 = Column(String, nullable=False)
    geometry = Column(JSON, nullable=False)


class AnnotationProject(Base):
    __tablename__ = "annotation_projects"
    id = Column(String, primary_key=True, default=uid)
    series_id = Column(ForeignKey("series.id"), nullable=False)
    status = Column(String, default="not_started", nullable=False)
    label_schema = Column(String, default="cardiac7-v1", nullable=False)
    head_version_id = Column(String, nullable=True)
    revision = Column(Integer, default=0, nullable=False)
    updated_at = Column(String, default=now, nullable=False)


class Assignment(Base):
    __tablename__ = "assignments"
    __table_args__ = (UniqueConstraint("project_id", "user_id"),)
    id = Column(String, primary_key=True, default=uid)
    project_id = Column(ForeignKey("annotation_projects.id"), nullable=False)
    user_id = Column(ForeignKey("users.id"), nullable=False)


class AnnotationVersion(Base):
    __tablename__ = "annotation_versions"
    __table_args__ = (UniqueConstraint("project_id", "number"),)
    id = Column(String, primary_key=True, default=uid)
    project_id = Column(ForeignKey("annotation_projects.id"), nullable=False)
    parent_id = Column(ForeignKey("annotation_versions.id"), nullable=True)
    number = Column(Integer, nullable=False)
    author_id = Column(ForeignKey("users.id"), nullable=False)
    created_at = Column(String, default=now, nullable=False)
    ct_sha256 = Column(String, nullable=False)
    label_schema = Column(String, nullable=False)
    source = Column(String, default="manual", nullable=False)
    document = Column(JSON, nullable=False)
    mask_path = Column(Text, nullable=False)
    mask_sha256 = Column(String, nullable=False)
    geometry_verified = Column(Boolean, default=True, nullable=False)


class AnnotationOperation(Base):
    __tablename__ = "annotation_operations"
    id = Column(String, primary_key=True, default=uid)
    version_id = Column(ForeignKey("annotation_versions.id"), nullable=False)
    position = Column(Integer, nullable=False)
    kind = Column(String, nullable=False)
    payload = Column(JSON, nullable=False)


class AnatomyNotebookVersion(Base):
    """Append-only CT observation notebook, independent of annotation masks."""
    __tablename__ = "anatomy_notebook_versions"
    __table_args__ = (UniqueConstraint("project_id", "number"),)
    id = Column(String, primary_key=True, default=uid)
    project_id = Column(ForeignKey("annotation_projects.id"), nullable=False)
    number = Column(Integer, nullable=False)
    author_id = Column(ForeignKey("users.id"), nullable=False)
    created_at = Column(String, default=now, nullable=False)
    ct_sha256 = Column(String, nullable=False)
    document = Column(JSON, nullable=False)


for action in ("UPDATE", "DELETE"):
    event.listen(AnatomyNotebookVersion.__table__, "after_create", DDL(
        f"CREATE TRIGGER anatomy_notebook_no_{action.lower()} BEFORE {action} ON anatomy_notebook_versions "
        "BEGIN SELECT RAISE(ABORT, 'Anatomy notebook versions are immutable'); END"
    ).execute_if(dialect="sqlite"))


class Review(Base):
    __tablename__ = "reviews"
    id = Column(String, primary_key=True, default=uid)
    version_id = Column(ForeignKey("annotation_versions.id"), nullable=False)
    surgeon_id = Column(ForeignKey("users.id"), nullable=False)
    decision = Column(String, nullable=False)
    note = Column(Text, default="", nullable=False)
    created_at = Column(String, default=now, nullable=False)


class Approval(Base):
    __tablename__ = "approvals"
    id = Column(String, primary_key=True, default=uid)
    version_id = Column(ForeignKey("annotation_versions.id"), unique=True, nullable=False)
    surgeon_id = Column(ForeignKey("users.id"), nullable=False)
    created_at = Column(String, default=now, nullable=False)


class Session(Base):
    __tablename__ = "sessions"
    token_hash = Column(String, primary_key=True)
    user_id = Column(ForeignKey("users.id"), nullable=False)
    csrf = Column(String, nullable=False)
    expires_at = Column(String, nullable=False)


class AuditEvent(Base):
    __tablename__ = "audit_events"
    id = Column(String, primary_key=True, default=uid)
    actor_id = Column(String, nullable=True)
    action = Column(String, nullable=False)
    project_id = Column(String, nullable=True)
    created_at = Column(String, default=now, nullable=False)


def database(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    engine = create_engine("sqlite:///" + str((root / "annotation.sqlite3").resolve()), connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def configure(connection, _):
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=5000")

    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False)
    with factory.begin() as db:
        for name in ("admin", "resident", "surgeon"):
            if not db.get(Role, name):
                db.add(Role(name=name))
    return engine, factory
