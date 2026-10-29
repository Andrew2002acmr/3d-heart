"""Local owner administration. Passwords use a prompt or stdin, never CLI flags."""
import argparse
from getpass import getpass
import hashlib
import json
import re
import shutil
import sqlite3
import sys
from pathlib import Path

from sqlalchemy import select, delete
from .app import PASSWORDS, data_root
from .db import database, User, Patient, Study, Series, AnnotationProject, Assignment, Session
from .imaging import phantom
from heart3d.storage import sha256_file


def create_user(factory, username, password, role="resident"):
    if not re.fullmatch(r"[a-zA-Z0-9_.-]{3,64}", username) or len(password) < 12:
        raise ValueError("Username: 3..64 ASCII letters/numbers/_.-; password: at least 12 characters")
    with factory.begin() as db:
        if db.scalar(select(User).where(User.username == username)):
            raise ValueError("User exists; use reset-password")
        user = User(username=username, password_hash=PASSWORDS.hash(password), role=role)
        db.add(user)
        db.flush()
        return user.id


def seed_project(root, factory, user_id, suffix="001"):
    root = Path(root)
    relative = f"raw/synthetic_{suffix}.nii.gz"
    grid = phantom(root / relative)
    with factory.begin() as db:
        patient = Patient(pseudonym="P-SYN-"+suffix)
        db.add(patient); db.flush()
        study = Study(patient_id=patient.id, pseudonym="S-SYN-"+suffix, synthetic=True)
        db.add(study); db.flush()
        series = Series(study_id=study.id, ct_path=relative, ct_sha256=sha256_file(root / relative), geometry=grid)
        db.add(series); db.flush()
        project = AnnotationProject(series_id=series.id)
        db.add(project); db.flush()
        db.add(Assignment(project_id=project.id, user_id=user_id))
        return project.id


def backup(root, destination):
    root, destination = Path(root), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    with sqlite3.connect(root / "annotation.sqlite3") as source, sqlite3.connect(destination / "annotation.sqlite3") as target:
        source.backup(target)
    # Version files and CT are immutable; a DB-first snapshot cannot reference future files.
    for name in ("raw", "annotations"):
        if (root / name).exists():
            shutil.copytree(root / name, destination / name)
    entries = [{"path": p.relative_to(destination).as_posix(), "SHA256": sha256_file(p)}
               for p in destination.rglob("*") if p.is_file()]
    (destination / "backup_manifest.json").write_text(json.dumps(entries, indent=2)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=data_root())
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "create-user", "reset-password"):
        p = commands.add_parser(name)
        p.add_argument("--username", required=True)
        p.add_argument("--password-stdin", action="store_true")
        if name == "create-user":
            p.add_argument("--role", choices=("resident", "admin", "surgeon"), default="resident")
    p = commands.add_parser("assign")
    p.add_argument("--username", required=True)
    p.add_argument("--project", required=True)
    p = commands.add_parser("backup")
    p.add_argument("--out", type=Path, required=True)
    p = commands.add_parser("import-experiment")
    p.add_argument("--username", required=True)
    p.add_argument("--ct", type=Path, required=True)
    p.add_argument("--admission", type=Path, required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--v1", type=Path, required=True, help="Model output directory with prediction/provenance")
    p.add_argument("--v2", type=Path, required=True)
    p.add_argument("--alias", default="C-001")
    p = commands.add_parser("import-slice-draft", help="Append one exploratory axial plane to an existing assigned project")
    p.add_argument("--username", required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--out", type=Path, required=True, help="Directory containing prediction.nii.gz and provenance.json")
    p = commands.add_parser("import-anatomy-bundle", help="Append an evidence-linked unapproved anatomy slice candidate")
    p.add_argument("--username", required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--bundle", type=Path, required=True, help="Verified anatomy bundle containing gzip mask and saved notebook export")
    args = parser.parse_args()
    engine, factory = database(args.root)
    if args.command in ("init", "create-user", "reset-password"):
        password = sys.stdin.readline().rstrip("\r\n") if args.password_stdin else getpass("New password (12+ characters): ")
        if args.command == "reset-password":
            if len(password) < 12:
                parser.error("Password must contain at least 12 characters")
            with factory.begin() as db:
                user = db.scalar(select(User).where(User.username == args.username))
                if not user:
                    parser.error("Unknown user")
                user.password_hash = PASSWORDS.hash(password)
                db.execute(delete(Session).where(Session.user_id == user.id))
        else:
            user_id = create_user(factory, args.username, password, getattr(args, "role", "resident"))
            if args.command == "init":
                print("Synthetic project:", seed_project(args.root, factory, user_id))
        print("User ready:", args.username)
    elif args.command == "assign":
        with factory.begin() as db:
            user = db.scalar(select(User).where(User.username == args.username))
            if not user or not db.get(AnnotationProject, args.project):
                parser.error("Unknown user/project")
            db.add(Assignment(project_id=args.project, user_id=user.id))
        print("Assignment created")
    elif args.command == "backup":
        backup(args.root, args.out)
        print("Backup created")
    elif args.command == "import-experiment":
        from heart3d.ml.clinical_experiment import admit
        from .predictions import import_predictions
        admission = json.loads(args.admission.read_text(encoding="utf-8-sig"))
        adapter = json.loads(args.adapter.read_text(encoding="utf-8"))
        admit(args.ct, admission, adapter)
        predictions = [(p / "prediction.nii.gz", json.loads((p / "provenance.json").read_text(encoding="utf-8")))
                       for p in (args.v1, args.v2)]
        print("Imported unapproved AI versions for project:", import_predictions(args.root, factory, args.username, args.ct, predictions, args.alias))
    elif args.command == "import-slice-draft":
        from .predictions import append_slice_draft
        provenance = json.loads((args.out / "provenance.json").read_text(encoding="utf-8-sig"))
        version = append_slice_draft(args.root, factory, args.username, args.project,
                                     args.out / "prediction.nii.gz", provenance)
        print("Appended unapproved single-slice version (head unchanged):", version)
    elif args.command == "import-anatomy-bundle":
        from .anatomy_predictions import append_anatomy_bundle
        version = append_anatomy_bundle(args.root, factory, args.username, args.project, args.bundle)
        print("Appended unapproved anatomy slice version (head unchanged):", version)
    engine.dispose()


if __name__ == "__main__":
    main()
