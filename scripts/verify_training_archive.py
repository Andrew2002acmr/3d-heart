"""Verify compressed archive and every allowlisted payload without a second data copy."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tarfile
from heart3d.ml.bundle import safe_relative
from heart3d.storage import sha256_file


def verify_archive(archive,receipt):
    archive=Path(archive);receipt=json.loads(Path(receipt).read_text(encoding="utf-8"))
    if archive.stat().st_size!=receipt["archive_bytes"] or sha256_file(archive)!=receipt["archive_SHA256"]:
        raise ValueError("Archive SHA/size mismatch")
    seen=set()
    with archive.open("rb") as source,gzip.GzipFile(fileobj=source) as decompressed,tarfile.open(fileobj=decompressed,mode="r|") as tar:
        members=iter(tar);first=next(members)
        if first.name!="bundle_manifest.json" or not first.isfile():raise ValueError("Manifest must be first regular entry")
        payload=tar.extractfile(first).read()
        if hashlib.sha256(payload).hexdigest()!=receipt["manifest_SHA256"]:raise ValueError("Manifest SHA mismatch")
        manifest=json.loads(payload);allowed={safe_relative(r["path"]):r for r in manifest["files"]}
        if len(allowed)!=len(manifest["files"]):raise ValueError("Duplicate manifest paths")
        for member in members:
            name=safe_relative(member.name)
            if not member.isfile() or name not in allowed or name in seen:raise ValueError("Unexpected/duplicate archive entry")
            row=allowed[name]
            if member.size!=row["bytes"]:raise ValueError("Payload size differs")
            digest=hashlib.sha256()
            with tar.extractfile(member) as stream:
                for block in iter(lambda:stream.read(4<<20),b""):digest.update(block)
            if digest.hexdigest()!=row["SHA256"]:raise ValueError("Payload SHA mismatch")
            seen.add(name)
            if len(seen)%50==0:print("ARCHIVE_PAYLOADS_VERIFIED",len(seen),flush=True)
        # Consume gzip trailer as well, validating the compressed stream CRC.
        while decompressed.read(4<<20):pass
    if seen!=set(allowed):raise ValueError("Missing payloads")
    print("ARCHIVE_ALL_SHA256_VERIFIED",len(seen),flush=True)
    return {"archive_SHA256":receipt["archive_SHA256"],"payload_files_verified":len(seen),"gzip_CRC_verified":True}

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--archive",type=Path,required=True);p.add_argument("--receipt",type=Path,required=True)
    a=p.parse_args();verify_archive(a.archive,a.receipt)
