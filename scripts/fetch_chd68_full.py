"""Bounded public CHD68 split-ZIP download and CRC/SHA-verified extraction.

All large files live in CLI directories. No credentials, no source deletion.
Central offsets are disk-relative; entries may span multiple disks.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import struct
import urllib.request
import zipfile
import zlib

from scripts.fetch_sample import split_directory

BASE = "https://www.kaggle.com/api/v1/datasets/"
DATASET = "xiaoweixumedicalai/chd68-segmentation-dataset-miccai19"

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()

def ensure_space(path, required, reserve):
    path = Path(path)
    while not path.exists():
        path = path.parent
    if shutil.disk_usage(path).free < required + reserve:
        raise ValueError("Storage budget would violate free-space reserve")

def span(parts, disk, offset, size):
    """Yield exactly size bytes, continuing on the next split disk."""
    while size:
        if disk >= len(parts):
            raise ValueError("Entry runs past final disk")
        with Path(parts[disk]).open("rb") as stream:
            stream.seek(offset)
            while size:
                block = stream.read(min(size, 1 << 20))
                if not block:
                    break
                size -= len(block)
                yield block
        disk += 1
        offset = 0

def extract(parts, entry, destination):
    """Validate original compressed-entry CRC; never overwrite an existing file."""
    destination = Path(destination)
    header = b"".join(span(parts, entry.volume, entry.header_offset, 30))
    fields = struct.unpack("<4s5H3I2H", header)
    if fields[0] != b"PK\x03\x04" or fields[2] & 1 or fields[3] != entry.compress_type:
        raise ValueError("Unsupported local ZIP header")
    if entry.compress_type not in (0, 8):
        raise ValueError("Unsupported compression")
    position = entry.header_offset + 30 + fields[-2] + fields[-1]
    disk = entry.volume
    while position >= Path(parts[disk]).stat().st_size:
        position -= Path(parts[disk]).stat().st_size
        disk += 1
    decoder = zlib.decompressobj(-15) if entry.compress_type == 8 else None
    crc, total, h = 0, 0, hashlib.sha256()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists():
        raise ValueError("Incomplete extraction exists; preserve it and choose another directory")
    stream = None if destination.exists() else temporary.open("xb")
    try:
        for block in span(parts, disk, position, entry.compress_size):
            payload = decoder.decompress(block) if decoder else block
            total += len(payload)
            crc = zlib.crc32(payload, crc)
            h.update(payload)
            if stream:
                stream.write(payload)
        tail = decoder.flush() if decoder else b""
        total += len(tail); crc = zlib.crc32(tail, crc); h.update(tail)
        if stream:
            stream.write(tail)
        if total != entry.file_size or crc != entry.CRC or (decoder and not decoder.eof):
            raise ValueError("Original entry CRC/size/stream mismatch")
    finally:
        if stream:
            stream.close()
    sha = h.hexdigest()
    if destination.exists():
        if destination.stat().st_size != total or digest(destination) != sha:
            raise ValueError("Existing source differs; refusing overwrite")
    else:
        temporary.rename(destination)
    return {"name": destination.name, "bytes": total, "sha256": sha, "crc32": f"{crc:08x}"}

def download_part(cache, item, reserve):
    name, size = item["name"], item["totalBytes"]
    target, wrapper = cache / name, cache / (name + ".wrapper.zip")
    ensure_space(cache, size * 2 + 2_000_000, reserve)
    if not wrapper.exists():
        partial = wrapper.with_suffix(".download")
        if partial.exists():
            raise ValueError("Incomplete download retained; choose a new cache or inspect it")
        print("DOWNLOAD", name, f"{size/1e9:.3f} GB", flush=True)
        request = urllib.request.Request(BASE + "download/" + DATASET + "/" + name,
                                         headers={"User-Agent": "heart3d-public-data-audit/1"})
        received = 0
        with urllib.request.urlopen(request, timeout=90) as response, partial.open("xb") as out:
            for block in iter(lambda: response.read(4 << 20), b""):
                received += len(block)
                if received > size + 5_000_000:
                    raise ValueError("Wrapper download exceeds published part budget")
                out.write(block)
        partial.rename(wrapper)
    with zipfile.ZipFile(wrapper) as archive:
        entry = archive.getinfo(name)
        if entry.file_size != size:
            raise ValueError("Outer ZIP size differs from public inventory")
        if not target.exists():
            partial = target.with_name(target.name + ".partial")
            if partial.exists():
                raise ValueError("Incomplete split part retained")
            with archive.open(entry) as src, partial.open("xb") as dst:
                shutil.copyfileobj(src, dst, 4 << 20)  # also verifies outer CRC
            partial.rename(target)
        elif target.stat().st_size != size:
            raise ValueError("Existing split part has wrong size")
    result = {"name": name, "bytes": size, "sha256": digest(target)}
    print("VERIFIED_PART", name, flush=True)
    return result

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--max-download-GB", type=float, default=6)
    p.add_argument("--storage-budget-GB", type=float, default=18)
    p.add_argument("--reserve-GB", type=float, default=80)
    p.add_argument("--inspect-only", action="store_true")
    a = p.parse_args()
    request = urllib.request.Request(BASE + "list/" + DATASET, headers={"User-Agent":"heart3d-public-data-audit/1"})
    with urllib.request.urlopen(request, timeout=90) as response:
        listing = json.load(response)
    items = sorted(listing["datasetFiles"], key=lambda i: (not i["name"].endswith("change2zip"), i["name"]))
    cache, pairs = a.root / "cache", a.root / "pairs"
    remaining = sum(i["totalBytes"] + 5_000_000 for i in items if not (cache/(i["name"]+".wrapper.zip")).exists())
    plan = {"dataset":DATASET, "published_inner_bytes":sum(i["totalBytes"] for i in items),
            "remaining_wrapper_upper_bytes":remaining, "storage_budget_GB":a.storage_budget_GB,
            "reserve_GB":a.reserve_GB, "free_bytes":shutil.disk_usage(a.root.parent if a.root.parent.exists() else a.root.anchor).free}
    print(json.dumps(plan), flush=True)
    if remaining > a.max_download_GB*1e9:
        raise ValueError("Download budget exceeded")
    ensure_space(a.root, int(a.storage_budget_GB*1e9), int(a.reserve_GB*1e9))
    if a.inspect_only:
        return
    cache.mkdir(parents=True, exist_ok=True)
    (a.root/"kaggle_files.json").write_text(json.dumps(listing,indent=2)+"\n",encoding="utf-8")
    with ThreadPoolExecutor(max_workers=2) as pool:
        receipts = list(pool.map(lambda i:download_part(cache,i,int(a.reserve_GB*1e9)),items))
    final = next(cache/i["name"] for i in items if i["name"].endswith("change2zip"))
    entries = split_directory(final)
    count = max(e.volume for e in entries)
    parts = [cache/f"CHD68_segmentation_dataset_miccai19.z{i+1:02}" for i in range(count)]+[final]
    selected = [e for e in entries if Path(e.filename).name.endswith(("_image.nii.gz","_label.nii.gz"))
                and not Path(e.filename).name.startswith("._")]
    names = [Path(e.filename).name for e in selected]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate archive basenames require explicit handling")
    files=[]
    for e in selected:
        files.append(extract(parts,e,pairs/Path(e.filename).name))
        print("EXTRACTED",Path(e.filename).name,flush=True)
    (a.root/"download_provenance.json").write_text(json.dumps(
        {"dataset":DATASET,"parts":receipts,"files":files,"original_entry_CRC_verified":True},indent=2)+"\n",encoding="utf-8")

if __name__ == "__main__":
    main()
