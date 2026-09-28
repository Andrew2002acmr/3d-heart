"""Fetch one ImageCHD pair from the public split ZIP, without the full 9 GB.

Kaggle wraps each split part in another ZIP. We download the small final part
for its central directory, then stream only the required prefix of part z01.
Only entries wholly in disk zero are supported. Original entry CRCs are checked.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import struct
import urllib.request
import zipfile
import zlib
from types import SimpleNamespace

BASE = "https://www.kaggle.com/api/v1/datasets/"
DATASET = "xiaoweixumedicalai/imagechd"


def request(url):
    return urllib.request.urlopen(url, timeout=90)


def read_exact(stream, size):
    result = bytearray()
    while len(result) < size:
        chunk = stream.read(size - len(result))
        if not chunk:
            raise RuntimeError("Unexpected end of download")
        result.extend(chunk)
    return bytes(result)


def split_directory(path):
    """Read this dataset's central directory; offsets are relative to each disk."""
    data = path.read_bytes()
    end = data.rfind(b"PK\x05\x06")
    if end < 0:
        raise RuntimeError("ZIP directory end missing")
    _, disk, cd_disk, _, count, size, offset, _ = struct.unpack_from("<4s4H2IH", data, end)
    if disk != cd_disk or offset + size > len(data):
        raise RuntimeError("Central directory spans disks; use 7-Zip")
    entries = []
    position = offset
    for _ in range(count):
        f = struct.unpack_from("<4s6H3I5H2I", data, position)
        if f[0] != b"PK\x01\x02" or max(f[8], f[9], f[16]) == 0xFFFFFFFF:
            raise RuntimeError("Unsupported central entry; use full archive and 7-Zip")
        name = data[position + 46:position + 46 + f[10]].decode("utf-8" if f[3] & 2048 else "cp437")
        entries.append(SimpleNamespace(filename=name, volume=f[13], header_offset=f[16],
                                       compress_size=f[8], file_size=f[9], CRC=f[7], compress_type=f[4]))
        position += 46 + f[10] + f[11] + f[12]
    if position != offset + size:
        raise RuntimeError("Central directory size mismatch")
    return entries


def part_prefix(name, required, expected_size):
    """Decode the outer ZIP's first entry, stopping after required inner bytes."""
    with request(f"{BASE}download/{DATASET}/{name}") as stream:
        header = struct.unpack("<4s5H3I2H", read_exact(stream, 30))
        if header[0] != b"PK\x03\x04" or header[2] & 1:
            raise RuntimeError("Expected an unencrypted ZIP wrapper")
        filename = read_exact(stream, header[-2]).decode("utf-8")
        read_exact(stream, header[-1])
        if filename != name or header[3] not in (0, 8):
            raise RuntimeError(f"Unsupported wrapper: {filename}, method {header[3]}")
        decoder = zlib.decompressobj(-15) if header[3] == 8 else None
        data = bytearray()
        downloaded = 0
        next_progress = 20_000_000
        while len(data) < required:
            block = stream.read(1 << 20)
            if not block:
                raise RuntimeError("Archive ended before selected files")
            downloaded += len(block)
            decoded = decoder.decompress(block) if decoder else block
            data.extend(decoded)
            if len(data) > expected_size:
                raise RuntimeError("Part exceeds published size")
            if downloaded >= next_progress:
                print(f"  received {downloaded / 1e6:.0f} MB", flush=True)
                next_progress += 20_000_000
        return data, downloaded


def fetch(destination):
    destination.mkdir(parents=True, exist_ok=True)
    with request(f"{BASE}list/{DATASET}") as response:
        listing = json.load(response)
    (destination / "kaggle_files.json").write_text(json.dumps(listing, indent=2), encoding="utf-8")
    sizes = {item["name"]: item["totalBytes"] for item in listing["datasetFiles"]}
    final_name = "ImageCHD_dataset.change2zip"
    cache = destination / final_name
    if not cache.exists() or cache.stat().st_size != sizes[final_name]:
        print("Downloading final split part (129 MB) to read archive directory...", flush=True)
        with request(f"{BASE}download/{DATASET}/{final_name}") as response:
            wrapper = response.read()
        with zipfile.ZipFile(io.BytesIO(wrapper)) as outer:
            payload = outer.read(final_name)  # also verifies the outer CRC
        if len(payload) != sizes[final_name]:
            raise RuntimeError("Final split part size mismatch")
        cache.write_bytes(payload)
    entries = split_directory(cache)
    final_payload = cache.read_bytes()
    for entry in entries:
        if entry.volume == 12 and entry.filename.endswith(".xlsx"):
            fields = struct.unpack_from("<4s5H3I2H", final_payload, entry.header_offset)
            start = entry.header_offset + 30 + fields[-2] + fields[-1]
            payload = zlib.decompress(final_payload[start:start + entry.compress_size], -15)
            if len(payload) != entry.file_size or zlib.crc32(payload) != entry.CRC:
                raise RuntimeError("Metadata table CRC mismatch")
            (destination / Path(entry.filename).name).write_bytes(payload)
    catalog = [{"name": i.filename, "disk": i.volume, "offset": i.header_offset,
                "compressed_bytes": i.compress_size, "bytes": i.file_size,
                "crc32": f"{i.CRC:08x}"} for i in entries]
    (destination / "archive_inventory.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    by_name = {i.filename: i for i in entries}
    part_size = sizes["ImageCHD_dataset.z01"]
    candidates = []
    for entry in entries:
        if not entry.filename.endswith("_image.nii.gz"):
            continue
        label = by_name.get(entry.filename.replace("_image.nii.gz", "_label.nii.gz"))
        if label and entry.volume == label.volume == 0:
            # Account for local header, filename and extra fields conservatively.
            end = max(i.header_offset + i.compress_size + 65536 for i in (entry, label))
            if end < part_size:
                candidates.append((end, entry, label))
    if not candidates:
        raise RuntimeError("No complete pair in first disk. Use full download and 7-Zip.")
    required, ct, mask = min(candidates, key=lambda row: row[0])
    if all((destination / Path(i.filename).name).exists() and
           zlib.crc32((destination / Path(i.filename).name).read_bytes()) == i.CRC for i in (ct, mask)):
        print("Sample pair already downloaded and CRC verified.")
        return
    print(f"Selected {ct.filename} + {mask.filename}; reading {required / 1e6:.1f} MB prefix", flush=True)
    prefix, network_bytes = part_prefix("ImageCHD_dataset.z01", required, part_size)
    provenance = {"dataset": f"https://www.kaggle.com/datasets/{DATASET}",
                  "method": "Public Kaggle API; final part + prefix of z01; original ZIP entry CRC32 verified",
                  "z01_download_bytes": network_bytes, "files": []}
    for entry in (ct, mask):
        position = entry.header_offset
        fields = struct.unpack_from("<4s5H3I2H", prefix, position)
        if fields[0] != b"PK\x03\x04" or fields[2] & 1:
            raise RuntimeError("Invalid local ZIP header")
        start = position + 30 + fields[-2] + fields[-1]
        compressed = prefix[start:start + entry.compress_size]
        if entry.compress_type == 8:
            payload = zlib.decompress(compressed, -15)
        elif entry.compress_type == 0:
            payload = bytes(compressed)
        else:
            raise RuntimeError(f"Unsupported ZIP method: {entry.compress_type}")
        if len(payload) != entry.file_size or zlib.crc32(payload) != entry.CRC:
            raise RuntimeError(f"CRC/size mismatch: {entry.filename}")
        target = destination / Path(entry.filename).name
        target.write_bytes(payload)
        provenance["files"].append({"archive_name": entry.filename, "local_name": target.name,
                                    "sha256": hashlib.sha256(payload).hexdigest(),
                                    "crc32": f"{entry.CRC:08x}", "bytes": len(payload)})
        print(f"Saved {target} ({len(payload) / 1e6:.1f} MB); CRC OK", flush=True)
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/imagechd"))
    args = parser.parse_args()
    fetch(args.out)
