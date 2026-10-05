"""Fetch a few CHD68 pairs from split ZIPs, verifying each original entry CRC.

The archive is shuffled: image/mask entries may be on different split disks.
Only entries wholly contained in one disk are supported. No training.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import zipfile
import zlib

import fetch_sample as split
from fetch_pediatric_metadata import fetch

DATASET = "xiaoweixumedicalai/chd68-segmentation-dataset-miccai19"


def entry_end(entry):
    return entry.header_offset + entry.compress_size + 65536


def extract_entry(data, entry):
    fields = struct.unpack_from("<4s5H3I2H", data, entry.header_offset)
    if fields[0] != b"PK\x03\x04" or fields[2] & 1 or fields[3] != entry.compress_type:
        raise ValueError("Unsupported local ZIP header")
    start = entry.header_offset + 30 + fields[-2] + fields[-1]
    compressed = data[start:start + entry.compress_size]
    if entry.compress_type not in (0, 8):
        raise ValueError("Unsupported ZIP compression")
    payload = zlib.decompress(compressed, -15) if entry.compress_type == 8 else compressed
    if len(payload) != entry.file_size or zlib.crc32(payload) != entry.CRC:
        raise ValueError("CHD68 original entry size/CRC mismatch")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--count", type=int, default=3, choices=range(1, 6))
    parser.add_argument("--max-prefix-mb", type=float, default=250)
    args = parser.parse_args()
    args.data.mkdir(parents=True, exist_ok=True)
    split.DATASET = DATASET
    listing_path = args.data / "kaggle_files.json"
    fetch(f"{split.BASE}list/{DATASET}", listing_path)
    listing = json.loads(listing_path.read_text())
    sizes = {i["name"]: i["totalBytes"] for i in listing["datasetFiles"]}
    last_name = "CHD68_segmentation_dataset_miccai19.change2zip"
    last = args.data / last_name
    if not last.exists():
        wrapper = args.data / "final_wrapper.zip"
        fetch(f"{split.BASE}download/{DATASET}/{last_name}", wrapper)
        with zipfile.ZipFile(wrapper) as z:
            last.write_bytes(z.read(last_name))
    if last.stat().st_size != sizes[last_name]:
        raise ValueError("Final split size differs from published inventory")
    entries = split.split_directory(last)
    final_disk = max(i.volume for i in entries)
    by_name = {i.filename: i for i in entries}
    part_names = {i: f"CHD68_segmentation_dataset_miccai19.z{i+1:02}" for i in range(final_disk)}
    part_names[final_disk] = last_name
    candidates = []
    for image in entries:
        if not image.filename.endswith("_image.nii.gz") or Path(image.filename).name.startswith("._"):
            continue
        mask = by_name.get(image.filename.replace("_image.nii.gz", "_label.nii.gz"))
        if mask is None or any(entry_end(i) >= sizes[part_names[i.volume]] for i in (image, mask)):
            continue
        cost = sum(max(entry_end(i) for i in (image, mask) if i.volume == disk)
                   for disk in {image.volume, mask.volume} if disk != final_disk)
        candidates.append((cost, image, mask))
    selected = sorted(candidates, key=lambda r: r[0])[:args.count]
    if len(selected) != args.count:
        raise ValueError("Not enough pairs wholly contained in single disks")
    chosen = [i for _, ct, mask in selected for i in (ct, mask)]
    # Already verified pairs need no network transfer on repeat runs.
    missing = [i for i in chosen if not (args.data / Path(i.filename).name).exists()]
    for i in chosen:
        target = args.data / Path(i.filename).name
        if target.exists() and (target.stat().st_size != i.file_size or zlib.crc32(target.read_bytes()) != i.CRC):
            raise ValueError(f"Existing file differs; refusing overwrite: {target}")
    disks = {i.volume for i in missing}
    required = {d: max(entry_end(i) for i in missing if i.volume == d) for d in disks if d != final_disk}
    if sum(required.values()) > args.max_prefix_mb * 1e6:
        raise ValueError("Prefix budget exceeded; inspect archive inventory before a larger download")
    data = {final_disk: last.read_bytes()}
    received = {}
    for disk, end in required.items():
        data[disk], received[disk] = split.part_prefix(part_names[disk], end, sizes[part_names[disk]])
    for i in missing:
        (args.data / Path(i.filename).name).write_bytes(extract_entry(data[i.volume], i))
    receipt = {"dataset": DATASET, "prefix_received_bytes": received,
               "method": "public Kaggle API; original ZIP entry CRC32 verified; no per-file age inferred",
               "files": [{"name": Path(i.filename).name, "disk": i.volume, "crc32": f"{i.CRC:08x}",
                   "sha256": hashlib.sha256((args.data / Path(i.filename).name).read_bytes()).hexdigest()} for i in chosen]}
    (args.data / "download_provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (args.data / "archive_inventory.json").write_text(json.dumps([vars(i) for i in entries], indent=2) + "\n")
    print("Verified", [Path(ct.filename).name for _, ct, _ in selected])


if __name__ == "__main__":
    main()
