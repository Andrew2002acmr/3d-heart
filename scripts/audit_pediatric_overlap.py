"""Compare public archive inventories without assuming independent cohorts."""
import argparse
import json
from pathlib import Path


def inventory(path):
    rows = json.loads(Path(path).read_text())
    result = {}
    for r in rows:
        name = Path(r.get("name", r.get("filename"))).name
        if name.startswith("._") or not name.endswith(("_image.nii.gz", "_label.nii.gz")):
            continue
        size = r["bytes"] if "bytes" in r else r["file_size"]
        crc = int(r["crc32"], 16) if "crc32" in r else r["CRC"]
        result[name] = (size, crc)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagechd-inventory", type=Path, required=True)
    parser.add_argument("--chd68-inventory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    a, b = inventory(args.imagechd_inventory), inventory(args.chd68_inventory)
    shared = sorted(set(a) & set(b))
    equal = [name for name in shared if a[name] == b[name]]
    complete = [name.split("_image")[0] for name in equal if name.endswith("_image.nii.gz")
                and name.replace("_image.nii.gz", "_label.nii.gz") in equal]
    summary = {"shared_file_names": len(shared), "matching_size_crc32": len(equal),
               "matching_complete_pairs": len(complete), "matching_files": equal,
               "note": "Size+CRC is strong file overlap evidence, not cryptographic proof of patient linkage. Group potential duplicates before train/test split."}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print("Matching complete pairs:", len(complete))


if __name__ == "__main__":
    main()
