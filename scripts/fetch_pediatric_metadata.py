"""Download public audit inputs. No credentials, DUA acceptance, images or training."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

BASE = "https://services.cancerimagingarchive.net/nbia-api/services/v1/"


def fetch(url, target, md5=None):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        partial = target.with_suffix(target.suffix + ".part")
        with urllib.request.urlopen(url, timeout=90) as r, partial.open("wb") as f:
            while block := r.read(1 << 20):
                f.write(block)
        if md5 and hashlib.md5(partial.read_bytes()).hexdigest() != md5:
            raise ValueError(f"Checksum mismatch: {target.name}")
        partial.replace(target)
    digest = hashlib.md5(target.read_bytes()).hexdigest()
    if md5 and digest != md5:
        raise ValueError(f"Existing file checksum mismatch; preserve and inspect: {target}")
    return {"name": target.name, "url": url, "bytes": target.stat().st_size,
            "md5": digest, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}


def extract_safe(archive, destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            path = (destination / item.filename).resolve()
            if not path.is_relative_to(destination) or (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Unsafe ZIP entry")
            if path.exists() and not item.is_dir() and path.read_bytes() != z.read(item):
                raise ValueError(f"Refuse to overwrite: {path}")
        z.extractall(destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="External/ignored data directory")
    parser.add_argument("--atlas", action="store_true", help="Also download public Zenodo atlas (~77 MB)")
    args = parser.parse_args()
    args.data.mkdir(parents=True, exist_ok=True)
    receipt = []
    for file_id, name, md5 in [(44566088, "hvsmr_clinical.csv", "b280b832e209f37e20b611449bb6a789"),
                               (44566091, "hvsmr_technical.csv", "87a96a29132bba80644446c692ad388b")]:
        receipt.append(fetch(f"https://ndownloader.figshare.com/files/{file_id}", args.data / name, md5))
    files = {
        "chd68_source_readme.md": "https://raw.githubusercontent.com/XiaoweiXu/Whole-heart-and-great-vessel-segmentation-of-chd_segmentation/master/README.md",
        "tcia_series.txt": BASE + "getSeries?Collection=Pediatric-CT-SEG&format=json",
        "tcia_patients.txt": BASE + "getPatient?Collection=Pediatric-CT-SEG&format=json",
        "pediatric_ct_manifest.tcia": "https://www.cancerimagingarchive.net/wp-content/uploads/Pediatric-CT-SEG-Mar-22-2022-manifest.tcia",
        "pediatric_ct_digest.xlsx": "https://www.cancerimagingarchive.net/wp-content/uploads/Pediatric-CT-SEG-Mar-22-2022-manifes-nbia-digest.xlsx",
    }
    for name, url in files.items():
        receipt.append(fetch(url, args.data / name))
    if args.atlas:
        receipt.append(fetch("https://zenodo.org/api/records/17554213", args.data / "zenodo_record.json"))
        record = json.loads((args.data / "zenodo_record.json").read_text())
        for item in record["files"]:
            path = args.data / Path(item["key"]).name
            receipt.append(fetch(item["links"]["self"], path, item["checksum"].split(":")[1]))
            extract_safe(path, args.data / "atlas")
    (args.data / "metadata_download_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("Downloaded/verified public audit inputs. No CT/MRI training was started.")


if __name__ == "__main__":
    main()
