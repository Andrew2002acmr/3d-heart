"""Select a 9-case DICOM pilot, covering represented age/scanner cells."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from heart3d.dicom.tcia import fetch_full_series
from heart3d.pediatric import write_json
from heart3d.storage import require_space


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--registry", type=Path, required=True)
    p.add_argument("--series", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--reserve-gb", type=float, default=80, help="Minimum free decimal GB during download/extraction")
    args = p.parse_args()
    by = {r["patient_id"]: r for r in json.loads(args.registry.read_text(encoding="utf-8"))["records"]
          if r["dataset"] == "Pediatric-CT-SEG" and r["eligible_by_reported_age"] is True}
    series = json.loads(args.series.read_text())
    ct = [s for s in series if s["Modality"] == "CT" and s["PatientID"] in by]
    selected = []
    groups = sorted({(by[s["PatientID"]]["age_group"], s["ManufacturerModelName"]) for s in ct})
    for group in groups:
        choices = sorted([s for s in ct if (by[s["PatientID"]]["age_group"], s["ManufacturerModelName"]) == group],
                         key=lambda s: (s["FileSize"], s["PatientID"]))
        selected.append(choices[len(choices) // 2])
    known = next(s for s in ct if s["PatientID"] == "Pediatric-CT-SEG-E03568A6")
    if known not in selected:
        selected.append(known)
    manifest = []
    for s in selected:
        pid = s["PatientID"]
        rt = [r for r in series if r["PatientID"] == pid and r["Modality"] == "RTSTRUCT"]
        if len(rt) != 1:
            raise ValueError("RT pilot identity ambiguous")
        manifest.append({"patient_id": pid, "age": by[pid]["age"], "age_group": by[pid]["age_group"],
                         "scanner": s["ManufacturerModelName"], "ct_series_uid": s["SeriesInstanceUID"],
                         "rt_series_uid": rt[0]["SeriesInstanceUID"], "expected_slices": s["ImageCount"],
                         "published_CT_bytes": s["FileSize"], "published_RT_bytes": rt[0]["FileSize"]})
    reserve_bytes = int(args.reserve_gb * 10**9)
    # Includes source archive + extracted DICOM, with a small ZIP overhead allowance.
    planned_bytes = int(2.1 * sum(r['published_CT_bytes'] + r['published_RT_bytes'] for r in manifest))
    require_space(args.data, planned_bytes, reserve_bytes)
    print('Storage preflight: conservative pilot bytes', planned_bytes, 'reserve', reserve_bytes, flush=True)
    write_json(args.data / "pilot_selection.json", {"method": "median-size CT per represented age/scanner cell + previously probed case; no outcome selection", "cases": manifest})
    for index, row in enumerate(manifest):
        pid = row["patient_id"]
        print("PILOT", index + 1, "/", len(manifest), pid, row["age"], row["scanner"], flush=True)
        destination = args.data / pid
        ct_receipt = fetch_full_series(row["ct_series_uid"], destination / "ct", reserve_bytes=reserve_bytes)
        rt_receipt = fetch_full_series(row["rt_series_uid"], destination / "rtstruct", reserve_bytes=reserve_bytes)
        write_json(destination / "full_download_receipt.json", {"patient_id": pid, "CT": ct_receipt, "RTSTRUCT": rt_receipt})
        print("FULL CT + RTSTRUCT saved", pid, flush=True)


if __name__ == "__main__":
    main()
