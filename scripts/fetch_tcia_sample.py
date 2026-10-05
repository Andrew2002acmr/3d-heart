"""Inspect one public CT slice + RTSTRUCT. This is not a DICOM stack converter."""
import argparse
import json
from pathlib import Path
import urllib.request
import pydicom
from fetch_pediatric_metadata import BASE, fetch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--patient", default="Pediatric-CT-SEG-E03568A6")
    args = parser.parse_args()
    series = json.loads((args.data / "tcia_series.txt").read_text())
    related = [r for r in series if r["PatientID"] == args.patient]
    if {r["Modality"] for r in related} != {"CT", "RTSTRUCT"}:
        raise ValueError("Expected CT and RTSTRUCT series for public patient")
    out = args.data / "tcia_sample" / args.patient
    summary = {"patient_id": args.patient, "full_CT_series_downloaded": False,
               "DICOM_pipeline_complete": False, "series": []}
    for row in related:
        uid = row["SeriesInstanceUID"]
        with urllib.request.urlopen(BASE + "getSOPInstanceUIDs?SeriesInstanceUID=" + uid, timeout=45) as r:
            sop = json.load(r)[0]["SOPInstanceUID"]
        path = out / f"{row['Modality']}_{uid}.dcm"
        receipt = fetch(BASE + f"getSingleImage?SeriesInstanceUID={uid}&SOPInstanceUID={sop}", path)
        d = pydicom.dcmread(path, stop_before_pixels=True)
        item = {"modality": row["Modality"], "series_uid": uid, "sha256": receipt["sha256"], "bytes": receipt["bytes"]}
        if row["Modality"] == "CT":
            item.update({key: str(getattr(d, key, "")) for key in
                ("PatientAge", "PixelSpacing", "SliceThickness", "ImagePositionPatient",
                 "ImageOrientationPatient", "RescaleSlope", "RescaleIntercept", "FrameOfReferenceUID")})
        else:
            item["ROIs"] = [str(r.ROIName) for r in d.StructureSetROISequence]
            item["heart_contour_available"] = any(r.ROIName.lower() == "heart" for r in d.StructureSetROISequence)
            item["referenced_ct_series"] = [str(s.SeriesInstanceUID)
                for f in d.ReferencedFrameOfReferenceSequence
                for study in f.RTReferencedStudySequence for s in study.RTReferencedSeriesSequence]
        summary["series"].append(item)
    args.data.joinpath("tcia_sample_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
