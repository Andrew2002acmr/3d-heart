"""327-patient evidence census. Heart eligibility is distinct from full-stack QA."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import json
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from heart3d.dicom.tcia import fetch_ct_probe, fetch_heart_extract
from heart3d.dicom.rtstruct import validate_references
from heart3d.pediatric import dicom_age, write_json


def census_one(row, series, data_root):
    pid = row["patient_id"]
    out = {"patient_id": pid, "age": row["age"], "age_group": row["age_group"],
        "age_precision": row["age_precision"], "healthy_status": "unknown",
        "ct_series_uid": None, "rt_series_uid": None, "scanner": None,
        "heart_roi_present": None, "heart_contours_complete": None,
        "full_original_rtstruct_downloaded": False, "full_ct_geometry_verified": False,
        "can_rasterize_on_full_CT": None, "scan_coverage": "unknown_pending_full_stack_QA",
        "suitable_for_training": False, "status": "requires_full_stack_QA",
        "exclusion_reasons": [], "pending_checks": ["complete_CT_geometry", "mask_rasterization", "scan_coverage", "visual_QA"]}
    related = [s for s in series if s["PatientID"] == pid]
    ctrows = [s for s in related if s["Modality"] == "CT"]
    rtrows = [s for s in related if s["Modality"] == "RTSTRUCT"]
    try:
        if len(ctrows) != 1 or len(rtrows) != 1:
            raise ValueError("Expected one CT and one RTSTRUCT series")
        ctrow, rtrow = ctrows[0], rtrows[0]
        out.update(ct_series_uid=ctrow["SeriesInstanceUID"], rt_series_uid=rtrow["SeriesInstanceUID"],
                   scanner=ctrow["ManufacturerModelName"], slices=ctrow["ImageCount"],
                   CT_published_bytes=ctrow["FileSize"], RT_published_bytes=rtrow["FileSize"])
        ct, sops, ct_receipt = fetch_ct_probe(out["ct_series_uid"], data_root / pid / "census")
        if str(ct.SeriesInstanceUID) != out["ct_series_uid"] or str(ct.PatientID) != pid or len(sops) != out["slices"]:
            raise ValueError("CT probe/public inventory identity mismatch")
        age = dicom_age(str(getattr(ct, "PatientAge", "")))
        if age["eligible_by_reported_age"] is not True or age["age"] != row["age"]:
            raise ValueError("DICOM age differs from eligible digest")
        out.update(age_from_DICOM=age["age"], age_raw=str(ct.PatientAge),
                   pixel_spacing_rc_mm=[float(v) for v in ct.PixelSpacing],
                   orientation_iop=[float(v) for v in ct.ImageOrientationPatient],
                   probe_position_lps_mm=[float(v) for v in ct.ImagePositionPatient],
                   slice_thickness_mm=float(ct.SliceThickness) if hasattr(ct, "SliceThickness") else None,
                   actual_slice_spacing_mm=None,
                   rescale_slope=float(ct.RescaleSlope) if hasattr(ct, "RescaleSlope") else None,
                   rescale_intercept=float(ct.RescaleIntercept) if hasattr(ct, "RescaleIntercept") else None,
                   contrast_agent=str(getattr(ct, "ContrastBolusAgent", "")),
                   contrast_route=str(getattr(ct, "ContrastBolusRoute", "")),
                   convolution_kernel=str(getattr(ct, "ConvolutionKernel", "")),
                   frame_uid=str(ct.FrameOfReferenceUID), CT_probe_sha256=ct_receipt["sha256"])
        out["contrast_status"] = "contrast_agent_reported" if out["contrast_agent"].strip() else "unknown"
        rt, receipt = fetch_heart_extract(out["rt_series_uid"], data_root / pid / "census")
        out.update(heart_roi_present=receipt["heart_defined"], heart_contours_complete=receipt["heart_contour_item_complete"],
                   heart_contours=receipt["heart_contours"], rt_prefix_bytes=receipt["received_prefix_bytes"],
                   RT_heart_extract_sha256=receipt["derived_sha256"])
        if str(rt.PatientID) != pid:
            raise ValueError("RT patient identity differs from CT")
        if not receipt["heart_defined"] or not receipt["heart_contours"]:
            out.update(status="excluded", exclusion_reasons=["missing_or_empty_Heart_ROI"], pending_checks=[])
            return out
        item = validate_references(rt, out["ct_series_uid"], out["frame_uid"], sops)
        planes, errors = [], []
        normal = np.cross(np.array(out["orientation_iop"][:3]), np.array(out["orientation_iop"][3:]))
        for c in item.ContourSequence:
            points = np.asarray(c.ContourData, dtype=float).reshape(-1, 3)
            if len(points) != int(c.NumberOfContourPoints) or len(points) < 3 or not np.isfinite(points).all():
                raise ValueError("Invalid Heart contour coordinate payload")
            if str(c.ContourGeometricType) not in {"CLOSED_PLANAR", "CLOSEDPLANAR_XOR"}:
                raise ValueError("Unsupported Heart contour type")
            positions = points @ normal
            errors.append(float(np.ptp(positions)))
            planes.append(float(np.mean(positions)))
        if max(errors) > .05:
            raise ValueError("Nonplanar Heart contour")
        out.update(Heart_SOP_references_verified=True, FrameOfReference_verified=True,
                   contour_planarity_max_mm=max(errors), heart_plane_range_lps_mm=[min(planes), max(planes)],
                   status="metadata_candidate_full_QA_pending")
    except Exception as error:
        out.update(status="requires_review", exclusion_reasons=[f"{type(error).__name__}: {error}"])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--registry", type=Path, required=True)
    p.add_argument("--series", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4, choices=range(1, 5))
    args = p.parse_args()
    args.data.mkdir(parents=True, exist_ok=True)
    rows = [r for r in json.loads(args.registry.read_text(encoding="utf-8"))["records"]
            if r["dataset"] == "Pediatric-CT-SEG" and r["eligible_by_reported_age"] is True]
    series = json.loads(args.series.read_text())
    existing = json.loads(args.out.read_text()).get("records", []) if args.out.exists() else []
    results = {r["patient_id"]: r for r in existing if r["status"] in {"metadata_candidate_full_QA_pending", "excluded"}}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(census_one, r, series, args.data): r["patient_id"] for r in rows if r["patient_id"] not in results}
        for future in as_completed(futures):
            result = future.result(); results[result["patient_id"]] = result
            records = sorted(results.values(), key=lambda r: r["patient_id"])
            write_json(args.out, {"version": 1, "age_rule": "2 <= reported age <= 17", "expected_patients": len(rows),
                "census_complete": len(records) == len(rows), "healthy_controls": False,
                "scope": "CT probe + complete Heart item, not full CT QA for all patients", "records": records})
            if len(results) % 10 == 0 or len(results) == len(rows):
                print("CENSUS", len(results), "/", len(rows), dict(Counter(r["status"] for r in records)), flush=True)
    print("Census complete. Full-stack/visual gates remain explicit per patient.", flush=True)


if __name__ == "__main__":
    main()
