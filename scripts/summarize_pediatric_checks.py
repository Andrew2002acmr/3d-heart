"""Commit-safe numeric summaries; patient volumes, meshes, paths stay external."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from heart3d.pediatric import validate_registry, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    registry = json.loads((args.metadata / "registry.json").read_text(encoding="utf-8"))
    by = {(r["dataset"], r["patient_id"]): r for r in registry["records"]}
    cases = []
    for path in sorted((args.data / "chd68/outputs").glob("ct_*/report.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        pid = path.parent.name
        row = by["CHD68", pid]
        row.update(downloaded_image=True, download_extent="complete_nifti_pair",
                   spacing_available=None if report["geometry"]["unit"] == "unknown" else True,
                   header_spacing_available=True, spacing_evidence="matching_NIfTI_affines_physical_units_unverified",
                   spatial_unit=report["geometry"]["unit"],
                   physical_scale_verified=report["geometry"]["physical_scale_independently_verified"],
                   structures_evidence="locally_verified_label_ids_authors_mapping")
        row["structures"] = [s["short_name"] for s in report["structures"]]
        cases.append({"patient_id": pid, "age_verified": False,
                      "image_sha256": report["ct"]["sha256"], "mask_sha256": report["mask"]["sha256"],
                      "geometry": report["geometry"], "original_ct_shape": report["ct"]["shape"],
                      "original_axis_codes": report["ct"]["axis_codes"],
                      "qform_code": report["ct"]["qform_code"], "sform_code": report["ct"]["sform_code"],
                      "labels": report["label_counts"], "ignored_labels": report["ignored_labels"],
                      "structures": [{k: v for k, v in s.items() if k not in {"file", "color"}} for s in report["structures"]],
                      "warnings": report["warnings"], "preview_generated": (path.parent / "preview.png").is_file()})
    if len(cases) < 3:
        raise ValueError("Need at least three completed CHD68 smoke cases")
    write_json(args.metadata / "chd68_smoke_summary.json", {"pipeline": "existing heart3d build --preview",
               "training_started": False, "cases": cases})
    for name in ("overlap_audit.json", "tcia_sample_summary.json", "metadata_download_receipt.json"):
        data = json.loads((args.data / name).read_text())
        write_json(args.metadata / name, data)
    sample = json.loads((args.data / "tcia_sample_summary.json").read_text())
    row = by["Pediatric-CT-SEG", sample["patient_id"]]
    rt = next(s for s in sample["series"] if s["modality"] == "RTSTRUCT")
    row.update(downloaded_image=True, download_extent="single_CT_slice_plus_RTSTRUCT",
               age_confirmed_against_downloaded_DICOM=True,
               heart_contour_available=rt["heart_contour_available"],
               structures=["HEART_REGION"] if rt["heart_contour_available"] else [],
               structures_evidence="locally_checked_RTSTRUCT_ROI_names",
               in_plane_spacing_available=True, spacing_evidence="single_slice_tags_only_not_full_stack")
    validate_registry(registry["records"])
    write_json(args.metadata / "registry.json", registry)
    print("Summarized", len(cases), "CHD68 cases and one partial TCIA case; no binary healthy inference.")


if __name__ == "__main__":
    main()
