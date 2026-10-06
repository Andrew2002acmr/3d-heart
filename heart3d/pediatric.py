"""Evidence-preserving pediatric metadata. No training or inferred healthy labels."""
import csv
import json
import math
import re
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

STATUSES = {"confirmed_healthy", "no_structural_cardiac_abnormality_reported",
            "unknown", "pathological"}
STRUCTURES_CT = ["LV", "RV", "LA", "RA", "MYO", "AO", "PA"]
STRUCTURES_MR = ["LV", "RV", "LA", "RA", "AO", "PA", "SVC", "IVC"]
SURGERIES = {"ArterialSwitch", "AtrialSwitch", "Rastelli", "Glenn", "Fontan",
             "PABanding", "AOPAAnastamosis"}
ARTIFACTS = {"CMRArtifactAO", "CMRArtifactPA"}
STRESS = {"SingleVentricle", "DILV", "DIDORV", "DORV", "DLoopTGA", "LLoopTGA",
          "CommonAtrium", "PAAtresiaOrMPAStump", "Glenn", "Fontan", "Heterotaxy",
          "SuperoinferiorVentricles", "InvertedVentricles", "InvertedAtria"}


def age_fields(age, precision="unknown", raw=None):
    """Apply requested inclusive numeric rule to REPORTED age; never invent age."""
    if age is not None and (not math.isfinite(age) or age < 0):
        raise ValueError("Age must be finite and nonnegative")
    eligible = None if age is None else 2 <= age <= 17
    group = "unknown" if age is None else "outside_target"
    if eligible:
        # Working groups cover fractional ages without a 5-to-6 gap.
        group = "2-5" if age < 6 else "6-11" if age < 12 else "12-17"
    return {"age": age, "age_raw": raw, "age_precision": precision,
            "age_group": group, "eligible_by_reported_age": eligible,
            "exact_age_available": precision == "exact_years"}


def dicom_age(value):
    if not value:
        return age_fields(None)
    match = re.fullmatch(r"(\d{3})([DWMY])", value.strip())
    if not match:
        raise ValueError(f"Invalid DICOM AS: {value}")
    n, unit = int(match[1]), match[2]
    years = n / {"D": 365.25, "W": 365.25 / 7, "M": 12, "Y": 1}[unit]
    return age_fields(years, f"reported_{unit}", value)


def clinical_status(normal, findings):
    if findings:
        # Includes dilation and surgical history even when no named CHD is flagged.
        return "pathological"
    if normal:
        # Structurally normal in a clinical cohort is not a healthy volunteer.
        return "no_structural_cardiac_abnormality_reported"
    return "unknown"


def record(dataset, patient_id, modality, source, structures):
    return {"dataset": dataset, "patient_id": patient_id, **age_fields(None),
            "modality": modality, "healthy_status": "unknown", "diagnosis": [],
            "healthy_evidence": "no_case_level_evidence",
            "pathology_group": [], "segmentation_available": True,
            "segmentation_evidence": "dataset_documentation",
            "structures": structures, "structures_evidence": "dataset_protocol_not_case_verified",
            "image_format": "NIfTI", "spacing_available": None,
            "spacing_evidence": "not_locally_checked", "source": source,
            "access_type": "public", "downloaded_image": False,
            "binary_classification_eligible": False}


def read_hvsmr(clinical_path, technical_path):
    clinical = list(csv.DictReader(Path(clinical_path).open(encoding="utf-8-sig")))
    technical = list(csv.DictReader(Path(technical_path).open(encoding="utf-8-sig")))
    by = {}
    for r in clinical:
        if not r["Pat"]:
            continue
        if r["Pat"] in by:
            raise ValueError("Duplicate HVSMR clinical patient")
        by[r["Pat"]] = r
    tech = {r["Pat"]: r for r in technical if r["Pat"]}
    result = []
    for pid in sorted(set(by) | set(tech), key=int):
        out = record("HVSMR-2.0", f"pat{pid}", "MRI",
                     "https://doi.org/10.6084/m9.figshare.25226360.v2", STRUCTURES_MR)
        r = by.get(pid)
        out.update(severity=None, previous_surgery=[], surgery_status="unknown",
                   stress_tags=[], artifacts=[], clinical_metadata_available=r is not None)
        if r:
            flags = [k for k, v in r.items() if v.strip().upper() == "X"]
            disease = [f for f in flags if f not in SURGERIES | ARTIFACTS | {"Normal"}]
            surgery = [f for f in flags if f in SURGERIES]
            out.update(age_fields(float(r["Age"]), "reported_integer_years", r["Age"]))
            out.update(healthy_status=clinical_status("Normal" in flags, disease + surgery),
                       diagnosis=disease, pathology_group=disease,
                       severity=r["Category"], previous_surgery=surgery,
                       surgery_status="reported" if surgery else "not_reported",
                       stress_tags=[f for f in flags if f in STRESS],
                       artifacts=[f for f in flags if f in ARTIFACTS],
                       healthy_evidence="clinical_csv_explicit_flags", normal_reported="Normal" in flags)
        out["technical"] = {k: v for k, v in tech.get(pid, {}).items() if k != "Pat"}
        out["binary_classification_eligible"] = out["healthy_status"] in {"confirmed_healthy", "pathological"}
        result.append(out)
    return result


def read_chd68(readme_path):
    header, result = None, []
    for line in Path(readme_path).read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [s.strip() for s in line.strip().strip("|").split("|")]
        if cells[0] == "Index":
            header = cells
        elif header and cells[0].isdigit():
            if len(cells) != len(header) or any(v not in {"0", "1"} for v in cells[1:]):
                raise ValueError("Malformed CHD68 diagnosis row")
            flags = [k for k, v in zip(header[1:], cells[1:]) if v == "1"]
            disease = [f for f in flags if f != "Normal"]
            out = record("CHD68", f"ct_{cells[0]}", "CT",
                         "https://github.com/XiaoweiXu/Whole-heart-and-great-vessel-segmentation-of-chd_segmentation",
                         STRUCTURES_CT)
            out.update(healthy_status=clinical_status("Normal" in flags, disease),
                       healthy_evidence="authors_readme_explicit_diagnosis_table",
                       diagnosis=disease, pathology_group=disease, normal_reported="Normal" in flags,
                       binary_classification_eligible=bool(disease))
            result.append(out)
    if not result or len({r["patient_id"] for r in result}) != len(result):
        raise ValueError("Empty or duplicate CHD68 table")
    return result


def xlsx_rows(path):
    """Read the official TCIA digest as values; no Excel runtime needed."""
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as z:
        strings = []
        if "xl/sharedStrings.xml" in z.namelist():
            strings = ["".join(n.itertext()) for n in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", ns)]
        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
        rows = []
        for row in sheet.findall("m:sheetData/m:row", ns):
            values = {}
            for cell in row:
                column = re.match(r"[A-Z]+", cell.attrib["r"])[0]
                v = cell.find("m:v", ns)
                value = v.text if v is not None else "".join(cell.itertext())
                if cell.get("t") == "s":
                    value = strings[int(value)]
                values[column] = value
            rows.append(values)
    header = rows[0]
    return [{name: r.get(col, "") for col, name in header.items()} for r in rows[1:]]


def read_tcia(digest_path, current_series_path):
    rows = xlsx_rows(digest_path)
    series = json.loads(Path(current_series_path).read_text(encoding="utf-8"))
    grouped = {}
    for r in rows:
        if r["Modality"] == "CT":
            grouped.setdefault(r["Patient ID"], []).append(r)
    result = []
    for pid, ctrows in sorted(grouped.items()):
        ages = {r["Patient Age"] for r in ctrows}
        if len(ages) != 1:
            raise ValueError(f"Conflicting TCIA ages for {pid}")
        out = record("Pediatric-CT-SEG", pid, "CT",
                     "https://doi.org/10.7937/TCIA.X0H0-1706", [])
        out.update(dicom_age(next(iter(ages))))
        related = [r for r in series if r["PatientID"] == pid]
        out.update(image_format="DICOM+RTSTRUCT", healthy_status="unknown",
                   healthy_evidence="diagnosis_not_available_in_collection",
                   segmentation_available=any(r["Modality"] == "RTSTRUCT" for r in related),
                   segmentation_evidence="live_tcia_series_inventory",
                   structures_evidence="ROI_names_not_yet_checked_per_case",
                   heart_contour_available=None,
                   ct_series=[r["SeriesInstanceUID"] for r in related if r["Modality"] == "CT"],
                   scanners=sorted({r["ManufacturerModelName"] for r in related if r["Modality"] == "CT"}),
                   license_current=sorted({r.get("LicenseURI", "") for r in related}),
                   license_digest=sorted({r["License URI"] for r in ctrows}),
                   age_source="official_2022_digest_DICOM_PatientAge",
                   age_confirmed_against_downloaded_DICOM=False)
        # The series endpoint has no spacing/ROI names; RTSTRUCT does not prove Heart exists.
        result.append(out)
    live_patients = {r["PatientID"] for r in series if r["Modality"] == "CT"}
    if live_patients != set(grouped):
        raise ValueError("Digest/live TCIA patient inventories differ")
    return result


def validate_registry(rows):
    seen = set()
    for row in rows:
        key = row["dataset"], row["patient_id"]
        if key in seen:
            raise ValueError(f"Duplicate record: {key}")
        seen.add(key)
        if row["healthy_status"] not in STATUSES:
            raise ValueError("Unsupported healthy_status")
        if row["eligible_by_reported_age"] != age_fields(row["age"])["eligible_by_reported_age"]:
            raise ValueError("Inconsistent age eligibility")
        if row["binary_classification_eligible"] and row["healthy_status"] not in {"confirmed_healthy", "pathological"}:
            raise ValueError("Unconfirmed binary classification label")


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Match Git's *.json eol=lf policy so frozen manifest hashes survive checkout.
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
