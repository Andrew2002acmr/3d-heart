"""Build a patient registry from downloaded public metadata, never from absent labels."""
import argparse
from collections import Counter
import csv
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from heart3d.pediatric import read_chd68, read_hvsmr, read_tcia, validate_registry, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = read_hvsmr(args.data / "hvsmr_clinical.csv", args.data / "hvsmr_technical.csv")
    rows += read_chd68(args.data / "chd68_source_readme.md")
    rows += read_tcia(args.data / "pediatric_ct_digest.xlsx", args.data / "tcia_series.txt")
    validate_registry(rows)
    rows.sort(key=lambda r: (r["dataset"], r["patient_id"]))
    write_json(args.out / "registry.json", {"schema_version": 1, "age_rule": "2 <= reported_age_years <= 17",
               "note": "Reported ages are not exact chronological ages. Dataset-level claims are distinguished from local checks.",
               "records": rows})
    hvsmr = [r for r in rows if r["dataset"] == "HVSMR-2.0" and r["eligible_by_reported_age"]]
    with (args.out / "hvsmr_2_17.csv").open("w", encoding="utf-8", newline="") as f:
        names = ["patient", "age", "age_group", "age_precision", "severity", "healthy_status",
                 "diagnoses", "previous_surgery", "surgery_status", "available_structures", "structure_evidence", "stress_tags", "artifacts"]
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        for r in hvsmr:
            writer.writerow({"patient": r["patient_id"], "age": r["age"], "age_group": r["age_group"],
                "age_precision": r["age_precision"], "severity": r["severity"], "healthy_status": r["healthy_status"],
                "diagnoses": ";".join(r["diagnosis"]), "previous_surgery": ";".join(r["previous_surgery"]),
                "surgery_status": r["surgery_status"], "available_structures": ";".join(r["structures"]),
                "structure_evidence": r["structures_evidence"], "stress_tags": ";".join(r["stress_tags"]),
                "artifacts": ";".join(r["artifacts"])})
    summary = {"age_rule": "inclusive numeric 2..17 applied to published age; not precision beyond source",
               "datasets": {}, "hvsmr_stress_counts": dict(Counter(t for r in hvsmr for t in r["stress_tags"])),
               "hvsmr_severity": dict(Counter(r["severity"] for r in hvsmr)),
               "source_sha256": {name: hashlib.sha256((args.data / name).read_bytes()).hexdigest() for name in
                   ("hvsmr_clinical.csv", "hvsmr_technical.csv", "chd68_source_readme.md", "pediatric_ct_digest.xlsx", "tcia_series.txt")}}
    for dataset in sorted({r["dataset"] for r in rows}):
        group = [r for r in rows if r["dataset"] == dataset]
        target = [r for r in group if r["eligible_by_reported_age"]]
        summary["datasets"][dataset] = {"records": len(group), "age_eligible_reported": len(target),
            "unknown_age": sum(r["age"] is None for r in group),
            "healthy_status": dict(Counter(r["healthy_status"] for r in group)),
            "target_age_groups": dict(Counter(r["age_group"] for r in target)),
            "target_ids": [r["patient_id"] for r in target]}
    summary["hvsmr_missing_clinical"] = [r["patient_id"] for r in rows if r["dataset"] == "HVSMR-2.0" and not r["clinical_metadata_available"]]
    write_json(args.out / "registry_summary.json", summary)
    print({k: {a: b for a, b in v.items() if a != "target_ids"} for k, v in summary["datasets"].items()})


if __name__ == "__main__":
    main()
