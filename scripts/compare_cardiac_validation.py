"""Pair two original-grid validation evaluations; never compare test or resized grids."""
import argparse
import math
from pathlib import Path

from heart3d.ml.cardiac_data import read_json, write_json
from heart3d.storage import sha256_file

CLASSES = ("LV", "RV", "LA", "RA", "MYO", "AO", "PA")


def compare(baseline_path, variant_path, output):
    if Path(output).exists():
        raise ValueError("Comparison output exists")
    baseline, variant = read_json(baseline_path), read_json(variant_path)
    for result in (baseline, variant):
        if result["partition"] != "validation":
            raise ValueError("Only validation comparison is authorized")
        if result["grid"] != "original release NIfTI grid, probabilities restored before argmax":
            raise ValueError("Only original-grid evaluations are comparable")
    for key in ("cohort_SHA256", "split_SHA256"):
        if baseline[key] != variant[key]:
            raise ValueError("Frozen protocol differs")
    a, b = [{row["case_id"]: row for row in result["records"]} for result in (baseline, variant)]
    if (len(a) != len(baseline["records"]) or len(b) != len(variant["records"])
            or not a or set(a) != set(b)):
        raise ValueError("Duplicate/mismatched validation cases")
    records, class_values = [], {name: [] for name in CLASSES}
    for case in sorted(a):
        before, after = a[case], b[case]
        x, y = float(before["macro_foreground_Dice"]), float(after["macro_foreground_Dice"])
        if not math.isfinite(x + y):
            raise ValueError("Non-finite validation Dice")
        classes = {}
        for name in CLASSES:
            old, new = before["classes"][name], after["classes"][name]
            if old["target_voxels"] != new["target_voxels"]:
                raise ValueError("Original GT voxel counts differ")
            if old["target_voxels"]:
                first, second = float(old["Dice"]), float(new["Dice"])
                if not math.isfinite(first + second):
                    raise ValueError("Non-finite class Dice")
                class_values[name].append((first, second))
                classes[name] = {"baseline_Dice": first, "variant_Dice": second, "delta": second-first}
        records.append({"case_id": case, "baseline_macro_Dice": x,
                        "variant_macro_Dice": y, "delta": y-x, "classes": classes})
    mean = lambda values: sum(values) / len(values)
    result = {
        "scope": "paired validation comparison; exploratory, not independent test/clinical validation",
        "grid": baseline["grid"], "case_count": len(records),
        "cohort_SHA256": baseline["cohort_SHA256"], "split_SHA256": baseline["split_SHA256"],
        "baseline_metrics_SHA256": sha256_file(baseline_path),
        "variant_metrics_SHA256": sha256_file(variant_path),
        "baseline_checkpoint_SHA256": baseline["checkpoint_SHA256"],
        "variant_checkpoint_SHA256": variant["checkpoint_SHA256"],
        "baseline_mean_case_macro_Dice": mean([r["baseline_macro_Dice"] for r in records]),
        "variant_mean_case_macro_Dice": mean([r["variant_macro_Dice"] for r in records]),
        "mean_paired_delta": mean([r["delta"] for r in records]),
        "cases_improved": sum(r["delta"] > 0 for r in records),
        "cases_worsened": sum(r["delta"] < 0 for r in records),
        "per_class": {name: {"reference_present_cases": len(values),
                            "baseline_mean_Dice": mean([x for x, _ in values]) if values else None,
                            "variant_mean_Dice": mean([y for _, y in values]) if values else None,
                            "mean_delta": mean([y-x for x, y in values]) if values else None}
                      for name, values in class_values.items()},
        "records": records, "test_evaluated": False, "physical_geometry_claimed": False,
        "age_verified_pediatric_claim": False,
    }
    write_json(output, result)
    print("VALIDATION_PAIRED_DELTA", result["mean_paired_delta"], flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "variant", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    compare(args.baseline, args.variant, args.output)
