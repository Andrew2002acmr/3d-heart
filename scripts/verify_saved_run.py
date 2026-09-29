"""Verify a saved four-case run against recorded evidence without rebuilding it.

Passing means file/metric reproducibility, not validated patient geometry.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import pyvista as pv

from heart3d.mesh_quality import quality
from heart3d.volume import sha256


def differences(actual, expected, path="root"):
    """Exact keys/counts; floating-point metrics allow rounding on VTP reload."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            return [path + ": keys differ"]
        return [d for k in expected for d in differences(actual[k], expected[k], path + "." + k)]
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return [path + ": list length differs"]
        return [d for i, (a, b) in enumerate(zip(actual, expected))
                for d in differences(a, b, f"{path}[{i}]")]
    if isinstance(expected, float):
        same = isinstance(actual, (float, int)) and not isinstance(actual, bool) and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-8)
    else:
        same = type(actual) is type(expected) and actual == expected
    return [] if same else [f"{path}: {actual!r} != {expected!r}"]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def checked_quality(mesh, expected):
    if not np.isfinite(mesh.points).all():
        return ["mesh contains non-finite coordinates"]
    actual = quality(mesh)
    return differences(actual, {k: expected[k] for k in actual})


def verify(run, data, reference, out):
    if out.exists():
        raise ValueError("Choose a new report filename to preserve previous checks")
    issues, documents, inputs, meshes = [], [], [], []
    for relative, ref in (("audit/audit.json", "audit.json"),
                          ("audit/headers.json", "headers.json"),
                          ("audit/metadata.json", "metadata.json"),
                          ("postprocess/experiment.json", "experiment.json")):
        actual, expected = read_json(run / relative), read_json(reference / ref)
        diff = differences(actual, expected, relative)
        issues.extend(diff)
        documents.append({"file": relative, "matches_reference": not diff,
                          "exact_json_value_match": actual == expected})
    headers = read_json(run / "audit/headers.json")
    audit_cases = {c["case"]: c for c in read_json(run / "audit/audit.json")["cases"]}
    experiment = read_json(run / "postprocess/experiment.json")
    if not experiment.get("all_input_hashes_unchanged"):
        issues.append("Postprocessing completion marker missing")
    provenance = {f["local_name"]: f["sha256"] for f in read_json(data / "provenance.json")["files"]}
    for case in headers["cases"]:
        for kind in ("ct", "mask"):
            source = case[kind]
            digest = sha256(data / source["file"])
            matches = digest == source["sha256"] == provenance.get(source["file"])
            if not matches:
                issues.append(source["file"] + ": input hash mismatch")
            inputs.append({"file": source["file"], "sha256": digest, "matches_report_and_provenance": matches})
    slice_count = 0
    audit_structures = {}
    for case in headers["cases"]:
        case_id = case["case"]
        report = read_json(run / "audit" / case_id / "report.json")
        for key in ("geometry", "structures", "warnings"):
            issues.extend(differences(report[key], audit_cases[case_id][key], case_id + "." + key))
        for kind in ("ct", "mask"):
            issues.extend(differences(report[kind]["sha256"], case[kind]["sha256"], case_id + "." + kind + ".sha256"))
            header = case[kind]["nibabel"]
            issues.extend(differences({k: report[kind][k] for k in header}, header, case_id + "." + kind + ".header"))
        with Image.open(run / "audit" / case_id / "slices.png") as image:
            image.verify()
        slice_count += 1
        for s in report["structures"]:
            key = (case_id, s["label"])
            if key in audit_structures:
                issues.append(f"Duplicate audit structure: {key}")
            audit_structures[key] = s
    expected_meshes = set()
    for s in experiment["structures"]:
        case, label, name = s["case"], s["label"], s["name"]
        print(case, name, "reload and recompute", flush=True)
        folder = Path("postprocess") / case / name
        original_path = folder / "original.vtp"
        original = pv.read(run / original_path)
        audit_info = audit_structures.pop((case, label))
        audit_path = Path("audit") / case / audit_info["file"]
        audit_mesh = pv.read(run / audit_path)
        same = np.array_equal(audit_mesh.points, original.points) and np.array_equal(audit_mesh.faces, original.faces)
        unit = str(audit_mesh.field_data["unit"][0])
        if not same or unit != s["unit"]:
            issues.append(f"{audit_path}: audit/baseline geometry or unit mismatch")
        if audit_info["points"] != original.n_points or audit_info["triangles"] != original.n_cells:
            issues.append(f"{audit_path}: report mesh size mismatch")
        expected_meshes.add(audit_path.as_posix())
        meshes.append({"file": audit_path.as_posix(), "sha256": sha256(run / audit_path),
                       "audit_matches_postprocess_baseline": same})
        for variant, expected in {"original": s["original"], **s["variants"]}.items():
            relative = folder / (variant + ".vtp")
            mesh = original if variant == "original" else pv.read(run / relative)
            diff = checked_quality(mesh, expected)
            issues.extend(str(relative) + ": " + d for d in diff)
            expected_meshes.add(relative.as_posix())
            meshes.append({"file": relative.as_posix(), "sha256": sha256(run / relative),
                           "recomputed_metrics_match": not diff,
                           "boundary_edges": expected["boundary_edges"],
                           "non_manifold_edges": expected["non_manifold_edges"],
                           "volume_is_qualified": expected["enclosed_volume"] is not None})
        del original, audit_mesh, mesh
    if audit_structures:
        issues.append("Audit structures missing in experiment")
    actual_meshes = {p.relative_to(run).as_posix() for root in ("audit", "postprocess") for p in (run / root).rglob("*.vtp")}
    if actual_meshes != expected_meshes:
        issues.append("Missing or extra VTP files: " + str(sorted(actual_meshes ^ expected_meshes)))
    result = {"checked_utc": datetime.now(timezone.utc).isoformat(), "run": run.name,
              "verifier_sha256": sha256(Path(__file__)),
              "reference": reference.as_posix(), "reproducibility_passed": not issues,
              "physical_geometry_validated": False, "clinical_segmentation_validated": False,
              "numeric_tolerance": {"rtol": 1e-9, "atol": 1e-8},
              "documents": documents, "inputs": inputs, "readable_slice_panels": slice_count,
              "saved_mesh_count": len(meshes), "mesh_quality_recomputed_count": sum("recomputed_metrics_match" in m for m in meshes),
              "meshes": meshes, "issues": issues,
              "limits": ["Reference agreement is not anatomical ground truth",
                         "Unknown physical units and patient orientation remain unresolved",
                         "Self-intersections and non-manifold vertex fans not tested",
                         "Sampled surface deviations compared to reference JSON, not recomputed"]}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path("data/imagechd"))
    p.add_argument("--reference", type=Path, default=Path("docs/validation"))
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    result = verify(a.run, a.data, a.reference, a.out)
    print("PASS (reproducibility only)" if result["reproducibility_passed"] else "FAIL", flush=True)
    sys.exit(0 if result["reproducibility_passed"] else 1)
