"""Reproducible smoothing-only experiment on SAVED, unprocessed VTP surfaces."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import shutil

import numpy as np
import pyvista as pv

from .labels import LABELS
from .mesh_quality import quality
from .postprocess import delta, deviation
from .smoothing import PROFILES, smooth_surface, umbrella_operator, validate_surface
from .volume import sha256


# Exploratory engineering limits declared before the real-data run, NOT clinical
# tolerances. Area reduction is recorded, not used to reward excessive smoothing.
LIMITS = {"absolute_volume_change_percent": 2., "symmetric_mean_distance": .35,
          "max_vertex_displacement": 1.5}


def compare(original, surface, baseline_quality, count=10000):
    if original.n_points != surface.n_points or not np.array_equal(original.faces, surface.faces):
        raise ValueError("Corresponding-vertex bound requires identical point ordering/connectivity")
    q = quality(surface)
    distance = deviation(original, surface, count=count)
    displacement = np.linalg.norm(surface.points.astype(float)-original.points, axis=1)
    distance["max_corresponding_vertex_displacement"] = float(displacement.max())
    distance["mean_corresponding_vertex_displacement"] = float(displacement.mean())
    # For any barycentric point of a triangle, corresponding displacement is a
    # convex combination of its vertex displacements. Thus this is an upper
    # bound on continuous bidirectional Hausdorff distance, NOT an exact value.
    distance["hausdorff_upper_bound"] = float(displacement.max())
    q.update({"deviation_from_baseline": distance,
              "area_change_percent": delta(q["surface_area"], baseline_quality["surface_area"]),
              "volume_change_percent": delta(q["enclosed_volume"], baseline_quality["enclosed_volume"]),
              "algebraic_volume_change_percent": delta(q["algebraic_signed_volume"], baseline_quality["algebraic_signed_volume"]),
              "connectivity_identical": True})
    return q


def assess(baseline, candidate):
    reasons = []
    for key in ("boundary_edges", "non_manifold_edges", "inconsistent_winding_edges", "zero_area_triangles"):
        if candidate[key] > baseline[key]:
            reasons.append(f"increased_{key}")
    if candidate["surface_components_vertex_connected"] != baseline["surface_components_vertex_connected"]:
        reasons.append("changed_component_count")
    change = candidate["volume_change_percent"]
    if baseline["enclosed_volume"] is not None and candidate["enclosed_volume"] is None:
        reasons.append("lost_volume_validity")
    if change is not None and abs(change) > LIMITS["absolute_volume_change_percent"]:
        reasons.append("volume_change_over_limit")
    d = candidate["deviation_from_baseline"]
    if d["symmetric_mean"] > LIMITS["symmetric_mean_distance"]:
        reasons.append("mean_distance_over_limit")
    if d["max_corresponding_vertex_displacement"] > LIMITS["max_vertex_displacement"]:
        reasons.append("max_displacement_over_limit")
    status = "rejected" if reasons else "review_invalid_baseline" if baseline["enclosed_volume"] is None else "candidate"
    return {"status": status, "reasons": reasons, "is_clinically_validated": False}


def input_manifest(audit, cases):
    inputs, records = {}, []
    for case_id in cases:
        if Path(case_id).name != case_id:
            raise ValueError("Case ID must be a directory name")
        path = audit / case_id / "report.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        inputs[str(path)] = sha256(path)
        structures = record["structures"]
        if not structures or len({s["label"] for s in structures}) != len(structures):
            raise ValueError(f"{case_id}: no structures or duplicate labels")
        for s in structures:
            mesh_path = (path.parent / s["file"]).resolve()
            if not mesh_path.is_relative_to(audit) or mesh_path.suffix.lower() != ".vtp":
                raise ValueError("Expected a VTP within the source audit")
            if s.get("smoothing") is not False or s.get("decimation") is not False or not s.get("normal_policy"):
                raise ValueError("Expected a current unprocessed baseline report with explicit normal policy")
            if s["label"] not in LABELS:
                raise ValueError("Unknown cardiac label in report")
            inputs[str(mesh_path)] = sha256(mesh_path)
        records.append((case_id, path, record))
    return inputs, records


def run(audit, cases, out, count=10000):
    audit, out = Path(audit).resolve(), Path(out).resolve()
    if out.is_relative_to(audit) or audit.is_relative_to(out):
        raise ValueError("Input audit and output directories must not overlap")
    if count < 100:
        raise ValueError("Use at least 100 surface samples per direction")
    if out.exists() and any(out.iterdir()):
        raise ValueError("Choose a new, empty --out directory")
    cases = cases or sorted(p.parent.name for p in audit.glob("*/report.json"))
    if not cases or len(cases) != len(set(cases)):
        raise ValueError("Expected one or more distinct cases")
    input_hashes, reports = input_manifest(audit, cases)
    out.mkdir(parents=True, exist_ok=True)
    result = {"started_utc": datetime.now(timezone.utc).isoformat(), "completed": False,
              "coordinate_units": "current model coordinates (u); NOT confirmed mm; area u^2, volume u^3",
              "method_note": "taubin = explicit lambda/mu pairs; windowed_sinc = VTK via PyVista smooth_taubin",
              "profiles": PROFILES, "exploratory_limits": LIMITS,
              "versions": {n: version(n) for n in ("numpy", "scipy", "pyvista", "vtk")},
              "code_sha256": {n: sha256(Path(__file__).parent / n) for n in
                              ("smoothing.py", "smoothing_experiment.py", "mesh_quality.py", "postprocess.py")},
              "source_hashes": input_hashes, "missing_labels": {}, "structures": []}
    def checkpoint():
        (out / "smoothing_metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    checkpoint()
    for case_id, report_path, report in reports:
        result["missing_labels"][case_id] = [LABELS[n][0] for n in LABELS if n not in {s["label"] for s in report["structures"]}]
        for s in report["structures"]:
            label, name = s["label"], LABELS[s["label"]][0]
            source = (report_path.parent / s["file"]).resolve()
            mesh = pv.read(source)
            validate_surface(mesh)
            if mesh.n_points != s["points"] or mesh.n_cells != s["triangles"]:
                raise ValueError(f"{source}: geometry counts differ from report")
            if "unit" in mesh.field_data and str(mesh.field_data["unit"][0]) != report["geometry"]["unit"]:
                raise ValueError("Baseline unit differs from report")
            folder = out / case_id / name
            folder.mkdir(parents=True)
            baseline_path = folder / "baseline.vtp"
            shutil.copy2(source, baseline_path)
            if sha256(baseline_path) != input_hashes[str(source)]:
                raise RuntimeError("Baseline copy differs from input bytes")
            points, faces = mesh.points.copy(), mesh.faces.copy()
            print(f"{case_id} {name}: baseline", flush=True)
            baseline = quality(mesh)
            operator = umbrella_operator(mesh)
            entry = {"case": case_id, "label": label, "name": name, "header_unit": report["geometry"]["unit"],
                     "source_vtp": str(source), "source_sha256": input_hashes[str(source)],
                     "baseline_file": str(baseline_path.relative_to(out)), "baseline": baseline,
                     "taubin_fixed_vertices": int(operator[1].sum()), "variants": {}}
            for method, profiles in PROFILES.items():
                (folder / method).mkdir()
                for level in profiles:
                    print(f"{case_id} {name}: {method}/{level}", flush=True)
                    surface = smooth_surface(mesh, method, level, operator=operator)
                    file = folder / method / f"{level}.vtp"
                    surface.save(file)
                    metrics = compare(mesh, surface, baseline, count)
                    metrics["assessment"] = assess(baseline, metrics)
                    metrics["file"] = str(file.relative_to(out))
                    metrics["sha256"] = sha256(file)
                    entry["variants"][f"{method}/{level}"] = metrics
                    del surface
            if not np.array_equal(mesh.points, points) or not np.array_equal(mesh.faces, faces):
                raise RuntimeError("Baseline modified in memory")
            entry["baseline_unmodified"] = True
            result["structures"].append(entry)
            checkpoint()
            del mesh, points, faces, operator
    if any(sha256(Path(path)) != digest for path, digest in input_hashes.items()):
        raise RuntimeError("Source VTP/report changed during experiment")
    result["all_source_hashes_unchanged"] = True
    result["completed"] = True
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    checkpoint()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True, help="Directory containing CASE/report.json and meshes")
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10000)
    args = parser.parse_args()
    run(args.audit, args.cases, args.out, args.samples)
