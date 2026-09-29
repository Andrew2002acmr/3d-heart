"""Locate newly degenerate faces in baseline connected components; read only."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pyvista as pv
from heart3d.volume import sha256


def areas(mesh):
    triangles = mesh.points.astype(float)[mesh.faces.reshape(-1, 4)[:, 1:]]
    return np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1)/2


def inspect(root, out):
    if out.exists():
        raise ValueError("Choose a new output file")
    report = json.loads((root / "smoothing_metrics.json").read_text(encoding="utf-8"))
    if not report.get("completed"):
        raise ValueError("Experiment incomplete")
    results = []
    for entry in report["structures"]:
        candidates = [(k, q) for k, q in entry["variants"].items()
                      if q["zero_area_triangles"] > entry["baseline"]["zero_area_triangles"]]
        if not candidates:
            continue
        path = root / entry["baseline_file"]
        if sha256(path) != entry["source_sha256"]:
            raise ValueError("Baseline hash differs")
        original = pv.read(path)
        original.cell_data["source_face_id"] = np.arange(original.n_cells)
        connected = original.connectivity()
        regions = np.empty(original.n_cells, dtype=np.int64)
        regions[connected.cell_data["source_face_id"]] = connected.cell_data["RegionId"]
        before = areas(original)
        for key, metrics in candidates:
            path = root / metrics["file"]
            if sha256(path) != metrics["sha256"]:
                raise ValueError("Variant hash differs")
            variant = pv.read(path)
            if not np.array_equal(original.faces, variant.faces):
                raise ValueError("Face correspondence lost")
            after = areas(variant)
            new = (after <= 1e-12) & (before > 1e-12)
            parts = []
            for region in np.unique(regions[new]):
                selected = regions == region
                parts.append({"baseline_component_id": int(region), "component_triangles": int(selected.sum()),
                              "new_degenerate_triangles": int((new & selected).sum()),
                              "baseline_area": float(before[selected].sum()), "processed_area": float(after[selected].sum())})
            # Global volume can hide severe loss of small components even for
            # profiles that have not crossed the numerical degeneracy threshold.
            for profile, q in entry["variants"].items():
                candidate_path = root / q["file"]
                if sha256(candidate_path) != q["sha256"]:
                    raise ValueError("Variant hash differs")
                candidate_areas = areas(pv.read(candidate_path))
                for part in parts:
                    selected = regions == part["baseline_component_id"]
                    change = 100*(float(candidate_areas[selected].sum())/part["baseline_area"]-1)
                    part.setdefault("area_change_percent_by_profile", {})[profile] = change
            results.append({"case": entry["case"], "structure": entry["name"], "profile": key, "components": parts})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2)+"\n", encoding="utf-8")
    print(f"Inspected {len(results)} affected structure/profile pairs", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    inspect(args.run, args.out)
