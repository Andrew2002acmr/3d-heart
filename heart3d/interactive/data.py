"""Discover existing cases and load verified report-associated surfaces only."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pyvista as pv

from ..labels import LABELS
from ..volume import Case, discover, load_case, load_ct_only


@dataclass
class CaseSource:
    case_id: str
    ct: Path
    mask: Path | None
    reports: list[tuple[Path, dict]]
    experiments: list[tuple[Path, dict]]


@dataclass
class LoadedCase:
    source: CaseSource
    volume: Case
    meshes: dict[int, dict[str, pv.PolyData]]
    warnings: list[str]
    report_path: Path | None
    window: tuple[float, float]


def catalog(data_root, results_root):
    """IDs are discovered from file pairs, never from a fixed four-case list."""
    sources = {}
    discovery = discover(data_root)
    notes = list(discovery["issues"])
    for pair in discovery["pairs"]:
        if pair["case"] in sources:
            raise ValueError(f"Duplicate case ID {pair['case']}; select a narrower --data directory")
        sources[pair["case"]] = CaseSource(pair["case"], Path(pair["ct"]), Path(pair["mask"]), [], [])
    for filename in ("report.json", "experiment.json"):
        for path in sorted(Path(results_root).rglob(filename)):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                if filename == "report.json":
                    case_id = Path(record["ct"]["path"]).name.split("_image.nii")[0]
                    if case_id in sources:
                        sources[case_id].reports.append((path, record))
                elif record.get("all_input_hashes_unchanged"):
                    for case_id in {s["case"] for s in record["structures"]}:
                        if case_id in sources:
                            sources[case_id].experiments.append((path, record))
            except (OSError, ValueError, KeyError, TypeError) as error:
                notes.append(f"Skipped unreadable report {path.name}: {error}")
    return sorted(sources.values(), key=lambda s: s.case_id), notes


def checked_mesh(path, expected_points=None, expected_cells=None):
    mesh = pv.read(path)
    if not isinstance(mesh, pv.PolyData) or not mesh.n_cells or not mesh.is_all_triangles:
        raise ValueError(f"{path.name}: expected a nonempty triangular VTP surface")
    if not np.isfinite(mesh.points).all():
        raise ValueError(f"{path.name}: non-finite mesh coordinates")
    if expected_points is not None and mesh.n_points != expected_points:
        raise ValueError(f"{path.name}: vertex count differs from report")
    if expected_cells is not None and mesh.n_cells != expected_cells:
        raise ValueError(f"{path.name}: triangle count differs from report")
    return mesh


def variant_allowed(baseline, candidate):
    """Reject missing metrics and filters known to worsen recorded topology."""
    keys = ("boundary_edges", "non_manifold_edges", "inconsistent_winding_edges", "zero_area_triangles")
    for key in keys:
        a, b = baseline.get(key), candidate.get(key)
        if type(a) is not int or type(b) is not int or min(a, b) < 0 or b > a:
            return False
    return True


def load_source(source):
    if source.mask is None:
        volume = load_ct_only(source.ct)
        low, high = np.percentile(volume.ct[::4, ::4, ::4], [1, 99])
        return LoadedCase(source, volume, {}, volume.report["warnings"], None,
                          (float(low), float(max(high, low+1))))
    volume = load_case(source.ct, source.mask)
    warnings = list(volume.report["warnings"])
    available = {int(k) for k in volume.report["label_counts"]} & LABELS.keys()
    matches = []
    for path, report in source.reports:
        try:
            same_inputs = all(report[k]["sha256"] == volume.report[k]["sha256"] for k in ("ct", "mask"))
            same_grid = (report["geometry"]["shape"] == list(volume.ct.shape) and
                         report["geometry"]["unit"] == volume.unit and
                         np.allclose(report["geometry"]["affine"], volume.affine, atol=1e-5, rtol=0))
            if same_inputs and same_grid:
                current_normals = all("normal_policy" in s for s in report["structures"])
                matches.append((current_normals, path.stat().st_mtime_ns, path, report))
        except (KeyError, TypeError, ValueError, OSError):
            continue
    meshes = {label: {} for label in available}
    report_path = None
    initial_window = None
    if matches:
        _, _, report_path, report = max(matches, key=lambda x: (x[0], x[1], str(x[2])))
        warnings.extend(report.get("warnings", []))
        initial_window = report.get("slices", {}).get("window")
        for structure in report["structures"]:
            label = structure["label"]
            if label not in available:
                continue
            try:
                mesh = checked_mesh(report_path.parent / structure["file"], structure.get("points"), structure.get("triangles"))
                if "unit" in mesh.field_data and str(mesh.field_data["unit"][0]) != volume.unit:
                    raise ValueError("mesh coordinate units differ from volume")
                meshes[label]["Original"] = mesh
            except (OSError, ValueError, KeyError) as error:
                warnings.append(f"{LABELS[label][0]}: 3D unavailable: {error}")
    else:
        warnings.append("No saved report with matching input hashes and geometry. CT/overlay remain available; no reconstruction is run.")
    for label in available:
        if not meshes[label]:
            warnings.append(f"{LABELS[label][0]}: mask present, saved 3D surface unavailable")
    # Read each optional experiment, newest first. No combined variant is exposed.
    for path, experiment in sorted(source.experiments, key=lambda x: x[0].stat().st_mtime_ns, reverse=True):
        for s in experiment["structures"]:
            label = s["label"]
            if s["case"] != source.case_id or label not in meshes or "Original" not in meshes[label]:
                continue
            if s.get("input_sha256") != volume.report["mask"]["sha256"] or s.get("unit") != volume.unit:
                continue
            folder = path.parent / source.case_id / LABELS[label][0]
            pending = [(caption, key) for caption, key in (("Smoothed", "taubin"), ("Simplified", "decimate50"))
                       if caption not in meshes[label] and key in s.get("variants", {})]
            if not pending or not (folder / "original.vtp").is_file():
                continue
            try:
                baseline = checked_mesh(folder / "original.vtp", s["original"]["vertices"], s["original"]["triangles"])
                original = meshes[label]["Original"]
                if not (np.array_equal(baseline.points, original.points) and np.array_equal(baseline.faces, original.faces)):
                    warnings.append(f"{LABELS[label][0]}: experiment baseline differs from selected Original; variants skipped")
                    continue
                for caption, key in pending:
                    q = s["variants"][key]
                    if not variant_allowed(s["original"], q):
                        warnings.append(f"{LABELS[label][0]} {caption}: blocked by missing/worsened topology metrics")
                        continue
                    meshes[label][caption] = checked_mesh(folder / f"{key}.vtp", q["vertices"], q["triangles"])
            except (OSError, ValueError, KeyError) as error:
                warnings.append(f"{LABELS[label][0]}: optional experiment unavailable: {error}")
    if initial_window is None or len(initial_window) != 2 or not np.isfinite(initial_window).all() or initial_window[0] >= initial_window[1]:
        low, high = np.percentile(volume.ct[::4, ::4, ::4], [1, 99])
        initial_window = (float(low), float(max(high, low + 1)))
    return LoadedCase(source, volume, meshes, list(dict.fromkeys(warnings)), report_path, tuple(initial_window))


def catalog_ct(ct_paths):
    """Explicit CT-only selection; local display aliases are not patient IDs."""
    sources = []
    for index, value in enumerate(ct_paths, 1):
        path = Path(value)
        if not path.is_file():
            raise ValueError(f"CT file not found: {path}")
        sources.append(CaseSource(f"CT_{index:03d}", path, None, [], []))
    return sources, []
