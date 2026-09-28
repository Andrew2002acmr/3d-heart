"""Controlled smoothing/decimation experiment; baseline and variants separate."""
import argparse
import json
from pathlib import Path
from importlib.metadata import version
import nibabel as nib
import numpy as np
import pyvista as pv

from .labels import LABELS
from .mesh_quality import quality
from .surfaces import build_surface
from .volume import image_geometry, sha256


def variants(mesh):
    smooth = mesh.smooth_taubin(n_iter=20, pass_band=0.1, boundary_smoothing=False,
                                feature_smoothing=False, non_manifold_smoothing=False,
                                normalize_coordinates=True, window_function="nuttall", inplace=False)
    def decimate(surface):
        return surface.decimate_pro(0.5, preserve_topology=True, splitting=False,
                                    boundary_vertex_deletion=False, inplace=False)
    return {"taubin": smooth, "decimate50": decimate(mesh), "taubin_decimate50": decimate(smooth)}


def sample_surface(mesh, count, seed):
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    t = np.asarray(mesh.points, dtype=float)[faces]
    areas = np.linalg.norm(np.cross(t[:, 1]-t[:, 0], t[:, 2]-t[:, 0]), axis=1) / 2
    rng = np.random.default_rng(seed)
    selected = t[rng.choice(len(t), size=count, p=areas / areas.sum())]
    u, v = rng.random((2, count))
    u = np.sqrt(u)
    return (1-u[:, None])*selected[:, 0] + (u*(1-v))[:, None]*selected[:, 1] + (u*v)[:, None]*selected[:, 2]


def deviation(a, b, count=10000):
    distances = []
    for source, target, seed in ((a, b, 20260928), (b, a, 20260929)):
        points = pv.PolyData(sample_surface(source, count, seed))
        # Closest point on triangles, not nearest mesh vertex. Signs discarded.
        distances.append(np.abs(points.compute_implicit_distance(target)["implicit_distance"]))
    means = [float(d.mean()) for d in distances]
    p95 = [float(np.percentile(d, 95)) for d in distances]
    return {"sample_count_per_direction": count, "seeds": [20260928, 20260929],
            "sampling": "uniform by triangle area; unsigned point-to-triangle distance",
            "symmetric_mean": sum(means)/2, "bidirectional_p95_max": max(p95),
            "sampled_max": max(float(d.max()) for d in distances),
            "is_exact_hausdorff": False}


def delta(value, baseline):
    return None if value is None or baseline in (None, 0) else 100 * (value / baseline - 1)


def run(data, cases, process_cases, labels, out):
    if out.exists() and any(out.iterdir()):
        raise ValueError("Choose a new empty output directory")
    out.mkdir(parents=True, exist_ok=True)
    result = {"unit": "per-case header coordinates; unknown units are NOT mm",
              "versions": {n: version(n) for n in ("numpy", "nibabel", "scikit-image", "pyvista", "vtk")},
              "code_sha256": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
              "parameters": {"taubin": {"n_iter": 20, "pass_band": 0.1, "window": "nuttall",
                                         "boundary_smoothing": False, "non_manifold_smoothing": False,
                                         "feature_smoothing": False, "normalize_coordinates": True},
                             "decimate": {"target_reduction": 0.5, "preserve_topology": True,
                                          "splitting": False, "boundary_vertex_deletion": False}},
              "structures": []}
    for case in cases:
        path = data / f"{case}_label.nii.gz"
        before = sha256(path)
        image = nib.load(path)
        _, affine, unit = image_geometry(image, case, [])
        canonical = nib.as_closest_canonical(nib.Nifti1Image(np.asanyarray(image.dataobj).astype(np.uint16), affine))
        mask = np.asarray(canonical.dataobj)
        for label in LABELS:
            original, info = build_surface(mask, label, canonical.affine)
            if original is None:
                continue
            print(case, LABELS[label][0], "quality", flush=True)
            folder = out / case / LABELS[label][0]
            folder.mkdir(parents=True)
            original.save(folder / "original.vtp")
            source_points, source_faces = original.points.copy(), original.faces.copy()
            baseline = quality(original)
            entry = {"case": case, "label": label, "name": LABELS[label][0], "unit": unit,
                     "input_sha256": before, "voxel_diagnostics": info, "original": baseline,
                     "variants": {}, "small_components_removed": False}
            if case in process_cases and label in labels:
                for name, surface in variants(original).items():
                    print(case, LABELS[label][0], name, flush=True)
                    surface.save(folder / f"{name}.vtp")
                    q = quality(surface)
                    entry["variants"][name] = {**q, "deviation_from_original": deviation(original, surface),
                        "area_change_percent": delta(q["surface_area"], baseline["surface_area"]),
                        "volume_change_percent": delta(q["enclosed_volume"], baseline["enclosed_volume"]),
                        "algebraic_volume_change_percent": delta(q["algebraic_signed_volume"], baseline["algebraic_signed_volume"]),
                        "triangle_change_percent": delta(q["triangles"], baseline["triangles"])}
            if not np.array_equal(source_points, original.points) or not np.array_equal(source_faces, original.faces):
                raise RuntimeError("In-place mutation of baseline mesh")
            entry["baseline_unmodified"] = True
            result["structures"].append(entry)
            (out / "experiment.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        if before != sha256(path):
            raise RuntimeError("Source mask changed")
        del mask, canonical, image
    result["all_input_hashes_unchanged"] = True
    (out / "experiment.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--cases", nargs="+", required=True)
    p.add_argument("--process-cases", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", type=int, default=[1, 5, 7])
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    run(a.data, a.cases, a.process_cases, a.labels, a.out)
