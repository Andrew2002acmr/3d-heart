"""Targeted follow-up: separate shading effects from stronger mesh smoothing."""
import argparse
import json
from importlib.metadata import version
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pyvista as pv
from heart3d.labels import LABELS
from heart3d.protected_smoothing import PROFILES, smooth_primary
from heart3d.smoothing_experiment import compare, assess
from heart3d.volume import sha256


def render(folder, entry, output, interactive=False):
    """Two rows isolate shading; all eight panels share the same camera."""
    if output.exists():
        raise ValueError("Choose a new image filename")
    output.parent.mkdir(parents=True, exist_ok=True)
    names = ["baseline", *PROFILES]
    plotter = pv.Plotter(shape=(2, 4), window_size=(1800, 900), off_screen=not interactive)
    camera = None
    try:
        for row in range(2):
            for col, name in enumerate(names):
                plotter.subplot(row, col)
                mesh = pv.read(folder / f"{name}.vtp")
                points, faces = mesh.points.copy(), mesh.faces.copy()
                plotter.set_background("#17212b")
                plotter.add_mesh(mesh, color=LABELS[entry["label"]][2], smooth_shading=bool(row),
                    split_sharp_edges=False, show_edges=False, specular=0, copy_mesh=True)
                assert np.array_equal(mesh.points, points) and np.array_equal(mesh.faces, faces)
                caption = "Original" if col == 0 else f"WS {PROFILES[name][0]} / {PROFILES[name][1]}"
                plotter.add_text(f"{entry['case']} {entry['name']} | {caption}\n"
                    f"{'Smooth' if row else 'Flat'} shading | u, NOT mm", font_size=10, color="white")
                if col:
                    q = entry["variants"][name]
                    vol = "n/a" if q["volume_change_percent"] is None else f"{q['volume_change_percent']:+.2f}%"
                    plotter.add_text(f"Area {q['area_change_percent']:+.2f}% | Volume {vol}\n"
                        f"Mean {q['deviation_from_baseline']['symmetric_mean']:.3f} u | "
                        f"degenerate {q['zero_area_triangles']}", position="lower_left", font_size=9, color="white")
        plotter.subplot(0, 0)
        plotter.view_isometric()
        plotter.enable_parallel_projection()
        plotter.reset_camera()
        plotter.camera.zoom(1.35)
        plotter.link_views()
        camera = {"position": list(plotter.camera.position), "focal_point": list(plotter.camera.focal_point),
                  "view_up": list(plotter.camera.up), "parallel_scale": plotter.camera.parallel_scale}
        if interactive:
            plotter.show(auto_close=False, interactive_update=True)
        plotter.screenshot(str(output))
        if interactive:
            plotter.show()
    finally:
        plotter.close()
    return camera


def run(root, out, selections, samples):
    root, out = root.resolve(), out.resolve()
    if root == out or root in out.parents or out in root.parents:
        raise ValueError("Input and output must not overlap")
    if out.exists() and any(out.iterdir()):
        raise ValueError("Choose a new empty output directory")
    record = json.loads((root / "smoothing_metrics.json").read_text(encoding="utf-8"))
    if not record.get("completed"):
        raise ValueError("Input experiment is incomplete")
    wanted = set(selections)
    entries = [e for e in record["structures"] if f"{e['case']}:{e['name']}" in wanted]
    if len(entries) != len(wanted):
        raise ValueError("Some requested case/structure pairs are absent")
    out.mkdir(parents=True, exist_ok=True)
    result = {"completed": False, "profiles": PROFILES, "units": "u, NOT mm",
        "policy": "retain all but largest triangle-count component exactly; no branch protection",
        "versions": {n: version(n) for n in ("pyvista", "vtk", "numpy")},
        "code_sha256": {str(p.relative_to(Path(__file__).resolve().parents[1])): sha256(p)
            for p in [Path(__file__).resolve(), *sorted((Path(__file__).resolve().parents[1] / "heart3d").glob("*.py"))]},
        "input_metrics_sha256": sha256(root / "smoothing_metrics.json"), "structures": []}
    source_hashes = {}
    for source in entries:
        path = (root / source["baseline_file"]).resolve()
        if root not in path.parents or sha256(path) != source["source_sha256"]:
            raise ValueError("Source path/hash mismatch")
        source_hashes[path] = sha256(path)
        original = pv.read(path)
        before = original.points.copy()
        folder = out / source["case"] / source["name"]
        if out not in folder.resolve().parents:
            raise ValueError("Output case/structure path escapes output directory")
        folder.mkdir(parents=True)
        shutil.copy2(path, folder / "baseline.vtp")
        entry = {k: source[k] for k in ("case", "name", "label", "baseline", "source_sha256")}
        entry["source_file"] = str(path)
        entry["variants"] = {}
        for name in PROFILES:
            print(source["case"], source["name"], name, flush=True)
            mesh, locked = smooth_primary(original, name)
            q = compare(original, mesh, source["baseline"], count=samples)
            q["screening"] = assess(source["baseline"], q)
            q["protected_vertices"] = int(locked.sum())
            q["protected_vertices_unchanged"] = bool(np.array_equal(mesh.points[locked], before[locked]))
            assert q["protected_vertices_unchanged"] and np.array_equal(original.points, before)
            mesh.save(folder / f"{name}.vtp")
            q["file"] = str((folder / f"{name}.vtp").relative_to(out))
            q["sha256"] = sha256(out / q["file"])
            entry["variants"][name] = q
        # Only three illustrations; all selected structures get metrics.
        if (source["case"], source["name"]) in (("ct_1001", "LV"), ("ct_1004", "MYO"), ("ct_1001", "PA")):
            entry["camera"] = render(folder, entry, folder / "comparison.png")
        result["structures"].append(entry)
        (out / "probe.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    assert all(sha256(p) == digest for p, digest in source_hashes.items())
    result.update(completed=True, all_source_hashes_unchanged=True)
    (out / "probe.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--select", nargs="+", default=["ct_1001:LV", "ct_1001:RV", "ct_1001:MYO",
        "ct_1001:PA", "ct_1002:LV", "ct_1003:RV", "ct_1004:MYO"])
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--view", help="Render an existing probe: CASE:STRUCTURE; --out is a new PNG")
    parser.add_argument("--interactive", action="store_true", help="Open linked views with --view")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("samples must be positive")
    if args.view:
        record = json.loads((args.run / "probe.json").read_text(encoding="utf-8"))
        if not record.get("completed"):
            parser.error("Probe is incomplete")
        entry = next((e for e in record["structures"] if f"{e['case']}:{e['name']}" == args.view), None)
        if entry is None:
            parser.error("Requested case/structure absent")
        folder = args.run / entry["case"] / entry["name"]
        for name, digest in [("baseline", entry["source_sha256"]),
                             *((n, q["sha256"]) for n, q in entry["variants"].items())]:
            if sha256(folder / f"{name}.vtp") != digest:
                parser.error("Saved mesh hash mismatch")
        if args.out.with_suffix(".json").exists():
            parser.error("Camera metadata already exists")
        camera = render(folder, entry, args.out, args.interactive)
        args.out.with_suffix(".json").write_text(json.dumps(camera, indent=2)+"\n", encoding="utf-8")
    else:
        if args.interactive:
            parser.error("Use --interactive together with --view")
        run(args.run, args.out, args.select, args.samples)
