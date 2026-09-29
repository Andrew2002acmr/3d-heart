"""Render saved Original / Taubin lambda-mu / Windowed Sinc with linked cameras."""
import argparse
import itertools
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pyvista as pv

from heart3d.labels import LABELS
from heart3d.volume import sha256


def render(root, case_id, labels, levels, output, interactive=False, zoom=.92):
    record = json.loads((root / "smoothing_metrics.json").read_text(encoding="utf-8"))
    if not record.get("completed"):
        raise ValueError("Wait for the smoothing experiment to finish")
    entries = {s["label"]: s for s in record["structures"] if s["case"] == case_id}
    rows = [(entries[n], level) for n, level in itertools.product(labels, levels) if n in entries]
    if not rows or len(rows) > 4:
        raise ValueError("Choose 1-4 existing structure/profile rows per comparison image")
    if not 0 < zoom <= 5:
        raise ValueError("Use 0 < zoom <= 5")
    if output.exists() or output.with_suffix(".json").exists():
        raise ValueError("Comparison output already exists; choose a new filename")
    output.parent.mkdir(parents=True, exist_ok=True)
    plotter = pv.Plotter(shape=(len(rows), 3), window_size=(1500, 430*len(rows)), off_screen=not interactive)
    metadata = {"case": case_id, "metrics_sha256": sha256(root / "smoothing_metrics.json"),
                "flat_shading": True, "parallel_projection": True, "rows": []}
    try:
        for row, (entry, level) in enumerate(rows):
            for col, (caption, key) in enumerate((("Original", None), ("Taubin lambda/mu", f"taubin/{level}"),
                                                 ("Windowed Sinc", f"windowed_sinc/{level}"))):
                plotter.subplot(row, col)
                q = entry["baseline"] if key is None else entry["variants"][key]
                file = entry["baseline_file"] if key is None else q["file"]
                digest = entry["source_sha256"] if key is None else q["sha256"]
                if sha256(root / file) != digest:
                    raise ValueError("Saved VTP differs from the measured file")
                mesh = pv.read(root / file)
                plotter.set_background("#17212b")
                plotter.add_mesh(mesh, color=LABELS[entry["label"]][2], smooth_shading=False)
                plotter.add_text(f"{case_id} {entry['name']} | {caption}\n{level if key else 'baseline'} | u (NOT mm) | zoom {zoom:g}",
                                 position="upper_left", font_size=10, color="white")
                if key:
                    volume = "n/a" if q["volume_change_percent"] is None else f"{q['volume_change_percent']:+.3f}%"
                    d = q["deviation_from_baseline"]
                    text = (f"Area {q['area_change_percent']:+.2f}% | Volume {volume}\n"
                            f"Mean distance {d['symmetric_mean']:.3f} u | degenerate {q['zero_area_triangles']}")
                else:
                    text = f"{q['vertices']:,} vertices | {q['triangles']:,} triangles\nNon-manifold edges: {q['non_manifold_edges']}"
                plotter.add_text(text, position="lower_left", font_size=9, color="#e2e9ef")
            plotter.subplot(row, 0)
            plotter.view_isometric()
            plotter.enable_parallel_projection()
            plotter.reset_camera()
            plotter.camera.zoom(zoom)
            # Every row has its own camera, shared across the three methods.
            plotter.link_views([row*3+i for i in range(3)])
            camera = plotter.camera
            metadata["rows"].append({"structure": entry["name"], "level": level,
                "camera": {"position": list(camera.position), "focal_point": list(camera.focal_point),
                           "view_up": list(camera.up), "parallel_scale": camera.parallel_scale}})
        plotter.screenshot(str(output))
        output.with_suffix(".json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
        if interactive:
            plotter.show()
    finally:
        plotter.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--labels", nargs="+", type=int, default=[1, 2, 5, 7])
    parser.add_argument("--levels", nargs="+", choices=["mild", "medium", "strong"], default=["medium"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--zoom", type=float, default=.92)
    args = parser.parse_args()
    render(args.run, args.case, args.labels, args.levels, args.out, args.interactive, args.zoom)
