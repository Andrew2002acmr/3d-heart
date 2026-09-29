"""Export small, reviewable validation evidence; never copy volumes or meshes."""
import argparse
import csv
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(audit, experiment, out):
    if out.exists() and any(out.iterdir()):
        raise ValueError("Choose a new empty evidence directory")
    out.mkdir(parents=True, exist_ok=True)
    report = json.loads(experiment.read_text(encoding="utf-8"))
    if not report.get("all_input_hashes_unchanged"):
        raise ValueError("Experiment is incomplete")
    audit_report = json.loads((audit / "audit.json").read_text(encoding="utf-8"))
    if {s["case"] for s in report["structures"]} != {c["case"] for c in audit_report["cases"]}:
        raise ValueError("Audit and experiment case sets differ (possibly incomplete audit)")
    for name in ("audit.json", "headers.json", "metadata.json"):
        shutil.copyfile(audit / name, out / name)
    shutil.copyfile(experiment, out / "experiment.json")
    metrics = ("vertices", "triangles", "surface_area", "enclosed_volume",
               "algebraic_signed_volume", "boundary_edges", "non_manifold_edges",
               "inconsistent_winding_edges", "zero_area_triangles",
               "surface_components_vertex_connected", "self_intersections_checked")
    raw, processed = [], []
    for s in report["structures"]:
        q, d = s["original"], s["voxel_diagnostics"]
        base = {"case": s["case"], "structure": s["name"], "unit": s["unit"]}
        raw.append({**base, **{k: q[k] for k in metrics},
                    "voxel_components_26": d["components_26_connected"],
                    "voxels_outside_largest_component": d["voxels"]-d["largest_component_voxels"],
                    "touches_scan_boundary": d["touches_scan_boundary"]})
        if not s["variants"]:
            continue
        for name, v in {"original": q, **s["variants"]}.items():
            dev = v.get("deviation_from_original", {})
            processed.append({**base, "variant": name, **{k: v[k] for k in metrics},
                              "area_change_percent": v.get("area_change_percent", 0),
                              "volume_change_percent": v.get("volume_change_percent", 0 if q["enclosed_volume"] is not None else None),
                              "mean_deviation": dev.get("symmetric_mean", 0),
                              "p95_deviation": dev.get("bidirectional_p95_max", 0),
                              "sampled_max_deviation": dev.get("sampled_max", 0)})
    write_csv(out / "raw_mesh_metrics.csv", raw)
    write_csv(out / "postprocess_metrics.csv", processed)
    lines = ["# Postprocessing measurements", "",
             "Coordinates have unknown units: area is in coordinate units squared, volume in coordinate units cubed.",
             "A dash denotes invalid topology; even other volumes have not been checked for self-intersections.",
             "Deviation uses 10,000 samples per direction, not exact Hausdorff distance.", "",
             "| Case / structure | Variant | Vertices | Triangles | Area | Volume | Mean deviation | Sampled max | Non-manifold edges |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in processed:
        volume = "—" if r["enclosed_volume"] is None else f'{r["enclosed_volume"]:.3f}'
        lines.append(f'| {r["case"]} {r["structure"]} | {r["variant"]} | {r["vertices"]} | {r["triangles"]} | {r["surface_area"]:.3f} | {volume} | {r["mean_deviation"]:.6g} | {r["sampled_max_deviation"]:.6g} | {r["non_manifold_edges"]} |')
    (out / "postprocess_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    selected = [s for s in report["structures"] if s["variants"]]
    x = np.arange(len(selected))
    names = ("taubin", "decimate50", "taubin_decimate50")
    colors = ("#2878b5", "#cf7b24", "#ad4444")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    panels = [("area_change_percent", "Area change (%)"),
              ("volume_change_percent", "Volume change (%); crosses = invalid topology"),
              ("deviation", "Mean surface distance (unknown units)"),
              ("non_manifold_edges", "Non-manifold edges")]
    for ax, (key, title) in zip(axes.ravel(), panels):
        for j, (name, color) in enumerate(zip(names, colors)):
            values = []
            for s in selected:
                v = s["variants"][name]
                value = v["deviation_from_original"]["symmetric_mean"] if key == "deviation" else v[key]
                values.append(np.nan if value is None else value)
            ax.bar(x+(j-1)*0.23, values, width=0.23, color=color, label=name)
            missing = np.isnan(values)
            if missing.any():
                ax.scatter((x+(j-1)*0.23)[missing], np.full(missing.sum(), 0.0025),
                           marker="x", s=35, color=color)
        if key == "non_manifold_edges":
            ax.scatter(x, [s["original"][key] for s in selected], marker="_", s=230, color="black", label="original")
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x, [s["case"][3:]+"\n"+s["name"] for s in selected])
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    axes[0, 0].legend(fontsize=8)
    axes[1, 1].legend(fontsize=8)
    fig.suptitle("ImageCHD: processing relative to the unchanged marching-cubes surface", fontsize=14)
    fig.savefig(out / "postprocess_comparison.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    summarize(a.audit, a.experiment, a.out)
