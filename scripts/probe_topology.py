"""Diagnostic isovalue experiment, not an automatic anatomical repair."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nibabel as nib
import numpy as np
import pyvista as pv
from skimage.measure import marching_cubes
from heart3d.mesh_quality import quality
from heart3d.volume import sha256


def probe(path, label):
    image = nib.load(path)
    array = np.asanyarray(image.dataobj) == label
    locations = [np.flatnonzero(array.any(axis=tuple(j for j in range(3) if j != i))) for i in range(3)]
    start = np.array([x[0] for x in locations])
    crop = np.pad(array[tuple(slice(x[0], x[-1]+1) for x in locations)].astype(np.uint8), 1)
    results = []
    for level in (0.49, 0.5, 0.51):
        v, f, _, _ = marching_cubes(crop, level=level, gradient_direction="ascent", allow_degenerate=False)
        v = nib.affines.apply_affine(image.affine, v - 1 + start)
        if np.linalg.det(image.affine[:3, :3]) < 0:
            f = f[:, ::-1]
        mesh = pv.PolyData(v, np.column_stack((np.full(len(f), 3), f)))
        raw = quality(mesh)
        adjusted = quality(mesh.compute_normals(consistent_normals=True, auto_orient_normals=False, split_vertices=False))
        results.append({"isovalue": level, "raw_marching_cubes": raw, "after_consistent_normals": adjusted})
        print(level, raw["non_manifold_edges"], raw["inconsistent_winding_edges"], adjusted["inconsistent_winding_edges"], flush=True)
    return {"input": path.name, "sha256": sha256(path), "label": label,
            "interpretation": "Sensitivity test only. 0.49/0.51 move the material boundary and can change topology; neither is adopted as a repair.", "results": results}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mask", type=Path, required=True)
    p.add_argument("--label", type=int, default=5)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    result = probe(a.mask, a.label)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
