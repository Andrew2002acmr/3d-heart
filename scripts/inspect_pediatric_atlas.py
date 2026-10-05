"""Inspect a public ED+ES atlas and render mean/PCA modes. No patient diagnosis."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat


def load_atlas(path):
    atlas = loadmat(path, simplify_cells=True)["EDESatlas"]
    mean, coeff, latent = atlas["mean"], atlas["coeff"], atlas["latent"]
    if mean.shape != (34860,) or coeff.shape != (34860, 100) or latent.shape != (100,):
        raise ValueError("Unexpected pediatric atlas layout")
    if not all(np.isfinite(atlas[k]).all() for k in ("mean", "coeff", "latent", "explained")) or np.any(latent <= 0):
        raise ValueError("Invalid PCA parameters")
    if not np.allclose(coeff.T @ coeff, np.eye(100), atol=1e-10):
        raise ValueError("PCA basis is not orthonormal")
    return atlas


def shape_at(atlas, mode=None, z=0):
    shape = atlas["mean"].copy()
    if mode is not None:
        if not 1 <= mode <= 100:
            raise ValueError("Modes use one-based indexing 1..100")
        shape += z * np.sqrt(atlas["latent"][mode - 1]) * atlas["coeff"][:, mode - 1]
    return shape[:17430].reshape(-1, 3), shape[17430:].reshape(-1, 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Ignored/external preview and generated meshes directory")
    args = parser.parse_args()
    mat = next(args.atlas_root.rglob("EDESatlas.mat"))
    indices = next(args.atlas_root.rglob("ETIndices.txt"))
    atlas = load_atlas(mat)
    faces = np.loadtxt(indices, dtype=int) - 1
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.min() < 0 or faces.max() >= 5810:
        raise ValueError("Invalid triangle topology")
    summary = {"format": "MATLAB struct EDESatlas", "features": 34860, "points_per_phase": 5810,
               "phases": ["ED", "ES"], "training_subjects": int(atlas["score"].shape[0]),
               "modes": 100, "triangles": len(faces), "explained_first_8_percent": float(sum(atlas["explained"][:8])),
               "explained_percent": atlas["explained"].tolist(),
               "eigenvalues": atlas["latent"].tolist(), "orthonormality_max_error": float(np.max(np.abs(atlas["coeff"].T @ atlas["coeff"] - np.eye(100)))),
               "surfaces": ["LV endocardium", "RV septum", "RV free wall", "epicardium", "valve patches"],
               "physical_units_independently_verified": False,
               "patient_projection_ready": False,
               "blockers": ["5810 homologous points required per phase", "paired ED+ES needed for shipped PCA",
                            "upstream projectOntoAtlas.m targets absent adult UKBRVLV.h5",
                            "need phase, scale and alignment validation", "individual ages/anthropometrics require DUA"]}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    if args.output:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pyvista as pv
        args.output.mkdir(parents=True, exist_ok=True)
        columns = [(None, 0), (1, -3), (1, 3), (8, -3), (8, 3)]
        fig = plt.figure(figsize=(16, 7))
        for col, (mode, z) in enumerate(columns):
            for phase, pts in enumerate(shape_at(atlas, mode, z)):
                ax = fig.add_subplot(2, 5, phase * 5 + col + 1, projection="3d")
                for start, stop, color in [(0, 3072, "#4fa984"), (3072, 6752, "#4a80c1"), (6752, 11616, "#c67370")]:
                    ax.plot_trisurf(*pts.T, triangles=faces[start:stop], color=color, alpha=.65, linewidth=0)
                ax.view_init(15, 65)
                ax.set(xlim=(-70, 70), ylim=(-70, 70), zlim=(-70, 70))
                ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
                ax.set_title(("ED" if phase == 0 else "ES") + (" mean" if mode is None else f" mode {mode}: {z:+d} SD"))
                if mode is None:
                    pv.PolyData(pts, np.column_stack([np.full(len(faces), 3), faces])).save(args.output / f"atlas_mean_{'ED' if phase == 0 else 'ES'}.vtp")
        fig.suptitle("Healthy pediatric atlas: mean and selected PCA modes (model coordinates)")
        fig.tight_layout(); fig.savefig(args.output / "atlas_mean_modes.png", dpi=130); plt.close(fig)
    print({k: v for k, v in summary.items() if k not in {"explained_percent", "eigenvalues"}})


if __name__ == "__main__":
    main()
