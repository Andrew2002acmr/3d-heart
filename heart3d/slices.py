"""Three native slice levels in each reoriented plane, with shared label colors."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
import nibabel as nib
import numpy as np

from .labels import LABELS


def save_slices(case, path, window=None):
    foreground = np.isin(case.mask, list(LABELS))
    if not foreground.any():
        raise ValueError("No cardiac labels for slice overlays")
    if window is None:
        low, high = np.percentile(case.ct[foreground], [1, 99])
        if high <= low:
            low, high = float(case.ct.min()), float(case.ct.max()) + 1
    else:
        low, high = window
        if high <= low:
            raise ValueError("Window maximum must exceed minimum")
    spacing = nib.affines.voxel_sizes(case.affine)
    cmap = np.zeros((65536, 4), dtype=np.float32)
    for label, (_, _, color) in LABELS.items():
        cmap[label] = to_rgba(color, 0.42)
    fig, axes = plt.subplots(3, 3, figsize=(13, 12), layout="constrained")
    selections = []
    # Neurological convention after RAS reorientation. Obliquity stays intact.
    for row, (fixed, title, horizontal, vertical) in enumerate([
        (2, "Axial / native k", "L -> R", "P -> A"),
        (1, "Coronal / native j", "L -> R", "I -> S"),
        (0, "Sagittal / native i", "P -> A", "I -> S"),
    ]):
        other = tuple(i for i in range(3) if i != fixed)
        occupied = np.flatnonzero(foreground.any(axis=other))
        indices = [int(round(x)) for x in np.quantile(occupied, [0.25, 0.5, 0.75])]
        selections.append({"axis": fixed, "indices": indices})
        for ax, index in zip(axes[row], indices):
            ct_slice = np.take(case.ct, index, axis=fixed).T
            mask_slice = np.take(case.mask, index, axis=fixed).T
            aspect = spacing[other[1]] / spacing[other[0]]
            ax.imshow(ct_slice, origin="lower", cmap="gray", vmin=low, vmax=high,
                      interpolation="nearest", aspect=aspect)
            ax.imshow(cmap[mask_slice], origin="lower", interpolation="nearest", aspect=aspect)
            ax.set_title(f"{title} = {index}", fontsize=10)
            ax.set_xlabel(f"{horizontal} (voxel index)", fontsize=9)
            ax.set_ylabel(f"{vertical} (voxel index)", fontsize=9)
            ax.tick_params(labelsize=8)
    fig.legend(handles=[Patch(facecolor=color, label=f"{i}: {short}")
                        for i, (short, _, color) in LABELS.items()],
               loc="outside lower center", ncol=7, frameon=False)
    units = "mm" if case.unit == "mm" else "unknown header units (NOT verified mm)"
    fig.suptitle(f"CT + segmentation | Header RAS (anatomical orientation unverified) | scale: {units}\n"
                 f"Intensity window [{low:.0f}, {high:.0f}] | native planes, no resampling", fontsize=12)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    case.report["slices"] = {"file": path.name, "window": [float(low), float(high)],
                              "selection": selections, "convention": "neurological, RAS header directions",
                              "interpolation": "none; original voxel labels"}
