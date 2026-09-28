"""Per-label marching cubes with full affine applied exactly once."""
import numpy as np
import nibabel as nib
import pyvista as pv
from scipy import ndimage
from skimage.measure import marching_cubes

from .labels import LABELS


def build_surface(mask, label, affine):
    occupied = mask == label
    if not occupied.any():
        return None, None
    locations = [np.flatnonzero(occupied.any(axis=tuple(j for j in range(3) if j != i))) for i in range(3)]
    start = np.array([x[0] for x in locations])
    stop = np.array([x[-1] + 1 for x in locations])
    crop = occupied[tuple(slice(a, b) for a, b in zip(start, stop))]
    touches_boundary = bool(np.any(start == 0) or np.any(stop == mask.shape))
    # A background rim closes structures at the crop boundary. If the structure
    # reaches the original scan boundary, this also creates artificial end caps.
    padded = np.pad(crop.astype(np.uint8), 1)
    vertices, faces, _, _ = marching_cubes(padded, level=0.5, step_size=1,
                                          allow_degenerate=False, gradient_direction="ascent")
    vertices = nib.affines.apply_affine(affine, vertices - 1 + start)
    if np.linalg.det(affine[:3, :3]) < 0:
        faces = faces[:, ::-1]
    vtk_faces = np.column_stack((np.full(len(faces), 3), faces))
    mesh = pv.PolyData(vertices, vtk_faces)
    # Marching cubes already orients the boundary of the occupied material.
    # Auto-orient would turn enclosed cavity shells outwards, adding their
    # volume instead of subtracting it. Preserve the signed boundary winding.
    mesh = mesh.compute_normals(auto_orient_normals=False, consistent_normals=True, split_vertices=False)
    components, number = ndimage.label(crop, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(components.ravel())[1:]
    voxel_count = int(crop.sum())
    boundary_edges = mesh.extract_feature_edges(boundary_edges=True, non_manifold_edges=False,
                                               feature_edges=False, manifold_edges=False).n_cells
    non_manifold_edges = mesh.extract_feature_edges(boundary_edges=False, non_manifold_edges=True,
                                                   feature_edges=False, manifold_edges=False).n_cells
    info = {"label": label, "short_name": LABELS[label][0], "name": LABELS[label][1],
            "color": LABELS[label][2], "voxels": voxel_count,
            "components_26_connected": int(number), "largest_component_voxels": int(sizes.max()),
            "touches_scan_boundary": touches_boundary, "artificial_boundary_caps": touches_boundary,
            "points": mesh.n_points, "triangles": mesh.n_cells,
            "boundary_edges": boundary_edges, "non_manifold_edges": non_manifold_edges,
            "bounds": list(mesh.bounds),
            "voxel_volume_in_coordinate_units_cubed": float(voxel_count * abs(np.linalg.det(affine[:3, :3]))),
            "surface_area_in_coordinate_units_squared": float(mesh.area),
            "smoothing": False, "decimation": False, "components_removed": False}
    info["normal_policy"] = "preserve marching-cubes material boundary; no automatic shell orientation"
    return mesh, info


def export_surfaces(case, directory):
    directory.mkdir(parents=True, exist_ok=True)
    meshes, summaries = {}, []
    for label in LABELS:
        mesh, summary = build_surface(case.mask, label, case.affine)
        if mesh is None:
            continue
        mesh.field_data["label_id"] = [label]
        mesh.field_data["coordinate_system"] = ["RAS+ (NIfTI header)"]
        mesh.field_data["unit"] = [case.unit]
        filename = f"{label:02d}_{LABELS[label][0]}.vtp"
        mesh.save(directory / filename)
        summary["file"] = f"meshes/{filename}"
        summaries.append(summary)
        meshes[label] = mesh
        print(f"  {LABELS[label][0]}: {mesh.n_points:,} vertices, {mesh.n_cells:,} triangles", flush=True)
        if summary["touches_scan_boundary"]:
            case.report["warnings"].append(f"{LABELS[label][0]} touches the scan boundary; marching cubes creates artificial end caps")
        if summary["components_26_connected"] > 1:
            case.report["warnings"].append(f"{LABELS[label][0]} has {summary['components_26_connected']} voxel components (26-neighbour); all retained")
        if summary["boundary_edges"] or summary["non_manifold_edges"]:
            case.report["warnings"].append(f"{LABELS[label][0]} mesh: {summary['boundary_edges']} boundary edges, {summary['non_manifold_edges']} non-manifold edges; no automatic topology repair")
    if not meshes:
        raise ValueError("No cardiac labels 1-7 to reconstruct")
    case.report["structures"] = summaries
    return meshes
