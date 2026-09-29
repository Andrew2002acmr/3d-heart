"""Triangle-level diagnostics and explicitly qualified volume estimates."""
import numpy as np


def quality(mesh):
    if not mesh.is_all_triangles or not mesh.n_cells:
        raise ValueError("Expected a nonempty triangular surface")
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    points = np.asarray(mesh.points, dtype=np.float64)
    triangles = points[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = np.linalg.norm(cross, axis=1) / 2
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    low, high = edges.min(axis=1), edges.max(axis=1)
    _, inverse, counts = np.unique(low * mesh.n_points + high, return_inverse=True, return_counts=True)
    balance = np.bincount(inverse, weights=np.where(edges[:, 0] < edges[:, 1], 1, -1))
    boundary = int(np.count_nonzero(counts == 1))
    non_manifold = int(np.count_nonzero(counts > 2))
    inconsistent = int(np.count_nonzero((counts == 2) & (balance != 0)))
    degenerate = int(np.count_nonzero(areas <= 1e-12))
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    shifted = triangles - center
    signed = float(np.einsum("ij,ij->i", shifted[:, 0], np.cross(shifted[:, 1], shifted[:, 2])).sum() / 6)
    valid = not (boundary or non_manifold or inconsistent or degenerate)
    connectivity = mesh.connectivity()
    regions, sizes = np.unique(connectivity.cell_data["RegionId"], return_counts=True)
    return {"vertices": mesh.n_points, "triangles": mesh.n_cells,
            "surface_area": float(areas.sum()),
            "algebraic_signed_volume": signed,
            "enclosed_volume": abs(signed) if valid else None,
            "volume_status": "closed_consistently_oriented; self-intersections not tested" if valid else "invalid_topology_or_winding; algebraic value is diagnostic only",
            "boundary_edges": boundary, "non_manifold_edges": non_manifold,
            "inconsistent_winding_edges": inconsistent, "zero_area_triangles": degenerate,
            "surface_components_vertex_connected": len(regions),
            "component_triangle_counts_desc": sorted((int(n) for n in sizes), reverse=True),
            "euler_characteristic": int(mesh.n_points - len(counts) + mesh.n_cells),
            "self_intersections_checked": False}
