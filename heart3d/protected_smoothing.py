"""Exploratory stronger filtering: retain every non-largest component exactly.

Largest means triangle count, not anatomical importance. This is not a topology
repair or protection of thin branches within the largest connected component.
"""
import numpy as np

from .smoothing import validate_surface


PROFILES = {"protected_40": (40, .05), "protected_60": (60, .01),
            "protected_100": (100, .001)}


def protected_points(mesh):
    validate_surface(mesh)
    tagged = mesh.copy(deep=True)
    tagged.point_data["source_point_id"] = np.arange(mesh.n_points)
    tagged.cell_data["source_face_id"] = np.arange(mesh.n_cells)
    connected = tagged.connectivity()
    regions, sizes = np.unique(connected.cell_data["RegionId"], return_counts=True)
    largest = regions[np.argmax(sizes)]
    locked = np.ones(mesh.n_points, dtype=bool)
    locked[connected.point_data["source_point_id"][connected.point_data["RegionId"] == largest]] = False
    return locked


def smooth_primary(mesh, profile):
    """Filter a copy and restore secondary components, preserving vertex order."""
    n_iter, band = PROFILES[profile]
    locked = protected_points(mesh)
    result = mesh.copy(deep=True)
    result.points = mesh.points.astype(np.float64).copy()
    result = result.smooth_taubin(n_iter=n_iter, pass_band=band, window_function="nuttall",
        boundary_smoothing=False, feature_smoothing=False, non_manifold_smoothing=False,
        normalize_coordinates=True, inplace=False)
    if result.n_points != mesh.n_points or not np.array_equal(result.faces, mesh.faces):
        raise RuntimeError("Filter changed connectivity")
    points = result.points.copy()
    points[locked] = mesh.points[locked]
    result.points = points
    result.compute_normals(point_normals=True, cell_normals=True, split_vertices=False,
        consistent_normals=False, auto_orient_normals=False, inplace=True)
    if not np.isfinite(result.points).all():
        raise RuntimeError("Non-finite coordinates")
    return result, locked
