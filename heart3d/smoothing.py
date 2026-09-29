"""Two distinct smoothing implementations; never decimate or mutate the input.

PyVista smooth_taubin is Windowed Sinc, NOT a separate lambda/mu algorithm.
Here 'taubin' means the explicit two-step filter from Taubin (SIGGRAPH 1995).
"""
import numpy as np
from scipy.sparse import csr_matrix


LAMBDA = 0.6307
PASS_BAND = 0.1
MU = 1 / (PASS_BAND - 1 / LAMBDA)
PROFILES = {
    "taubin": {level: {"pairs": pairs, "lambda": LAMBDA, "mu": MU, "pass_band": PASS_BAND}
               for level, pairs in (("mild", 5), ("medium", 10), ("strong", 20))},
    "windowed_sinc": {level: {"n_iter": n, "pass_band": band, "window": "nuttall",
                            "boundary_smoothing": False, "feature_smoothing": False,
                            "non_manifold_smoothing": False, "normalize_coordinates": True}
                      for level, n, band in (("mild", 10, .15), ("medium", 20, .1), ("strong", 40, .05))},
}


def validate_surface(mesh):
    if not mesh.is_all_triangles or not mesh.n_cells or not np.isfinite(mesh.points).all():
        raise ValueError("Expected a nonempty triangular mesh with finite coordinates")


def umbrella_operator(mesh):
    """Uniform UNIQUE one-ring neighbours; lock boundary/non-manifold endpoints.

    This conservative policy does not detect non-manifold vertex-only contacts.
    No feature-angle classification or removal of connected components is done.
    """
    validate_surface(mesh)
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges.sort(axis=1)
    keys, counts = np.unique(edges[:, 0]*mesh.n_points + edges[:, 1], return_counts=True)
    u, v = keys // mesh.n_points, keys % mesh.n_points
    rows, cols = np.r_[u, v], np.r_[v, u]
    degree = np.bincount(rows, minlength=mesh.n_points)
    weights = csr_matrix((1/degree[rows], (rows, cols)), shape=(mesh.n_points, mesh.n_points))
    locked = degree == 0
    locked[np.r_[u[counts != 2], v[counts != 2]]] = True
    return weights, locked


def taubin_coordinates(points, weights, locked, pairs, lam=LAMBDA, mu=MU):
    if type(pairs) is not int or pairs < 1 or not (0 < lam < 1 and mu < -lam):
        raise ValueError("Expected positive pair count and 0 < lambda < 1, mu < -lambda")
    # Simultaneous updates, with the mu step using the NEW lambda-step coordinates.
    center = (points.min(axis=0) + points.max(axis=0))/2
    original = np.asarray(points, dtype=np.float64) - center
    result = original.copy()
    for _ in range(pairs):
        for factor in (lam, mu):
            result += factor * (weights @ result - result)
            result[locked] = original[locked]
    return result + center


def smooth_surface(mesh, method, level, operator=None):
    """Each profile starts from the supplied baseline, with double coordinates."""
    validate_surface(mesh)
    if method not in PROFILES or level not in PROFILES[method]:
        raise ValueError("Unknown smoothing method/profile")
    params = PROFILES[method][level]
    surface = mesh.copy(deep=True)
    surface.points = np.asarray(mesh.points, dtype=np.float64).copy()
    if method == "taubin":
        weights, locked = operator if operator is not None else umbrella_operator(mesh)
        surface.points = taubin_coordinates(surface.points, weights, locked, params["pairs"], params["lambda"], params["mu"])
    else:
        surface = surface.smooth_taubin(n_iter=params["n_iter"], pass_band=params["pass_band"],
                                       window_function=params["window"], boundary_smoothing=False,
                                       feature_smoothing=False, non_manifold_smoothing=False,
                                       normalize_coordinates=True, inplace=False)
    # Stored source normals are stale after moving vertices. Recompute lighting
    # normals without splitting vertices or auto-orienting cavity shells.
    surface.compute_normals(point_normals=True, cell_normals=True, split_vertices=False,
                            consistent_normals=False, auto_orient_normals=False, inplace=True)
    if surface.n_points != mesh.n_points or not np.array_equal(surface.faces, mesh.faces):
        raise RuntimeError("Smoothing unexpectedly changed connectivity")
    if not np.isfinite(surface.points).all():
        raise RuntimeError("Smoothing produced non-finite coordinates")
    return surface
