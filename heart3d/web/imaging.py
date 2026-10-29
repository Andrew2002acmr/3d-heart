"""Canonical Python rasterization; browser points are Cornerstone LPS millimetres."""
from pathlib import Path
import numpy as np
import nibabel as nib
from skimage.draw import polygon

from heart3d.dicom.ct import LPS_TO_RAS
from heart3d.volume import image_geometry
from heart3d.storage import sha256_file
from heart3d.labels import LABELS

SCHEMA = "cardiac7-v1"


def labels():
    return [{"id": k, "code": v[0], "name": v[1], "color": v[2]} for k, v in LABELS.items()]


def geometry(image):
    warnings = []
    info, affine, unit = image_geometry(image, "CT", warnings)
    if unit != "mm":
        raise ValueError("Web annotation requires a verified millimetre grid")
    spacing = nib.affines.voxel_sizes(affine)
    directions = affine[:3, :3] / spacing
    if not np.allclose(directions.T @ directions, np.eye(3), atol=1e-5):
        raise ValueError("Sheared grids need a separately validated adapter")
    lps = LPS_TO_RAS @ affine
    return {"shape": list(image.shape), "affine_ras": affine.tolist(),
            "origin_lps": lps[:3, 3].tolist(), "spacing": spacing.tolist(),
            "direction_lps": (lps[:3, :3] / spacing).flatten(order="F").tolist(),
            "unit": "mm", "axis_codes": info["axis_codes"], "physical_geometry_verified": True}


def phantom(destination):
    destination = Path(destination)
    x, y, z = np.indices((96, 96, 64), dtype=np.float32)
    body = ((x - 48) / 40) ** 2 + ((y - 48) / 34) ** 2 < 1
    ct = np.where(body, 35, -1000).astype(np.float32)
    for cx in (28, 68):
        lung = ((x - cx) / 15) ** 2 + ((y - 47) / 23) ** 2 + ((z - 32) / 35) ** 2 < 1
        ct[lung] = -650
    heart = ((x - 48) / 20) ** 2 + ((y - 47) / 19) ** 2 + ((z - 31) / 23) ** 2 < 1
    ct[heart] = 115
    for cx, cy, cz, rx, ry, rz in [(43, 44, 28, 9, 10, 15), (56, 45, 28, 8, 9, 14),
                                 (42, 55, 39, 8, 7, 9), (56, 56, 40, 8, 7, 9)]:
        inside = ((x-cx)/rx)**2 + ((y-cy)/ry)**2 + ((z-cz)/rz)**2 < 1
        ct[inside] = 270
    for cx, cy in [(47, 58), (55, 59)]:
        ct[((x-cx)**2 + (y-cy)**2 < 20) & (z > 39)] = 350
    ct += np.random.default_rng(20261022).normal(0, 3, ct.shape).astype(np.float32)
    affine = np.diag([.9, 1.1, 1.5, 1.])
    affine[:3, 3] = [-43.2, -52.8, -48]
    image = nib.Nifti1Image(ct, affine)
    image.header.set_xyzt_units("mm")
    image.set_qform(affine, 1)
    image.set_sform(affine, 1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, destination)
    return geometry(image)


def rasterize(document, grid, base=None):
    shape = tuple(grid["shape"])
    inverse = np.linalg.inv(LPS_TO_RAS @ np.asarray(grid["affine_ras"]))
    mask = np.zeros(shape, dtype=np.uint8) if base is None else np.array(base, dtype=np.uint8, copy=True)
    if mask.shape != shape or np.any(mask > 7):
        raise ValueError("Base prediction differs from CT grid/label schema")
    contours = {c["id"]: c for c in document.get("contours", [])}
    if len(contours) != len(document.get("contours", [])):
        raise ValueError("Duplicate contour IDs")
    for contour in contours.values():
        points = np.asarray(contour["points"], dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not 1 <= len(points) <= 2048 or not np.isfinite(points).all():
            raise ValueError("Invalid control points")
        axis, index = contour["axis"], contour["slice"]
        if axis not in (0, 1, 2) or not 0 <= index < shape[axis] or contour["label"] not in LABELS:
            raise ValueError("Invalid contour plane/class")
        xyz = nib.affines.apply_affine(inverse, points)
        if np.max(np.abs(xyz[:, axis] - index)) > .05 or np.any(xyz < -.01) or np.any(xyz > np.asarray(shape) - 1 + .01):
            raise ValueError("Contour does not belong to the original CT grid/plane")
    replaced = set()
    for operation in document.get("operations", []):
        if operation["kind"] == "contour":
            c = contours.get(operation["contour_id"])
            if not c or not c["applied"] or not c["closed"]:
                continue
            if len(c["points"]) < 3:
                raise ValueError("Closed contour needs at least three control points")
            xyz = nib.affines.apply_affine(inverse, c["points"])
            # An affine round trip may put an exact boundary at 1.999999999.
            # Snap only numerical noise, preserving genuinely subvoxel vertices.
            rounded = np.rint(xyz)
            xyz = np.where(np.abs(xyz - rounded) < 1e-6, rounded, xyz)
            other = [i for i in range(3) if i != c["axis"]]
            a, b = polygon(xyz[:, other[0]], xyz[:, other[1]], shape=tuple(shape[i] for i in other))
            selection = [None] * 3
            selection[c["axis"]] = c["slice"]
            selection[other[0]], selection[other[1]] = a, b
            key = (c["axis"], c["slice"], c["label"])
            if c.get("replace_label_slice") and key not in replaced:
                plane_selection = [slice(None)] * 3
                plane_selection[c["axis"]] = c["slice"]
                plane = mask[tuple(plane_selection)]
                plane[plane == c["label"]] = 0
                replaced.add(key)
            mask[tuple(selection)] = c["label"]
        elif operation["kind"] == "brush":
            indices = np.asarray(operation["indices"], dtype=np.int64)
            values = np.asarray(operation["values"], dtype=np.int64)
            if len(indices) != len(values) or indices.ndim != 1 or len(indices) > np.prod(shape):
                raise ValueError("Invalid native brush patch")
            if np.any(indices < 0) or np.any(indices >= np.prod(shape)) or np.any(values < 0) or np.any(values > 7):
                raise ValueError("Brush patch outside original grid/label schema")
            # Cornerstone's scalar layout is x-fastest, exactly NumPy Fortran order.
            x, y, z = np.unravel_index(indices, shape, order="F")
            mask[x, y, z] = values.astype(np.uint8)
        else:
            raise ValueError("Unknown annotation operation")
    return mask


def trace_mask_contours(mask, grid, label, axis, index):
    """Expose editable original-grid boundaries, refusing holes/FOV clipping."""
    from scipy.ndimage import binary_fill_holes
    from skimage.measure import find_contours, approximate_polygon
    from uuid import uuid4
    if axis not in (0, 1, 2) or label not in LABELS or not 0 <= index < mask.shape[axis]:
        raise ValueError("Invalid plane/class")
    plane = np.take(mask == label, index, axis=axis)
    if np.any(binary_fill_holes(plane) != plane):
        raise ValueError("Slice contains holes; use brush/eraser to preserve topology")
    if plane[0].any() or plane[-1].any() or plane[:, 0].any() or plane[:, -1].any():
        raise ValueError("Boundary reaches CT edge; use brush/eraser")
    other = [i for i in range(3) if i != axis]
    result = []
    for points in find_contours(np.pad(plane.astype(float), 1), .5):
        points = approximate_polygon(points - 1, tolerance=.2)[:-1]
        if not 3 <= len(points) <= 2048:
            raise ValueError("Boundary too complex for point editing; use brush/eraser")
        xyz = np.empty((len(points), 3))
        xyz[:, axis] = index
        xyz[:, other] = points
        world = nib.affines.apply_affine(LPS_TO_RAS @ np.asarray(grid["affine_ras"]), xyz)
        result.append({"id": str(uuid4()), "label": label, "axis": axis, "slice": index,
                       "points": world.tolist(), "closed": True, "applied": True, "replace_label_slice": True})
    return result


def save_mask(mask, source, destination):
    source = nib.load(source)
    image = nib.Nifti1Image(mask.astype(np.uint8), source.affine, source.header.copy())
    for name in ("qform", "sform"):
        matrix, code = getattr(source, "get_" + name)(coded=True)
        getattr(image, "set_" + name)(matrix, int(code))
    image.set_data_dtype(np.uint8)
    image.header.set_slope_inter(1, 0)
    nib.save(image, destination)
    reloaded = nib.load(destination)
    if reloaded.shape != source.shape or not np.allclose(reloaded.affine, source.affine, atol=1e-6):
        raise ValueError("Saved mask geometry verification failed")
    return sha256_file(destination)
