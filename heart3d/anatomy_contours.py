"""Rasterize supplied native-XY research contours on one admitted CT plane.

This deterministic geometry operation performs no inference, image interpretation
or network transfer. It cannot establish the anatomical correctness of contours.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion
from skimage.draw import polygon

from heart3d.anatomy_harness import load_admitted_ct
from heart3d.storage import sha256_file


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError("Nonfinite JSON value")


def _safe_json(value, depth=0, budget=None):
    if budget is None:
        budget = [20000]
    budget[0] -= 1
    if budget[0] < 0 or depth > 12:
        raise ValueError("Annotation protocol exceeds bounded JSON size/depth")
    if value is None or type(value) is bool:
        return
    if type(value) in (int, float):
        if not np.isfinite(value):
            raise ValueError("Annotation protocol must contain finite numbers")
        return
    if isinstance(value, str):
        if len(value) > 8000:
            raise ValueError("Annotation protocol string is too long")
        return
    if isinstance(value, list):
        for child in value:
            _safe_json(child, depth + 1, budget)
        return
    if isinstance(value, dict) and all(isinstance(key, str) and len(key) <= 200 for key in value):
        for child in value.values():
            _safe_json(child, depth + 1, budget)
        return
    raise ValueError("Annotation protocol must be plain JSON")


def _cross(a, b, c):
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _on_segment(a, b, point):
    return abs(_cross(a, b, point)) <= 1e-8 and all(
        min(a[axis], b[axis]) - 1e-8 <= point[axis] <= max(a[axis], b[axis]) + 1e-8
        for axis in (0, 1))


def _intersects(a, b, c, d):
    ab_c, ab_d, cd_a, cd_b = _cross(a, b, c), _cross(a, b, d), _cross(c, d, a), _cross(c, d, b)
    if ((ab_c > 1e-8 and ab_d < -1e-8) or (ab_c < -1e-8 and ab_d > 1e-8)) and (
            (cd_a > 1e-8 and cd_b < -1e-8) or (cd_a < -1e-8 and cd_b > 1e-8)):
        return True
    return any((_on_segment(a, b, c), _on_segment(a, b, d),
                _on_segment(c, d, a), _on_segment(c, d, b)))


def _polygon_mask(vertices, shape_xy):
    if not isinstance(vertices, list) or not 3 <= len(vertices) <= 2048:
        raise ValueError("Polygon needs 3..2048 native-XY vertices")
    if any(not isinstance(point, list) or len(point) != 2 or
           any(type(coordinate) not in (int, float) for coordinate in point) for point in vertices):
        raise ValueError("Polygon vertices must be finite numeric [x,y] pairs")
    points = np.array(vertices, dtype=np.float64)
    if not np.isfinite(points).all():
        raise ValueError("Polygon vertices must be finite")
    if np.array_equal(points[0], points[-1]):
        points = points[:-1]
    if len(points) < 3 or len(np.unique(points, axis=0)) != len(points):
        raise ValueError("Polygon vertices must be distinct")
    if np.any(points < 0) or np.any(points > np.array(shape_xy) - 1):
        raise ValueError("Polygon vertices outside original CT; clipping is forbidden")
    area_twice = np.sum(points[:, 0] * np.roll(points[:, 1], -1) -
                        points[:, 1] * np.roll(points[:, 0], -1))
    if abs(area_twice) < 1e-8:
        raise ValueError("Polygon is degenerate")
    for first in range(len(points)):
        for second in range(first + 1, len(points)):
            if second == first + 1 or (first == 0 and second == len(points) - 1):
                continue
            if _intersects(points[first], points[(first + 1) % len(points)],
                           points[second], points[(second + 1) % len(points)]):
                raise ValueError("Self-intersecting polygon is forbidden")
    # skimage receives row=Y, column=X; the stored plane is explicitly [X,Y].
    rows_y, columns_x = polygon(points[:, 1], points[:, 0], shape=(shape_xy[1], shape_xy[0]))
    mask = np.zeros(shape_xy, dtype=bool)
    mask[columns_x, rows_y] = True
    if not mask.any():
        raise ValueError("Polygon contains no voxel centres")
    return mask


def validate_recipe(recipe, ct_digest, shape):
    expected = {"schema_version", "ct_sha256", "target", "source", "annotation_protocol", "labels"}
    if not isinstance(recipe, dict) or set(recipe) != expected:
        raise ValueError("Recipe fields differ from schema 1")
    if type(recipe["schema_version"]) is not int or recipe["schema_version"] != 1:
        raise ValueError("Recipe schema_version must be 1")
    if recipe["ct_sha256"] != ct_digest:
        raise ValueError("Recipe CT hash differs from admitted CT")
    if recipe["source"] not in ("cloud_model", "local_model", "human"):
        raise ValueError("Recipe source must preserve cloud_model/local_model/human provenance")
    target = recipe["target"]
    if (not isinstance(target, dict) or set(target) != {"axis", "index"} or
            type(target["axis"]) is not int or target["axis"] != 2 or
            type(target["index"]) is not int or not 0 <= target["index"] < shape[2]):
        raise ValueError("Target must be one original-grid axial slice")
    if not isinstance(recipe["annotation_protocol"], dict):
        raise ValueError("Annotation protocol must be a JSON object")
    _safe_json(recipe["annotation_protocol"])
    labels = recipe["labels"]
    if not isinstance(labels, list) or not 1 <= len(labels) <= 7:
        raise ValueError("Recipe needs 1..7 explicit label contours")
    masks, details = {}, {}
    for row in labels:
        if not isinstance(row, dict) or set(row) != {"label", "outer", "holes", "confidence", "boundary_uncertainty"}:
            raise ValueError("Label contour fields differ from schema 1")
        label = row["label"]
        if type(label) is not int or label not in range(1, 8) or label in masks:
            raise ValueError("Label IDs must be distinct integers 1..7")
        if row["confidence"] not in ("low", "medium", "high"):
            raise ValueError("Invalid contour confidence")
        uncertainty = row["boundary_uncertainty"]
        if (not isinstance(uncertainty, list) or len(uncertainty) > 50 or
                any(not isinstance(item, str) or not item.strip() or len(item) > 4000 for item in uncertainty)):
            raise ValueError("Boundary uncertainty must contain bounded nonempty text")
        outer = _polygon_mask(row["outer"], shape[:2])
        holes = row["holes"]
        if not isinstance(holes, list) or len(holes) > 16:
            raise ValueError("At most 16 explicit polygon holes are allowed")
        mask = outer.copy()
        for hole in holes:
            removed = _polygon_mask(hole, shape[:2])
            if np.any(removed & ~outer):
                raise ValueError("Hole voxel centres extend outside their outer polygon")
            # Hole boundaries are removed inclusively; a matching LV outer contour
            # therefore owns its endocardial pixels instead of the MYO ring.
            mask &= ~removed
        if not mask.any():
            raise ValueError("Label has no foreground after subtracting holes")
        masks[label] = mask
        details[label] = {"confidence": row["confidence"], "boundary_uncertainty": uncertainty,
                          "contour_voxels_before_boundary_ownership": int(mask.sum())}
    boundaries = {label: mask & ~binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)
                  for label, mask in masks.items()}
    conflicts = []
    order = sorted(masks)
    for index, first in enumerate(order):
        for second in order[index + 1:]:
            overlap = masks[first] & masks[second]
            if not overlap.any():
                continue
            if np.any(overlap & ~(boundaries[first] & boundaries[second])):
                raise ValueError(f"Interior positive overlap between labels {first} and {second}")
            conflicts.append({"labels": [first, second], "shared_boundary_voxels": int(overlap.sum()),
                              "owner": first})
    plane = np.zeros(shape[:2], dtype=np.uint8)
    for label in order:
        plane[masks[label] & (plane == 0)] = label
        count = int(np.count_nonzero(plane == label))
        if not count:
            raise ValueError("Boundary ownership would remove an entire positive label")
        details[label]["saved_voxels"] = count
    return plane, details, conflicts


def rasterize(ct, admission, recipe_path, out):
    ct, admission, recipe_path, out = map(Path, (ct, admission, recipe_path, out))
    if out.exists() or out.is_symlink():
        raise ValueError("Choose a new output directory; existing files are preserved")
    if recipe_path.stat().st_size > 2 << 20:
        raise ValueError("Recipe exceeds 2 MiB")
    raw_recipe = recipe_path.read_bytes()
    recipe = json.loads(raw_recipe.decode("utf-8-sig"), object_pairs_hook=_unique_object,
                        parse_constant=_nonfinite)
    hashes = {name: sha256_file(path) for name, path in
              (("ct", ct), ("admission", admission), ("recipe", recipe_path))}
    source = load_admitted_ct(ct, admission)
    plane, details, conflicts = validate_recipe(recipe, hashes["ct"], source.shape)
    target = recipe["target"]["index"]
    data = np.zeros(source.shape, dtype=np.uint8)
    data[:, :, target] = plane
    header = source.header.copy()
    header.extensions.clear()
    for field in ("data_type", "db_name", "descrip", "aux_file", "intent_name"):
        if field in header:
            header[field] = b""
    header.set_data_dtype(np.uint8)
    header.set_slope_inter(1., 0.)
    header["intent_code"] = 0
    header["cal_min"], header["cal_max"] = 0, 7
    image = source.__class__(data, source.affine, header)
    image.set_qform(source.get_qform(), int(source.header["qform_code"]))
    image.set_sform(source.get_sform(), int(source.header["sform_code"]))
    image.header.set_slope_inter(1., 0.)
    image.header.set_xyzt_units("mm")
    if any(sha256_file(path) != hashes[name] for name, path in
           (("ct", ct), ("admission", admission), ("recipe", recipe_path))):
        raise ValueError("Input changed during contour validation")
    # No output is created until admission, recipe, geometry and overlap checks pass.
    out.mkdir(parents=True, exist_ok=False)
    candidate = out / "candidate.nii.gz"
    with candidate.open("xb") as handle:
        with gzip.GzipFile(filename="", fileobj=handle, mode="wb", mtime=0) as compressed:
            image.to_file_map({"image": nib.FileHolder(fileobj=compressed)})
    with (out / "recipe.json").open("xb") as handle:
        handle.write(raw_recipe)
    manifest = {"schema_version": 1, "kind": "anatomy_contour_rasterization",
        "created_utc": datetime.now(timezone.utc).isoformat(), "source": recipe["source"],
        "expert_approved": False, "medical_correctness_verified": False,
        "rasterizer_inference_performed": False, "rasterizer_image_cloud_transfer": False,
        "scope": {"axis": 2, "index": target, "complete_volume": False},
        "unannotated_planes_are_background_GT": False,
        "ct_sha256": hashes["ct"], "admission_sha256": hashes["admission"],
        "recipe": {"file": "recipe.json", "sha256": hashes["recipe"], "copied_without_modification": True},
        "candidate": {"file": candidate.name, "sha256": sha256_file(candidate), "dtype": "uint8"},
        "shape_ijk": list(source.shape), "affine_ras": source.affine.tolist(),
        "spacing_mm": nib.affines.voxel_sizes(source.affine).tolist(), "unit": "mm",
        "pixel_rule": "native voxel centres at integer XY; polygon boundary included; holes removed inclusively",
        "overlap_policy": "interior overlap rejected; shared raster boundary only; lower label ID owns each shared pixel",
        "boundary_conflicts": conflicts, "boundary_conflict_pair_voxels": sum(row["shared_boundary_voxels"] for row in conflicts),
        "positive_labels": sorted(details), "labels": {str(label): details[label] for label in sorted(details)},
        "annotation_protocol": recipe["annotation_protocol"],
        "code_sha256": {"heart3d/anatomy_contours.py": sha256_file(Path(__file__)),
                        "heart3d/anatomy_harness.py": sha256_file(Path(load_admitted_ct.__code__.co_filename))}}
    if any(sha256_file(path) != hashes[name] for name, path in
           (("ct", ct), ("admission", admission), ("recipe", recipe_path))):
        raise ValueError("Input changed during rasterization; output is incomplete")
    manifest["manifest_sha256"] = hashlib.sha256(_json_bytes(manifest)).hexdigest()
    with (out / "manifest.json").open("xb") as handle:
        handle.write(_json_bytes(manifest) + b"\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("ct", "admission", "recipe", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        result = rasterize(args.ct, args.admission, args.recipe, args.out)
    except (ValueError, OSError, KeyError, TypeError, OverflowError, np.linalg.LinAlgError,
            nib.filebasedimages.ImageFileError) as error:
        parser.exit(2, f"Contour rasterization rejected: {error}\n")
    print(json.dumps({"candidate_created": True, "scope": result["scope"],
                      "positive_labels": result["positive_labels"],
                      "boundary_conflict_pair_voxels": result["boundary_conflict_pair_voxels"],
                      "expert_approved": False, "medical_correctness_verified": False}))


if __name__ == "__main__":
    main()
