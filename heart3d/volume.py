"""NIfTI loading and explicit geometry checks; no registration or resampling."""
from dataclasses import dataclass
import hashlib
from pathlib import Path

import nibabel as nib
import numpy as np

from .labels import LABELS


@dataclass
class Case:
    ct: np.ndarray
    mask: np.ndarray
    affine: np.ndarray
    unit: str
    report: dict


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def image_geometry(image, name, warnings):
    if len(image.shape) != 3 or min(image.shape) < 2:
        raise ValueError(f"{name}: expected a 3D volume with at least 2 voxels per axis")
    affine = np.asarray(image.affine, dtype=float)
    if (not np.isfinite(affine).all() or
            abs(np.linalg.det(affine[:3, :3])) < 1e-12 or
            not np.allclose(affine[3], [0, 0, 0, 1])):
        raise ValueError(f"{name}: invalid or singular affine")
    zooms = np.asarray(image.header.get_zooms()[:3], dtype=float)
    if not np.isfinite(zooms).all() or np.any(zooms <= 0):
        raise ValueError(f"{name}: invalid header spacing")
    spacing = nib.affines.voxel_sizes(affine)
    if not np.allclose(zooms, spacing, atol=1e-4, rtol=1e-4):
        warnings.append(f"{name}: pixdim differs from affine spacing; using selected affine")
    qform, qcode = image.get_qform(coded=True)
    sform, scode = image.get_sform(coded=True)
    if not qcode and not scode:
        raise ValueError(f"{name}: neither qform nor sform defines orientation")
    if qcode and scode and not np.allclose(qform, sform, atol=1e-3, rtol=0):
        warnings.append(f"{name}: qform and sform disagree; nibabel selected sform. Verify orientation manually.")
    unit = image.header.get_xyzt_units()[0]
    if unit not in ("unknown", "mm", "meter", "micron"):
        raise ValueError(f"{name}: unsupported spatial unit {unit}")
    scale = {"unknown": 1.0, "mm": 1.0, "meter": 1000.0, "micron": 0.001}[unit]
    normalized = affine.copy()
    normalized[:3, :] *= scale
    info = {
        "shape": list(image.shape), "dtype": str(image.header.get_data_dtype()),
        "header_spacing": zooms.tolist(), "affine_spacing": spacing.tolist(),
        "spatial_unit": unit, "axis_codes": list(nib.aff2axcodes(affine)),
        "affine": affine.tolist(), "qform_code": int(qcode), "sform_code": int(scode),
        "selected_transform": "sform" if scode else "qform",
        "qform": None if qform is None else qform.tolist(),
        "sform": None if sform is None else sform.tolist(),
        "obliquity_degrees": np.rad2deg(nib.affines.obliquity(affine)).tolist(),
    }
    return info, normalized, "unknown" if unit == "unknown" else "mm"


def load_case(ct_path, mask_path):
    warnings = []
    ct_image, mask_image = nib.load(str(ct_path)), nib.load(str(mask_path))
    ct_info, ct_affine, unit = image_geometry(ct_image, "CT", warnings)
    mask_info, mask_affine, mask_unit = image_geometry(mask_image, "Mask", warnings)
    if unit != mask_unit:
        raise ValueError("CT/mask spatial units cannot be compared: one is unspecified")
    if unit == "unknown":
        warnings.append("Spatial units are unspecified. Header coordinates are NOT confirmed millimetres; physical dimensions and volumes are unverified.")
    warnings.append("Anatomical orientation is taken from the NIfTI header, not independently verified against DICOM or landmarks.")
    ct = ct_image.get_fdata(dtype=np.float32)
    raw_mask = np.asanyarray(mask_image.dataobj)
    if not np.isfinite(ct).all() or not np.isfinite(raw_mask).all():
        raise ValueError("CT/mask contains NaN or infinity")
    values = np.unique(raw_mask)
    if np.any(values < 0) or np.any(values > 65535) or not np.equal(values, np.rint(values)).all():
        raise ValueError("Mask must contain nonnegative integer label IDs, not probabilities")
    if not np.any(values > 0):
        raise ValueError("Mask is empty")
    unknown = sorted(set(int(v) for v in values) - {0, *LABELS})
    if unknown:
        warnings.append(f"Non-cardiac/unknown labels {unknown} are reported but excluded from cardiac meshes and overlays")
    missing = sorted(set(LABELS) - set(values))
    if missing:
        warnings.append(f"Cardiac labels absent: {missing}. Absence is not interpreted as a diagnosis.")
    # Axis permutation/flip only: no interpolation and no change to original files.
    ct_canonical = nib.as_closest_canonical(nib.Nifti1Image(ct, ct_affine))
    mask_canonical = nib.as_closest_canonical(nib.Nifti1Image(raw_mask.astype(np.uint16), mask_affine))
    if ct_canonical.shape != mask_canonical.shape:
        raise ValueError(f"CT/mask shapes differ after lossless reorientation: {ct_canonical.shape} vs {mask_canonical.shape}")
    if not np.allclose(ct_canonical.affine, mask_canonical.affine, atol=1e-4, rtol=0):
        raise ValueError("CT/mask affines differ after lossless reorientation; overlay would be invalid. No automatic resampling performed.")
    ct, mask = np.asarray(ct_canonical.dataobj), np.asarray(mask_canonical.dataobj)
    affine = ct_canonical.affine
    if np.max(nib.affines.obliquity(affine)) > np.deg2rad(0.1):
        warnings.append("Oblique acquisition: overlays show native reoriented planes, not resampled anatomical MPR planes")
    spacing = nib.affines.voxel_sizes(affine)
    directions = affine[:3, :3] / spacing
    if not np.allclose(directions.T @ directions, np.eye(3), atol=1e-4):
        warnings.append("Affine contains shear: 3D uses full affine; 2D rectangular plots do not reproduce shear angles")
    ids, counts = np.unique(mask, return_counts=True)
    report = {
        "ct": {"path": str(Path(ct_path).resolve()), "sha256": sha256(ct_path), **ct_info},
        "mask": {"path": str(Path(mask_path).resolve()), "sha256": sha256(mask_path), **mask_info},
        "geometry": {"grids_match_after_lossless_reorientation": True, "shape": list(ct.shape),
                     "affine": affine.tolist(), "spacing": spacing.tolist(), "unit": unit,
                     "axis_codes": list(nib.aff2axcodes(affine)), "coordinates": "RAS+ according to NIfTI header",
                     "resampled": False, "spatial_unit_defined": unit != "unknown",
                     "physical_scale_independently_verified": False,
                     "anatomical_orientation_independently_verified": False},
        "intensity": {"min": float(ct.min()), "max": float(ct.max()),
                      "note": "NIfTI scaling applied; HU calibration is not independently verified"},
        "label_counts": {str(i): int(n) for i, n in zip(ids, counts)},
        "ignored_labels": unknown, "missing_labels": missing, "warnings": warnings,
    }
    return Case(ct, mask, affine, unit, report)


def discover(root):
    pairs = []
    issues = []
    files = sorted(p for p in Path(root).rglob("*") if p.name.endswith((".nii", ".nii.gz")))
    seen = set()
    for ct in files:
        if "_image.nii" not in ct.name:
            continue
        mask = ct.with_name(ct.name.replace("_image.nii", "_label.nii"))
        if mask.is_file():
            pairs.append({"case": ct.name.split("_image.nii")[0], "ct": str(ct), "mask": str(mask)})
            seen.update((ct, mask))
        else:
            issues.append(f"No paired mask: {ct}")
    issues.extend(f"Unpaired/unrecognized NIfTI: {p}" for p in files if p not in seen and "_image.nii" not in p.name)
    return {"root": str(Path(root).resolve()), "pairs": pairs, "issues": issues}
