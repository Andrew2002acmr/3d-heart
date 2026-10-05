"""Classic CT series -> (column,row,slice) grid with explicit LPS geometry."""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import numpy as np
import pydicom
from pydicom.uid import CTImageStorage

LPS_TO_RAS = np.diag([-1., -1., 1., 1.])


@dataclass
class CTGeometry:
    files: list
    headers: list
    affine_lps: np.ndarray
    sop_to_slice: dict
    report: dict

    @property
    def shape(self):
        return tuple(self.report["shape_xyz"])

    @property
    def affine_ras(self):
        return LPS_TO_RAS @ self.affine_lps


def require_vector(value, length, name):
    vector = np.asarray(value, dtype=float)
    if vector.shape != (length,) or not np.isfinite(vector).all():
        raise ValueError(f"Invalid {name}")
    return vector


def inspect_headers(headers, files=None, expected_sops=None, expected_series=None):
    if len(headers) < 2:
        raise ValueError("Need a complete CT stack with at least two slices")
    files = list(files) if files is not None else [None] * len(headers)
    if len(files) != len(headers):
        raise ValueError("File/header counts differ")
    for h in headers:
        if str(getattr(h, "Modality", "")) != "CT" or int(getattr(h, "NumberOfFrames", 1)) != 1:
            raise ValueError("Only classic single-frame CT supported")
        if str(getattr(h, 'SOPClassUID', '')) != str(CTImageStorage):
            raise ValueError('Classic CT Image Storage required for HU conversion')
    def same(name):
        values = {str(getattr(h, name, "")) for h in headers}
        if len(values) != 1 or "" in values:
            raise ValueError(f"Missing/mixed {name}")
        return next(iter(values))
    series, frame, patient, age = [same(k) for k in ("SeriesInstanceUID", "FrameOfReferenceUID", "PatientID", "PatientAge")]
    if expected_series and series != expected_series:
        raise ValueError("CT series differs from manifest")
    sops = [str(h.SOPInstanceUID) for h in headers]
    if len(set(sops)) != len(sops):
        raise ValueError("Duplicated SOP instance")
    if expected_sops is not None and set(sops) != set(expected_sops):
        raise ValueError(f"Released SOP inventory mismatch: missing={len(set(expected_sops)-set(sops))}, extra={len(set(sops)-set(expected_sops))}")
    orientation = require_vector(headers[0].ImageOrientationPatient, 6, "IOP")
    column_dir, row_dir = orientation[:3], orientation[3:]
    if not np.allclose([column_dir @ column_dir, row_dir @ row_dir, column_dir @ row_dir], [1, 1, 0], atol=1e-5):
        raise ValueError("IOP directions not orthonormal")
    normal = np.cross(column_dir, row_dir)
    pixel_spacing = require_vector(headers[0].PixelSpacing, 2, "PixelSpacing")
    if np.any(pixel_spacing <= 0):
        raise ValueError("Pixel spacing must be positive")
    positions = []
    dimensions = int(headers[0].Rows), int(headers[0].Columns)
    slopes, intercepts, thickness = [], [], []
    for h in headers:
        if (int(h.Rows), int(h.Columns)) != dimensions:
            raise ValueError("Mixed slice dimensions")
        if not np.allclose(require_vector(h.ImageOrientationPatient, 6, "IOP"), orientation, atol=1e-5, rtol=0):
            raise ValueError("Mixed slice orientation")
        if not np.allclose(require_vector(h.PixelSpacing, 2, "PixelSpacing"), pixel_spacing, atol=1e-6, rtol=0):
            raise ValueError("Mixed PixelSpacing")
        positions.append(require_vector(h.ImagePositionPatient, 3, "IPP"))
        if not hasattr(h, "RescaleSlope") or not hasattr(h, "RescaleIntercept"):
            raise ValueError("HU gate: missing RescaleSlope/RescaleIntercept")
        slope, intercept = float(h.RescaleSlope), float(h.RescaleIntercept)
        if not np.isfinite([slope, intercept]).all() or slope == 0:
            raise ValueError("Invalid HU rescale tags")
        if str(getattr(h, "RescaleType", "HU")).strip() not in ("", "HU"):
            raise ValueError("RescaleType does not establish HU")
        slopes.append(slope); intercepts.append(intercept)
        thickness.append(float(h.SliceThickness) if hasattr(h, "SliceThickness") else None)
    positions = np.asarray(positions)
    order = np.argsort(positions @ normal)
    positions = positions[order]
    projected = positions @ normal
    distances = np.diff(projected)
    if np.any(distances < .01):
        raise ValueError("Duplicated/indistinguishable slice positions")
    spacing_z = float(np.median(distances))
    if np.max(np.abs(distances - spacing_z)) > max(.02, .01 * spacing_z):
        raise ValueError("Irregular spacing or missing slice gap; no silent resampling")
    residual = positions - (positions[0] + np.arange(len(headers))[:, None] * normal * spacing_z)
    if np.max(np.linalg.norm(residual, axis=1)) > .05:
        raise ValueError("Non-affine/tilted stack; requires explicit geometry review")
    affine = np.eye(4)
    affine[:3, 0] = column_dir * pixel_spacing[1]
    affine[:3, 1] = row_dir * pixel_spacing[0]
    affine[:3, 2] = normal * spacing_z
    affine[:3, 3] = positions[0]
    sorted_headers = [headers[i] for i in order]
    instance_numbers = [int(h.InstanceNumber) for h in sorted_headers if hasattr(h, "InstanceNumber")]
    report = {"patient_id": patient, "patient_age": age, "series_uid": series, "frame_of_reference_uid": frame,
        "shape_xyz": [dimensions[1], dimensions[0], len(headers)],
        "spacing_xyz_mm": [float(pixel_spacing[1]), float(pixel_spacing[0]), spacing_z],
        "coordinate_system": "LPS", "array_axes": ["column", "row", "slice"],
        "origin_lps_mm": positions[0].tolist(), "orientation_iop": orientation.tolist(),
        "affine_lps": affine.tolist(), "affine_ras": (LPS_TO_RAS @ affine).tolist(),
        "slice_distance_min_max_mm": [float(distances.min()), float(distances.max())],
        "slice_thickness_values_mm": sorted(set(t for t in thickness if t is not None)),
        "max_grid_residual_mm": float(np.max(np.linalg.norm(residual, axis=1))),
        "released_SOP_inventory_complete": expected_sops is not None,
        "source_acquisition_missing_slices_independently_verified": False,
        "sop_set_sha256": hashlib.sha256("\n".join(sorted(sops)).encode()).hexdigest(),
        "regular_spacing": True, "physical_geometry_confirmed_from_DICOM": True,
        "HU_rescale_tags_confirmed": True, "rescale_slope_values": sorted(set(slopes)),
        "rescale_intercept_values": sorted(set(intercepts)), "resampled": False,
        "instance_number_gaps": len(instance_numbers) > 1 and bool(np.any(np.abs(np.diff(instance_numbers)) != 1))}
    return CTGeometry([files[i] for i in order], sorted_headers, affine,
                      {str(h.SOPInstanceUID): k for k, h in enumerate(sorted_headers)}, report)


def inspect_series(directory, expected_sops=None, expected_series=None):
    paths, headers = [], []
    for path in sorted(Path(directory).rglob("*")):
        if not path.is_file():
            continue
        try:
            header = pydicom.dcmread(path, stop_before_pixels=True)
        except pydicom.errors.InvalidDicomError:
            continue  # ZIP license text is not an image.
        if getattr(header, "Modality", None) != "CT":
            raise ValueError("Non-CT DICOM mixed into CT directory")
        paths.append(path); headers.append(header)
    return inspect_headers(headers, paths, expected_sops, expected_series)


def load_volume(geometry):
    data = np.empty(geometry.shape, dtype=np.float32)
    for k, (path, header) in enumerate(zip(geometry.files, geometry.headers)):
        image = pydicom.dcmread(path)
        if str(image.SOPInstanceUID) != str(header.SOPInstanceUID):
            raise ValueError("DICOM changed since header inspection")
        pixels = image.pixel_array
        if pixels.shape != geometry.shape[:2][::-1]:
            raise ValueError("Decoded pixel shape differs")
        data[:, :, k] = pixels.T.astype(np.float32) * float(header.RescaleSlope) + float(header.RescaleIntercept)
    if not np.isfinite(data).all():
        raise ValueError("Nonfinite CT values")
    return data


def save_nifti(data, affine_ras, path):
    import nibabel as nib
    image = nib.Nifti1Image(data, affine_ras)
    image.header.set_xyzt_units("mm")
    image.set_qform(affine_ras, 1); image.set_sform(affine_ras, 1)
    nib.save(image, path)
