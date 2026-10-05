"""RTSTRUCT Heart polygons -> original CT grid, with no interpolation."""
import numpy as np
from skimage.draw import polygon
from scipy import ndimage
from pydicom.uid import CTImageStorage


def referenced_series(rt):
    return [(str(f.FrameOfReferenceUID), str(s.SeriesInstanceUID),
             [str(i.ReferencedSOPInstanceUID) for i in getattr(s, "ContourImageSequence", [])])
            for f in getattr(rt, "ReferencedFrameOfReferenceSequence", [])
            for study in getattr(f, "RTReferencedStudySequence", [])
            for s in getattr(study, "RTReferencedSeriesSequence", [])]


def heart_item(rt):
    definitions = [r for r in getattr(rt, "StructureSetROISequence", []) if str(r.ROIName).strip().casefold() == "heart"]
    if len(definitions) != 1:
        raise ValueError("Missing/ambiguous Heart ROI")
    matches = [r for r in getattr(rt, "ROIContourSequence", []) if int(r.ReferencedROINumber) == int(definitions[0].ROINumber)]
    if len(matches) != 1 or not getattr(matches[0], "ContourSequence", []):
        raise ValueError("Heart ROI has no complete nonempty contours")
    return definitions[0], matches[0]


def validate_references(rt, series_uid, frame_uid, ct_sops):
    definition, item = heart_item(rt)
    if str(definition.ReferencedFrameOfReferenceUID) != frame_uid:
        raise ValueError("Heart FrameOfReference mismatch")
    refs = referenced_series(rt)
    if not refs or {s for _, s, _ in refs} != {series_uid} or {f for f, _, _ in refs} != {frame_uid}:
        raise ValueError("RT referenced CT series/frame mismatch or mixed references")
    if any(set(sops) - set(ct_sops) for _, _, sops in refs):
        raise ValueError("RT global SOP references absent from CT inventory")
    for frame in rt.ReferencedFrameOfReferenceSequence:
        for study in frame.RTReferencedStudySequence:
            for series in study.RTReferencedSeriesSequence:
                for reference in getattr(series, 'ContourImageSequence', []):
                    if str(getattr(reference, 'ReferencedSOPClassUID', '')) != str(CTImageStorage):
                        raise ValueError('RT global reference is not classic CT Image Storage')
    for contour in item.ContourSequence:
        refs = [str(r.ReferencedSOPInstanceUID) for r in getattr(contour, "ContourImageSequence", [])]
        if len(refs) != 1 or refs[0] not in ct_sops:
            raise ValueError("Heart contour lacks unique released CT SOP reference")
        if str(getattr(contour.ContourImageSequence[0], 'ReferencedSOPClassUID', '')) != str(CTImageStorage):
            raise ValueError('Heart reference is not classic CT Image Storage')
    return item


def rasterize_heart(rt, geometry):
    item = validate_references(rt, geometry.report["series_uid"], geometry.report["frame_of_reference_uid"], geometry.sop_to_slice)
    kinds = {str(c.ContourGeometricType) for c in item.ContourSequence}
    if not kinds <= {"CLOSED_PLANAR", "CLOSEDPLANAR_XOR"} or len(kinds) != 1:
        raise ValueError("Unsupported/mixed contour geometric types")
    xor = "CLOSEDPLANAR_XOR" in kinds
    inverse = np.linalg.inv(geometry.affine_lps)
    mask = np.zeros(geometry.shape, dtype=np.uint8)
    polygons = []
    residuals = []
    for c in item.ContourSequence:
        points = np.asarray(c.ContourData, dtype=float).reshape(-1, 3)
        if len(points) != int(c.NumberOfContourPoints) or len(points) < 3 or not np.isfinite(points).all():
            raise ValueError("Invalid Heart contour points")
        xyz = points @ inverse[:3, :3].T + inverse[:3, 3]
        k = geometry.sop_to_slice[str(c.ContourImageSequence[0].ReferencedSOPInstanceUID)]
        error = float(np.max(np.abs(xyz[:, 2] - k)) * geometry.report["spacing_xyz_mm"][2])
        residuals.append(error)
        if error > .05:
            raise ValueError("Contour plane differs from referenced CT slice by >0.05 mm")
        if np.any(xyz[:, :2] < -.51) or np.any(xyz[:, :2] > np.array(geometry.shape[:2]) - .49):
            raise ValueError("Heart contour extends outside image field of view")
        x, y = polygon(xyz[:, 0], xyz[:, 1], shape=geometry.shape[:2])
        if xor:
            mask[x, y, k] ^= 1
        else:
            mask[x, y, k] = 1
        polygons.append({"slice": k, "xy": xyz[:, :2]})
    if not mask.any():
        raise ValueError("Rasterized Heart is empty")
    active_slices = np.flatnonzero(mask.any(axis=(0, 1)))
    missing_planes = sorted(set(range(int(active_slices[0]), int(active_slices[-1]) + 1)) - set(active_slices))
    touches = [bool(mask.take(i, axis=a).any()) for a in range(3) for i in (0, -1)]
    components, count = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(components.ravel())[1:]
    report = {"roi": "Heart", "binary_labels": [0, 1], "contours": len(polygons),
        "combination": "XOR" if xor else "union CLOSED_PLANAR", "shape_xyz": list(mask.shape),
        "resampled": False, "max_contour_slice_residual_mm": max(residuals),
        "contoured_slice_range": [int(active_slices[0]), int(active_slices[-1])],
        "internal_uncontoured_slices": missing_planes, "touches_grid_faces": touches,
        "scan_coverage": "requires_review_truncated" if any(touches) else "ROI_inside_grid_anatomical_completeness_requires_visual_review",
        "components_26": int(count), "component_sizes_voxels": sizes.tolist(),
        "heart_voxels": int(mask.sum()),
        "volume_ml": float(mask.sum() * abs(np.linalg.det(geometry.affine_lps[:3, :3])) / 1000),
        "initial_status": "requires_review" if any(touches) or missing_planes else "geometry_passed_visual_QA_pending"}
    return mask, report, polygons
