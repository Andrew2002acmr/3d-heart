"""Prepare local raw-CT context packets, never anatomical findings or labels.

No clinical image is returned through stdout/network. This module has no HTTP,
cloud, model-pull or inference client. A missing visual observer stays missing.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import map_coordinates

from heart3d.storage import sha256_file


def load_admitted_ct(path, admission_path):
    admission = json.loads(Path(admission_path).read_text(encoding="utf-8-sig"))
    fields = ("user_attested_permission_and_basis", "local_only", "geometry_verified",
              "independent_reader_verified", "metadata_deidentified", "pixel_identifier_screening_passed")
    if not all(admission.get(k) is True for k in fields) or admission.get("ct_sha256") != sha256_file(path):
        raise ValueError("Specific admitted deidentified CT required")
    image = nib.load(path)
    if len(image.shape) != 3 or min(image.shape) < 2 or image.header.get_xyzt_units()[0] != "mm":
        raise ValueError("Three-dimensional millimetre CT required")
    if image.header.extensions or any(bytes(image.header[k]).strip(b"\0 ") for k in ("descrip", "aux_file", "intent_name", "db_name")):
        raise ValueError("CT must have clean metadata")
    direction = image.affine[:3, :3]/nib.affines.voxel_sizes(image.affine)
    # Saved native planes are labelled AX/COR/SAG only for aligned source axes.
    # Oblique physical MPR remains supported in the web notebook, not this exporter.
    if not np.allclose(np.abs(direction), np.eye(3), atol=1e-5):
        raise ValueError("Packet export currently requires axis-aligned XYZ; use physical web MPR for oblique CT")
    return image


def plane_geometry(shape, spacing, view, index, maximum_edge=768):
    axis = {"AX": 2, "COR": 1, "SAG": 0}[view]
    if type(index) is not int or not 0 <= index < shape[axis]:
        raise ValueError("Source plane outside CT")
    horizontal, vertical = {"AX": (0, 1), "COR": (0, 2), "SAG": (1, 2)}[view]
    pixel_mm = min(spacing[horizontal], spacing[vertical])
    width = max(1, round(shape[horizontal]*spacing[horizontal]/pixel_mm))
    height = max(1, round(shape[vertical]*spacing[vertical]/pixel_mm))
    scale = min(1., maximum_edge/max(width, height))
    width, height = max(1, round(width*scale)), max(1, round(height*scale))
    # Mapping takes integer PNG pixel centres (u,v,1) to native IJK voxel centres.
    matrix = np.zeros((3, 3), dtype=float); matrix[axis, 2] = index
    matrix[horizontal, 0] = shape[horizontal]/width
    matrix[horizontal, 2] = .5*shape[horizontal]/width-.5
    flip_vertical = view in ("COR", "SAG")
    step = shape[vertical]/height
    matrix[vertical, 1] = -step if flip_vertical else step
    matrix[vertical, 2] = shape[vertical]-.5-.5*step if flip_vertical else .5*step-.5
    return axis, width, height, matrix


def render_plane(array, affine_ras, spacing, view, index, width_hu, level_hu, maximum_edge=768):
    if width_hu <= 0 or not np.isfinite([width_hu, level_hu]).all():
        raise ValueError("Finite display window required")
    axis, width, height, mapping = plane_geometry(array.shape, spacing, view, index, maximum_edge)
    u, v = np.meshgrid(np.arange(width), np.arange(height), indexing="xy")
    pixels = np.stack([u, v, np.ones_like(u)]).reshape(3, -1)
    coordinates = mapping @ pixels
    hu = map_coordinates(array, coordinates, order=1, mode="nearest", prefilter=False).reshape(height, width)
    gray = np.rint(np.clip((hu-(level_hu-width_hu/2))/width_hu, 0, 1)*255).astype(np.uint8)
    # Add generated orientation/index markers only; no source headers/free text.
    pad = 32
    content = Image.fromarray(gray).convert("RGB")
    image = Image.new("RGB", (width+pad*2, height+pad*2), (12, 18, 24))
    image.paste(content, (pad, pad))
    draw = ImageDraw.Draw(image)
    codes = nib.aff2axcodes(affine_ras)
    opposite = {"R":"L", "L":"R", "A":"P", "P":"A", "S":"I", "I":"S"}
    horizontal, vertical = {"AX": (0, 1), "COR": (0, 2), "SAG": (1, 2)}[view]
    top = codes[vertical] if view != "AX" else opposite[codes[vertical]]
    right = codes[horizontal]
    draw.text((pad, 5), f"{view} source {index+1}; W{width_hu:g} L{level_hu:g}", fill="white")
    draw.text((5, height//2+pad), opposite[right], fill="white")
    draw.text((width+pad+10, height//2+pad), right, fill="white")
    draw.text((width//2+pad, 20), top, fill="white")
    draw.text((width//2+pad, height+pad+10), opposite[top], fill="white")
    lps = np.diag([-1., -1., 1., 1.]) @ affine_ras
    anchor = mapping @ np.array([(width-1)/2, (height-1)/2, 1.])
    normal = lps[:3, axis]/np.linalg.norm(lps[:3, axis])
    locator = {"view": view, "native_axis": axis, "native_index": index,
        "anchor_ijk": anchor.tolist(), "focal_point_lps_mm": nib.affines.apply_affine(lps, anchor).tolist(),
        "normal_lps": normal.tolist(), "window_width": width_hu, "window_level": level_hu,
        "overlay_visible": False}
    png_mapping = mapping.copy(); png_mapping[:, 2] -= pad*(mapping[:, 0]+mapping[:, 1])
    return image, {"locator": locator, "png_width": image.width, "png_height": image.height,
                   "content_rectangle_xywh": [pad, pad, width, height],
                   "orientation_markers": {"top": top, "bottom": opposite[top], "right": right, "left": opposite[right]},
                   "source_ijk_from_png_pixel_centre": png_mapping.tolist(),
                   "display_interpolation": "linear HU for display only", "source_grid_changed": False}


def prepare(ct, admission, protocol_path, out, target_hint=None, request_path=None):
    out = Path(out)
    if out.exists():
        raise ValueError("Choose a new private output directory")
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    source = load_admitted_ct(ct, admission)
    spacing = nib.affines.voxel_sizes(source.affine)
    array = source.get_fdata(dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Finite CT required")
    if target_hint is not None and not 0 <= target_hint < source.shape[2]:
        raise ValueError("Target hint outside CT")
    requests = []
    windows = {w["name"]: w for w in protocol["window_presets"]}
    if request_path is not None:
        requested = json.loads(Path(request_path).read_text(encoding="utf-8"))
        if set(requested) != {"ct_sha256", "planes"} or requested["ct_sha256"] != sha256_file(ct):
            raise ValueError("Follow-up request must name only this CT hash and plane requests")
        if not isinstance(requested["planes"], list) or not 1 <= len(requested["planes"]) <= 32:
            raise ValueError("Bounded follow-up plane requests required")
        for row in requested["planes"]:
            if not isinstance(row, dict) or set(row) != {"view", "indices", "window"} or row["view"] not in ("AX", "COR", "SAG") or row["window"] not in windows:
                raise ValueError("Follow-up request contains unsupported fields/view/window")
            if not isinstance(row["indices"], list) or not 1 <= len(row["indices"]) <= 32:
                raise ValueError("Bounded source indices required")
            axis = {"AX": 2, "COR": 1, "SAG": 0}[row["view"]]
            for index in row["indices"]:
                if type(index) is not int or not 0 <= index < source.shape[axis]:
                    raise ValueError("Requested source index outside CT")
                requests.append((row["view"], index, row["window"]))
    for view, axis in (("AX", 2), ("COR", 1), ("SAG", 0)):
        if request_path is not None:
            break
        indices = set(np.rint(np.linspace(0, source.shape[axis]-1, protocol["overview_counts"][view])).astype(int).tolist())
        if view == "AX" and target_hint is not None:
            indices.update(z for z in range(target_hint-2, target_hint+3) if 0 <= z < source.shape[2])
        for index in sorted(indices):
            for window in protocol["window_presets"]:
                requests.append((view, index, window["name"]))
    # File creation is after all admission, geometry and untrusted request checks.
    (out/"packets").mkdir(parents=True)
    packets = []
    for view, index, window_name in sorted(set(requests)):
        window = windows[window_name]
        image, detail = render_plane(array, source.affine, spacing, view, index,
            window["width"], window["level"], protocol["maximum_display_edge"])
        packet_id = f"{view}_{index:04d}_{window['name']}"
        relative = f"packets/{packet_id}.png"; image.save(out/relative)
        packets.append({"id": packet_id, "file": relative, "sha256": sha256_file(out/relative),
            "review_state": "not_reviewed", "evidence_kind": "raw_CT", **detail})
    manifest = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "ct_sha256": sha256_file(ct), "admission_sha256": sha256_file(admission),
        "protocol_sha256": sha256_file(protocol_path), "protocol": protocol,
        "shape_ijk": list(source.shape), "spacing_mm": spacing.tolist(), "affine_ras": source.affine.tolist(),
        "source_axis_codes": list(nib.aff2axcodes(source.affine)),
        "target_hint_native_z": target_hint, "target_anatomically_selected": False,
        "phase": "requested_followup" if request_path else "orientation_survey",
        "request_sha256": sha256_file(request_path) if request_path else None,
        "source_inventory_plane_counts": {"AX": source.shape[2], "COR": source.shape[1], "SAG": source.shape[0]},
        "all_source_planes_available": True, "all_planes_reviewed": False,
        "observer_status": "unavailable; no observations or anatomical findings generated",
        "observations": [], "structure_tracks": [], "target_annotation": None,
        "prediction_maps_used": False, "image_cloud_transfer": False, "packets": packets}
    (out/"packet_manifest.private.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    (out/"review_log.json").write_text(json.dumps({"schema_version": 1, "ct_sha256": manifest["ct_sha256"],
        "status": "awaiting_local_visual_observer", "reviewed_packet_ids": [], "observations": [],
        "tracks": [], "decisions": [], "target": None}, indent=2)+"\n", encoding="utf-8")
    if sha256_file(ct) != manifest["ct_sha256"]:
        raise ValueError("CT changed during packet creation")
    print(json.dumps({"raw_CT_packets_created_locally": len(packets), "anatomical_observations_generated": 0,
        "observer_available": False, "target_annotation_created": False, "prediction_maps_used": False}))
    return manifest


def inspect_runtime_receipt(receipt):
    """Preflight only. No network request/image submission or cloud fallback exists."""
    reasons = []
    if receipt.get("owner_controlled") is not True:
        reasons.append("Runtime ownership is not verified")
    if receipt.get("offline_mode_verified") is not True or receipt.get("nonloopback_egress_block_verified") is not True:
        reasons.append("Offline execution and process egress isolation are not verified")
    if receipt.get("remote_host") or receipt.get("remote_model") or receipt.get("cloud_enabled") is not False:
        reasons.append("Cloud relay must be explicitly disabled and absent")
    if "vision" not in receipt.get("capabilities", []):
        reasons.append("Visual input capability is not verified")
    for kind in ("executable", "weights", "projector"):
        row = receipt.get(kind, {})
        path = Path(row.get("path", ""))
        if not path.is_absolute() or not path.is_file() or not isinstance(row.get("sha256"), str):
            reasons.append(f"Local {kind} file and hash are required")
        elif sha256_file(path) != row["sha256"]:
            reasons.append(f"Local {kind} hash differs")
    return {"ready_for_transport": not reasons, "medical_validity_verified": False,
            "attestations_not_independently_audited": True, "execution_enabled": False,
            "issues": reasons, "image_submission_performed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "request-packets"):
        p = commands.add_parser(name)
        for field in ("ct", "admission", "protocol", "out"):
            p.add_argument("--"+field, type=Path, required=True)
        p.add_argument("--target-hint", type=int)
        p.add_argument("--request", type=Path, required=name=="request-packets")
    p = commands.add_parser("check-runtime")
    p.add_argument("--receipt", type=Path, required=True)
    a = parser.parse_args()
    if a.command in ("prepare", "request-packets"):
        prepare(a.ct, a.admission, a.protocol, a.out, a.target_hint, a.request)
    else:
        result = inspect_runtime_receipt(json.loads(a.receipt.read_text(encoding="utf-8")))
        print(json.dumps(result, indent=2))
        if not result["ready_for_transport"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
