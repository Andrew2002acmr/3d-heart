"""Explicitly admitted local exploratory CT, reusing recovered CHD68 inference.

Does not relax the public/test guards of infer_cardiac or training experiments.
No GT, optimization, cloud calls, anatomy-based tuning, smoothing or cleanup.
"""
import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import threading
import time

import nibabel as nib
import numpy as np
import psutil
import torch

from heart3d.ml.infer_cardiac import load_recovered, prepare_volume, infer_volume, save_prediction, LABELS
from heart3d.storage import sha256_file


def canonical_input(image):
    source_orientation = nib.orientations.io_orientation(image.affine)
    canonical = nib.as_closest_canonical(image)
    inverse = nib.orientations.ornt_transform(nib.orientations.io_orientation(canonical.affine), source_orientation)
    if nib.aff2axcodes(canonical.affine) != ("R", "A", "S"):
        raise ValueError("Cannot establish RAS index orientation")
    return canonical.get_fdata(dtype=np.float32), inverse


def admit(ct, admission, adapter):
    required = ("user_attested_permission_and_basis", "local_only", "geometry_verified",
                "independent_reader_verified", "metadata_deidentified", "pixel_identifier_screening_passed")
    if not all(admission.get(k) is True for k in required):
        raise ValueError("Clinical admission gate incomplete; no inference")
    if sha256_file(ct) != admission.get("ct_sha256"):
        raise ValueError("CT differs from admitted derivative")
    if not (adapter.get("exploratory_only") is True and adapter.get("gain") == 1
            and adapter.get("offset") == 1024 and adapter.get("source_intensity_convention_confirmed") is False):
        raise ValueError("Only the preregistered exploratory HU+1024 adapter is implemented")
    image = nib.load(ct)
    if image.header.get_xyzt_units()[0] != "mm" or len(image.shape) != 3:
        raise ValueError("Verified 3D millimetre CT required")
    if image.header.extensions or any(bytes(image.header[k]).strip(b"\0 ") for k in ("descrip", "aux_file", "intent_name", "db_name")):
        raise ValueError("CT contains unexpected metadata text/extensions")
    spacing = nib.affines.voxel_sizes(image.affine)
    direction = image.affine[:3, :3] / spacing
    if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5):
        raise ValueError("Sheared CT cannot use this adapter")
    return image


def run(ct, admission_path, adapter_path, checkpoint, variant, out, batch_size=2):
    ct, out = Path(ct), Path(out)
    if out.exists() or batch_size not in (1, 2):
        raise ValueError("New output and batch 1 or 2 required")
    admission = json.loads(Path(admission_path).read_text(encoding="utf-8-sig"))
    adapter = json.loads(Path(adapter_path).read_text(encoding="utf-8"))
    source = admit(ct, admission, adapter)
    checkpoint_digest = sha256_file(checkpoint)
    # No fallback and no inference with fabricated/unsupported source geometry.
    if not torch.cuda.is_available():
        raise ValueError("CUDA required; no implicit CPU fallback")
    process = psutil.Process()
    rss = [process.memory_info().rss]
    stopped = threading.Event()
    def sample():
        while not stopped.wait(.1):
            rss[0] = max(rss[0], process.memory_info().rss)
    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    started = time.perf_counter()
    try:
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        t = time.perf_counter()
        model, saved, config, pre = load_recovered(variant, checkpoint)
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - t
        t = time.perf_counter()
        array, inverse = canonical_input(source)
        if not np.isfinite(array).all():
            raise ValueError("Nonfinite CT")
        source_stats = {"HU_min": float(array.min()), "HU_max": float(array.max())}
        array += adapter["offset"]
        low, high = pre["clip_source_intensity"]
        clipping = {"below_frozen_clip_fraction": float(np.mean(array < low)),
                    "above_frozen_clip_fraction": float(np.mean(array > high))}
        prepared, transform = prepare_volume(array, pre)
        del array
        gc.collect()
        prep_seconds = time.perf_counter() - t
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t = time.perf_counter()
        canonical_prediction = infer_volume(model, prepared, transform, pre, batch_size)
        torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - t
        allocated = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
        prediction = nib.orientations.apply_orientation(canonical_prediction, inverse).astype(np.uint8)
        if prediction.shape != source.shape:
            raise ValueError("Inverse reorientation differs from source CT shape")
        out.mkdir(parents=True)
        destination = out / "prediction.nii.gz"
        save_prediction(prediction, source, destination)
        restored = nib.load(destination)
        if (restored.shape != source.shape or not np.allclose(restored.affine, source.affine, atol=1e-6)
                or restored.header.get_xyzt_units()[0] != "mm"):
            raise ValueError("Original-grid prediction verification failed")
        for name in ("qform", "sform"):
            a, ac = getattr(source, "get_"+name)(coded=True)
            b, bc = getattr(restored, "get_"+name)(coded=True)
            if int(ac) != int(bc) or (a is not None and not np.allclose(a, b, atol=1e-6)):
                raise ValueError("NIfTI form differs from CT")
        counts = {name: int(np.count_nonzero(prediction == i)) for i, name in LABELS.items()}
        metrics = {"variant": variant, "source": "AI prediction", "created_utc": datetime.now(timezone.utc).isoformat(),
                   "exploratory_only": True, "expert_approved": False, "quality_evaluated": False,
                   "source_intensity_convention_confirmed": False, "intensity_adapter": adapter,
                   "adapter_sha256": sha256_file(adapter_path), "admission_sha256": sha256_file(admission_path),
                   "ct_sha256": sha256_file(ct), "prediction_sha256": sha256_file(destination),
                   "checkpoint_sha256": checkpoint_digest, "checkpoint_epoch": saved["epoch"],
                   "frozen_config_sha256": saved["config_SHA256"], "frozen_preprocessing_sha256": saved["preprocessing_SHA256"],
                   "batch_size": batch_size, "precision": "FP32; TF32 disabled", "torch": torch.__version__,
                   "GPU": torch.cuda.get_device_name(0), "GPU_total_bytes": torch.cuda.get_device_properties(0).total_memory,
                   "source_grid_shape": list(source.shape), "affine_ras": source.affine.tolist(),
                   "inverse_index_orientation": inverse.tolist(), "prepared_shape": list(prepared.shape),
                   "physical_context_span_mm": float(4 * nib.affines.voxel_sizes(source.affine)[2]),
                   "frozen_preprocessing": pre, "intensity_QA": source_stats, **clipping,
                   "checkpoint_load_seconds": load_seconds, "preprocessing_seconds": prep_seconds,
                   "inference_seconds": inference_seconds, "total_seconds": time.perf_counter()-started,
                   "peak_VRAM_allocated_bytes": allocated, "peak_VRAM_reserved_bytes": reserved,
                   "peak_process_RSS_sampled_bytes": max(rss[0], process.memory_info().rss),
                   "peak_process_working_set_OS_bytes": getattr(process.memory_info(), "peak_wset", None),
                   "memory_scope": "RSS sampled every 100ms; torch peaks exclude context/other applications",
                   "predicted_voxels": counts, "absent_classes": [name for name in list(LABELS.values())[1:] if not counts[name]],
                   "scan_face_contacts": {name: [bool(np.any(np.take(prediction == i, k, axis=axis))) for axis in range(3) for k in (0,-1)] for i,name in LABELS.items() if i},
                   "geometry_verified": True, "training_started": False, "GT_used": False, "postprocessing": "none"}
        # Recheck both immutable inputs after computation.
        if sha256_file(ct) != admission["ct_sha256"]:
            raise ValueError("Admitted CT changed during inference")
        if sha256_file(checkpoint) != metrics["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed during inference")
        (out / "provenance.json").write_text(json.dumps(metrics, indent=2)+"\n", encoding="utf-8")
        print(json.dumps({k:metrics[k] for k in ("variant","inference_seconds","peak_VRAM_allocated_bytes","peak_process_RSS_sampled_bytes","predicted_voxels","absent_classes","exploratory_only")},indent=2))
        return metrics
    finally:
        stopped.set()
        monitor.join(timeout=1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("ct", "admission", "adapter", "checkpoint", "out"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--variant", choices=("v1", "v2"), required=True)
    p.add_argument("--batch-size", type=int, default=2)
    a = p.parse_args()
    run(a.ct, a.admission, a.adapter, a.checkpoint, a.variant, a.out, a.batch_size)


if __name__ == "__main__":
    main()
