"""Recover CHD68 inference using the frozen protocol and existing project helpers.

Inputs are synthetic or SHA-verified public development CT. No GT, training,
clinical input, test evaluation, preprocessing refit or automatic cleanup.
"""
import argparse
import gc
import json
from pathlib import Path
import statistics
import subprocess
import time

import nibabel as nib
import numpy as np
import torch

from heart3d.ml.cardiac_data import (
    index_transform, load_protocol, read_json, resize_plane,
    restore_probability_plane, write_json,
)
from heart3d.ml.cardiac_train import load_checkpoint, probability_batches
from heart3d.storage import sha256_file

REPO = Path(__file__).resolve().parents[2]
VARIANTS = {
    "v1": ("configs/cardiac_chd68_runpod_v1.json", "cardiac_chd68_results_v1.json"),
    "v2": ("configs/cardiac_chd68_resolution384_v2.json", "cardiac_chd68_resolution384_results_v2.json"),
}
LABELS = {0: "background", 1: "LV", 2: "RV", 3: "LA", 4: "RA", 5: "MYO", 6: "AO", 7: "PA"}


def protocol(variant):
    config_name, result_name = VARIANTS[variant]
    path = REPO / config_name
    original = read_json(path)
    config = dict(original, _config_path=str(path))
    for key in ("cohort", "split", "preprocessing"):
        config[key] = str(REPO / original[key])
    pre = read_json(config["preprocessing"])
    load_protocol(config)
    if (config["model"].get("input_channels") != 5
            or config["model"].get("output_channels") != 8
            or pre["input_size"] != config["input_size"]
            or pre["context_offsets_indices"] != [-2, -1, 0, 1, 2]
            or pre["clip_source_intensity"] != [0.0, 2015.0]
            or pre["fitted_partition"] != "train"):
        raise ValueError("Not the recovered CHD68 protocol")
    history = read_json(REPO / "metadata/pediatric" / result_name)
    expected = (history.get("checkpoints") or history["training"]["checkpoints"])["best.pt"]
    return config, original, pre, expected


def load_recovered(variant, checkpoint, device="cuda"):
    config, original, pre, expected = protocol(variant)
    checkpoint = Path(checkpoint)
    if checkpoint.stat().st_size != expected["bytes"] or sha256_file(checkpoint) != expected["SHA256"]:
        raise ValueError("Recovered checkpoint size/SHA256 differs from historical best.pt")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; no implicit CPU fallback")
    model, saved = load_checkpoint(checkpoint, config, device)
    if saved["config"] != original or saved["preprocessing"] != pre:
        raise ValueError("Embedded config/preprocessing differs from frozen files")
    if sum(p.numel() for p in model.parameters()) != 489112:
        raise ValueError("Unexpected recovered model architecture")
    model.eval()
    return model, saved, config, pre


def public_development_ct(path, case_id, config):
    rows, split = load_protocol(config)
    if case_id not in split["development_cases"] or case_id not in split["partitions"]["train"]:
        raise ValueError("Only declared public development/train cases are allowed; no test or clinical input")
    path = Path(path)
    row = rows[case_id]
    if sha256_file(path) != row["image_SHA256"]:
        raise ValueError("CT differs from SHA-verified public development source")
    image = nib.load(path)
    if tuple(image.shape) != tuple(row["shape"]) or nib.aff2axcodes(image.affine) != ("R", "A", "S"):
        raise ValueError("Public source shape/orientation differs")
    return image.get_fdata(dtype=np.float32), image


def prepare_volume(ct, pre):
    ct = np.asarray(ct)
    if ct.ndim != 3 or min(ct.shape) < 2 or not np.isfinite(ct).all():
        raise ValueError("Finite 3D source CT with all dimensions >=2 required")
    transform = index_transform(ct.shape, pre["input_size"])
    low, high = pre["clip_source_intensity"]
    if high <= low:
        raise ValueError("Invalid frozen intensity bounds")
    image = np.empty((ct.shape[2], pre["input_size"], pre["input_size"]), dtype=np.float32)
    for z in range(ct.shape[2]):
        plane = resize_plane(ct[:, :, z], transform, 1, low)
        image[z] = np.clip((plane - low) / (high - low), 0, 1) * 2 - 1
    return image, transform


def infer_volume(model, image, transform, pre, batch_size, device="cuda"):
    prediction = np.empty(transform["original_shape_xyz"], dtype=np.uint8)
    with torch.inference_mode():
        for centers, probability in probability_batches(
                model, image, pre["context_offsets_indices"], batch_size, device):
            if not np.isfinite(probability).all():
                raise ValueError("Nonfinite predicted probabilities")
            for center, plane in zip(centers, probability):
                prediction[:, :, center] = restore_probability_plane(plane, transform).argmax(0)
    return prediction


def save_prediction(prediction, source_image, destination):
    if source_image is None:
        output = nib.Nifti1Image(prediction, np.eye(4))
        output.header.set_xyzt_units("unknown")
    else:
        output = nib.Nifti1Image(prediction, source_image.affine, source_image.header.copy())
        for name in ("qform", "sform"):
            matrix, code = getattr(source_image, "get_" + name)(coded=True)
            getattr(output, "set_" + name)(matrix, int(code))
    output.set_data_dtype(np.uint8)
    output.header.set_slope_inter(1, 0)
    nib.save(output, destination)


def run(args):
    out = Path(args.out)
    if out.exists():
        raise ValueError("Choose a new output directory; previous results are preserved")
    if not 1 <= args.batch_size <= 4 or not 1 <= args.repeats <= 10:
        raise ValueError("Local inference requires batch_size 1..4 and repeats 1..10")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    load_start = time.perf_counter()
    model, saved, config, pre = load_recovered(args.variant, args.checkpoint)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start
    source_image = None
    if args.synthetic:
        ct = np.random.default_rng(20261022).uniform(0, 2015, (128, 112, 16)).astype(np.float32)
    else:
        if not args.case:
            raise ValueError("--case is required with --ct")
        ct, source_image = public_development_ct(args.ct, args.case, config)
    prep_start = time.perf_counter()
    image, transform = prepare_volume(ct, pre)
    preprocessing_seconds = time.perf_counter() - prep_start
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    prediction = infer_volume(model, image, transform, pre, args.batch_size)
    torch.cuda.synchronize()
    cold_seconds = time.perf_counter() - start
    cold_memory = {"allocated": torch.cuda.max_memory_allocated(), "reserved": torch.cuda.max_memory_reserved()}
    torch.cuda.reset_peak_memory_stats()
    timings = []
    for _ in range(args.repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        prediction = infer_volume(model, image, transform, pre, args.batch_size)
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
    out.mkdir(parents=True)
    filename = "synthetic_prediction.nii.gz" if args.synthetic else "cardiac_prediction_original.nii.gz"
    save_prediction(prediction, source_image, out / filename)
    try:
        smi = subprocess.check_output([
            "nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"
        ], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        smi = None
    result = {
        "variant": args.variant, "checkpoint_SHA256": sha256_file(args.checkpoint),
        "checkpoint_epoch": saved["epoch"], "config_SHA256": saved["config_SHA256"],
        "preprocessing_SHA256": saved["preprocessing_SHA256"],
        "split_SHA256": saved["split_SHA256"], "cohort_SHA256": saved["cohort_SHA256"],
        "torch": torch.__version__, "CUDA_runtime": torch.version.cuda,
        "GPU": torch.cuda.get_device_name(0), "GPU_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "precision": "FP32; TF32 disabled", "parameters": 489112, "labels": LABELS,
        "synthetic": args.synthetic, "case_id": args.case,
        "source_CT_SHA256": None if args.synthetic else sha256_file(args.ct),
        "source_shape_xyz": list(ct.shape), "prepared_shape_zyx": list(image.shape),
        "transform": transform, "preprocessing": pre, "batch_size": args.batch_size,
        "checkpoint_load_seconds": load_seconds, "preprocessing_seconds": preprocessing_seconds,
        "first_volume_seconds": cold_seconds, "first_volume_peak_bytes": cold_memory,
        "measured_volume_seconds": timings, "mean_seconds": statistics.mean(timings),
        "median_seconds": statistics.median(timings), "minimum_seconds": min(timings),
        "timing_scope": "prepared input through GPU forward, CPU transfer and original-grid probability restoration/argmax; excludes load/preprocess/save",
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "nvidia_smi_total_used_MiB_after_run": smi,
        "memory_scope": "torch peaks include model and inference tensors, exclude CUDA context and other applications",
        "prediction_SHA256": sha256_file(out / filename),
        "predicted_voxels": {name: int(np.count_nonzero(prediction == label)) for label, name in LABELS.items()},
        "physical_geometry_verified": False, "GT_used": False, "quality_evaluated": False,
        "clinical_data_used": False, "test_data_used": False, "training_started": False,
    }
    write_json(out / "inference.json", result)
    print(json.dumps({k: result[k] for k in (
        "variant", "checkpoint_epoch", "GPU", "batch_size", "prepared_shape_zyx",
        "first_volume_seconds", "mean_seconds", "peak_allocated_bytes", "peak_reserved_bytes",
        "nvidia_smi_total_used_MiB_after_run", "synthetic", "predicted_voxels"
    )}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic", action="store_true")
    source.add_argument("--ct", type=Path)
    parser.add_argument("--case")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
