"""Build a resolution-only train/validation cache directly from original public CT."""
import argparse
from pathlib import Path
import shutil

import nibabel as nib
import numpy as np

from heart3d.ml.bundle import pack, safe_relative, verify
from heart3d.ml.cardiac_data import (
    index_transform, load_protocol, read_json, resize_plane, write_json,
)
from heart3d.storage import require_space, sha256_file

VARIANT_KEYS = {
    "experiment_scope", "data_root", "preprocessing", "cache_relative_path",
    "outputs_relative_path", "checkpoint_relative_path", "input_size",
    "allowed_partitions", "baseline_config", "single_changed_factor",
}


def protocol(config_path, baseline_path):
    config, baseline = read_json(config_path), read_json(baseline_path)
    if {k: v for k, v in config.items() if k not in VARIANT_KEYS} != {
        k: v for k, v in baseline.items() if k not in VARIANT_KEYS
    }:
        raise ValueError("Resolution-only experiment changed a training hyperparameter")
    if baseline["input_size"] != 256 or config["input_size"] != 384:
        raise ValueError("Expected the declared 256 -> 384 experiment")
    if config["allowed_partitions"] != ["train", "validation"]:
        raise ValueError("Test must remain sealed")
    if Path(config["baseline_config"]).resolve() != Path(baseline_path).resolve():
        raise ValueError("Baseline config identity differs")
    for key in ("cache_relative_path", "outputs_relative_path", "checkpoint_relative_path"):
        if safe_relative(config[key]) == safe_relative(baseline[key]):
            raise ValueError("Variant would overwrite a baseline directory")
    rows, split = load_protocol(config)
    pre, old_pre = read_json(config["preprocessing"]), read_json(baseline["preprocessing"])
    expected = dict(old_pre, input_size=384)
    if pre != expected:
        raise ValueError("Only preprocessing input_size may change; do not refit intensities")
    if pre["fit_cases"] != split["partitions"]["train"] or pre["fitted_partition"] != "train":
        raise ValueError("Intensity fit is not from the frozen train split")
    for key, source in (("split_SHA256", config["split"]), ("cohort_SHA256", config["cohort"])):
        if pre[key] != sha256_file(source):
            raise ValueError("Frozen preprocessing identity differs")
    return config, baseline, rows, split, pre


def file_record(root, path):
    path = Path(path)
    return {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
            "SHA256": sha256_file(path)}


def baseline_provenance(root, baseline, case, partition, row, entries):
    path = root / baseline["cache_relative_path"] / case / "provenance.json"
    record = entries.get(path.relative_to(root).as_posix())
    if record is None or file_record(root, path) != record:
        raise ValueError("Baseline provenance changed")
    p = read_json(path)
    if p["case_id"] != case or p["partition"] != partition:
        raise ValueError("Baseline case/partition differs")
    for key, source in (("split_SHA256", baseline["split"]),
                        ("cohort_SHA256", baseline["cohort"]),
                        ("preprocessing_SHA256", baseline["preprocessing"])):
        if p[key] != sha256_file(source):
            raise ValueError("Baseline cache protocol differs")
    if p["source_image_SHA256"] != row["image_SHA256"] or p["source_mask_SHA256"] != row["mask_SHA256"]:
        raise ValueError("Baseline source hashes differ")
    count = row["shape"][2]
    positive, negative = p["positive_indices"], p["negative_indices"]
    if (len(positive) != len(set(positive)) or len(negative) != len(set(negative))
            or set(positive) & set(negative)
            or set(positive + negative) != set(range(count))):
        raise ValueError("Baseline slice selection is invalid")
    return p


def prepare(config_path, baseline_path, data_root, summary_path, checkpoint=None):
    config, baseline, rows, split, pre = protocol(config_path, baseline_path)
    root = Path(data_root)
    cache = root / safe_relative(config["cache_relative_path"])
    if cache.exists() or Path(summary_path).exists():
        raise ValueError("Variant output exists; no implicit overwrite/resume")
    old_manifest = read_json(root / baseline["cache_relative_path"] / "prepared_manifest.json")
    entries = {r["path"]: r for r in old_manifest["files"]}
    if len(entries) != len(old_manifest["files"]):
        raise ValueError("Duplicate baseline manifest paths")
    selected = [(p, c) for p in config["allowed_partitions"] for c in split["partitions"][p]]
    # Budget includes the worst-case archive plus a validation reference; reserve is local.
    prepared_bytes = sum(rows[c]["shape"][2] * 384**2 * 5 + 256 for _, c in selected)
    free = require_space(root, 2 * prepared_bytes + 2_000_000_000, int(80e9))
    print("STORAGE_PREFLIGHT", free, "cache_estimate", prepared_bytes, flush=True)
    reference = None
    if checkpoint is not None:
        checkpoint = Path(checkpoint).resolve()
        relative = safe_relative(checkpoint.relative_to(root.resolve()).as_posix())
        import torch
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        for key, source in (("config_SHA256", baseline_path), ("split_SHA256", baseline["split"]),
                            ("cohort_SHA256", baseline["cohort"]), ("preprocessing_SHA256", baseline["preprocessing"])):
            if state[key] != sha256_file(source):
                raise ValueError("Reference checkpoint protocol differs")
        if state["training_scope"] != "public_chd68_multiclass_scratch_v1":
            raise ValueError("Invalid baseline reference checkpoint")
        reference = {"checkpoint_relative_path": relative,
                     "checkpoint_SHA256": sha256_file(checkpoint), "checkpoint_epoch": state["epoch"],
                     "config_SHA256": sha256_file(baseline_path),
                     "prepared_manifest_relative_path": config["cache_relative_path"] + "/baseline_validation_prepared_manifest.json"}
        del state
    cache.mkdir(parents=True)
    records, cases = [], []
    low, high = pre["clip_source_intensity"]
    if not np.isfinite([low, high]).all() or high <= low:
        raise ValueError("Invalid frozen intensity range")
    for partition, case in selected:
        row = rows[case]
        old = baseline_provenance(root, baseline, case, partition, row, entries)
        ct_path, gt_path = [root / safe_relative(row[k]) for k in ("image_relative_path", "mask_relative_path")]
        if sha256_file(ct_path) != row["image_SHA256"] or sha256_file(gt_path) != row["mask_SHA256"]:
            raise ValueError("Original CT/GT integrity failure")
        ct, gt = nib.load(ct_path), nib.load(gt_path)
        if (ct.shape != gt.shape or tuple(ct.shape) != tuple(row["shape"])
                or not np.allclose(ct.affine, gt.affine, atol=1e-5, rtol=0)
                or not np.allclose(ct.affine, old["original_affine"], atol=1e-5, rtol=0)
                or nib.aff2axcodes(ct.affine) != ("R", "A", "S")):
            raise ValueError("Original pair geometry changed")
        image, target = ct.get_fdata(dtype=np.float32), np.asanyarray(gt.dataobj)
        if (not np.isfinite(image).all() or not np.isfinite(target).all()
                or target.min() < 0 or not np.equal(target, np.rint(target)).all()):
            raise ValueError("Invalid original data values")
        transform = index_transform(ct.shape, 384)
        directory = cache / case
        directory.mkdir()
        shape = (transform["slice_count"], 384, 384)
        x = np.lib.format.open_memmap(directory / "image.npy", mode="w+", dtype=np.float32, shape=shape)
        y = np.lib.format.open_memmap(directory / "mask.npy", mode="w+", dtype=np.uint8, shape=shape)
        counts = np.zeros(8, dtype=np.int64)
        positives = []
        rx, ry = transform["resized_xy"]
        px, py = transform["pad_xy"]
        for z in range(shape[0]):
            # Resize source values before clipping, identically to v1; padding remains -1.
            plane = resize_plane(image[:, :, z], transform, 1, low)
            x[z] = np.clip((plane - low) / (high - low), 0, 1) * 2 - 1
            x[z, :py] = -1
            x[z, py + ry:] = -1
            x[z, :, :px] = -1
            x[z, :, px + rx:] = -1
            resized = resize_plane(target[:, :, z], transform, 0, 255)
            y[z] = np.where(resized <= 7, resized, 255).astype(np.uint8)
            counts += np.bincount(y[z][y[z] < 8], minlength=8)
            if np.any((y[z] >= 1) & (y[z] <= 7)):
                positives.append(z)
        x.flush()
        y.flush()
        x._mmap.close()
        y._mmap.close()
        if np.any(counts[1:] == 0):
            raise ValueError("A class vanished during resampling; requires review")
        p = {**old, "transform": transform, "preprocessing_SHA256": sha256_file(config["preprocessing"]),
             "image_SHA256": sha256_file(directory / "image.npy"),
             "mask_SHA256": sha256_file(directory / "mask.npy"),
             "slice_selection_from_baseline": True,
             "baseline_provenance_SHA256": entries[(root / baseline["cache_relative_path"] / case / "provenance.json").relative_to(root).as_posix()]["SHA256"],
             "resized_foreground_indices": positives}
        write_json(directory / "provenance.json", p)
        records.extend(file_record(root, directory / name) for name in ("image.npy", "mask.npy", "provenance.json"))
        cases.append({"case_id": case, "partition": partition, "source_shape": list(ct.shape),
                      "prepared_shape": list(shape), "class_voxels": counts[1:].tolist(),
                      "baseline_positive_slices": len(old["positive_indices"]),
                      "positive_slice_selection_unchanged": True,
                      "new_foreground_slice_difference": sorted(set(positives) ^ set(old["positive_indices"]))})
        del image, target
        print("PREPARED_384", partition, case, shape[0], flush=True)
    manifest = {"version": 1, "files": records, "source_bytes": sum(r["bytes"] for r in records),
                "cohort_SHA256": sha256_file(config["cohort"]), "split_SHA256": sha256_file(config["split"]),
                "preprocessing_SHA256": sha256_file(config["preprocessing"]), "case_count": len(selected),
                "prepared_partitions": config["allowed_partitions"], "physical_geometry_verified": False,
                "raw_DICOM_included": False, "test_opened": False, "preprocessing_refitted": False,
                "baseline_prepared_manifest_SHA256": sha256_file(root / baseline["cache_relative_path"] / "prepared_manifest.json"),
                "validation_reference": reference}
    write_json(cache / "prepared_manifest.json", manifest)
    verify(root, cache / "prepared_manifest.json")
    if reference:
        prefix = baseline["cache_relative_path"] + "/"
        val_files = [r for r in old_manifest["files"] if any(
            r["path"].startswith(prefix + case + "/") for case in split["partitions"]["validation"])]
        if len(val_files) != 3 * len(split["partitions"]["validation"]):
            raise ValueError("Incomplete baseline validation cache")
        val_manifest = {**old_manifest, "files": val_files, "case_count": len(split["partitions"]["validation"]),
                        "source_bytes": sum(r["bytes"] for r in val_files),
                        "prepared_partitions": ["validation"], "test_opened": False}
        path = root / reference["prepared_manifest_relative_path"]
        write_json(path, val_manifest)
        verify(root, path)
    summary = {k: v for k, v in manifest.items() if k != "files"}
    summary.update(config_SHA256=sha256_file(config_path), baseline_config_SHA256=sha256_file(baseline_path),
                   prepared_manifest_SHA256=sha256_file(cache / "prepared_manifest.json"),
                   train_cases=len(split["partitions"]["train"]), validation_cases=len(split["partitions"]["validation"]),
                   train_samples_per_epoch=sum(len(cases_p["positive_indices"]) + min(
                       len(cases_p["positive_indices"]), len(cases_p["negative_indices"]))
                       for c in split["partitions"]["train"]
                       for cases_p in [read_json(cache / c / "provenance.json")]),
                   free_bytes_before=free, free_bytes_after=shutil.disk_usage(root).free,
                   reserve_bytes=80_000_000_000, cases=cases, full_training_started=False, GPU_benchmark_completed=False)
    write_json(summary_path, summary)
    return summary


def bundle(config_path, baseline_path, root, archive, manifest_path):
    config, baseline, rows, split, _ = protocol(config_path, baseline_path)
    root = Path(root)
    prepared_path = root / config["cache_relative_path"] / "prepared_manifest.json"
    prepared = verify(root, prepared_path)
    expected_cases = {c for p in config["allowed_partitions"] for c in split["partitions"][p]}
    expected_paths = {f'{config["cache_relative_path"]}/{c}/{n}' for c in expected_cases
                      for n in ("image.npy", "mask.npy", "provenance.json")}
    if {r["path"] for r in prepared["files"]} != expected_paths or prepared["prepared_partitions"] != ["train", "validation"]:
        raise ValueError("Variant bundle contains unexpected cases/partitions")
    paths = [r["path"] for r in prepared["files"]] + [prepared_path.relative_to(root).as_posix()]
    for case in split["partitions"]["validation"]:
        path = safe_relative(rows[case]["mask_relative_path"])
        if sha256_file(root / path) != rows[case]["mask_SHA256"]:
            raise ValueError("Validation GT changed")
        paths.append(path)
    reference = prepared["validation_reference"]
    if reference:
        if sha256_file(root / reference["checkpoint_relative_path"]) != reference["checkpoint_SHA256"]:
            raise ValueError("Baseline checkpoint changed")
        ref_path = root / reference["prepared_manifest_relative_path"]
        ref = verify(root, ref_path)
        paths += [r["path"] for r in ref["files"]]
        paths += [reference["checkpoint_relative_path"], reference["prepared_manifest_relative_path"]]
    repository = [{"path": str(p).replace("\\", "/"), "SHA256": sha256_file(p)} for p in (
        config_path, baseline_path, config["cohort"], config["split"],
        config["preprocessing"], baseline["preprocessing"])]
    return pack(root, paths, archive, manifest_path, repository, int(80e9))


def stage_reference(config_path, data_root):
    """Install a validation-only baseline cache manifest in a NEW remote data root."""
    config = read_json(config_path)
    root = Path(data_root)
    prepared = verify(root, root / config["cache_relative_path"] / "prepared_manifest.json")
    reference = prepared["validation_reference"]
    if not reference:
        raise ValueError("Baseline validation reference was not bundled")
    baseline = read_json(config["baseline_config"])
    source = root / safe_relative(reference["prepared_manifest_relative_path"])
    verify(root, source)
    target = root / safe_relative(baseline["cache_relative_path"]) / "prepared_manifest.json"
    if target.exists():
        raise ValueError("Do not overwrite an existing baseline manifest")
    shutil.copyfile(source, target)
    print("BASELINE_VALIDATION_MANIFEST_STAGED", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "bundle", "stage-reference"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline-config", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--reference-checkpoint", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    if args.command == "stage-reference":
        stage_reference(args.config, args.data)
        return
    if args.baseline_config is None:
        parser.error("--baseline-config is required")
    if args.command == "prepare":
        if args.summary is None:
            parser.error("--summary is required")
        prepare(args.config, args.baseline_config, args.data, args.summary, args.reference_checkpoint)
    else:
        if args.archive is None or args.manifest is None:
            parser.error("--archive and --manifest are required")
        bundle(args.config, args.baseline_config, args.data, args.archive, args.manifest)


if __name__ == "__main__":
    main()
