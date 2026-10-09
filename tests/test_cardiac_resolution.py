"""Resolution-only regression tests use tiny synthetic public-style volumes."""
from pathlib import Path
import copy

import nibabel as nib
import numpy as np
import pytest
import torch

from heart3d.ml.cardiac_data import (
    CardiacDataset, index_transform, read_json, resize_plane, write_json,
)
from heart3d.ml.bundle import unpack
from heart3d.storage import sha256_file
from scripts import prepare_cardiac_resolution as resolution


@pytest.fixture
def resolution_toy(tmp_path, monkeypatch):
    base = read_json("configs/cardiac_chd68_runpod_v1.json")
    old_pre = read_json(base["preprocessing"])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(resolution, "require_space", lambda *args: 200_000_000_000)
    monkeypatch.setattr("heart3d.ml.bundle.require_space", lambda *args: 200_000_000_000)
    root = tmp_path / "data"
    root.mkdir()
    partitions = {"train": ["a"], "validation": ["b"], "test": ["c"]}
    rows, files = [], []
    for case in ("a", "b", "c"):
        mask = np.broadcast_to(np.arange(8, dtype=np.uint8)[:, None, None], (8, 8, 8)).copy()
        mask[:, :, 0] = 0
        mask[0, 0, 1] = 9
        image = (mask.astype(np.float32) * 100 + np.arange(8)[None, :, None] + ord(case)).astype(np.float32)
        ct, gt = root / f"{case}_ct.nii.gz", root / f"{case}_gt.nii.gz"
        nib.save(nib.Nifti1Image(image, np.eye(4)), ct)
        nib.save(nib.Nifti1Image(mask, np.eye(4)), gt)
        rows.append({"case_id": case, "group_id": case, "shape": [8, 8, 8],
                     "image_relative_path": ct.name, "mask_relative_path": gt.name,
                     "image_SHA256": sha256_file(ct), "mask_SHA256": sha256_file(gt)})
    write_json("cohort.json", {"records": rows})
    write_json("split.json", {"cohort_manifest_SHA256": sha256_file("cohort.json"),
                            "partitions": partitions, "frozen": True, "development_cases": ["a"]})
    base.update(cohort="cohort.json", split="split.json", preprocessing="pre256.json",
                cache_relative_path="cache256", outputs_relative_path="runs256",
                checkpoint_relative_path="checkpoints256", data_root=str(root))
    old_pre.update(fit_cases=["a"], split_SHA256=sha256_file("split.json"),
                   cohort_SHA256=sha256_file("cohort.json"))
    write_json("pre256.json", old_pre)
    write_json("baseline.json", base)
    for partition, ids in partitions.items():
        for case in ids:
            row = next(r for r in rows if r["case_id"] == case)
            directory = root / "cache256" / case
            directory.mkdir(parents=True)
            # Intentionally unrelated image values prove new pixels come from ORIGINAL CT.
            np.save(directory / "image.npy", np.zeros((8, 256, 256), dtype=np.float32))
            np.save(directory / "mask.npy", np.zeros((8, 256, 256), dtype=np.uint8))
            write_json(directory / "provenance.json", {
                "case_id": case, "partition": partition, "transform": index_transform((8, 8, 8), 256),
                "positive_indices": list(range(1, 8)), "negative_indices": [0],
                "source_image_SHA256": row["image_SHA256"], "source_mask_SHA256": row["mask_SHA256"],
                "original_affine": np.eye(4).tolist(), "physical_geometry_verified": False,
                "split_SHA256": sha256_file("split.json"), "cohort_SHA256": sha256_file("cohort.json"),
                "preprocessing_SHA256": sha256_file("pre256.json"),
            })
            files.extend(resolution.file_record(root, directory / name) for name in
                         ("image.npy", "mask.npy", "provenance.json"))
    write_json(root / "cache256/prepared_manifest.json", {
        "files": files, "split_SHA256": sha256_file("split.json"),
        "cohort_SHA256": sha256_file("cohort.json"), "preprocessing_SHA256": sha256_file("pre256.json"),
    })
    new = copy.deepcopy(base)
    new.update(input_size=384, preprocessing="pre384.json", cache_relative_path="cache384",
               outputs_relative_path="runs384", checkpoint_relative_path="checkpoints384",
               allowed_partitions=["train", "validation"], baseline_config="baseline.json")
    write_json("variant.json", new)
    write_json("pre384.json", dict(old_pre, input_size=384))
    checkpoint = root / "reference.pt"
    torch.save({"config_SHA256": sha256_file("baseline.json"),
                "cohort_SHA256": sha256_file("cohort.json"), "split_SHA256": sha256_file("split.json"),
                "preprocessing_SHA256": sha256_file("pre256.json"), "epoch": 23,
                "training_scope": "public_chd68_multiclass_scratch_v1"}, checkpoint)
    return root, checkpoint


def test_only_resolution_changes_and_test_sealed_before_data_io(resolution_toy):
    root, _ = resolution_toy
    with pytest.raises(ValueError, match="sealed"):
        CardiacDataset("variant.json", "test", root)
    for file, key, value, message in [
        ("variant.json", "learning_rate", 0.0005, "hyperparameter"),
        ("pre384.json", "clip_source_intensity", [0, 1000], "Only preprocessing"),
    ]:
        original = read_json(file)
        altered = dict(original, **{key: value})
        write_json(file, altered)
        with pytest.raises(ValueError, match=message):
            resolution.protocol("variant.json", "baseline.json")
        write_json(file, original)


def test_preparation_original_pixels_fixed_sampling_and_no_test(resolution_toy):
    root, _ = resolution_toy
    before = sha256_file(root / "cache256/prepared_manifest.json")
    # A sealed case's missing files cannot affect preparation; they must never be opened.
    (root / "c_ct.nii.gz").unlink()
    (root / "c_gt.nii.gz").unlink()
    result = resolution.prepare("variant.json", "baseline.json", root, "summary.json")
    assert result["case_count"] == 2 and not result["test_opened"]
    assert not result["preprocessing_refitted"] and result["train_samples_per_epoch"] == 8
    assert not (root / "cache384/c").exists()
    assert before == sha256_file(root / "cache256/prepared_manifest.json")
    original = nib.load(root / "a_ct.nii.gz").get_fdata(dtype=np.float32)
    expected = resize_plane(original[:, :, 1], index_transform((8, 8, 8), 384), 1, 0)
    expected = np.clip(expected / 2015, 0, 1) * 2 - 1
    actual = np.load(root / "cache384/a/image.npy", mmap_mode="r")
    assert np.allclose(actual[1], expected)
    actual._mmap.close()
    mask = np.load(root / "cache384/a/mask.npy")
    assert 255 in np.unique(mask) and set(np.unique(mask)) <= set(range(8)) | {255}
    dataset = CardiacDataset("variant.json", "train", root, augmentation=False)
    assert dataset[1]["image"].shape == (5, 384, 384)
    assert dataset[1]["target"].shape == (384, 384)
    assert dataset.provenance["a"]["positive_indices"] == list(range(1, 8))
    assert {c for c, _ in dataset.indices} == {"a"}
    dataset.close()
    with pytest.raises(ValueError, match="exists"):
        resolution.prepare("variant.json", "baseline.json", root, "another_summary.json")


def test_bundle_reference_validation_only_and_sha_roundtrip(resolution_toy):
    root, checkpoint = resolution_toy
    resolution.prepare("variant.json", "baseline.json", root, "summary.json", checkpoint)
    receipt = resolution.bundle("variant.json", "baseline.json", root, "bundle.tar.gz", "bundle_manifest.json")
    manifest = read_json("bundle_manifest.json")
    assert receipt["files"] == 13  # 2x3 variant + 1 manifest + 1 val GT + 3 baseline + ref manifest/checkpoint.
    assert not any("/c/" in r["path"] or r["path"].startswith("c_") for r in manifest["files"])
    restored = Path("restored")
    unpack("bundle.tar.gz", restored)
    resolution.stage_reference("variant.json", restored)
    baseline = CardiacDataset("baseline.json", "validation", restored, augmentation=False)
    assert {r["case_id"] for r in baseline.rows} == {"b"}
    baseline.close()
    with pytest.raises(ValueError, match="overwrite"):
        resolution.stage_reference("variant.json", restored)
