"""Train-only visual QA: original CT/GT versus the 256 and 384 input grids."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import nibabel as nib
import numpy as np

from heart3d.labels import LABELS
from heart3d.ml.cardiac_data import read_json, write_json
from heart3d.storage import sha256_file
from scripts.prepare_cardiac_resolution import protocol


def render(config_path, data_root, output, case_ids):
    config = read_json(config_path)
    config, baseline, rows, split, pre = protocol(config_path, config["baseline_config"])
    if not case_ids or len(set(case_ids)) != len(case_ids) or not set(case_ids) <= set(split["partitions"]["train"]):
        raise ValueError("QA selection must contain distinct frozen TRAIN cases")
    output, root = Path(output), Path(data_root)
    if output.exists():
        raise ValueError("QA output exists")
    output.mkdir(parents=True)
    colors = ListedColormap(["none"] + [LABELS[i][2] for i in range(1, 8)])
    records = []
    for case in case_ids:
        row = rows[case]
        sources = {}
        for key, hash_key in (("image_relative_path", "image_SHA256"), ("mask_relative_path", "mask_SHA256")):
            path = root / row[key]
            if sha256_file(path) != row[hash_key]:
                raise ValueError("Original QA source changed")
            sources[key] = nib.load(path)
        original_image = sources["image_relative_path"].get_fdata(dtype=np.float32)
        original_mask = np.asanyarray(sources["mask_relative_path"].dataobj)
        arrays = []
        for cfg in (baseline, config):
            directory = root / cfg["cache_relative_path"] / case
            p = read_json(directory / "provenance.json")
            if p["partition"] != "train" or p["preprocessing_SHA256"] != sha256_file(cfg["preprocessing"]):
                raise ValueError("Stale QA input")
            for name, key in (("image.npy", "image_SHA256"), ("mask.npy", "mask_SHA256")):
                if sha256_file(directory / name) != p[key]:
                    raise ValueError("QA cache changed")
            arrays.append((np.load(directory / "image.npy", mmap_mode="r"),
                           np.load(directory / "mask.npy", mmap_mode="r"), p))
        if arrays[0][2]["positive_indices"] != arrays[1][2]["positive_indices"]:
            raise ValueError("QA sampling changed")
        positive = arrays[0][2]["positive_indices"]
        slices = [positive[round((len(positive)-1)*fraction)] for fraction in (.25, .5, .75)]
        low, high = pre["clip_source_intensity"]
        fig, axes = plt.subplots(3, 3, figsize=(11, 11))
        for column, z in enumerate(slices):
            panels = [(np.clip((original_image[:, :, z].T-low)/(high-low), 0, 1)*2-1,
                       original_mask[:, :, z].T)]
            panels += [(image[z], mask[z]) for image, mask, _ in arrays]
            for row_index, (image, mask) in enumerate(panels):
                ax = axes[row_index, column]
                ax.imshow(image, cmap="gray", vmin=-1, vmax=1, interpolation="nearest")
                overlay = np.ma.masked_where((mask < 1) | (mask > 7), mask)
                ax.imshow(overlay, cmap=colors, vmin=-.5, vmax=7.5, alpha=.5, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                if column == 0:
                    ax.set_ylabel(("Original release grid", "Baseline input 256", "Variant input 384")[row_index])
                if row_index == 0:
                    ax.set_title(f"Original axial index {z}")
        fig.suptitle(f"{case}: TRAIN preprocessing QA, not predictions\nSame source, window and slices; HU/mm/age unverified")
        fig.legend(handles=[Patch(color=LABELS[i][2], label=LABELS[i][0]) for i in range(1, 8)],
                   loc="lower center", ncol=7)
        fig.tight_layout(rect=(0, .04, 1, .94))
        path = output / f"{case}_resolution_QA.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        for image, mask, _ in arrays:
            image._mmap.close()
            mask._mmap.close()
        records.append({"case_id": case, "partition": "train", "slices": slices,
                        "image_SHA256": row["image_SHA256"], "mask_SHA256": row["mask_SHA256"],
                        "figure": path.name, "figure_SHA256": sha256_file(path)})
    write_json(output / "QA_summary.json", {"scope": "preprocessing QA only; no model/quality measurement",
                                          "test_used": False, "physical_geometry_claimed": False,
                                          "records": records})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-ids", nargs="+", required=True)
    args = parser.parse_args()
    render(args.config, args.data, args.output, args.case_ids)
