"""Post-hoc paired validation QA; no test access or preprocessing fitting."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from heart3d.ml.cardiac_data import read_json, write_json
from heart3d.storage import require_space, sha256_file
from scripts.compare_cardiac_validation import compare
from scripts.qa_cardiac_predictions import overlay, decorate


def render(row, old_metric, new_metric, role, data, baseline, variant, output, clip, hashes):
    case = row["case_id"]
    paths = [data/row["image_relative_path"], data/row["mask_relative_path"]]
    expected = [row["image_SHA256"], row["mask_SHA256"]]
    for folder, checkpoint_sha in zip((baseline, variant), hashes):
        provenance = read_json(folder/case/"provenance.json")
        if provenance["checkpoint_SHA256"] != checkpoint_sha:
            raise ValueError("Prediction checkpoint mismatch")
        paths.append(folder/case/"cardiac_prediction_original.nii.gz")
        expected.append(provenance["prediction_SHA256"])
    if any(sha256_file(p) != h for p, h in zip(paths, expected)):
        raise ValueError("Original source or prediction SHA mismatch")
    loaded = [nib.load(p) for p in paths]
    if any(v.shape != loaded[0].shape or not np.allclose(v.affine, loaded[0].affine, atol=1e-5, rtol=0) for v in loaded[1:]):
        raise ValueError("Original-grid alignment differs")
    arrays = [loaded[0].get_fdata(dtype=np.float32)] + [np.asanyarray(v.dataobj) for v in loaded[1:]]
    if not np.isfinite(arrays[0]).all() or any(np.any((v < 0) | (v > 7)) for v in arrays[2:]):
        raise ValueError("Invalid prediction or image")
    foreground = (arrays[1] > 0) & (arrays[1] <= 7)
    positive = [np.flatnonzero(foreground.any(axis=tuple(j for j in range(3) if j != i))) for i in range(3)]
    if any(not len(p) for p in positive):
        raise ValueError("Empty cardiac reference")
    center = [int((p[0]+p[-1])//2) for p in positive]
    specs = [("Axial20%", 2, int(positive[2][round(.2*(len(positive[2])-1))])),
             ("Axial50%", 2, int(positive[2][round(.5*(len(positive[2])-1))])),
             ("Axial80%", 2, int(positive[2][round(.8*(len(positive[2])-1))])),
             ("Coronal", 1, center[1]), ("Sagittal", 0, center[0])]
    fig, axes = plt.subplots(5, 4, figsize=(15, 18))
    for axis_row, (name, axis, index) in zip(axes, specs):
        planes = [np.take(v, index, axis=axis).T for v in arrays]
        for col, ax in enumerate(axis_row):
            ax.imshow(planes[0], cmap="gray", vmin=clip[0], vmax=clip[1], origin="lower", interpolation="nearest")
            if col: ax.imshow(overlay(planes[col]), origin="lower", interpolation="nearest")
            ax.set_title(name+f", index {index}\n"+["CT", "GT", "Prediction256 v1", "Prediction384 v2"][col], fontsize=10)
            ax.axis("off")
    title = f'{role}: {case} | validation original-grid macro Dice v1={old_metric["macro_foreground_Dice"]:.3f}, v2={new_metric["macro_foreground_Dice"]:.3f}'
    decorate(fig, title, include_errors=False)
    fig.tight_layout(rect=(0, .04, 1, .95))
    image = output/f"{case}_paired_validation.png"
    fig.savefig(image, dpi=115)
    plt.close(fig)
    return {"case_id":case, "role":role, "image":image.name,
            "baseline_Dice":old_metric["macro_foreground_Dice"], "variant_Dice":new_metric["macro_foreground_Dice"],
            "sampling":[{"plane":n,"axis":a,"index":i} for n,a,i in specs]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "baseline", "variant", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--cohort", type=Path, default=Path("metadata/pediatric/cardiac_chd68_cohort_v1.json"))
    parser.add_argument("--preprocessing", type=Path, default=Path("configs/cardiac_chd68_preprocessing_v1.json"))
    a = parser.parse_args()
    if a.output.exists(): raise ValueError("QA output exists; choose a new directory")
    require_space(a.output, 100_000_000, 80_000_000_000)
    old, new = [read_json(p/"metrics.json") for p in (a.baseline, a.variant)]
    if old["cohort_SHA256"] != sha256_file(a.cohort): raise ValueError("Cohort mismatch")
    pre = read_json(a.preprocessing)
    if pre["fitted_partition"] != "train" or old["preprocessing_SHA256"] != sha256_file(a.preprocessing):
        raise ValueError("Baseline display preprocessing differs")
    clip = np.asarray(pre["clip_source_intensity"], dtype=float)
    if clip.shape != (2,) or not np.isfinite(clip).all() or clip[0] >= clip[1]: raise ValueError("Invalid clip")
    a.output.mkdir(parents=True)
    compare(a.baseline/"metrics.json", a.variant/"metrics.json", a.output/"paired_validation.json")
    cohort = {r["case_id"]:r for r in read_json(a.cohort)["records"]}
    baseline = {r["case_id"]:r for r in old["records"]}
    ordered = sorted(new["records"], key=lambda r:(r["macro_foreground_Dice"],r["case_id"]))
    selected = [("Best v2 validation",ordered[-1]), ("Middle rank v2 validation",ordered[len(ordered)//2]), ("Worst v2 validation",ordered[0])]
    result = [render(cohort[r["case_id"]], baseline[r["case_id"]], r, role, a.data, a.baseline, a.variant, a.output, clip,
                     [old["checkpoint_SHA256"],new["checkpoint_SHA256"]]) for role,r in selected]
    write_json(a.output/"QA_summary.json", {"partition":"validation", "test_used":False, "cases":result,
               "baseline_metrics_SHA256":sha256_file(a.baseline/"metrics.json"), "variant_metrics_SHA256":sha256_file(a.variant/"metrics.json")})


if __name__ == "__main__": main()
