"""One-plane local AI-assisted draft; not visual/manual expert annotation.

No training, new weights, normal-anatomy enforcement, filling or island removal.
The image-conditioned refinement is a hypothesis for the user's review.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import nibabel as nib
import numpy as np
from scipy import ndimage
import torch

from heart3d.labels import LABELS
from heart3d.storage import sha256_file
from .clinical_experiment import admit, canonical_input
from .infer_cardiac import load_recovered, prepare_volume, restore_probability_plane, save_prediction


def select_slice(first, second, spacing, protocol):
    if first.shape != second.shape or first.ndim != 3:
        raise ValueError("Same original-grid masks required")
    lo, hi = protocol["selection"]["axial_fraction_range"]
    area = float(np.prod(spacing[:2]))
    middle = (first.shape[2]-1)/2
    threshold = protocol["selection"]["interior_agreement_component_min_area_mm2"]
    rows = []
    for z in range(int(np.ceil(first.shape[2]*lo)), int(np.floor(first.shape[2]*hi))+1):
        agreement = first[:, :, z] == second[:, :, z]
        areas = {}
        for label, (code, *_) in LABELS.items():
            components, count = ndimage.label(agreement & (first[:, :, z] == label))
            sizes = np.bincount(components.ravel()); sizes[0] = 0
            best = 0
            for i in np.argsort(sizes)[::-1]:
                if i == 0:
                    continue
                xy = np.argwhere(components == i)
                if xy.min() > 0 and np.all(xy.max(axis=0) < np.array(components.shape)-1):
                    best = int(sizes[i]); break
            areas[code] = best*area
        rows.append({"index": z, "eligible_classes": sum(a >= threshold for a in areas.values()),
                     "distance_from_middle": abs(z-middle), "agreement_largest_component_mm2": areas})
    return min(rows, key=lambda r: (-r["eligible_classes"], r["distance_from_middle"], r["index"])), rows


def edge_weights(hu, sigma):
    return (np.exp(-np.diff(hu, axis=0)**2/(2*sigma*sigma)),
            np.exp(-np.diff(hu, axis=1)**2/(2*sigma*sigma)))


def energy(labels, unary, weights, strength):
    wx, wy = weights
    return float(np.take_along_axis(unary, labels[None], axis=0).sum() + strength*(
        (wx*(labels[1:] != labels[:-1])).sum() + (wy*(labels[:, 1:] != labels[:, :-1])).sum()))


def refine_plane(probability, hu, first_labels, second_labels, config):
    probability = np.asarray(probability, dtype=np.float32)
    hu = np.asarray(hu, dtype=np.float32)
    if probability.shape != (8, *hu.shape) or not np.isfinite(probability).all() or not np.isfinite(hu).all():
        raise ValueError("Finite original-grid probability plane and HU required")
    if (probability < 0).any() or not np.allclose(probability.sum(axis=0), 1, atol=1e-4):
        raise ValueError("Normalized probabilities required")
    c = config["refinement"]
    unary = -np.log(np.maximum(probability, 1e-8))
    supported = np.array([True]+[bool(np.any(first_labels == i) or np.any(second_labels == i)) for i in range(1, 8)])
    labels = np.where(supported[:, None, None], probability, -1).argmax(axis=0).astype(np.uint8)
    initial = labels.copy()
    maxima = probability.max(axis=0)
    frozen = (first_labels == second_labels) & (maxima >= c["freeze_agreement_probability"])
    candidates = probability >= c["minimum_candidate_probability"]
    # A class unsupported by both model label maps is never invented by smoothing.
    for label in range(1, 8):
        if not supported[label]:
            candidates[label] = False
    candidates[labels, *np.indices(labels.shape)] = True
    weights = edge_weights(hu, c["HU_edge_sigma"])
    strength = c["pairwise_weight"]
    wx, wy = weights
    parity = np.indices(hu.shape).sum(axis=0) % 2
    history = [energy(labels, unary, weights, strength)]
    for _ in range(c["checkerboard_sweeps"]):
        for color in (0, 1):
            costs = unary.copy()
            for label in range(8):
                costs[label, 1:] += strength*wx*(labels[:-1] != label)
                costs[label, :-1] += strength*wx*(labels[1:] != label)
                costs[label, :, 1:] += strength*wy*(labels[:, :-1] != label)
                costs[label, :, :-1] += strength*wy*(labels[:, 1:] != label)
            costs[~candidates] = np.inf
            proposal = costs.argmin(axis=0).astype(np.uint8)
            current_cost = np.take_along_axis(costs, labels[None], axis=0)[0]
            best_cost = costs.min(axis=0)
            update = (~frozen) & (parity == color) & (best_cost < current_cost-1e-6)
            labels[update] = proposal[update]
        history.append(energy(labels, unary, weights, strength))
    if np.any(labels[frozen] != initial[frozen]) or any(b > a+.5 for a, b in zip(history, history[1:])):
        raise ValueError("Refinement changed frozen anchors or increased its declared energy")
    sorted_p = np.sort(probability, axis=0)
    flags = config["review_flags"]
    uncertainty = ((first_labels != second_labels) | (maxima < flags["probability_below"])
                   | ((sorted_p[-1]-sorted_p[-2]) < flags["margin_below"]))
    return labels, uncertainty, {"energy_history": history, "frozen_pixels": int(frozen.sum()),
        "pixels_changed_from_ensemble": int(np.count_nonzero(labels != initial)),
        "model_disagreement_pixels": int(np.count_nonzero(first_labels != second_labels))}


def plane_probabilities(source, index, adapter, checkpoint, variant):
    # The admitted series is axial with +S slice indexing. No through-plane
    # resampling or data-dependent crop; exactly the trained five-plane context.
    if nib.aff2axcodes(source.affine)[2] != "S" or not 2 <= index < source.shape[2]-2:
        raise ValueError("Interior axial +S source slice required")
    slab = np.asarray(source.dataobj[:, :, index-2:index+3], dtype=np.float32)
    offset = np.eye(4); offset[2, 3] = index-2
    slab_image = nib.Nifti1Image(slab, source.affine @ offset)
    canonical, inverse = canonical_input(slab_image)
    if not np.array_equal(inverse[:, 0], [0, 1, 2]) or inverse[2, 1] != 1:
        raise ValueError("Axial plane must be preserved by canonical orientation")
    model, _, _, pre = load_recovered(variant, checkpoint)
    prepared, transform = prepare_volume(canonical + adapter["offset"], pre)
    context = prepared[np.asarray(pre["context_offsets_indices"])+2]
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); started = time.perf_counter()
    with torch.inference_mode():
        probability = model(torch.from_numpy(context[None].copy()).cuda()).softmax(1).cpu().numpy()[0]
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-started
    restored = restore_probability_plane(probability, transform)
    restored = np.stack([nib.orientations.apply_orientation(p[:, :, None], inverse)[:, :, 0] for p in restored])
    metrics = {"forward_seconds": elapsed, "peak_torch_VRAM_bytes": torch.cuda.max_memory_allocated(),
               "checkpoint_sha256": sha256_file(checkpoint), "batch_size": 1}
    del model
    torch.cuda.empty_cache()
    return restored, metrics


def run(ct, admission_path, adapter_path, v1, v2, checkpoint_v1, checkpoint_v2, protocol_path, out):
    out = Path(out)
    if out.exists():
        raise ValueError("New output directory required")
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    adapter = json.loads(Path(adapter_path).read_text(encoding="utf-8"))
    admission = json.loads(Path(admission_path).read_text(encoding="utf-8-sig"))
    source = admit(Path(ct), admission, adapter)
    masks, hashes = [], []
    for directory in (v1, v2):
        directory = Path(directory)
        provenance = json.loads((directory/"provenance.json").read_text())
        image = nib.load(directory/"prediction.nii.gz")
        digest = sha256_file(directory/"prediction.nii.gz")
        if (digest != provenance["prediction_sha256"] or provenance["ct_sha256"] != admission["ct_sha256"]
                or image.shape != source.shape or not np.allclose(image.affine, source.affine, atol=1e-6)):
            raise ValueError("Original AI input differs from admitted CT grid/provenance")
        masks.append(np.asanyarray(image.dataobj)); hashes.append(digest)
    selected, candidates = select_slice(*masks, nib.affines.voxel_sizes(source.affine), protocol)
    z = selected["index"]
    torch.set_num_threads(4); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    probabilities, measurements, differences = [], {}, {}
    for variant, checkpoint, previous in zip(("v1", "v2"), (checkpoint_v1, checkpoint_v2), masks):
        probability, metrics = plane_probabilities(source, z, adapter, checkpoint, variant)
        difference = float(np.mean(probability.argmax(axis=0) != previous[:, :, z]))
        if difference > .001:
            raise ValueError("One-plane probability path differs materially from the existing prediction")
        probabilities.append(probability); measurements[variant] = metrics; differences[variant] = difference
    mean_probability = (probabilities[0]+probabilities[1])*.5
    hu = np.asarray(source.dataobj[:, :, z], dtype=np.float32)
    plane, uncertain, refinement = refine_plane(mean_probability, hu, masks[0][:, :, z], masks[1][:, :, z], protocol)
    prediction = np.zeros(source.shape, np.uint8); prediction[:, :, z] = plane
    uncertainty = np.zeros(source.shape, np.uint8); uncertainty[:, :, z] = uncertain
    foreground = (plane > 0) | (masks[0][:, :, z] > 0) | (masks[1][:, :, z] > 0)
    counts = {"background": int(np.count_nonzero(prediction == 0)),
              **{code: int(np.count_nonzero(plane == i)) for i, (code, *_) in LABELS.items()}}
    warning = "AI-assisted draft of ONE slice; not visually reviewed by an expert; HU-domain mapping remains unconfirmed. All other slices are unannotated."
    out.mkdir(parents=True)
    save_prediction(prediction, source, out/"prediction.nii.gz")
    save_prediction(uncertainty, source, out/"uncertainty.nii.gz")
    np.savez_compressed(out/"slice_probabilities.private.npz", v1=probabilities[0], v2=probabilities[1])
    provenance = {"variant": "slice_draft", "source": "AI-assisted slice draft", "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {"axis": 2, "index": z, "complete_volume": False}, "selection": selected,
        "method": "equal v1/v2 original-grid probability ensemble + fixed local HU-edge Potts refinement",
        "warning": warning, "review_required_classes": list(code for code, *_ in LABELS.values()),
        "visual_expert_review_performed": False, "anatomical_identities_verified": False,
        "expert_approved": False, "exploratory_only": True, "geometry_verified": True,
        "ct_sha256": sha256_file(ct), "prediction_sha256": sha256_file(out/"prediction.nii.gz"),
        "uncertainty_sha256": sha256_file(out/"uncertainty.nii.gz"), "source_prediction_sha256": hashes,
        "protocol": protocol, "protocol_sha256": sha256_file(protocol_path), "adapter_sha256": sha256_file(adapter_path),
        "predicted_voxels": counts, "absent_classes": [c for c, n in counts.items() if c != "background" and n == 0],
        "uncertain_fraction_of_candidate_foreground": float(np.mean(uncertain[foreground])),
        "measurements": measurements, "recomputed_argmax_difference_fractions": differences,
        "refinement": refinement, "model_training_started": False, "GT_used": False,
        "source_intensity_convention_confirmed": False}
    (out/"provenance.json").write_text(json.dumps(provenance, indent=2)+"\n", encoding="utf-8")
    (out/"selection_candidates.private.json").write_text(json.dumps(candidates, indent=2)+"\n", encoding="utf-8")
    if sha256_file(ct) != admission["ct_sha256"]:
        raise ValueError("Admitted CT changed")
    print(json.dumps({k: provenance[k] for k in ("scope", "predicted_voxels", "uncertain_fraction_of_candidate_foreground", "refinement")}, indent=2))
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("ct", "admission", "adapter", "v1", "v2", "checkpoint-v1", "checkpoint-v2", "protocol", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    a = parser.parse_args()
    run(a.ct, a.admission, a.adapter, a.v1, a.v2, a.checkpoint_v1, a.checkpoint_v2, a.protocol, a.out)


if __name__ == "__main__":
    main()
