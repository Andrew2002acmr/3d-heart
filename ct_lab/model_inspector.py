"""Instrument the recovered production pipeline without substituting experiments."""
import numpy as np
import nibabel as nib
import torch
from heart3d.ml.infer_cardiac import protocol, prepare_volume, load_recovered
from heart3d.ml.cardiac_data import index_transform, resize_plane, index_neighbors
from ct_lab.core import stats


def inspect(image, variant, center):
    _, _, pre, _ = protocol(variant)
    if nib.aff2axcodes(image.affine) != ('R', 'A', 'S'):
        raise ValueError('Actual CHD68 source gate requires RAS; no silent reorientation')
    raw = image.get_fdata(dtype=np.float32)
    indices, outside = index_neighbors(center, raw.shape[2], pre['context_offsets_indices'])
    # Only five source planes: prepare_volume does not interpolate/resample Z.
    # Calling the production function on this exact context avoids allocating a full prepared volume.
    context = raw[:, :, indices]
    prepared, transform = prepare_volume(context, pre)
    tensor = torch.from_numpy(prepared.copy()).unsqueeze(0)
    t = index_transform(raw.shape, pre['input_size'])
    resized = np.stack([resize_plane(raw[:, :, z], t, 1, pre['clip_source_intensity'][0]) for z in indices])
    clipped = np.clip(resized, *pre['clip_source_intensity'])
    stages = {'01 Source context CXY': np.moveaxis(context, -1, 0),
              '02 Orientation: identity RAS (actual pipeline)': np.moveaxis(context, -1, 0),
              '03 XY resize/pad; no Z resampling': resized,
              '04 Clipping after XY resize': clipped,
              '05 Normalization; final BCHW tensor': tensor.numpy()[0]}
    return {'tensor': tensor, 'stages': stages, 'preprocessing': pre,
            'source_indices': indices.tolist(), 'replicated_edge': outside.tolist(),
            'transform': t, 'tensor_statistics': stats(tensor.numpy()),
            'channel_statistics': [stats(c) for c in tensor.numpy()[0]],
            'source': 'heart3d.ml.infer_cardiac.prepare_volume; no augmentation in eval',
            'tensor_shape': list(tensor.shape)}


def predict_slice(result, variant, checkpoint, source_shape):
    model, _, _, pre = load_recovered(variant, checkpoint, device='cpu')
    captured = []
    def hook(module, args):
        captured.append(args[0].detach().cpu().clone())
    handle = model.register_forward_pre_hook(hook)
    try:
        with torch.inference_mode():
            probability = model(result['tensor']).softmax(1).cpu().numpy()[0]
    finally:
        handle.remove()
    if len(captured) != 1 or not torch.equal(captured[0], result['tensor']):
        raise ValueError('Forward input differs from inspector tensor')
    from heart3d.ml.cardiac_data import restore_probability_plane
    t = index_transform(source_shape, pre['input_size'])
    restored = restore_probability_plane(probability, t)
    # Renormalize interpolation roundoff before entropy.
    restored /= np.maximum(restored.sum(0, keepdims=True), 1e-8)
    return restored, captured[0], -np.sum(restored*np.log(np.maximum(restored, 1e-8)), axis=0)


def diagnose_volume(image, mask, variant, checkpoint, components=False, progress=None):
    """Optional CPU full-volume inference; no fitting, source-grid probabilities before argmax."""
    from heart3d.ml.infer_cardiac import infer_volume
    from heart3d.ml.cardiac_data import confusion_counts, scores_from_confusion
    from ct_lab.core import mask_analysis, fingerprint
    model, _, _, pre = load_recovered(variant, checkpoint, device='cpu')
    if progress: progress('prepare')
    prepared, transform = prepare_volume(image.get_fdata(dtype=np.float32), pre)
    if progress: progress('infer full volume')
    prediction = infer_volume(model, prepared, transform, pre, batch_size=1, device='cpu')
    del prepared
    if progress: progress('prediction mask anatomy')
    report = {'variant': variant, 'checkpoint_sha256': fingerprint(checkpoint),
              'scope': 'full released volume, original source grid; exploratory analysis, no training',
              'prediction_anatomy': mask_analysis(prediction, image.affine, image.header.get_xyzt_units()[0], components)}
    if mask is not None:
        if progress: progress('expert mask anatomy')
        from ct_lab.core import geometry_matches
        if not geometry_matches(image, mask): raise ValueError('GT/source geometry mismatch')
        target = np.asanyarray(mask.dataobj)
        from ct_lab.core import cardiac_target
        report['metrics'] = scores_from_confusion(confusion_counts(prediction, cardiac_target(target)))
        report['ignored_gt_voxels'] = int(np.count_nonzero((target < 0) | (target > 7)))
        report['expert_anatomy'] = mask_analysis(target, image.affine, image.header.get_xyzt_units()[0], components)
        if components:
            from ct_lab.core import fragment_comparison
            if progress: progress('fragment/GT overlaps')
            report['fragment_gt_comparison']=fragment_comparison(prediction,target,image.affine)
    if progress: progress('complete')
    return prediction, report
