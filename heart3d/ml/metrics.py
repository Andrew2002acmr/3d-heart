"""Patient-level binary metrics; physical surface distances require confirmed geometry."""
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def overlap_metrics(prediction, target):
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if prediction.shape != target.shape:
        raise ValueError('Prediction and GT grids differ')
    tp = int(np.count_nonzero(prediction & target))
    p, t = int(np.count_nonzero(prediction)), int(np.count_nonzero(target))
    return {'Dice': 2*tp/(p+t) if p+t else 1.,
            'IoU': tp/(p+t-tp) if p+t-tp else 1.,
            'precision': tp/p if p else float(t == 0),
            'recall': tp/t if t else float(p == 0),
            'predicted_voxels': p, 'GT_voxels': t}


def physical_metrics(prediction, target, spacing_xyz_mm, geometry_confirmed):
    if not geometry_confirmed:
        raise ValueError('DICOM physical geometry must be confirmed for mm metrics')
    spacing = np.asarray(spacing_xyz_mm, dtype=float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError('Invalid physical spacing')
    p, t = np.asarray(prediction, bool), np.asarray(target, bool)
    result = overlap_metrics(p, t)
    result.update(predicted_volume_ml=result['predicted_voxels']*float(np.prod(spacing))/1000,
                  GT_volume_ml=result['GT_voxels']*float(np.prod(spacing))/1000)
    if not p.any() or not t.any():
        result.update(HD95_mm=None, ASSD_mm=None, surface_status='empty_prediction_or_target')
        return result
    # Union bounding box retains all surfaces and their nearest neighbours,
    # reducing distance-map RAM without changing distances.
    union = p | t
    coordinates = [np.flatnonzero(union.any(axis=tuple(j for j in range(3) if j != i))) for i in range(3)]
    box = tuple(slice(max(0, c[0]-1), min(p.shape[i], c[-1]+2)) for i, c in enumerate(coordinates))
    p, t = p[box], t[box]
    ps = p & ~binary_erosion(p, border_value=0)
    ts = t & ~binary_erosion(t, border_value=0)
    dpt = distance_transform_edt(~ts, sampling=spacing)[ps]
    dtp = distance_transform_edt(~ps, sampling=spacing)[ts]
    result.update(HD95_mm=float(max(np.quantile(dpt, .95), np.quantile(dtp, .95))),
                  ASSD_mm=float((dpt.sum()+dtp.sum())/(len(dpt)+len(dtp))),
                  surface_status='ok')
    return result


def aggregate(records, group_key=None):
    groups = {}
    for row in records:
        label = str(row.get(group_key, 'unknown')) if group_key else 'all'
        groups.setdefault(label, []).append(row)
    result = {}
    for label, rows in groups.items():
        statistics = {'patients': len(rows)}
        for name in ('Dice', 'IoU', 'precision', 'recall', 'HD95_mm', 'ASSD_mm'):
            values = [r[name] for r in rows if r.get(name) is not None]
            if values:
                statistics[name] = {'mean': float(np.mean(values)), 'median': float(np.median(values)),
                                    'std_population': float(np.std(values)), 'min': float(min(values)),
                                    'max': float(max(values)), 'valid_patients': len(values)}
        result[label] = statistics
    return result
