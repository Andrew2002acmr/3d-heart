"""Conservative, image-mask-only multiclass 3D postprocessing in index coordinates."""
import numpy as np
from scipy import ndimage


def _box(mask):
    coordinates = [np.flatnonzero(mask.any(axis=tuple(j for j in range(3) if j != i))) for i in range(3)]
    if any(not len(c) for c in coordinates):
        return None
    return tuple(slice(max(0, int(c[0])-1), min(mask.shape[i], int(c[-1])+2)) for i,c in enumerate(coordinates))


def small_changes(prediction, config):
    """Return proposals without GT, never remove largest component or overwrite labels."""
    p = np.asarray(prediction)
    if p.ndim != 3 or not np.issubdtype(p.dtype, np.integer) or np.any((p < 0) | (p > 7)):
        raise ValueError("Expected integer 3D labels 0..7")
    if config['foreground_connectivity'] != 26 or config['background_connectivity'] != 6:
        raise ValueError("Explicit complementary 26/6 connectivity required")
    cap = int(config['maximum_component_voxels'])
    fraction = float(config['maximum_fraction_of_largest_component'])
    if cap < 1 or not 0 < fraction <= .01:
        raise ValueError("Invalid conservative threshold")
    remove_labels, fill_labels = set(config['remove_labels']), set(config['fill_labels'])
    preserved = set(config['preserved_labels'])
    if not remove_labels <= {1,2,3,4,5} or not fill_labels <= {1,2,3,4} or preserved != {6,7}:
        raise ValueError("Vessels preserved; fill restricted to chambers")
    removed = np.zeros(p.shape, dtype=bool)
    fill = np.zeros(p.shape, dtype=np.uint8)
    conflicts = np.zeros(p.shape, dtype=bool)
    records = {}
    for label in sorted(remove_labels | fill_labels):
        box = _box(p == label)
        if box is None:
            records[str(label)] = {'components_before':0,'removed_components':0,'eligible_hole_components':0,'threshold_voxels':0}
            continue
        region = p[box]
        mask = region == label
        components, count = ndimage.label(mask, structure=np.ones((3,3,3),dtype=bool))
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        largest = int(sizes.argmax())
        threshold = min(cap, int(np.floor(fraction * sizes[largest])))
        drop = np.zeros(len(sizes),dtype=bool)
        if label in remove_labels:
            drop = (sizes > 0) & (sizes <= threshold)
            drop[largest] = False
            removed[box] |= drop[components]
        holes_count = 0
        if label in fill_labels and threshold:
            holes = ndimage.binary_fill_holes(mask, structure=ndimage.generate_binary_structure(3,1)) & ~mask
            hc, hn = ndimage.label(holes, structure=ndimage.generate_binary_structure(3,1))
            hs = np.bincount(hc.ravel()); hs[0] = 0
            allowed = (hs > 0) & (hs <= threshold)
            candidate = allowed[hc] & (region == 0)
            # All proposals come from the immutable original, hence class order is immaterial.
            conflicts[box] |= candidate & (fill[box] != 0) & (fill[box] != label)
            fill[box][candidate] = label
            holes_count = int(np.count_nonzero(allowed))
        records[str(label)] = {'components_before':int(count),'largest_component_voxels':int(sizes[largest]),
                              'threshold_voxels':threshold,'removed_components':int(drop.sum()),'eligible_hole_components':holes_count}
    fill[conflicts] = 0
    return removed, fill, records


def apply_changes(prediction, removed, fill, variant):
    if variant not in ('remove_small','fill_small','combined'):
        raise ValueError("Unknown variant")
    result = np.asarray(prediction).astype(np.uint8, copy=True)
    if variant in ('remove_small','combined'):
        result[removed] = 0
    if variant in ('fill_small','combined'):
        candidate = (fill != 0) & (np.asarray(prediction) == 0)
        result[candidate] = fill[candidate]
    for label in (6,7):
        if not np.array_equal(result == label, np.asarray(prediction) == label):
            raise ValueError("Vessel mask changed")
    return result
