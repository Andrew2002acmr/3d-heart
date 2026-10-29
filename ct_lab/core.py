"""Array axes XYZ, world coordinates RAS, units explicitly carried from NIfTI.

No implicit HU conversion or clinical geometry correction is performed.
"""
from pathlib import Path
import hashlib
import itertools
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
from scipy import ndimage as ndi

LABELS = {0: 'background', 1: 'LV', 2: 'RV', 3: 'LA', 4: 'RA', 5: 'MYO', 6: 'AO', 7: 'PA'}
PERCENTILES = [1, 5, 25, 50, 75, 95, 99]


def cardiac_target(mask):
    """Historical evaluation ignores source labels outside cardiac 0..7."""
    result = np.array(mask, dtype=np.uint8, copy=True)
    result[(mask < 0) | (mask > 7)] = 255
    return result


def discover(root):
    root = Path(root)
    return [{'id': p.name.removesuffix('_image.nii.gz').removesuffix('_image.nii'),
             'image': str(p), 'mask': str(p.with_name(p.name.replace('_image.', '_label.')))}
            for p in sorted(root.rglob('*_image.nii*')) if p.is_file()]


def fingerprint(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def stats(array):
    a = np.asarray(array)
    finite = np.isfinite(a)
    x = a[finite]
    result = {'count': int(a.size), 'finite': int(x.size), 'nan': int(np.isnan(a).sum()),
              'inf': int(np.isinf(a).sum()), 'dtype': str(a.dtype)}
    if x.size:
        quantiles = np.percentile(x, PERCENTILES).tolist()
        result.update(min=float(x.min()), max=float(x.max()), mean=float(x.mean(dtype=np.float64)),
                      std=float(x.std(dtype=np.float64)), median=quantiles[3],
                      percentiles=dict(zip(map(str, PERCENTILES), quantiles)))
    return result


def metadata(image):
    q, qc = image.get_qform(coded=True)
    s, sc = image.get_sform(coded=True)
    affine = image.affine
    spacing = nib.affines.voxel_sizes(affine)
    return {'shape_xyz': list(image.shape), 'spacing_xyz': spacing.tolist(),
            'origin_ras': affine[:3, 3].tolist(), 'direction_ras': (affine[:3, :3] / spacing).tolist(),
            'affine_ras': affine.tolist(), 'orientation': ''.join(nib.aff2axcodes(affine)),
            'spatial_unit': image.header.get_xyzt_units()[0], 'storage_dtype': str(image.get_data_dtype()),
            'slope': image.dataobj.slope if hasattr(image.dataobj, 'slope') else None,
            'intercept': image.dataobj.inter if hasattr(image.dataobj, 'inter') else None,
            'qform_code': int(qc), 'sform_code': int(sc),
            'qform_sform_disagree': bool(qc and sc and not np.allclose(q, s, atol=1e-5, rtol=0)),
            'HU_verified': False, 'physical_scale_verified': False,
            'center_extent_xyz': ((np.asarray(image.shape)-1)*spacing).tolist()}


def geometry_matches(image, mask):
    return (image.shape == mask.shape and image.header.get_xyzt_units()[0] == mask.header.get_xyzt_units()[0]
            and np.allclose(image.affine, mask.affine, atol=1e-5, rtol=0))


def load_pair(row):
    image = nib.load(row['image'])
    if len(image.shape) != 3:
        raise ValueError('Expected a 3D NIfTI volume')
    mask = nib.load(row['mask']) if Path(row['mask']).is_file() else None
    if mask is not None:
        if not geometry_matches(image, mask):
            raise ValueError('CT/mask geometry mismatch: shape, affine or spatial units')
        values = np.asanyarray(mask.dataobj)
        if not np.isfinite(values).all() or not np.equal(values, np.rint(values)).all():
            raise ValueError('Nonfinite or noninteger segmentation labels')
    return image, mask


def world_grid(image, spacing=None, max_voxels=100_000_000):
    """Axis aligned RAS grid covering all source voxel centers, including oblique CT."""
    corners = np.array(list(itertools.product(*[(0, n-1) for n in image.shape])))
    points = nib.affines.apply_affine(image.affine, corners)
    lo, hi = points.min(0), points.max(0)
    spacing = np.asarray(spacing if spacing is not None else nib.affines.voxel_sizes(image.affine), float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError('Positive finite XYZ spacing required')
    shape = np.ceil((hi-lo)/spacing).astype(int)+1
    if np.prod(shape.astype(float)) > max_voxels:
        raise ValueError('Requested grid exceeds 100 million voxels; use coarser spacing')
    affine = np.diag([*spacing, 1.])
    affine[:3, 3] = lo
    return tuple(shape), affine


def orthogonal_pair(image, mask=None, spacing=None):
    if mask is not None and not geometry_matches(image,mask):
        raise ValueError('CT/mask geometry mismatch before orthogonal resampling')
    linear = image.affine[:3, :3]
    # An explicitly verified axis-aligned RAS grid already represents true
    # orthogonal planes. Avoid a needless full-volume interpolation/copy.
    if (np.all(np.diag(linear) > 0) and np.allclose(linear, np.diag(np.diag(linear)), atol=1e-8, rtol=0)
            and (spacing is None or np.allclose(spacing, np.diag(linear), atol=1e-8, rtol=0))):
        return image, mask
    grid = world_grid(image, spacing)
    # Explicit reslicing in physical RAS, not an unchecked transpose.
    ct = resample_from_to(image, grid, order=1, mode='constant', cval=0)
    seg = resample_from_to(mask, grid, order=0, mode='constant', cval=0) if mask is not None else None
    for item in (ct, seg):
        if item is not None:
            item.header.set_xyzt_units(image.header.get_xyzt_units()[0])
    return ct, seg


def crop_pad(image, start, size, fill=0):
    """Index crop and/or padding with translated affine and unchanged units."""
    start, size = np.asarray(start, int), np.asarray(size, int)
    if start.shape != (3,) or size.shape != (3,) or np.any(size < 1) or np.prod(size.astype(float)) > 100_000_000:
        raise ValueError('Invalid crop dimensions')
    src = np.asanyarray(image.dataobj)
    output = np.full(tuple(size), fill, dtype=src.dtype)
    lo, hi = np.maximum(start, 0), np.minimum(start+size, src.shape)
    if np.all(hi > lo):
        output[tuple(slice(a, b) for a, b in zip(lo-start, hi-start))] = src[tuple(slice(a, b) for a, b in zip(lo, hi))]
    affine = image.affine.copy()
    affine[:3, 3] = nib.affines.apply_affine(image.affine, start)
    return nib.Nifti1Image(output, affine, image.header.copy())


def resize_dimensions(image, size, mask=None):
    """Change sample counts while retaining voxel-center physical endpoints."""
    if mask is not None and not geometry_matches(image,mask):
        raise ValueError('CT/mask geometry mismatch before dimension resize')
    size = np.asarray(size, int)
    if size.shape != (3,) or np.any(size < 2) or np.prod(size.astype(float)) > 100_000_000:
        raise ValueError('Size must be XYZ >=2, at most 100 million voxels')
    affine = image.affine.copy()
    affine[:3, :3] *= (np.asarray(image.shape)-1)/(size-1)
    grid = tuple(size), affine
    result = resample_from_to(image, grid, order=1, mode='nearest')
    target = resample_from_to(mask, grid, order=0, mode='nearest') if mask is not None else None
    return result, target


def window(array, low, high):
    if not np.isfinite([low, high]).all() or high <= low:
        raise ValueError('Upper bound must exceed lower bound')
    x = np.asarray(array, dtype=np.float32)
    valid = np.isfinite(x)
    return np.clip(x, low, high), {'below_fraction': float(np.mean(x[valid] < low)) if valid.any() else None,
                                  'above_fraction': float(np.mean(x[valid] > high)) if valid.any() else None}


def normalize(array, method, low=0., high=2015., mean=None, std=None):
    x = np.asarray(array, dtype=np.float32)
    finite = x[np.isfinite(x)]
    if not finite.size:
        raise ValueError('No finite values')
    if method == 'None':
        return x.copy()
    if method == 'Z-score':
        mean, std = float(finite.mean(dtype=np.float64)), float(finite.std(dtype=np.float64))
    if method in ('Z-score', 'Saved train Z-score'):
        if mean is None or std is None or not np.isfinite([mean, std]).all() or std <= 0:
            raise ValueError('Finite saved mean and positive std required')
        return (x-mean)/std
    if method == 'Min-Max':
        low, high = float(finite.min()), float(finite.max())
        if high == low:
            return np.zeros_like(x)
    elif method == 'Percentile':
        low, high = np.percentile(finite, [1, 99])
        if high == low:
            return np.zeros_like(x)
    elif method not in ('Fixed-window', 'Saved train window'):
        raise ValueError('Unknown normalization')
    return (window(x, low, high)[0]-low)/(high-low)*2-1


def filter_plane(plane, method, strength):
    from skimage import restoration, exposure, feature
    x = np.asarray(plane, np.float32)
    if method == 'None': return x.copy()
    if method == 'Gaussian': return ndi.gaussian_filter(x, strength)
    if method == 'Median': return ndi.median_filter(x, size=2*int(strength)+1)
    if method == 'Gradient Magnitude': return ndi.gaussian_gradient_magnitude(x, strength)
    if method == 'Sobel': return np.hypot(ndi.sobel(x, 0), ndi.sobel(x, 1))
    if method == 'Laplacian': return ndi.laplace(x)
    unit = (normalize(x, 'Min-Max')+1)/2
    if method == 'Bilateral': return restoration.denoise_bilateral(unit, sigma_color=.05*strength, sigma_spatial=strength, channel_axis=None)
    if method == 'Non-local Means': return restoration.denoise_nl_means(unit, h=.04*strength, fast_mode=True, patch_size=3, patch_distance=3, channel_axis=None)
    if method == 'CLAHE': return exposure.equalize_adapthist(unit, clip_limit=.01*strength)
    if method == 'Canny': return feature.canny(unit, sigma=strength).astype(np.float32)
    raise ValueError('Unknown filter')


def plane(array, axis, index):
    return np.take(array, index, axis=axis).T


def mask_analysis(mask, affine, unit='unknown', components=True):
    """26-connectivity in 3D and 8-connectivity per 2D plane; never conflate."""
    spacing = nib.affines.voxel_sizes(affine)
    rows = []
    area = {}
    for label in np.unique(mask):
        # np.take on a Fortran/strided NIfTI mask can copy the entire volume
        # for every requested slice. One contiguous boolean buffer bounds work.
        binary = np.ascontiguousarray(mask == label)
        count = int(binary.sum())
        row = {'label': int(label), 'name': LABELS.get(int(label), 'UNKNOWN'), 'voxels': count,
               'fraction': count/mask.size, 'volume_grid_units3': count*abs(float(np.linalg.det(affine[:3, :3]))),
               'unit': unit, 'volume_mm3_verified': None}
        if label and components:
            labels, n = ndi.label(binary, structure=np.ones((3, 3, 3)))
            sizes = np.bincount(labels.ravel())[1:]
            del labels
            coordinates = np.nonzero(binary)
            centroid = [float(c.mean()) for c in coordinates]
            row.update(components_3d_26=int(n), component_sizes_voxels=sorted(sizes.tolist(), reverse=True),
                       centroid_world_ras=nib.affines.apply_affine(affine, centroid).tolist())
            row['extent_grid_units_xyz'] = [(int(c.max())-int(c.min())+1)*float(d) for c,d in zip(coordinates,spacing)]
            del coordinates
            row['components_2d_8_per_axis'] = {str(axis): [int(ndi.label(np.take(binary, i, axis), np.ones((3,3)))[1])
                                                       for i in range(mask.shape[axis])] for axis in range(3)}
        rows.append(row)
        area[str(int(label))] = {str(axis): (binary.sum(axis=tuple(a for a in range(3) if a != axis))*
                                            np.prod(np.delete(spacing, axis))).tolist() for axis in range(3)}
    return {'classes': rows, 'slice_area_grid_units2': area}


def line_profile(original, processed, mask, p0, p1, spacing_2d):
    n = max(2, int(np.linalg.norm(np.asarray(p1)-p0)*2)+1)
    coords = np.linspace(p0, p1, n).T
    raw = ndi.map_coordinates(original, coords, order=1, mode='nearest')
    after = ndi.map_coordinates(processed, coords, order=1, mode='nearest')
    distance = np.linspace(0, np.linalg.norm((np.asarray(p1)-p0)*spacing_2d), n)
    return {'distance': distance, 'original': raw, 'processed': after,
            'gradient': np.gradient(raw, distance) if distance[-1] > 0 else np.zeros(n),
            'label': ndi.map_coordinates(mask, coords, order=0, mode='nearest') if mask is not None else np.zeros(n)}


def fragment_comparison(prediction, target, affine):
    """Localize 26-connected prediction components and measure their actual GT overlaps.

    A satellite component is not automatically an error. Preserve its overlap
    with every source label, including extra annotations, for review.
    """
    if prediction.shape != target.shape: raise ValueError('Prediction/GT shape mismatch')
    records=[]
    for c in range(1,8):
        labels,n=ndi.label(np.ascontiguousarray(prediction==c),np.ones((3,3,3)))
        sizes=np.bincount(labels.ravel())[1:]
        largest=int(np.argmax(sizes)+1) if n else 0
        for component,bbox in enumerate(ndi.find_objects(labels),1):
            if bbox is None: continue
            local=labels[bbox]==component
            truth=target[bbox][local]
            values,counts=np.unique(truth,return_counts=True)
            coordinates=np.nonzero(local)
            centroid=[float(p.mean()+s.start) for p,s in zip(coordinates,bbox)]
            records.append({'class':c,'name':LABELS[c],'component':component,'largest':component==largest,
                            'voxels':int(sizes[component-1]),'gt_same_class_fraction':float(np.mean(truth==c)),
                            'gt_overlap_voxels':dict(zip(map(str,values.tolist()),map(int,counts))),
                            'centroid_world_ras':nib.affines.apply_affine(affine,centroid).tolist(),
                            'bbox_xyz':[[s.start,s.stop] for s in bbox]})
    return records
