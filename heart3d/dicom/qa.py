"""Independent readers/rasterizers and native-grid inspection images."""
import numpy as np
from scipy import ndimage


def check_simpleitk(geometry, volume):
    import SimpleITK as sitk
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([str(p) for p in geometry.files])
    image = reader.Execute()
    affine = np.eye(4)
    affine[:3, :3] = np.array(image.GetDirection()).reshape(3, 3) @ np.diag(image.GetSpacing())
    affine[:3, 3] = image.GetOrigin()
    if not np.allclose(affine, geometry.affine_lps, atol=1e-4, rtol=0):
        raise ValueError('Independent SimpleITK LPS geometry differs')
    array = sitk.GetArrayViewFromImage(image).transpose(2, 1, 0)
    if array.shape != volume.shape:
        raise ValueError('Independent SimpleITK dimensions differ')
    maximum = max(float(np.max(np.abs(array[:, :, k] - volume[:, :, k]))) for k in range(volume.shape[2]))
    if maximum > 1e-3:
        raise ValueError('Independent SimpleITK HU values differ')
    return {'reader': 'SimpleITK.ImageSeriesReader', 'geometry_match': True,
            'max_HU_difference': maximum, 'array_match': True}


def vtk_polygon(xy, shape):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
    points = vtk.vtkPoints()
    for x, y in xy:
        points.InsertNextPoint(float(x), float(y), 0.)
    cells = vtk.vtkCellArray(); cells.InsertNextCell(len(xy)+1)
    for i in range(len(xy)): cells.InsertCellPoint(i)
    cells.InsertCellPoint(0)
    poly = vtk.vtkPolyData(); poly.SetPoints(points); poly.SetLines(cells)
    stencil = vtk.vtkPolyDataToImageStencil(); stencil.SetInputData(poly)
    stencil.SetOutputOrigin(0, 0, 0); stencil.SetOutputSpacing(1, 1, 1)
    stencil.SetOutputWholeExtent(0, shape[0]-1, 0, shape[1]-1, 0, 0)
    stencil.SetTolerance(0.)
    image = vtk.vtkImageData(); image.SetDimensions(shape[0], shape[1], 1)
    image.GetPointData().SetScalars(numpy_to_vtk(np.ones(shape[0]*shape[1], np.uint8), deep=True))
    filt = vtk.vtkImageStencil(); filt.SetInputData(image); filt.SetStencilConnection(stencil.GetOutputPort())
    filt.SetBackgroundValue(0); filt.Update()
    return vtk_to_numpy(filt.GetOutput().GetPointData().GetScalars()).reshape(shape, order='F')


def check_vtk(mask, polygons, xor=False):
    alternate = np.zeros_like(mask)
    for p in polygons:
        m = vtk_polygon(p['xy'], mask.shape[:2])
        k = p['slice']
        if xor: alternate[:, :, k] ^= m
        else: alternate[:, :, k] |= m
    difference = alternate != mask
    # Different boundary-inclusion conventions are allowed only at the boundary.
    boundary = np.zeros_like(mask, dtype=bool)
    for k in range(mask.shape[2]):
        plane = mask[:, :, k].astype(bool)
        boundary[:, :, k] = ndimage.binary_dilation(plane) ^ ndimage.binary_erosion(plane)
    nonboundary = int(np.count_nonzero(difference & ~boundary))
    if nonboundary:
        raise ValueError(f'Independent VTK differs away from polygon boundary: {nonboundary} voxels')
    intersection = int(np.count_nonzero(mask & alternate))
    dice = 2*intersection / (int(mask.sum())+int(alternate.sum()))
    if dice < .99:
        raise ValueError('Independent VTK agreement <0.99; requires review')
    return {'rasterizer': 'vtkPolyDataToImageStencil, index voxel centers',
            'Dice_between_rasterizers': dice, 'differing_boundary_voxels': int(difference.sum()),
            'nonboundary_differences': nonboundary}


def make_mosaic(volume, mask, geometry, polygons, destination, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    active = np.argwhere(mask)
    center = np.median(active, axis=0).astype(int)
    low, high = active.min(axis=0), active.max(axis=0)
    choices = [('native axial / first Heart plane', 2, int(low[2])),
               ('native axial / center', 2, int(center[2])),
               ('native axial / last Heart plane', 2, int(high[2])),
               ('native coronal / center', 1, int(center[1])),
               ('native sagittal / center', 0, int(center[0]))]
    spacing = geometry.report['spacing_xyz_mm']
    fig, axes = plt.subplots(len(choices), 2, figsize=(12, 16), constrained_layout=True)
    for row, (name, axis, index) in enumerate(choices):
        image = np.take(volume, index, axis=axis).T
        roi = np.take(mask, index, axis=axis).T
        remaining = [a for a in range(3) if a != axis]
        aspect = spacing[remaining[1]] / spacing[remaining[0]]
        for col in range(2):
            ax = axes[row, col]
            ax.imshow(image, cmap='gray', vmin=-150, vmax=250, origin='lower', aspect=aspect)
            if col:
                overlay = np.zeros((*roi.shape, 4)); overlay[roi > 0] = [.05, 1., .2, .28]
                ax.imshow(overlay, origin='lower', aspect=aspect)
                ax.contour(roi, levels=[.5], colors=['lime'], linewidths=.7)
                if axis == 2:
                    for p in polygons:
                        if p['slice'] == index:
                            points = np.vstack([p['xy'], p['xy'][0]])
                            ax.plot(points[:, 0], points[:, 1], color='magenta', linewidth=.6)
            ax.set_title(f'{name}, index={index}\n'+('CT + Heart (green), RT (magenta)' if col else 'CT HU'), fontsize=9)
            ax.set_xlabel('voxel column along native plane'); ax.set_ylabel('native-plane voxel row')
    fig.suptitle(title+'\nOriginal grid; physical aspect; native planes, no resampling', fontsize=11)
    fig.savefig(destination, dpi=120); plt.close(fig)
