"""Explicit original XYZ <-> network DHW mapping, preserving native directions."""
import numpy as np
from scipy.ndimage import map_coordinates


def make_transform(shape_xyz, affine_ras, size, z_spacing_mm):
    shape=np.asarray(shape_xyz,dtype=int); affine=np.asarray(affine_ras,dtype=float)
    spacing=np.linalg.norm(affine[:3,:3],axis=0)
    if np.any(shape<2) or np.any(spacing<=0) or size<8 or z_spacing_mm<=0:
        raise ValueError('Invalid physical grid or target size')
    fov=shape[:2]*spacing[:2]
    resized_xy=np.maximum(1,np.rint(size*fov/fov.max()).astype(int))
    pad_xy=(size-resized_xy)//2
    span=(shape[2]-1)*spacing[2]
    positions=np.arange(int(np.floor(span/z_spacing_mm))+1)*z_spacing_mm
    return {'original_shape_xyz':shape.tolist(),'original_affine_ras':affine.tolist(),
        'original_spacing_xyz_mm':spacing.tolist(),'size':size,
        'resized_xy':resized_xy.tolist(),'pad_xy':pad_xy.tolist(),
        'z_positions_mm':positions.tolist(),'target_z_spacing_mm':z_spacing_mm,
        'array_axes':'network [slice,row,column]; source [column,row,slice]',
        'z_origin':'relative distance along original slice normal',
        'coordinate_mapping':'pixel centers: (output+.5)*source/resized-.5',
        'crop':'none; full image FOV fit/pad, no ground-truth crop'}


def resample_xyz(volume, transform, order, fill_value=0):
    """Linear CT / nearest binary mask; one plane of interpolation workspace."""
    if order not in (0,1):raise ValueError('Only nearest/linear supported')
    if tuple(volume.shape)!=tuple(transform['original_shape_xyz']):raise ValueError('Source shape mismatch')
    nx,ny,_=volume.shape; rx,ry=transform['resized_xy']; px,py=transform['pad_xy']; size=transform['size']
    x=(np.arange(rx)+.5)*nx/rx-.5; y=(np.arange(ry)+.5)*ny/ry-.5
    gx,gy=np.meshgrid(x,y,indexing='xy')
    positions=np.asarray(transform['z_positions_mm']); dz=transform['original_spacing_xyz_mm'][2]
    result=np.full((len(positions),size,size),fill_value,dtype=np.float32 if order else np.uint8)
    for k,position in enumerate(positions):
        z=np.full_like(gx,position/dz)
        result[k,py:py+ry,px:px+rx]=map_coordinates(volume,[gx,gy,z],order=order,mode='nearest',prefilter=False)
    return result


def physical_context(positions_mm, center, offsets_mm):
    positions=np.asarray(positions_mm,dtype=float)
    if not len(positions) or np.any(np.diff(positions)<=0):raise ValueError('Monotonic physical positions required')
    requested=positions[center]+np.asarray(offsets_mm,dtype=float)
    outside=(requested<positions[0])|(requested>positions[-1])
    # Explicit replicate-edge context outside the acquired domain, never wrap.
    clamped=np.clip(requested,positions[0],positions[-1])
    high=np.searchsorted(positions,clamped,side='right').clip(1,len(positions)-1)
    low=high-1
    weights=(clamped-positions[low])/(positions[high]-positions[low]) if len(positions)>1 else np.zeros_like(clamped)
    if len(positions)==1:low=np.zeros_like(high);high=low.copy()
    return low,high,weights,outside


def restore_probability(probability_dhw, transform):
    """Restore floating probabilities to original XYZ; threshold after interpolation."""
    positions=np.asarray(transform['z_positions_mm']); size=transform['size']
    if probability_dhw.shape!=(len(positions),size,size):raise ValueError('Prediction grid mismatch')
    nx,ny,nz=transform['original_shape_xyz']; rx,ry=transform['resized_xy']; px,py=transform['pad_xy']
    x=(np.arange(nx)+.5)*rx/nx-.5+px; y=(np.arange(ny)+.5)*ry/ny-.5+py
    gx,gy=np.meshgrid(x,y,indexing='ij')
    dz=transform['original_spacing_xyz_mm'][2]
    output=np.empty((nx,ny,nz),dtype=np.float32)
    for z in range(nz):
        fractional=np.interp(z*dz,positions,np.arange(len(positions)))
        output[:,:,z]=map_coordinates(probability_dhw,[np.full_like(gx,fractional),gy,gx],order=1,mode='nearest',prefilter=False)
    return output
