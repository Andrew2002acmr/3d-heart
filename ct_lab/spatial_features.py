"""Source XYZ point features, bounded patches with exact halos, no GT inputs.

Header grid lengths remain unverified physical units. Texture padding is
half-sample reflect; context/profile indices replicate the nearest edge.
"""
import os
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
import ct_lab.launch
os.environ['NUMBA_CACHE_DIR']=str(Path(__file__).resolve().parents[1]/'outputs/spatial_feature_02/numba_cache')

PLANES={'Axial XY':(0,1,2),'Coronal XZ':(0,2,1),'Sagittal YZ':(1,2,0)}
STATS=['Mean','Std','Gradient','LoG']


def context_names(radius):
    return ([f'Z I {i:+d}' for i in range(-radius,radius+1) if i]+
            [f'Z diff {i:+d}' for i in range(-radius,radius+1) if i]+
            [f'Z{radius} Mean',f'Z{radius} Std',f'Z{radius} Gradient',f'Z{radius} Range'])


def feature_sets():
    xy=[f'XY {s}' for s in ('Gradient','Mean','Std','LoG')];tri=xy+[f'{p} {s}' for p in ('XZ','YZ') for s in STATS]
    return {'A Intensity':['Intensity'], 'B 2D':['Intensity',*xy],
            **{f'{chr(66+r)} Z±{r}':['Intensity',*xy,*context_names(r)] for r in (1,2,3)},
            'F Tri-planar':['Intensity',*tri],
            'G 3D':['Intensity',*[f'3D {s}' for s in STATS],*[f'3D Gaussian {s:g}' for s in (.5,1.,2.)]],
            'H Tri-planar+Z':['Intensity',*tri,*context_names(3)]}


def check_affine(affine):
    a=np.asarray(affine,float)
    if a.shape!=(4,4) or not np.isfinite(a).all() or not np.allclose(a[:3,:3],np.diag(np.diag(a[:3,:3])),atol=1e-8) or np.any(np.diag(a[:3,:3])<=0):
        raise ValueError('Frozen source-ID experiment requires native axis-aligned RAS; no silent oblique transpose/resampling')
    return np.diag(a[:3,:3]).copy()


def plane_coordinates(xyz,plane):
    a,b,fixed=PLANES[plane];xyz=np.asarray(xyz)
    return xyz[...,a],xyz[...,b],xyz[...,fixed]


def from_plane(u,v,fixed,plane):
    a,b,c=PLANES[plane];xyz=np.zeros(3,int);xyz[a]=int(round(float(u)));xyz[b]=int(round(float(v)));xyz[c]=int(fixed)
    return xyz


def reflect_indices(indices,size):
    if size<1:raise ValueError('Empty axis')
    j=np.asarray(indices)% (2*size)
    return np.where(j<size,j,2*size-1-j)


def gather(volume,xyz,radius):
    offsets=np.arange(-radius,radius+1)
    indices=[reflect_indices(xyz[:,a,None]+offsets,volume.shape[a]) for a in range(3)]
    return np.asarray(volume[indices[0][:,:,None,None],indices[1][:,None,:,None],indices[2][:,None,None,:]],np.float32)


def z_context(volume,xyz,radius=3,spacing=1.):
    xyz=np.asarray(xyz,int);offsets=np.arange(-radius,radius+1)
    wanted=xyz[:,2,None]+offsets
    indices=np.clip(wanted,0,volume.shape[2]-1)
    values=volume[xyz[:,0,None],xyz[:,1,None],indices].astype(np.float32)
    return values,indices!=wanted


def kernels(sigma):
    radius=int(4*sigma+.5);t=np.arange(-radius,radius+1,dtype=float)
    w=np.exp(-.5*(t/sigma)**2);w/=w.sum()
    return w,(t*t/sigma**4-1/sigma**2)*w


def stencil(patch,spacing,axes):
    """Means/variances use float64 accumulation; retained features are float32."""
    r=patch.shape[1]//2
    s=tuple([slice(None)]+[slice(r-2,r+3) if a in axes else r for a in range(3)])
    local=patch[s].astype(np.float64)
    dims=tuple(range(1,local.ndim));mean=local.mean(dims);std=local.std(dims)
    gm=np.zeros(len(patch),float);log=np.zeros(len(patch),float)
    for a in axes:
        plus=[slice(None),r,r,r];minus=plus.copy();plus[a+1]=r+1;minus[a+1]=r-1
        gradient=(patch[tuple(plus)].astype(float)-patch[tuple(minus)])/ (2*spacing[a])
        gm+=gradient*gradient
        weights=[];slices=[slice(None)]
        for b in range(3):
            if b in axes:
                w,w2=kernels(1./spacing[b]);weights.append(w2/spacing[b]**2 if b==a else w)
                length=len(w)//2;slices.append(slice(r-length,r+length+1))
            else:slices.append(r)
        q=patch[tuple(slices)]
        if len(axes)==2:log+=np.einsum('nij,i,j->n',q,*weights,optimize=True)
        else:log+=np.einsum('nijk,i,j,k->n',q,*weights,optimize=True)
    return [mean,std,np.sqrt(gm),log]


def bilateral_patch(patch,output_radius,spacing,dimensions):
    """Exact finite-support bilateral, sigma_space=1 and sigma_intensity=100.

    Filter the entire requested patch; its surrounding halo is raw source data.
    No independent per-plane fitting or intensity normalization.
    """
    from numba import njit
    return _bilateral_kernel(patch,output_radius,np.asarray(spacing,float),dimensions)


_bilateral_kernel=None
def _prepare_bilateral():
    global _bilateral_kernel
    if _bilateral_kernel is not None:return
    from numba import njit
    @njit(cache=False)
    def compute(patch,r,spacing,dimensions):
        count=patch.shape[0];center=patch.shape[1]//2;length=2*r+1
        result=np.empty((count,length,length,length),np.float32)
        rx=int(np.ceil(3/spacing[0]));ry=int(np.ceil(3/spacing[1]));rz=int(np.ceil(3/spacing[2])) if dimensions==3 else 0
        for n in range(count):
            for x in range(length):
                for y in range(length):
                    for z in range(length):
                        xx=x+center-r;yy=y+center-r;zz=z+center-r;v=patch[n,xx,yy,zz]
                        numerator=0.;denominator=0.
                        for dx in range(-rx,rx+1):
                            for dy in range(-ry,ry+1):
                                for dz in range(-rz,rz+1):
                                    other=patch[n,xx+dx,yy+dy,zz+dz]
                                    w=np.exp(-.5*((dx*spacing[0])**2+(dy*spacing[1])**2+(dz*spacing[2])**2)-.5*((other-v)/100.)**2)
                                    numerator+=w*other;denominator+=w
                        result[n,x,y,z]=numerator/denominator
        return result
    _bilateral_kernel=compute


def extract(volume,xyz,affine,chunk_size=64,filter_config=None,include_gaussians=True):
    """Extract only sampled patches. chunk_size changes batching, not features.

    Halo = feature radius + filter radius. Thus chunk boundaries cannot become
    image boundaries. There is no mask/label/distance input to this function.
    """
    spacing=check_affine(affine);xyz=np.asarray(xyz,int)
    if volume.ndim!=3 or xyz.ndim!=2 or xyz.shape[1]!=3 or np.any(xyz<0) or np.any(xyz>=np.asarray(volume.shape)):
        raise ValueError('Source XYZ coordinates outside volume')
    if chunk_size<1:raise ValueError('Positive chunk size required')
    config=filter_config or {'method':'Baseline','dimensions':3}
    radii=[int(4/step+.5) for step in spacing]
    radius=max(3,2,*radii,*([int(8/step+.5) for step in spacing] if include_gaussians else []))
    method=config['method'];dimensions=int(config.get('dimensions',3));sigma=float(config.get('sigma',1.))
    if dimensions not in (2,3) or method not in ('Baseline','Gaussian','Bilateral'):raise ValueError('Invalid filter specification')
    halo=0
    if method=='Gaussian':
        if sigma<=0:raise ValueError('Positive filter sigma required')
        halo=max(int(4*sigma/step+.5) for step in spacing[:dimensions])
    elif method=='Bilateral':
        if config.get('spatial_sigma',1.)!=1 or config.get('intensity_sigma',100.)!=100:raise ValueError('Frozen bilateral requires sigma_space=1, intensity=100')
        halo=max(int(np.ceil(3/step)) for step in spacing[:dimensions]);_prepare_bilateral()
    if radius+halo>24:raise ValueError('Requested patch radius exceeds bounded CPU memory protocol')
    estimated_patch_bytes=chunk_size*(2*(radius+halo)+1)**3*4*4
    if estimated_patch_bytes>256*1024**2:raise ValueError('Estimated temporary patch buffers exceed 256 MiB; reduce chunk_size')
    pieces=[];schema=None
    for start in range(0,len(xyz),chunk_size):
        points=xyz[start:start+chunk_size];patch=gather(volume,points,radius+halo)
        if not np.isfinite(patch).all():raise ValueError('Nonfinite source patch')
        if method=='Gaussian':
            sigmas=[0.,sigma/spacing[0],sigma/spacing[1],sigma/spacing[2] if dimensions==3 else 0.]
            patch=ndi.gaussian_filter(patch,sigmas,mode='reflect',truncate=4.)
            if halo:patch=patch[:,halo:-halo,halo:-halo,halo:-halo]
        elif method=='Bilateral':patch=bilateral_patch(patch,radius,spacing,dimensions)
        r=radius;center=patch[:,r,r,r]
        bank={'Intensity':center}
        for prefix,axes in [('XY',(0,1)),('XZ',(0,2)),('YZ',(1,2)),('3D',(0,1,2))]:
            values=stencil(patch,spacing,axes)
            # Match np.gradient's one-sided edge rule for the GM baseline.
            gradient_sq=np.zeros(len(points),float)
            for a in axes:
                plus=[slice(None),r,r,r];minus=plus.copy();plus[a+1]=r+1;minus[a+1]=r-1
                g=(patch[tuple(plus)].astype(float)-patch[tuple(minus)])/(2*spacing[a])
                at_lo=points[:,a]==0;at_hi=points[:,a]==volume.shape[a]-1
                g[at_lo]=(patch[tuple(plus)][at_lo]-center[at_lo])/spacing[a]
                g[at_hi]=(center[at_hi]-patch[tuple(minus)][at_hi])/spacing[a]
                gradient_sq+=g*g
            values[2]=np.sqrt(gradient_sq)
            bank.update({prefix+' '+key:value for key,value in zip(STATS,values)})
        if include_gaussians:
            for scale in (.5,1.,2.):
                weights=[kernels(scale/s)[0] for s in spacing]
                slices=tuple([slice(None)]+[slice(r-len(w)//2,r+len(w)//2+1) for w in weights])
                bank[f'3D Gaussian {scale:g}']=np.einsum('nijk,i,j,k->n',patch[slices],*weights,optimize=True)
        for a,name in enumerate('XYZ'):
            for offset in range(-3,4):
                # Replicate edge for profiles/context, including filtered data.
                actual=np.clip(points[:,a]+offset,0,volume.shape[a]-1)-points[:,a]
                index=[np.arange(len(points)),np.full(len(points),r),np.full(len(points),r),np.full(len(points),r)]
                index[a+1]=r+actual
                bank[f'{name} I {offset:+d}']=patch[tuple(index)]
        for offset in range(-3,4):
            if offset:bank[f'Z diff {offset:+d}']=bank[f'Z I {offset:+d}']-center
        for nr in (1,2,3):
            values=np.stack([bank[f'Z I {o:+d}'] for o in range(-nr,nr+1)],axis=1).astype(float)
            bank.update({f'Z{nr} Mean':values.mean(1),f'Z{nr} Std':values.std(1),
                         f'Z{nr} Gradient':(bank['Z I +1']-bank['Z I -1'])/(2*spacing[2]),f'Z{nr} Range':np.ptp(values,axis=1)})
        names=list(bank)
        if schema is None:schema=names
        if schema!=names:raise ValueError('Feature schema changed between chunks')
        pieces.append(np.column_stack(list(bank.values())).astype(np.float32))
    if not pieces:raise ValueError('No sampled voxels')
    return {'names':schema,'values':np.concatenate(pieces),'xyz':xyz,'spacing_xyz':spacing.tolist(),
            'patch_radius':radius,'filter_halo':halo,'filter':config,'chunk_size':chunk_size,
            'estimated_temporary_patch_bytes':estimated_patch_bytes,
            'GT_input':False,'units':'unverified header grid units'}


def columns(bank,names):
    return bank['values'][:,[bank['names'].index(n) for n in names]]
