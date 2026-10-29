"""Experiment 01: source-scale 2D filters, features and coordinate-preserving sampling.

GT is used only for sampling/evaluation, never by filters or feature extraction.
All axes of a displayed plane are row=source Y, column=source X. No HU/mm claim.
"""
import hashlib
import numpy as np
from scipy import ndimage as ndi
from skimage import exposure, restoration

SEED = 20261028
FEATURES = ['Intensity','Gradient Magnitude','Local Mean','Local Std','Laplacian of Gaussian']
PAIRS = {'LV / MYO': (1,5), 'RV / PA': (2,7)}
CONFIGS = [dict(name='Baseline',method='Baseline'),
           *[dict(name=f'Gaussian {s}',method='Gaussian',sigma=s) for s in (.5,1.,2.)],
           dict(name='Bilateral',method='Bilateral',spatial_sigma=1.,intensity_sigma=100.),
           dict(name='CLAHE',method='CLAHE',clip_limit=.01,kernel_size=32,input_range=[0.,4095.])]
FEATURE_PROTOCOL = {'local_window_pixels':5,'local_mode':'reflect','gradient':'central differences / header grid spacing',
                    'LoG_sigma_grid':1.,'LoG_mode':'reflect','intensity':'source scale; CLAHE mapped back to configured source range',
                    'GT_features':False,'physical_units_verified':False}


def apply_filter(source, config, spacing=(1.,1.)):
    x=np.array(source,dtype=np.float32,copy=True)
    spacing=np.asarray(spacing,float)
    if x.ndim!=2 or min(x.shape)<2 or not np.isfinite(x).all():raise ValueError('Finite 2D source plane required')
    if spacing.shape!=(2,) or not np.isfinite(spacing).all() or np.any(spacing<=0):raise ValueError('Positive grid spacing required')
    method=config['method'];clipped=0.
    if method=='Baseline' or method=='Gradient Magnitude':
        result=x.copy() # gradient is a feature channel, never replaces CT
    elif method=='Gaussian':
        if config['sigma']<=0:raise ValueError('Positive Gaussian sigma required')
        result=ndi.gaussian_filter(x,config['sigma']/spacing,mode='reflect')
    elif method=='Bilateral':
        spatial=float(config['spatial_sigma']);intensity=float(config['intensity_sigma'])
        if spatial<=0 or intensity<=0:raise ValueError('Positive bilateral sigmas required')
        # skimage bilateral has isotropic pixel support. Reject anisotropy rather
        # than silently interpret its scalar pixel sigma as a world-grid sigma.
        if not np.isclose(*spacing):raise ValueError('Bilateral requires isotropic in-plane grid spacing')
        result=restoration.denoise_bilateral(x,sigma_color=intensity,sigma_spatial=spatial/spacing[0],
                                            channel_axis=None,mode='reflect',bins=10000)
    elif method=='CLAHE':
        low,high=config.get('input_range',[0.,4095.])
        clip=float(config['clip_limit']);kernel=int(config['kernel_size'])
        if high<=low or not 0<clip<=1 or kernel<2:raise ValueError('Invalid CLAHE settings')
        clipped=float(np.mean((x<low)|(x>high)))
        unit=np.clip((x-low)/(high-low),0,1)
        result=exposure.equalize_adapthist(unit,kernel_size=kernel,clip_limit=clip,nbins=256)*(high-low)+low
    else:raise ValueError('Unknown filter')
    return np.asarray(result,np.float32),{'config':dict(config),'sequence':
           ['copy source','fixed-range clip and unit conversion','CLAHE','map to configured source range'] if method=='CLAHE' else ['copy source',method],
           'clipped_fraction':clipped,'spacing_row_col':spacing.tolist(),'feature_protocol':FEATURE_PROTOCOL}


def feature_maps(source,spacing=(1.,1.)):
    x=np.asarray(source,dtype=np.float64)
    if x.ndim!=2 or not np.isfinite(x).all():raise ValueError('Finite plane required')
    spacing=np.asarray(spacing,float)
    mean=ndi.uniform_filter(x,size=5,mode='reflect')
    var=np.maximum(ndi.uniform_filter(x*x,size=5,mode='reflect')-mean*mean,0)
    gradient=np.hypot(*np.gradient(x,*spacing))
    sigma=1./spacing
    log=(ndi.gaussian_filter(x,sigma,order=(2,0),mode='reflect')/spacing[0]**2+
         ndi.gaussian_filter(x,sigma,order=(0,2),mode='reflect')/spacing[1]**2)
    return np.stack([x,gradient,mean,np.sqrt(var),log],axis=-1).astype(np.float32)


def sample_voxels(mask,pair,case,z,shape_xyz,roi=None,per_class=1000,seed=SEED):
    """Stable global source IDs, unaffected by filter, feature axes or ROI display crop."""
    mask=np.asarray(mask)
    roi=roi or (0,mask.shape[0],0,mask.shape[1])
    y0,y1,x0,x1=map(int,roi)
    if not (0<=y0<y1<=mask.shape[0] and 0<=x0<x1<=mask.shape[1]):raise ValueError('Invalid half-open ROI')
    if tuple(mask.shape)!=(shape_xyz[1],shape_xyz[0]) or not 0<=z<shape_xyz[2]:raise ValueError('Source coordinate mismatch')
    seed_case=int.from_bytes(hashlib.sha256(f'{seed}|{case}|{z}'.encode()).digest()[:8],'little')
    rng=np.random.default_rng(seed_case);rows=[];cols=[];labels=[];fold=[]
    for label in pair:
        yy,xx=np.where(mask[y0:y1,x0:x1]==label)
        chosen=rng.choice(len(yy),size=min(int(per_class),len(yy)),replace=False)
        yy,xx=yy[chosen]+y0,xx[chosen]+x0
        n=len(yy);train=np.zeros(n,bool);train[:int(.8*n)]=True;rng.shuffle(train)
        rows.extend(yy.tolist());cols.extend(xx.tolist());labels.extend([label]*n);fold.extend(train.tolist())
    rows,cols=np.asarray(rows,dtype=np.int32),np.asarray(cols,dtype=np.int32)
    xyz=np.column_stack([cols,rows,np.full(len(rows),z)])
    ids=np.ravel_multi_index(xyz.T,shape_xyz) if len(rows) else np.array([],np.int64)
    return {'case':case,'z':int(z),'rows':rows,'cols':cols,'labels':np.array(labels,np.int16),
            'train':np.array(fold,bool),'voxel_ids':ids,'xyz':xyz,'seed':seed,'roi':list(roi)}


def matrix(maps,sample):
    return maps[sample['rows'],sample['cols']].copy()


def fixed_regions(source_maps,mask,pair,roi):
    y0,y1,x0,x1=roi;region=np.zeros(mask.shape,bool);region[y0:y1,x0:x1]=True
    a,b=(mask==c for c in pair)
    boundary=((ndi.binary_dilation(a)&b)|(ndi.binary_dilation(b)&a))&region
    homogeneous=np.zeros_like(region)
    for cls in (a,b):
        inside=ndi.binary_erosion(cls,iterations=3,border_value=0)&region
        values=source_maps[:,:,3][inside]
        if values.size:
            homogeneous |= inside & (source_maps[:,:,3]<=np.percentile(values,25))
    return region,boundary,homogeneous


def measurements(maps,mask,pair,regions,clip_fraction=0.):
    region,boundary,homogeneous=regions
    a,b=[maps[:,:,0][(mask==c)&region].astype(np.float64) for c in pair]
    if not len(a) or not len(b):return {'valid':False,'reason':'Both classes required in ROI'}
    ha,_=np.histogram(a,bins=128,range=(0,4095));hb,_=np.histogram(b,bins=128,range=(0,4095))
    contrast=abs(float(a.mean()-b.mean()));den=float(a.var()+b.var())
    return {'valid':True,'class_a_voxels':len(a),'class_b_voxels':len(b),
            'contrast_abs_mean':contrast,'fisher_ratio':contrast**2/den if den else None,
            'histogram_overlap':float(np.minimum(ha/ha.sum(),hb/hb.sum()).sum()) if ha.sum() and hb.sum() else None,
            'histogram_range':[0,4095],'histogram_bins':128,'histogram_outside_fraction':float(np.mean(np.concatenate([a,b])<0)+np.mean(np.concatenate([a,b])>4095)),
            'homogeneous_voxels':int(homogeneous.sum()),'boundary_voxels':int(boundary.sum()),
            'homogeneous_local_variance_median':float(np.median(maps[:,:,3][homogeneous].astype(float)**2)) if homogeneous.any() else None,
            'expert_interface_gradient_mean':float(maps[:,:,1][boundary].mean()) if boundary.any() else None,
            'clipped_fraction':float(clip_fraction)}


def fit_scaler(train):
    train=np.asarray(train,float)
    if train.ndim!=2 or len(train)<2 or not np.isfinite(train).all():raise ValueError('At least two finite training samples required')
    mean=train.mean(0);scale=train.std(0);scale[scale<1e-12]=1.
    return {'mean':mean.tolist(),'scale':scale.tolist(),'train_count':len(train)}


def standardize(values,scaler):
    return (np.asarray(values,float)-scaler['mean'])/scaler['scale']


def knn_score(train,labels,heldout,truth,columns,k=15,groups=None):
    from scipy.spatial import cKDTree
    columns=list(columns);scaler=fit_scaler(train[:,columns])
    count=min(k,len(train))
    tree=cKDTree(standardize(train[:,columns],scaler))
    distances,indices=tree.query(standardize(heldout[:,columns],scaler),k=count,workers=1)
    if count==1:distances,indices=distances[:,None],indices[:,None]
    zero=distances==0
    weights=1/np.maximum(distances,1e-300)
    weights=np.where(zero.any(1)[:,None],zero.astype(float),weights)
    classes=sorted(np.unique(labels).tolist())
    votes=np.stack([(weights*(labels[indices]==c)).sum(1) for c in classes],axis=1)
    prediction=np.asarray(classes)[votes.argmax(1)]
    def metrics(actual,pred):
        confusion=np.array([[np.count_nonzero((actual==a)&(pred==b)) for b in classes] for a in classes])
        den=confusion.sum(0)+confusion.sum(1)
        f1=np.divide(2*np.diag(confusion),den,out=np.zeros(len(classes),float),where=den!=0)
        return {'macro_f1':float(f1.mean()),'confusion_matrix':confusion.tolist()}
    result={**metrics(truth,prediction),'classes':classes,
            'scaler':scaler,'feature_columns':columns,'features':[FEATURES[i] for i in columns],
            'k':int(count),'weights':'distance','implementation':'scipy.spatial.cKDTree; zero distances vote exclusively',
            'train_samples':len(train),'evaluation_samples':len(heldout)}
    if groups is not None:
        groups=np.asarray(groups)
        result['case_metrics']={str(g):{**metrics(truth[groups==g],prediction[groups==g]),
                                      'n':int(np.count_nonzero(groups==g))} for g in np.unique(groups)}
        scores=[v['macro_f1'] for v in result['case_metrics'].values()]
        result['equal_case_macro_f1_mean']=float(np.mean(scores));result['equal_case_macro_f1_std']=float(np.std(scores))
    return result


def projection(values,train,kind='PCA',seed=SEED):
    scaler=fit_scaler(values[train]);scaled=standardize(values,scaler)
    if min(np.count_nonzero(train),np.count_nonzero(~train))<2:raise ValueError('At least two train and heldout samples required')
    if kind=='PCA':
        center=scaled[train].mean(0)
        _,singular,components=np.linalg.svd(scaled[train]-center,full_matrices=False)
        components=components[:2].copy()
        for row in components:
            if row[np.abs(row).argmax()]<0:row*=-1
        embedding=(scaled-center)@components.T
        total=float(np.sum(singular**2))
        settings={'method':'PCA','implementation':'NumPy exact SVD','explained_variance_ratio':(singular[:2]**2/total if total else np.zeros(2)).tolist(),
                  'components':components.tolist(),'center':center.tolist()}
    elif kind=='UMAP':
        from umap import UMAP
        neighbors=min(15,int(train.sum())-1)
        if neighbors<2:raise ValueError('UMAP requires at least four training samples')
        reducer=UMAP(n_components=2,n_neighbors=neighbors,min_dist=.1,metric='euclidean',random_state=seed,
                     transform_seed=seed,n_jobs=1,low_memory=True,n_epochs=200)
        reducer.fit(scaled[train])
        embedding=np.empty((len(values),2),np.float32);embedding[train]=reducer.embedding_
        embedding[~train]=reducer.transform(scaled[~train])
        settings={'method':'UMAP','n_neighbors':neighbors,'min_dist':.1,'metric':'euclidean','seed':seed,'transform_seed':seed,'n_epochs':200,'n_jobs':1}
    else:raise ValueError('Unknown projection')
    return embedding,{'scaler':scaler,'settings':settings,'fit_scope':'training samples only; labels not supplied to reducer'}


def selected_voxel_ids(event,allowed_ids):
    allowed=set(map(int,allowed_ids));selected=[]
    for point in event.get('selection',{}).get('points',[]):
        custom=point.get('customdata',[])
        if len(custom)>=4:
            voxel=int(custom[3])
            if voxel in allowed:selected.append(voxel)
    return np.unique(selected).astype(np.int64)
