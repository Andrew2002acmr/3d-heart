"""Reproducible exploratory measurements on real ct_1001, no global fit or training."""
import io
import json
from pathlib import Path
import ct_lab.launch
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from ct_lab.core import stats, normalize, filter_plane, fingerprint, LABELS, line_profile


def main():
    repo=Path(__file__).resolve().parents[1]
    root=repo.parent/'laboratories/data/chd68/pairs'
    path=root/'ct_1001_image.nii.gz'
    image=nib.load(path); x=image.get_fdata(dtype=np.float32)
    y=np.asanyarray(nib.load(root/'ct_1001_label.nii.gz').dataobj)
    z=x.shape[2]//2;p=x[:,:,z].T
    output=repo/'outputs/ct_preprocessing_lab/research';output.mkdir(parents=True,exist_ok=True)
    experiments={'Source':p,'Fixed train window':normalize(p,'Fixed-window'),
                 'Z-score (slice)':normalize(p,'Z-score'),'Percentile (slice)':normalize(p,'Percentile'),
                 'Gaussian sigma=1 + fixed window':normalize(ndi.gaussian_filter(p,1),'Fixed-window'),
                 'CLAHE + Min-Max':normalize(filter_plane(p,'CLAHE',1),'Min-Max')}
    multi=np.stack([normalize(p,'Fixed-window',low=a,high=b) for a,b in [(0,1000),(500,1500),(1000,2015)]])
    result={'case':'ct_1001','split':'train/development','source_sha256':fingerprint(path),
            'scope':f'source axial index {z} for experiments; full source volume for class overlap',
            'HU_verified':False,'physical_scale_verified':False,
            'experiment_sequence':'Each experimental output starts independently from raw source slice',
            'statistics':{k:stats(v) for k,v in experiments.items()},'multi_window_channels':[stats(c) for c in multi],
            'multi_window_bounds':[[0,1000],[500,1500],[1000,2015]],'randomness':'none; all measured voxels',
            'clipped_below_fraction':float(np.mean(p<0)),'clipped_above_fraction':float(np.mean(p>2015))}
    edges=np.linspace(float(x.min()),float(x.max()),129)
    hist={}
    for c in range(1,8):
        h,_=np.histogram(x[y==c],bins=edges);hist[c]=h/h.sum()
    result['overlap_coefficient_128bins_fullvolume']={f'{LABELS[a]} / {LABELS[b]}':float(np.minimum(hist[a],hist[b]).sum()) for a in hist for b in hist if a<b}
    context=[]
    for offset in range(-3,4):
        q=x[:,:,z+offset]
        context.append({'offset':offset,'distance_grid_units':offset*float(image.header.get_zooms()[2]),
                        'mean':float(q.mean(dtype=np.float64)),'mean_abs_difference_center':float(np.mean(np.abs(q-x[:,:,z]),dtype=np.float64))})
    result['neighbor_context']=context
    yy,xx=np.where(y[:,:,z].T==5)
    row=int(np.median(yy)) if yy.size else p.shape[0]//2
    result['line_profile']={k:v.tolist() for k,v in line_profile(p,experiments['Gaussian sigma=1 + fixed window'],y[:,:,z].T,
                                                              [row,0],[row,p.shape[1]-1],np.array([1.,1.])).items()}
    result['line_profile_endpoints_row_col']=[[row,0],[row,p.shape[1]-1]]
    (output/'measurements.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    fig,axes=plt.subplots(2,3,figsize=(12,8))
    for ax,(name,a) in zip(axes.flat,experiments.items()):
        ax.imshow(a,cmap='gray',origin='lower');ax.set_title(name);ax.axis('off')
    fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(output/f'experiments.{ext}',bbox_inches='tight')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4))
    for c,h in hist.items():ax.plot((edges[1:]+edges[:-1])/2,h,label=LABELS[c])
    ax.set_xlabel('Source intensity (HU unverified)');ax.set_ylabel('Probability / bin');ax.legend()
    for ext in ('png','svg'):fig.savefig(output/f'class_distributions.{ext}',bbox_inches='tight')
    plt.close(fig)
    fig=go.Figure()
    for name,a in experiments.items():
        h,e=np.histogram(a,bins=128);fig.add_trace(go.Scatter(x=(e[1:]+e[:-1])/2,y=h,name=name))
    (output/'experiments.html').write_text(fig.to_html(include_plotlyjs=True),encoding='utf-8')
    print('RESEARCH',result['clipped_above_fraction'],min(result['overlap_coefficient_128bins_fullvolume'].values()),max(result['overlap_coefficient_128bins_fullvolume'].values()),flush=True)


if __name__=='__main__':main()
