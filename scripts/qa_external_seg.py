"""Visual technical QA of imported source masks; no anatomy inference."""
import argparse,json
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--audit',required=True,type=Path)
    ap.add_argument('--ct-series',required=True)
    ap.add_argument('--objects',required=True,nargs='+')
    args=ap.parse_args();root=args.audit
    image=nib.load(root/'prepared'/args.ct_series/'ct.nii.gz')
    ct=image.get_fdata(dtype=np.float32);spacing=nib.affines.voxel_sizes(image.affine)
    masks=[];names=[]
    for obj in args.objects:
        path=root/'imported_seg'/obj
        info=json.loads((path/'import.json').read_text(encoding='utf-8'))
        assert info['source_ct_series']==args.ct_series
        number=info['segments'][0]['segment_number']
        mi=nib.load(path/f'segment_{number:03d}.nii.gz')
        if mi.shape!=image.shape or not np.allclose(mi.affine,image.affine,atol=1e-4,rtol=0):
            raise ValueError('CT/SEG original grids differ')
        masks.append(np.asarray(mi.dataobj))
        names.append(info['segments'][0]['source_label']+'\n'+obj.rsplit('/',1)[1])
    fig,axes=plt.subplots(3,len(masks)+1,figsize=(3.8*(len(masks)+1),10),constrained_layout=True)
    for row,(axis,name) in enumerate([(2,'native axial'),(1,'native coronal'),(0,'native sagittal')]):
        k=ct.shape[axis]//2;plane=np.take(ct,k,axis=axis).T
        remaining=[a for a in range(3) if a!=axis]
        aspect=spacing[remaining[1]]/spacing[remaining[0]]
        for col,ax in enumerate(axes[row]):
            ax.imshow(plane,cmap='gray',vmin=-200,vmax=600,origin='lower',aspect=aspect)
            if col:
                roi=np.take(masks[col-1],k,axis=axis).T
                overlay=np.zeros((*roi.shape,4));overlay[roi>0]=[.1,1.,.4,.35]
                ax.imshow(overlay,origin='lower',aspect=aspect)
            ax.set_title(f'{name}, {k}\n'+(names[col-1] if col else 'CT only'))
            ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle(args.ct_series+' | Imported source SEG, NOT approved ground truth\nImage-only center sampling; native grid; same physical aspect')
    destination=root/'visual_qa';destination.mkdir(exist_ok=True)
    fig.savefig(destination/(args.ct_series.replace('/','_')+'_source_masks.png'),dpi=95)
    plt.close(fig)

if __name__=='__main__':main()
