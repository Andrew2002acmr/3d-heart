"""Representative original-grid overlays; no ground truth or accuracy claim."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
def render(ct_path,mask_path,output):
    ct=nib.load(ct_path);mask=nib.load(mask_path)
    if ct.shape!=mask.shape or not np.allclose(ct.affine,mask.affine,atol=1e-5):raise ValueError("Grid mismatch")
    image=ct.get_fdata(dtype=np.float32);seg=np.asanyarray(mask.dataobj)
    fig,axes=plt.subplots(3,2,figsize=(9,12),facecolor="#17212b")
    for row,axis in enumerate([2,1,0]):
        counts=seg.sum(axis=tuple(i for i in range(3) if i!=axis))
        center=int(np.argmax(counts)) if counts.any() else seg.shape[axis]//2
        selection=[slice(None)]*3;selection[axis]=center
        plane=np.flipud(image[tuple(selection)].T);overlay=np.flipud(seg[tuple(selection)].T)
        other=[i for i in range(3) if i!=axis];spacing=nib.affines.voxel_sizes(ct.affine)
        extent=[0,ct.shape[other[0]]*spacing[other[0]],0,ct.shape[other[1]]*spacing[other[1]]]
        for col in (0,1):
            ax=axes[row,col];ax.imshow(plane,cmap="gray",vmin=-200,vmax=700,extent=extent)
            if col:ax.imshow(np.ma.masked_where(overlay==0,overlay),cmap="autumn",alpha=.45,vmin=0,vmax=1,extent=extent,interpolation="nearest")
            ax.set_title(f"{['Sagittal','Coronal','Axial'][axis]} index {center}: "+("draft Heart" if col else "CT"),color="white")
            ax.set_axis_off()
    fig.suptitle("Clinical pilot: model draft, no verified reference mask",color="white")
    fig.tight_layout();Path(output).parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=130);plt.close(fig)
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ["ct","mask","output"]:p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args();render(a.ct,a.mask,a.output)
