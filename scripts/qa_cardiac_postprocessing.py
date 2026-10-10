"""Show limited postprocessing changes and remaining chamber errors, validation only."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import nibabel as nib
from heart3d.ml.cardiac_data import read_json,write_json
from heart3d.storage import sha256_file,require_space
from scripts.qa_cardiac_predictions import overlay


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('data','baseline','experiment','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--cohort',type=Path,default=Path('metadata/pediatric/cardiac_chd68_cohort_v1.json'))
    a=p.parse_args();summary=read_json(a.experiment/'summary.json')
    if summary['partition']!='validation' or summary['test_evaluated']:raise ValueError('Validation only')
    if a.output.exists():raise ValueError('New QA output required')
    require_space(a.output,100_000_000,80_000_000_000);a.output.mkdir(parents=True)
    variant='combined';comparison=summary['comparisons'][variant]
    cohort={r['case_id']:r for r in read_json(a.cohort)['records']}
    if sha256_file(a.cohort)!=comparison['cohort_SHA256']:raise ValueError('Cohort mismatch')
    audits=read_json(a.experiment/'audit_progress.json');audits={r['case_id']:r for r in audits if r['variant']==variant}
    ranked=sorted(comparison['records'],key=lambda r:r['delta'])
    picks=[('Smallest macro delta',ranked[0]),('Largest macro delta',ranked[-1])]
    if picks[0][1]['case_id']==picks[1][1]['case_id']:picks=picks[:1]
    qa=[]
    for role,r in picks:
        case=r['case_id'];row=cohort[case]
        paths=[a.data/row['image_relative_path'],a.data/row['mask_relative_path'],a.baseline/case/'cardiac_prediction_original.nii.gz',a.experiment/variant/case/'cardiac_prediction_original.nii.gz']
        hashes=[row['image_SHA256'],row['mask_SHA256'],read_json(a.baseline/case/'provenance.json')['prediction_SHA256'],read_json(a.experiment/variant/case/'provenance.json')['prediction_SHA256']]
        if any(sha256_file(p)!=h for p,h in zip(paths,hashes)):raise ValueError('SHA mismatch')
        vols=[nib.load(p) for p in paths]
        if any(v.shape!=vols[0].shape or not np.allclose(v.affine,vols[0].affine,atol=1e-5,rtol=0) for v in vols):raise ValueError('Grid mismatch')
        # Load masks only; CT is read at selected slices for display.
        gt,before,after=[np.asanyarray(v.dataobj) for v in vols[1:]]
        changed=before!=after
        z=int(np.argmax(changed.sum(axis=(0,1))))
        xy=np.argwhere(changed[:,:,z]);center=xy[len(xy)//2] if len(xy) else np.array(vols[0].shape[:2])//2
        box=tuple(slice(max(0,int(v)-28),min(vols[0].shape[i],int(v)+29)) for i,v in enumerate(center))
        ct=np.asarray(vols[0].dataobj[:,:,z],dtype=np.float32)
        fig,axes=plt.subplots(2,5,figsize=(16,7))
        legend=[Patch(color='#ef476f',label='Correct class removed'),Patch(color='#20c997',label='Incorrect class removed'),Patch(color='#118ab2',label='Correct class added'),Patch(color='#ffd166',label='Incorrect class added')]
        diff=np.zeros((*ct.shape,4),np.float32)
        valid_gt=gt[:,:,z]<=7
        removed=changed[:,:,z]&(after[:,:,z]==0)&valid_gt;added=changed[:,:,z]&(before[:,:,z]==0)&valid_gt
        diff[removed&(before[:,:,z]==gt[:,:,z])]=[.94,.28,.44,.95]
        diff[removed&(before[:,:,z]!=gt[:,:,z])]=[.13,.79,.59,.95]
        diff[added&(after[:,:,z]==gt[:,:,z])]=[.07,.54,.7,.95]
        diff[added&(after[:,:,z]!=gt[:,:,z])]=[1,.82,.4,.95]
        for i,crop in enumerate((tuple(slice(None) for _ in range(2)),box)):
            for col,ax in enumerate(axes[i]):
                ax.imshow(ct[crop].T,cmap='gray',vmin=0,vmax=2015,origin='lower',interpolation='nearest')
                if col in (1,2,3):ax.imshow(overlay((gt,before,after)[col-1][crop+(z,)].T),origin='lower',interpolation='nearest')
                if col==4:ax.imshow(np.transpose(diff[crop],(1,0,2)),origin='lower',interpolation='nearest')
                ax.set_title(['CT','GT','v1 original','Combined candidate','Changes'][col]);ax.axis('off')
        fig.suptitle(f'{role}: {case}; macro delta {r["delta"]:+.6f}; axial index {z}\nFull slice and detail; exploratory candidate, not approved; physical scale unverified')
        fig.legend(handles=legend,loc='lower center',ncol=4,fontsize=9)
        fig.tight_layout(rect=(0,.06,1,.90));dst=a.output/f'{case}_postprocessing_changes.png';fig.savefig(dst,dpi=120);plt.close(fig)
        qa.append({'case_id':case,'role':role,'variant':variant,'axial_index':z,'display_crop':[{'start':s.start,'stop':s.stop} for s in box],
                   'delta':r['delta'],'image':dst.name,'image_SHA256':sha256_file(dst),'GT_used_only_for_metrics_and_QA':True})
    write_json(a.output/'QA_summary.json',{'partition':'validation','test_used':False,'cases':qa})


if __name__=='__main__':main()
