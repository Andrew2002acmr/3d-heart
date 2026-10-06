"""Streaming train-only statistics on original CT/GT; no held-out data is read."""
import argparse
from collections import Counter
import json
from pathlib import Path
import nibabel as nib
import numpy as np
from heart3d.ml.splits import load_frozen_split
from heart3d.pediatric import write_json
from heart3d.storage import sha256_file


def histogram_quantiles(hist, edges, quantiles):
    total=hist.sum()
    if not total:raise ValueError('Empty histogram')
    cumulative=hist.cumsum()
    return {str(q):float(edges[min(np.searchsorted(cumulative,q*total),len(hist)-1)]) for q in quantiles}


def analyse(root, split_path, cohort_path, output):
    split=load_frozen_split(split_path,cohort_path)
    edges=np.arange(-4096,8194,dtype=float)
    hist={key:np.zeros(len(edges)-1,dtype=np.int64) for key in ('CT','body','Heart')}
    summaries=[]
    for row in split['partitions']['train']:
        for relative,key in [('ct_relative_path','CT_SHA256'),('mask_relative_path','mask_SHA256')]:
            if sha256_file(root/row[relative])!=row['review'][key]:
                raise ValueError('Original reviewed train data changed')
        ct=nib.load(root/row['ct_relative_path']); gt=nib.load(root/row['mask_relative_path'])
        if ct.shape!=gt.shape or not np.allclose(ct.affine,gt.affine,atol=1e-5):
            raise ValueError('Original CT/GT geometry differs')
        # Bounded sampling, patient by patient; original full arrays never accumulated.
        sample=np.asanyarray(ct.dataobj)[::4,::4,::2].copy()
        labels=np.asanyarray(gt.dataobj)[::4,::4,::2].copy()
        selections={'CT':sample.ravel(),'body':sample[sample>-500], 'Heart':sample[labels>0]}
        stats={}
        for name,values in selections.items():
            hist[name]+=np.histogram(np.clip(values,edges[0],edges[-1]-.001),edges)[0]
            stats[name]={'samples':len(values),'min':float(values.min()),'max':float(values.max()),
                         'quantiles':np.quantile(values,[.005,.01,.5,.99,.995]).tolist()}
        geometry=row['geometry']; heart=row['Heart']; first,last=heart['contoured_slice_range']
        summaries.append({'patient_id':row['patient_id'],'age_group':row['age_group'],
            'scanner':row['scanner'],'contrast_status':row['contrast_status'],
            'shape_xyz':list(ct.shape),'spacing_xyz_mm':geometry['spacing_xyz_mm'],
            'orientation_iop':geometry['orientation_iop'],
            'FOV_xyz_mm':(np.array(ct.shape)*geometry['spacing_xyz_mm']).tolist(),
            'Heart_volume_ml':heart['volume_ml'],
            'Heart_voxel_fraction':heart['heart_voxels']/np.prod(ct.shape),
            'positive_slices':last-first+1,'negative_slices':ct.shape[2]-(last-first+1),
            'Heart_z_extent_mm':(last-first+1)*geometry['spacing_xyz_mm'][2],
            'intensity_sample':stats})
        print('TRAIN STATISTICS',row['patient_id'],flush=True)
    quantiles={name:histogram_quantiles(h,edges,[.005,.01,.5,.99,.995]) for name,h in hist.items()}
    result={'version':1,'partition':'train','split_SHA256':sha256_file(split_path),
        'cohort_SHA256':sha256_file(cohort_path),'patient_ids':[r['patient_id'] for r in summaries],
        'sampling':'fixed xyz stride [4,4,2], voxel-weighted aggregate; no validation/test volume opened',
        'body_definition_for_statistics':'image HU > -500, not a crop',
        'aggregate_HU_quantiles':quantiles,'histogram_range_HU':[-4096,8193],
        'histogram_bin_width_HU':1,'age_counts':dict(Counter(r['age_group'] for r in summaries)),
        'scanner_counts':dict(Counter(r['scanner'] for r in summaries)),
        'orientation_counts':dict(Counter(str(r['orientation_iop']) for r in summaries)),
        'original_total_slices':sum(r['shape_xyz'][2] for r in summaries),
        'positive_slices':sum(r['positive_slices'] for r in summaries),
        'negative_slices':sum(r['negative_slices'] for r in summaries),'records':summaries}
    write_json(output,result)
    print('TRAIN ANALYSIS',len(summaries),'HU',quantiles,flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','split','cohort','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();analyse(a.data,a.split,a.cohort,a.out)
