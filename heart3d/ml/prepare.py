"""Prepare only the selected partition, with train-fitted parameters and provenance."""
import argparse
import json
from pathlib import Path
import nibabel as nib
import numpy as np
from heart3d.ml.geometry import make_transform,resample_xyz
from heart3d.ml.splits import load_frozen_split
from heart3d.pediatric import write_json
from heart3d.storage import sha256_file,require_space


def prepare_partition(config_path, partition='train', data_root=None):
    config=json.loads(Path(config_path).read_text())
    root=Path(data_root or config['data_root'])
    split_path=Path(config['split']);cohort_path=Path(config['cohort'])
    split=load_frozen_split(split_path,cohort_path)
    preproc_path=Path(config['preprocessing']);pre=json.loads(preproc_path.read_text())
    if pre['fitted_partition']!='train' or pre['split_SHA256']!=sha256_file(split_path):
        raise ValueError('Preprocessing was not fitted to this frozen training partition')
    cache=root/config['cache_relative_path'];cache.mkdir(parents=True,exist_ok=True)
    records=[]
    for row in split['partitions'][partition]:
        destination=cache/row['patient_id'];provenance_path=destination/'provenance.json'
        if provenance_path.exists():
            provenance=json.loads(provenance_path.read_text())
            if provenance['preprocessing_SHA256']!=sha256_file(preproc_path) or provenance['split_SHA256']!=sha256_file(split_path):
                raise ValueError('Existing cache belongs to another experiment')
            for name,key in [('image.npy','image_SHA256'),('mask.npy','mask_SHA256')]:
                if sha256_file(destination/name)!=provenance[key]:raise ValueError('Prepared cache integrity failure')
        else:
            ct_path=root/row['ct_relative_path'];mask_path=root/row['mask_relative_path']
            if sha256_file(ct_path)!=row['review']['CT_SHA256'] or sha256_file(mask_path)!=row['review']['mask_SHA256']:
                raise ValueError('Reviewed original data changed')
            ct=nib.load(ct_path);mask=nib.load(mask_path)
            if ct.shape!=mask.shape or not np.allclose(ct.affine,mask.affine,atol=1e-5):raise ValueError('CT/GT geometry mismatch')
            if 'orientation_iop' in pre and not np.allclose(row['geometry']['orientation_iop'],pre['orientation_iop'],atol=1e-5):
                raise ValueError('Native orientation differs from train-fitted baseline convention; explicit reorientation required')
            transform=make_transform(ct.shape,ct.affine,pre['input_size'],pre['z_spacing_mm'])
            needed=len(transform['z_positions_mm'])*pre['input_size']**2*5
            require_space(root,needed,int(config['reserve_GB']*1e9))
            destination.mkdir(parents=True,exist_ok=True)
            if (destination/'image.npy').exists() or (destination/'mask.npy').exists():
                raise ValueError('Incomplete cache exists; use a new cache path, do not silently overwrite')
            image=resample_xyz(np.asanyarray(ct.dataobj),transform,1,pre['HU_clip'][0])
            low,high=pre['HU_clip'];np.clip(image,low,high,out=image)
            image-=low;image*=2/(high-low);image-=1
            target=resample_xyz(np.asanyarray(mask.dataobj),transform,0)
            if not np.isfinite(image).all() or not set(np.unique(target)).issubset({0,1}):raise ValueError('Invalid prepared tensors')
            np.save(destination/'image.npy',image);np.save(destination/'mask.npy',target)
            positive=np.flatnonzero(target.any(axis=(1,2))).tolist()
            if not positive:raise ValueError('Resampling removed the entire Heart target; review rather than train an empty case')
            provenance={'patient_id':row['patient_id'],'partition':partition,'source_series_uid':row['ct_series_uid'],
                'split_SHA256':sha256_file(split_path),'preprocessing_SHA256':sha256_file(preproc_path),
                'source_CT_SHA256':row['review']['CT_SHA256'],'source_GT_SHA256':row['review']['mask_SHA256'],
                'image_SHA256':sha256_file(destination/'image.npy'),'mask_SHA256':sha256_file(destination/'mask.npy'),
                'transform':transform,'positive_indices':positive,'negative_indices':[i for i in range(len(target)) if i not in set(positive)],
                'original_GT_modified':False,'image_interpolation':'linear','mask_interpolation':'nearest-neighbor'}
            write_json(provenance_path,provenance)
            del image,target
        records.append({'patient_id':row['patient_id'],'cache_relative_path':str(destination.relative_to(root)).replace('\\','/'),
            'provenance_SHA256':sha256_file(provenance_path),'positive_slices':len(provenance['positive_indices']),
            'negative_slices':len(provenance['negative_indices']),'total_slices':len(provenance['transform']['z_positions_mm'])})
        print('PREPARED',partition,row['patient_id'],flush=True)
    result={'partition':partition,'split_SHA256':sha256_file(split_path),'preprocessing_SHA256':sha256_file(preproc_path),'records':records}
    write_json(cache/f'{partition}_index.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--data',type=Path);p.add_argument('--partition',choices=['train','validation','test'],default='train')
    a=p.parse_args();prepare_partition(a.config,a.partition,a.data)
