"""Reproducible held-out comparison mosaics and unsmoothed binary Heart surfaces."""
import argparse
import json
from pathlib import Path
import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from heart3d.ml.splits import load_frozen_split
from heart3d.mesh_quality import quality
from heart3d.surfaces import build_surface
from heart3d.pediatric import write_json
from heart3d.storage import sha256_file


def compare_case(row, root, predictions, output, selection='first patient ID per age stratum in frozen test; no performance-based selection'):
    pid=row['patient_id'];destination=output/pid
    if destination.exists():raise ValueError('QA output exists; do not overwrite')
    paths=[root/row['ct_relative_path'],root/row['mask_relative_path'],predictions/pid/'heart_prediction_original.nii.gz']
    images=[nib.load(p) for p in paths]
    if any(im.shape!=images[0].shape or not np.allclose(im.affine,images[0].affine,atol=1e-5) for im in images):
        raise ValueError('CT/GT/prediction geometry mismatch')
    if sha256_file(paths[1])!=row['review']['mask_SHA256']:raise ValueError('Original GT changed')
    if sha256_file(paths[0])!=row['review']['CT_SHA256']:raise ValueError('Original CT changed')
    ct=images[0].get_fdata(dtype=np.float32);gt=np.asanyarray(images[1].dataobj)>0;pred=np.asanyarray(images[2].dataobj)>0
    destination.mkdir(parents=True)
    positive=[np.flatnonzero(gt.any(axis=tuple(j for j in range(3) if j!=i))) for i in range(3)]
    center=[int((p[0]+p[-1])//2) for p in positive]
    specs=[('Axial center',2,center[2]),('Coronal center',1,center[1]),('Sagittal center',0,center[0]),
           ('Axial lower OAR',2,max(0,int(positive[2][0])-1)),('Axial upper OAR',2,min(gt.shape[2]-1,int(positive[2][-1])+1))]
    fig,axes=plt.subplots(len(specs),3,figsize=(12,17),layout='constrained')
    spacing=nib.affines.voxel_sizes(images[0].affine)
    for j,(name,axis,index) in enumerate(specs):
        planes=[np.take(v,index,axis=axis).T for v in (ct,gt,pred)]
        remaining=[i for i in range(3) if i!=axis];aspect=spacing[remaining[1]]/spacing[remaining[0]]
        for k,ax in enumerate(axes[j]):
            ax.imshow(planes[0],cmap='gray',vmin=-150,vmax=250,origin='lower',aspect=aspect)
            if k in (0,2) and planes[1].any():ax.contour(planes[1],levels=[.5],colors=['lime'],linewidths=.8)
            if k in (1,2) and planes[2].any():ax.contour(planes[2],levels=[.5],colors=['red'],linewidths=.8)
            ax.set_title(f'{name} {index}: '+['GT','Prediction','Comparison'][k]);ax.axis('off')
    fig.suptitle(f'{pid} | {row["age"]} y | {row["scanner"]}\nGT green / prediction red; native grid, no contour edits')
    fig.savefig(destination/'comparison.png',dpi=120);plt.close(fig)
    summaries={};meshes={}
    for name,mask in [('GT',gt),('Prediction',pred)]:
        mesh,summary=build_surface(mask.astype(np.uint8),1,images[0].affine)
        if mesh is None:
            summaries[name]={'empty':True};continue
        # Existing multiclass label 1 is LV. Override its metadata explicitly:
        # this surface is the binary source Heart OAR, never LV.
        summary.update(short_name='Heart',name='Heart OAR',color='#00ff00' if name=='GT' else '#ff0000')
        mesh.field_data['structure']=['Heart OAR'];mesh.field_data['coordinate_system']=['RAS'];mesh.field_data['unit']=['mm']
        mesh.save(destination/(name+'_Heart.vtp'))
        diagnostics=quality(mesh);summaries[name]={'surface':summary,'quality':diagnostics}
        meshes[name]=mesh
    distances=None
    if len(meshes)==2:
        a=np.asarray(meshes['GT'].points);b=np.asarray(meshes['Prediction'].points)
        ab=cKDTree(b).query(a)[0];ba=cKDTree(a).query(b)[0]
        distances={'mean_pooled_vertex_nearest_distance_mm':float((ab.sum()+ba.sum())/(len(ab)+len(ba))),
                   'HD95_vertex_nearest_mm':float(max(np.quantile(ab,.95),np.quantile(ba,.95))),
                   'definition':'sampled mesh vertex-to-vertex distances; distinct from voxel-boundary HD95/ASSD'}
    result={'patient_id':pid,'selection':selection,
            'sampling':specs,'meshes':summaries,'mesh_distances':distances,
            'physical_geometry_confirmed':row['geometry']['physical_geometry_confirmed_from_DICOM'],
            'sources':[{'path':str(p),'SHA256':sha256_file(p)} for p in paths],
            'smoothing':False,'component_removal':False,'source_annotation_modified':False}
    write_json(destination/'comparison.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split',type=Path,default=Path('configs/splits/pediatric_ct_heart_v1.json'))
    p.add_argument('--cohort',type=Path,default=Path('metadata/pediatric/heart_approved_cohort_v1.json'))
    p.add_argument('--patient',action='append',help='Explicit frozen-test patient for post-hoc failure review; no model tuning')
    a=p.parse_args();split=load_frozen_split(a.split,a.cohort)
    selected={}
    selection='first patient ID per age stratum in frozen test; no performance-based selection'
    if a.patient:
        available={r['patient_id']:r for r in split['partitions']['test']}
        if len(a.patient)!=len(set(a.patient)) or any(pid not in available for pid in a.patient):
            raise ValueError('Failure review requires unique patient IDs from frozen test')
        selected={pid:available[pid] for pid in a.patient}
        selection='explicit post-hoc frozen-test failure review; no retraining, threshold fitting or postprocessing'
    else:
        for row in sorted(split['partitions']['test'],key=lambda r:r['patient_id']):selected.setdefault(row['age_group'],row)
    results=[compare_case(row,a.data,a.predictions,a.output,selection) for row in selected.values()]
    write_json(a.output/'summary.json',results)
