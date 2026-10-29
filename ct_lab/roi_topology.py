"""Common source-grid FG topology/HD95, with no component removal or GT cleanup."""
import gc
import json
import time
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from ct_lab.roi_core import *
from heart3d.ml.cardiac_data import confusion_counts,scores_from_confusion


def surface_hd95(pc,gt,valid,algorithm='auto'):
    """Identical symmetric Euclidean grid metric via two exact implementations."""
    if algorithm not in ('auto','KD','EDT'):raise ValueError('Unknown exact HD95 implementation')
    structure=ndi.generate_binary_structure(3,1);trusted=ndi.binary_erosion(valid,structure,border_value=0)
    ps=pc&~ndi.binary_erosion(pc,structure,border_value=0)&trusted;gs=gt&~ndi.binary_erosion(gt,structure,border_value=0)&trusted
    if not ps.any() or not gs.any():return None,None
    raster=algorithm=='EDT' or (algorithm=='auto' and pc.size>5_000_000 and int(ps.sum())+int(gs.sum())>2_000_000)
    if raster:
        d1=ndi.distance_transform_edt(~gs)[ps];d2=ndi.distance_transform_edt(~ps)[gs];name='exact raster EDT'
    else:
        pcoords=np.column_stack(np.nonzero(ps));gcoords=np.column_stack(np.nonzero(gs));d1=cKDTree(gcoords).query(pcoords,workers=1)[0];d2=cKDTree(pcoords).query(gcoords,workers=1)[0];name='exact cKDTree'
    return float(np.percentile(np.concatenate([d1,d2]),95)),name


def native_metrics(prediction,target):
    valid=target<8;cm=confusion_counts(prediction,target);summary=scores_from_confusion(cm);classes={};structure=ndi.generate_binary_structure(3,1)
    for c in range(1,8):
        p=(prediction==c)&valid;g=target==c;region=bbox(p|g)
        if region is None:
            classes[str(c)]={'components_6connected':0,'GT_components_6connected':0,'component_sizes_desc':[],'false_component_voxels':0,'false_component_fraction':0.,'largest_component_voxels':0,'false_components_no_GT_overlap':0,'HD95_index_units':None,'components':[]};continue
        region=expand_box(region,target.shape,0,1);sl=slices(region);pc=p[sl];gt=g[sl];vc=valid[sl]
        labels,n=ndi.label(pc,structure);sizes=np.bincount(labels.ravel())[1:];overlap=np.bincount(labels[gt],minlength=n+1)[1:];false=overlap==0
        bounds=ndi.find_objects(labels);components=[]
        for i,(size,area) in enumerate(zip(sizes,bounds)):
            if area is None:continue
            lo=np.array([s.start for s in area])+region[0];hi=np.array([s.stop for s in area])+region[0]
            components.append({'id':i+1,'voxels':int(size),'bbox_xyz':np.stack([lo,hi]).tolist(),'bbox_center_xyz':((lo+hi-1)/2).tolist(),
                               'GT_same_class_overlap_voxels':int(overlap[i]),'false_no_overlap':bool(false[i])})
        hd,algorithm=surface_hd95(pc,gt,vc)
        classes[str(c)]={'components_6connected':int(n),'GT_components_6connected':int(ndi.label(gt,structure)[1]),'component_sizes_desc':sorted(sizes.tolist(),reverse=True),
                         'false_components_no_GT_overlap':int(false.sum()),'false_component_voxels':int(sizes[false].sum()),
                         'false_component_fraction':float(sizes[false].sum()/max(1,sizes.sum())),'largest_component_voxels':int(sizes.max()) if n else 0,
                         'prediction_voxels':int(p.sum()),'GT_voxels':int(g.sum()),'false_positive_voxels':int((p&~g).sum()),'HD95_index_units':hd,'HD95_algorithm':algorithm,
                         'components':components}
    return {'confusion_matrix':cm.tolist(),**summary,'topology':classes,'BG_to_foreground_voxels':int(cm[0,1:].sum()),'valid_voxels':int(valid.sum()),
            'HD95_units':'index grid, not millimeters; source/ignore-adjacent surfaces censored','connectivity':6,'components_removed':False}


def export_meshes(case,method,prediction):
    directory=OUT/'surfaces'/case/method;receipt=directory/'index.json'
    if receipt.exists():return
    from heart3d.surfaces import build_surface
    source=nib.load(DEFAULT_DATA/f'{case}_image.nii.gz');rows=[];directory.mkdir(parents=True,exist_ok=True)
    for c in range(1,8):
        mesh,info=build_surface(prediction,c,source.affine)
        if mesh is None:continue
        mesh.field_data['coordinate_system']=['RAS+ header coordinates'];mesh.field_data['unit']=['unknown, NOT millimeters'];mesh.save(directory/f'{c}_{LABELS[c]}.vtp')
        visual=mesh.decimate(max(0.,1-30000/mesh.n_cells)) if mesh.n_cells>30000 else mesh
        np.savez_compressed(directory/f'{c}_display.npz',vertices=visual.points.astype(np.float32),faces=visual.faces.reshape(-1,4)[:,1:].astype(np.int32))
        rows.append({'class':c,'raw_surface':f'{c}_{LABELS[c]}.vtp','display_npz':f'{c}_display.npz','raw_points':mesh.n_points,'raw_triangles':mesh.n_cells,
                     'display_triangles':visual.n_cells,'display_only_decimation':mesh.n_cells>30000,'metrics_use_undecimated_voxel_mask':True,'Heart3D_metadata':info})
    write(receipt,rows);print('HEART3D SURFACES',case,method,flush=True)


def run():
    plan,_=freeze();completed={};mesh_methods={'CHD68_v1','CHD68_v2','B_full_256','C_oracle_256','D_predicted_256'}
    representatives={plan['evaluation_cases'][0],plan['validation_cases'][0]}
    for case in representatives:
        gt=cardiac_target(np.asanyarray(nib.load(DEFAULT_DATA/f'{case}_label.nii.gz').dataobj));export_meshes(case,'GT',gt)
    from concurrent.futures import ThreadPoolExecutor,as_completed
    workers=max(1,min(2,int(os.environ.get('ROI05_TOPOLOGY_WORKERS','2'))))
    if workers>1:
        try:
            import psutil
            if psutil.virtual_memory().available<4*1024**3:workers=1
        except ImportError:workers=1
    print('TOPOLOGY WORKERS',workers,'; VTK serial; adaptive exact KD/EDT; no metric changes',flush=True)
    def measure(scorefile):
        prediction=np.load(scorefile.with_name('prediction.npy'),mmap_mode='r');case=scorefile.parent.name
        target=cardiac_target(np.asanyarray(nib.load(DEFAULT_DATA/f'{case}_label.nii.gz').dataobj));metric=native_metrics(prediction,target)
        metric.update({'case':case,'method':scorefile.parent.parent.name,'oracle_GT_ROI':scorefile.parent.parent.name.startswith('C_oracle')})
        write(scorefile.with_name('topology.json'),metric);return scorefile,metric
    while True:
        paths=list((OUT/'evaluation'/'all_train').glob('*/*/scores.json'))
        pending=[]
        for scorefile in paths:
            method=scorefile.parent.parent.name;case=scorefile.parent.name;key=method+'|'+case;path=scorefile.with_name('topology.json')
            if path.exists():
                completed[key]=json.loads(path.read_text())
                if case in representatives and method in mesh_methods and not (OUT/'surfaces'/case/method/'index.json').exists():export_meshes(case,method,np.load(scorefile.with_name('prediction.npy'),mmap_mode='r'))
                continue
            if case not in plan['evaluation_cases']:raise ValueError('Unexpected case in dense evaluation')
            pred_path=scorefile.with_name('prediction.npy')
            if not pred_path.exists():continue
            pending.append(scorefile)
        # At most two native masks in metric computation; meshes remain serial (VTK).
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures=[executor.submit(measure,p) for p in pending]
            for future in as_completed(futures):
                scorefile,metric=future.result();case=metric['case'];method=metric['method'];completed[method+'|'+case]=metric
                if case in representatives and method in mesh_methods:export_meshes(case,method,np.load(scorefile.with_name('prediction.npy'),mmap_mode='r'))
                print('TOPOLOGY',case,method,metric['macro_foreground_Dice'],flush=True);gc.collect()
        expected=(len(plan['main_jobs'])+2)*len(plan['evaluation_cases'])
        if (OUT/'segmentation_results.json').exists() and (OUT/'baseline_results.json').exists() and len(completed)>=expected:break
        time.sleep(10)
    write(OUT/'topology_results.json',list(completed.values()));print('ALL TOPOLOGY COMPLETE',len(completed),flush=True)


if __name__=='__main__':run()
