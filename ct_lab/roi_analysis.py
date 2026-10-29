"""Train geometry and real-data roundtrip resampling diagnostics; no test reads."""
import json
import nibabel as nib
import numpy as np
import torch
from scipy import ndimage as ndi
from ct_lab.roi_core import *


def prepare_case(case,role,plan,record):
    if case not in plan[role+'_cases']:raise ValueError('Case outside permitted split')
    path=OUT/'coarse'/f'{case}.npz'
    if path.exists():return
    source=DEFAULT_DATA/f'{case}_image.nii.gz';assert fingerprint(source)==record['image_SHA256']
    image=nib.load(source);volume=image.get_fdata(dtype=np.float32)
    if nib.aff2axcodes(image.affine)!=('R','A','S') or not np.allclose(image.affine[:3,:3],np.diag(np.diag(image.affine[:3,:3]))):raise ValueError('Native axis-aligned RAS required')
    x=torch.from_numpy(np.ascontiguousarray(np.clip(volume,0,2015)/2015*2-1))[None,None]
    low=torch.nn.functional.interpolate(x,size=plan['localizer']['shape'],mode='trilinear',align_corners=False)[0,0].numpy()
    data={'image':low.astype(np.float32),'shape_xyz':np.array(volume.shape),'affine':image.affine}
    info={'case':case,'role':role,'shape_xyz':list(volume.shape),'affine':image.affine.tolist(),'image_sha256':record['image_SHA256'],'GT_input':False}
    if role=='train':
        mask_path=DEFAULT_DATA/f'{case}_label.nii.gz';assert fingerprint(mask_path)==record['mask_SHA256']
        target=cardiac_target(np.asanyarray(nib.load(mask_path).dataobj));fg=(target>0)&(target<8);b=bbox(fg)
        binary=np.where(target==255,255,fg.astype(np.uint8)).astype(np.uint8)
        coarse=torch.nn.functional.interpolate(torch.from_numpy(np.ascontiguousarray(binary.astype(np.float32)))[None,None],size=plan['localizer']['shape'],mode='nearest-exact')[0,0].numpy().astype(np.uint8)
        data['target']=coarse
        center=np.array(ndi.center_of_mass(fg));roi=target[slices(b)];bounds=b/np.array(volume.shape)
        classes={str(c):bbox(target==c).tolist() if np.any(target==c) else None for c in range(1,8)}
        info.update({'GT_bbox':b.tolist(),'GT_center_mass':center.tolist(),'GT_bbox_center':b.mean(0).tolist(),'GT_bbox_size':(b[1]-b[0]).tolist(),
                     'normalized_bbox':bounds.tolist(),'normalized_center':(center/volume.shape).tolist(),'bbox_to_FOV':float(np.prod(b[1]-b[0])/volume.size),
                     'background_fraction_inside_bbox':float(np.mean(roi==0)),'ignored_fraction_inside_bbox':float(np.mean(roi==255)),
                     'foreground_voxels':int(fg.sum()),'foreground_edge_distances_indices':[b[0].tolist(),(np.array(volume.shape)-b[1]).tolist()],
                     'class_bboxes':classes,'mask_sha256':record['mask_SHA256']})
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**data);write(path.with_suffix('.json'),info)
    print('COARSE DATA',role,case,flush=True)


def roundtrip_diagnostics(case,info):
    source=nib.load(DEFAULT_DATA/f'{case}_image.nii.gz');x=source.get_fdata(dtype=np.float32)
    target=cardiac_target(np.asanyarray(nib.load(DEFAULT_DATA/f'{case}_label.nii.gz').dataobj));b=expand_box(np.array(info['GT_bbox']),source.shape,.2,8)
    z=int(round((np.array(info['GT_bbox'])[0,2]+np.array(info['GT_bbox'])[1,2]-1)/2));raw=x[:,:,z];gt=target[:,:,z];rows=[];arrays={'source':raw,'GT':gt,'z':np.array(z)}
    for mode,box in [('Full FOV',np.array([[0,0,0],source.shape])),('Oracle ROI',b)]:
        for size in (256,384):
            t=transform(source.shape,box,size);image=prepare_plane(raw,t);small=prepare_plane(gt,t,0,255).astype(np.uint8)
            # Geometry-only nearest-mask inverse, without classifier or GT-guided cleanup.
            rx,ry=t['resize']['resized_xy'];px,py=t['resize']['pad_xy'];crop=small[py:py+ry,px:px+rx]
            back=torch.nn.functional.interpolate(torch.from_numpy(crop.copy().astype(np.float32))[None,None],size=tuple(map(int,(box[1,:2]-box[0,:2])[::-1])),mode='nearest-exact')[0,0].numpy().T.astype(np.uint8)
            restored=np.zeros(gt.shape,np.uint8);restored[slice(box[0,0],box[1,0]),slice(box[0,1],box[1,1])]=back
            valid=gt<8;gradient=ndi.gaussian_gradient_magnitude(image.astype(float),1).mean()
            recovered_image=restore_plane(image[None].astype(np.float32),t).numpy()[0]
            reference_image=raw[slice(box[0,0],box[1,0]),slice(box[0,1],box[1,1])]
            native_g=np.hypot(*np.gradient(reference_image.astype(float)));recovered_g=np.hypot(*np.gradient(recovered_image.astype(float)))
            # A restored ignore voxel is an unclassified location, never relabeled as background.
            classareas={str(c):{'source_area':int((gt==c).sum()),'restored_area':int(((restored==c)&valid).sum()),
                               'roundtrip_Dice':float(2*((gt==c)&(restored==c)).sum()/max(1,(gt==c).sum()+((restored==c)&valid).sum()))} for c in range(1,8)}
            # Thin MYO geometry: sampled GT interior widths before/after resampling, not mm.
            width=2*ndi.distance_transform_edt(gt==5);thin=(gt==5)&(width<=4)
            rows.append({'case':case,'mode':mode,'size':size,'transform':t,'XY_scale':(np.array(t['resize']['resized_xy'])/(box[1,:2]-box[0,:2])).tolist(),
                         'mean_gradient_on_resized_grid':float(gradient),'roundtrip_fg_Dice_mean':float(np.mean([classareas[str(c)]['roundtrip_Dice'] for c in range(1,8) if (gt==c).any()])),
                         'source_crop_gradient_mean':float(native_g.mean()),'roundtrip_source_grid_gradient_mean':float(recovered_g.mean()),
                         'native_grid_gradient_preservation_ratio':float(recovered_g[1:-1,1:-1].mean()/max(1e-9,native_g[1:-1,1:-1].mean())),
                         'roundtrip_image_MAE_source_counts':float(np.abs(recovered_image-reference_image).mean()),
                         'valid_changed_pixels':int(((gt!=restored)&valid).sum()),'thin_MYO_recall_width_le4_index':float(((restored==5)&thin).sum()/max(1,thin.sum())),
                         'valid_voxels_restored_as_ignore':int(((restored==255)&valid).sum()),
                         'class_areas':classareas,'scope':'GT oracle/resampling diagnostic only; no learned model, no segmentation-quality inference'})
            key=mode.replace(' ','_')+'_'+str(size);arrays[key+'_image']=image;arrays[key+'_mask_restored']=restored;arrays[key+'_error']=((gt!=restored)&valid)
    np.savez_compressed(OUT/'resolution'/f'{case}.npz',**arrays);return rows


def run():
    plan,records=freeze()
    from concurrent.futures import ThreadPoolExecutor,as_completed
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1),ThreadPoolExecutor(max_workers=2) as executor:
        jobs=[executor.submit(prepare_case,c,role,plan,records[c]) for role in ('train','validation') for c in plan[role+'_cases']]
        for f in as_completed(jobs):f.result()
    stats=[json.loads((OUT/'coarse'/f'{c}.json').read_text()) for c in plan['train_cases']]
    write(OUT/'train_geometry.json',stats)
    # Fixed ROI is a train-fit function; each CV uses its own fold train subset.
    (OUT/'resolution').mkdir(exist_ok=True);rows=[]
    for case in plan['resize_diagnostic_cases']:rows.extend(roundtrip_diagnostics(case,next(r for r in stats if r['case']==case)))
    write(OUT/'resolution_roundtrip.json',rows);print('GEOMETRY COMPLETE',len(stats),len(rows),flush=True)


if __name__=='__main__':run()
