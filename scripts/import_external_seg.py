"""Import external source SEG masks, preserving labels/variants and geometry."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
import pydicom
from heart3d.dicom.ct import inspect_headers,save_nifti
from heart3d.dicom.segmentation import map_binary_seg
from scripts.audit_external_ct import TAGS,sha256,safe_geometry


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--audit',required=True,type=Path)
    ap.add_argument('--reserve-gb',type=float,default=80)
    ap.add_argument('--mesh-series',action='append',default=[])
    args=ap.parse_args();root=args.audit.resolve()
    locator=json.loads((root/'source_locators.private.json').read_text(encoding='utf-8'))
    inventory=json.loads((root/'segmentation_inventory.json').read_text(encoding='utf-8'))
    source=Path(locator['input_root']).resolve()
    series_by_uid={(v['study_uid'],v['series_uid']):k for k,v in locator['series'].items()}
    geometries={};results=[]
    def geometry_for(alias):
        if alias not in geometries:
            item=locator['series'][alias];paths=[(source/p).resolve() for p in item['relative_paths']]
            if any(source not in p.parents for p in paths):raise ValueError('Source path escapes incoming')
            for p,digest in zip(paths,item['source_hashes']):
                if sha256(p)!=digest:raise ValueError('CT hash differs from inventory')
            headers=[pydicom.dcmread(p,stop_before_pixels=True,specific_tags=TAGS) for p in paths]
            geometries[alias]=inspect_headers(headers,paths,expected_series=item['series_uid'],require_demographic_metadata=False)
        return geometries[alias]
    for item in inventory:
        object_alias=item['object'];series_alias,object_name=object_alias.rsplit('/',1)
        index=int(object_name.split('_')[1])-1
        entry=locator['series'][series_alias]
        path=(source/entry['relative_paths'][index]).resolve()
        try:
            if source not in path.parents:raise ValueError('SEG source path escapes incoming')
            if sha256(path)!=entry['source_hashes'][index]:raise ValueError('SEG hash differs from inventory')
            seg=pydicom.dcmread(path)
            refs=getattr(seg,'ReferencedSeriesSequence',[])
            if len(refs)!=1:raise ValueError('SEG must identify one source series')
            source_alias=series_by_uid.get((str(seg.StudyInstanceUID),str(refs[0].SeriesInstanceUID)))
            if not source_alias:raise ValueError('SEG source series unavailable')
            geometry=geometry_for(source_alias)
            estimated=int(np.prod(geometry.shape))*max(1,len(seg.SegmentSequence))*3
            if shutil.disk_usage(root).free-estimated<args.reserve_gb*10**9:
                raise ValueError('Free-space reserve would be exceeded')
            masks,reports=map_binary_seg(seg,geometry)
            destination=root/'imported_seg'/object_alias;destination.mkdir(parents=True,exist_ok=True)
            for report in reports:
                number=report['segment_number'];mask=masks[number]
                mask_path=destination/f'segment_{number:03d}.nii.gz'
                save_nifti(mask,geometry.affine_ras,mask_path)
                report['mask_sha256']=sha256(mask_path)
                if source_alias in args.mesh_series and report['source_label'] in {'Heart','Left Ventricle'} and mask.any():
                    from heart3d.surfaces import build_surface
                    from heart3d.mesh_quality import quality
                    mesh,info=build_surface(mask,1,geometry.affine_ras)
                    # Binary SEG label 1 is NOT ImageCHD's LV label 1.
                    info.update({'name':report['source_label'],'short_name':f'segment_{number:03d}','color':'#38c9b9'})
                    mesh.field_data['source_label']=[report['source_label']]
                    mesh.field_data['coordinate_system']=['RAS'];mesh.field_data['unit']=['mm']
                    mesh.save(destination/f'segment_{number:03d}.vtp')
                    mesh.save(destination/f'segment_{number:03d}_mm.stl')
                    info['quality']=quality(mesh)
                    info['file_sha256']={p.name:sha256(p) for p in (
                        destination/f'segment_{number:03d}.vtp',
                        destination/f'segment_{number:03d}_mm.stl')}
                    info['physical_units']='mm';info['print_readiness']='not_expert_approved; self_intersections_not_checked'
                    report['mesh']=info
            result={'object':object_alias,'source_ct_series':source_alias,'status':'original_grid_import_passed',
                    'geometry':safe_geometry(geometry),'segments':reports,'source_sha256':entry['source_hashes'][index]}
            (destination/'import.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        except (ValueError,AttributeError,TypeError,RuntimeError) as exc:
            result={'object':object_alias,'status':'requires_review','error_type':type(exc).__name__,'reason':str(exc)}
        results.append(result)
        print(object_alias,result['status'],flush=True)
    (root/'seg_import_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'objects':len(results),'passed':sum(x['status']=='original_grid_import_passed' for x in results),'requires_review':sum(x['status']=='requires_review' for x in results)}))

if __name__=='__main__':main()
