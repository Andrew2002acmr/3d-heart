"""Strict BINARY DICOM SEG -> unchanged classic CT grid; no interpolation."""
import numpy as np
from pydicom.uid import SegmentationStorage, CTImageStorage
from .ct import require_vector


def functional(frame, shared, name):
    sequence = getattr(frame, name, None) or getattr(shared, name, None)
    if not sequence or len(sequence) != 1:
        raise ValueError(f'Missing/ambiguous SEG {name}')
    return sequence[0]


def source_references(group):
    return [source for derivation in getattr(group, 'DerivationImageSequence', [])
            for source in getattr(derivation, 'SourceImageSequence', [])]


def map_binary_seg(seg, geometry):
    if str(seg.SOPClassUID) != str(SegmentationStorage) or str(getattr(seg,'SegmentationType','')) != 'BINARY':
        raise ValueError('Only BINARY Segmentation Storage is supported')
    studies = {str(getattr(h, 'StudyInstanceUID', '') or '') for h in geometry.headers}
    if len(studies) != 1 or '' in studies or str(getattr(seg, 'StudyInstanceUID', '')) not in studies:
        raise ValueError('SEG study differs from CT or study identity missing')
    if str(getattr(seg,'FrameOfReferenceUID','')) != geometry.report['frame_of_reference_uid']:
        raise ValueError('SEG FrameOfReference mismatch')
    refs = getattr(seg,'ReferencedSeriesSequence',[])
    if len(refs) != 1 or str(refs[0].SeriesInstanceUID) != geometry.report['series_uid']:
        raise ValueError('SEG source series mismatch or ambiguous references')
    inventory = getattr(refs[0],'ReferencedInstanceSequence',[])
    if not inventory:
        raise ValueError('SEG lacks source SOP inventory')
    referenced_sops = [str(ref.ReferencedSOPInstanceUID) for ref in inventory]
    if len(set(referenced_sops)) != len(referenced_sops):
        raise ValueError('Duplicated SEG source SOP inventory')
    referenced_sops = set(referenced_sops)
    for ref in inventory:
        if (str(ref.ReferencedSOPClassUID) != str(CTImageStorage)
            or str(ref.ReferencedSOPInstanceUID) not in geometry.sop_to_slice):
            raise ValueError('SEG references absent/non-CT source instance')
    if (int(seg.Columns),int(seg.Rows)) != geometry.shape[:2]:
        raise ValueError('SEG dimensions differ; explicit resampling adapter required')
    shared_seq=getattr(seg,'SharedFunctionalGroupsSequence',[])
    if len(shared_seq)>1:raise ValueError('Ambiguous SEG shared functional groups')
    from pydicom.dataset import Dataset
    shared=shared_seq[0] if shared_seq else Dataset()
    definitions={int(s.SegmentNumber):s for s in seg.SegmentSequence}
    if len(definitions)!=len(seg.SegmentSequence):raise ValueError('Duplicate SEG segment numbers')
    frames=seg.PerFrameFunctionalGroupsSequence
    if len(frames)!=int(seg.NumberOfFrames):raise ValueError('SEG frame count mismatch')
    pixels=seg.pixel_array
    if pixels.ndim==2:pixels=pixels[None,...]
    if pixels.shape!=(len(frames),int(seg.Rows),int(seg.Columns)):
        raise ValueError('Decoded SEG frame shape mismatch')
    if not np.isin(pixels,[0,1]).all():raise ValueError('Nonbinary values in BINARY SEG')
    masks={number:np.zeros(geometry.shape,np.uint8) for number in definitions}
    inverse=np.linalg.inv(geometry.affine_lps)
    expected_iop=np.array(geometry.report['orientation_iop'])
    expected_spacing=np.array(geometry.report['spacing_xyz_mm'])[[1,0]]
    seen=set();residuals=[];per_frame_reference_count=0
    for index,frame in enumerate(frames):
        number=int(functional(frame,shared,'SegmentIdentificationSequence').ReferencedSegmentNumber)
        if number not in definitions:raise ValueError('Unknown SEG segment number')
        iop=require_vector(functional(frame,shared,'PlaneOrientationSequence').ImageOrientationPatient,6,'SEG IOP')
        spacing=require_vector(functional(frame,shared,'PixelMeasuresSequence').PixelSpacing,2,'SEG spacing')
        if not np.allclose(iop,expected_iop,atol=1e-5,rtol=0):raise ValueError('SEG orientation differs')
        if not np.allclose(spacing,expected_spacing,atol=1e-6,rtol=0):raise ValueError('SEG pixel spacing differs')
        position=require_vector(functional(frame,shared,'PlanePositionSequence').ImagePositionPatient,3,'SEG IPP')
        xyz=inverse[:3,:3]@position+inverse[:3,3]
        k=int(np.rint(xyz[2]))
        if not 0<=k<geometry.shape[2]:raise ValueError('SEG frame outside CT extent')
        if str(geometry.headers[k].SOPInstanceUID) not in referenced_sops:
            raise ValueError('SEG physical plane lacks global source SOP reference')
        expected_origin=geometry.affine_lps[:3,3]+k*geometry.affine_lps[:3,2]
        residual=float(np.linalg.norm(position-expected_origin))
        if residual>.05:raise ValueError('SEG frame origin/plane differs from CT grid')
        if (number,k) in seen:raise ValueError('Duplicated SEG segment/slice; no silent union')
        seen.add((number,k));residuals.append(residual)
        for source in source_references(frame):
            per_frame_reference_count+=1
            uid=str(source.ReferencedSOPInstanceUID)
            if (str(source.ReferencedSOPClassUID)!=str(CTImageStorage)
                or geometry.sop_to_slice.get(uid)!=k):
                raise ValueError('SEG per-frame SOP reference differs from physical plane')
        masks[number][:,:,k]=pixels[index].T
    for source in source_references(shared):
        if (str(source.ReferencedSOPClassUID)!=str(CTImageStorage)
            or str(source.ReferencedSOPInstanceUID) not in geometry.sop_to_slice):
            raise ValueError('SEG shared SOP reference absent/non-CT')
    reports=[]
    for number,mask in masks.items():
        definition=definitions[number]
        active=np.flatnonzero(mask.any(axis=(0,1)))
        reports.append({'segment_number':number,'source_label':str(definition.SegmentLabel),
            'source_algorithm_type':str(getattr(definition,'SegmentAlgorithmType','')),
            'coded_meanings':[str(e.value) for e in definition.iterall() if e.keyword=='CodeMeaning'],
            'voxels':int(mask.sum()),'volume_ml':float(mask.sum()*abs(np.linalg.det(geometry.affine_lps[:3,:3]))/1000),
            'nonempty_slice_range':[int(active[0]),int(active[-1])] if len(active) else None,
            'touches_grid_faces':[bool(mask.take(i,axis=a).any()) for a in range(3) for i in (0,-1)],
            'frame_count':sum(n==number for n,k in seen),
            'max_frame_origin_residual_mm':max(residuals,default=0),
            'per_frame_SOP_references_checked':per_frame_reference_count,
            'source_SOP_inventory_checked':True,'interpolated':False,
            'role':'imported_source_segmentation','expert_approval':'unknown',
            'ground_truth_certified':False,'clinical_status':'unknown'})
    return masks,reports
