"""Known voxel landmarks and strict source/plane failure cases for BINARY SEG."""
import copy
import numpy as np
import pytest
from pydicom.dataset import Dataset,FileDataset,FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import CTImageStorage,SegmentationStorage,ExplicitVRLittleEndian
from pydicom.pixels.utils import pack_bits
from heart3d.dicom.ct import inspect_headers
from heart3d.dicom.segmentation import map_binary_seg


def fixture():
    hs=[]
    for k in range(3):
        h=Dataset();h.Modality='CT';h.SOPClassUID=CTImageStorage;h.StudyInstanceUID='1.2.10'
        h.SeriesInstanceUID='1.2.3';h.FrameOfReferenceUID='1.2.4';h.SOPInstanceUID=f'1.2.3.{k+1}'
        h.Rows=4;h.Columns=5;h.PixelSpacing=[2,1]
        h.ImageOrientationPatient=[1,0,0,0,1,0];h.ImagePositionPatient=[10,20,30+k*3]
        h.RescaleSlope=1;h.RescaleIntercept=-1000;h.SliceThickness=7
        hs.append(h)
    geometry=inspect_headers(hs,require_demographic_metadata=False)
    meta=FileMetaDataset();meta.TransferSyntaxUID=ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID=SegmentationStorage;meta.MediaStorageSOPInstanceUID='1.2.9'
    seg=FileDataset(None,{},file_meta=meta,preamble=b'\0'*128)
    seg.SOPClassUID=SegmentationStorage;seg.SOPInstanceUID='1.2.9';seg.Modality='SEG'
    seg.StudyInstanceUID='1.2.10';seg.FrameOfReferenceUID='1.2.4';seg.SegmentationType='BINARY';seg.Rows=4;seg.Columns=5;seg.NumberOfFrames=3
    seg.SamplesPerPixel=1;seg.PhotometricInterpretation='MONOCHROME2'
    seg.BitsAllocated=1;seg.BitsStored=1;seg.HighBit=0;seg.PixelRepresentation=0
    definition=Dataset();definition.SegmentNumber=1;definition.SegmentLabel='Heart';definition.SegmentAlgorithmType='SEMIAUTOMATIC'
    seg.SegmentSequence=Sequence([definition])
    def ref(uid):
        d=Dataset();d.ReferencedSOPClassUID=CTImageStorage;d.ReferencedSOPInstanceUID=uid;return d
    series=Dataset();series.SeriesInstanceUID='1.2.3';series.ReferencedInstanceSequence=Sequence([ref(h.SOPInstanceUID) for h in hs])
    seg.ReferencedSeriesSequence=Sequence([series]);seg.SharedFunctionalGroupsSequence=Sequence([Dataset()])
    seg.PerFrameFunctionalGroupsSequence=Sequence();planes=[];expected=np.zeros(geometry.shape,np.uint8)
    for k in [2,0,1]:
        f=Dataset()
        pos=Dataset();pos.ImagePositionPatient=hs[k].ImagePositionPatient;f.PlanePositionSequence=Sequence([pos])
        orient=Dataset();orient.ImageOrientationPatient=hs[k].ImageOrientationPatient;f.PlaneOrientationSequence=Sequence([orient])
        measures=Dataset();measures.PixelSpacing=[2,1];f.PixelMeasuresSequence=Sequence([measures])
        ident=Dataset();ident.ReferencedSegmentNumber=1;f.SegmentIdentificationSequence=Sequence([ident])
        deriv=Dataset();deriv.SourceImageSequence=Sequence([ref(hs[k].SOPInstanceUID)]);f.DerivationImageSequence=Sequence([deriv])
        seg.PerFrameFunctionalGroupsSequence.append(f)
        plane=np.zeros((4,5),np.uint8);plane[2,3]=1;planes.append(plane);expected[3,2,k]=1
    seg.PixelData=pack_bits(np.asarray(planes))
    return seg,geometry,expected


def test_seg_frame_order_and_known_voxel_landmark():
    seg,g,expected=fixture();masks,reports=map_binary_seg(seg,g)
    np.testing.assert_array_equal(masks[1],expected)
    assert reports[0]['max_frame_origin_residual_mm']==0
    assert reports[0]['volume_ml']==pytest.approx(.018)
    assert reports[0]['per_frame_SOP_references_checked']==3
    assert not reports[0]['ground_truth_certified'] and not reports[0]['interpolated']


@pytest.mark.parametrize('failure',['study','frame','series','global_sop','per_frame_sop','origin','spacing','orientation','duplicate','unknown_segment'])
def test_seg_geometry_failures_are_not_fixed(failure):
    seg,g,_=fixture();frame=seg.PerFrameFunctionalGroupsSequence[0]
    if failure=='study':seg.StudyInstanceUID='1.2.99'
    elif failure=='frame':seg.FrameOfReferenceUID='1.2.99'
    elif failure=='series':seg.ReferencedSeriesSequence[0].SeriesInstanceUID='1.2.99'
    elif failure=='global_sop':seg.ReferencedSeriesSequence[0].ReferencedInstanceSequence[0].ReferencedSOPInstanceUID='1.2.99'
    elif failure=='per_frame_sop':frame.DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID='1.2.3.1'
    elif failure=='origin':frame.PlanePositionSequence[0].ImagePositionPatient[0]+=.2
    elif failure=='spacing':frame.PixelMeasuresSequence[0].PixelSpacing=[2,2]
    elif failure=='orientation':frame.PlaneOrientationSequence[0].ImageOrientationPatient=[0,1,0,1,0,0]
    elif failure=='duplicate':seg.PerFrameFunctionalGroupsSequence[1]=copy.deepcopy(frame)
    elif failure=='unknown_segment':frame.SegmentIdentificationSequence[0].ReferencedSegmentNumber=2
    with pytest.raises(ValueError):map_binary_seg(seg,g)


def test_global_source_inventory_and_geometry_without_per_frame_references():
    seg,g,expected=fixture()
    for frame in seg.PerFrameFunctionalGroupsSequence:del frame.DerivationImageSequence
    masks,report=map_binary_seg(seg,g)
    np.testing.assert_array_equal(masks[1],expected)
    assert report[0]['per_frame_SOP_references_checked']==0
    assert report[0]['source_SOP_inventory_checked']


def test_every_physical_frame_requires_a_global_reference():
    seg,g,_=fixture()
    for frame in seg.PerFrameFunctionalGroupsSequence:
        del frame.DerivationImageSequence
    seg.ReferencedSeriesSequence[0].ReferencedInstanceSequence.pop()
    with pytest.raises(ValueError,match='physical plane lacks'):
        map_binary_seg(seg,g)
