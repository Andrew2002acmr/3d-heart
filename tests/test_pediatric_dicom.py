"""Independent landmarks and failure cases for original-grid DICOM conversion."""
import io
import copy
import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import CTImageStorage, RTStructureSetStorage, ExplicitVRLittleEndian, ImplicitVRLittleEndian

from heart3d.dicom.ct import inspect_headers, load_volume, save_nifti
from heart3d.dicom.rtstruct import rasterize_heart
from heart3d.dicom.tcia import BoundedStream, extract_heart_stream


def ct_headers():
    headers = []
    # Oblique plane, unequal row/column spacing; thickness intentionally != step.
    col = np.array([1., 0, 0]); row = np.array([0, .8, .6])
    normal = np.array([0, -.6, .8])
    for k in (2, 0, 1, 3):
        h = Dataset()
        h.Modality = 'CT'; h.SOPClassUID = CTImageStorage
        h.SOPInstanceUID = f'1.2.3.{k+1}'; h.SeriesInstanceUID = '1.2.4'
        h.FrameOfReferenceUID = '1.2.5'; h.PatientID = 'synthetic'; h.PatientAge = '008Y'
        h.Rows = 9; h.Columns = 10; h.PixelSpacing = [2, 1]
        h.ImageOrientationPatient = [*col, *row]
        h.ImagePositionPatient = (np.array([10, 20, 30]) + k*3*normal).tolist()
        h.RescaleSlope = k+1; h.RescaleIntercept = -1000; h.SliceThickness = 7
        h.InstanceNumber = 100-k
        headers.append(h)
    return headers


def rt_for(geometry, polygons=None, kind='CLOSED_PLANAR'):
    rt = FileDataset(None, {}, file_meta=FileMetaDataset(), preamble=b'\0'*128)
    rt.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    rt.file_meta.MediaStorageSOPClassUID = RTStructureSetStorage
    rt.file_meta.MediaStorageSOPInstanceUID = '1.2.9'
    rt.SOPClassUID = RTStructureSetStorage; rt.SOPInstanceUID = '1.2.9'
    rt.Modality = 'RTSTRUCT'; rt.SeriesInstanceUID = '1.2.8'; rt.PatientID = 'synthetic'
    roi = Dataset(); roi.ROINumber = 1; roi.ROIName = 'Heart'; roi.ReferencedFrameOfReferenceUID = '1.2.5'
    rt.StructureSetROISequence = Sequence([roi])
    def sop_ref(uid):
        ref = Dataset(); ref.ReferencedSOPClassUID = CTImageStorage; ref.ReferencedSOPInstanceUID = uid
        return ref
    series = Dataset(); series.SeriesInstanceUID = '1.2.4'
    series.ContourImageSequence = Sequence([sop_ref(u) for u in geometry.sop_to_slice])
    study = Dataset(); study.RTReferencedSeriesSequence = Sequence([series])
    frame = Dataset(); frame.FrameOfReferenceUID = '1.2.5'; frame.RTReferencedStudySequence = Sequence([study])
    rt.ReferencedFrameOfReferenceSequence = Sequence([frame])
    item = Dataset(); item.ReferencedROINumber = 1; item.ContourSequence = Sequence()
    if polygons is None:
        polygons = [(1, [(2.25, 2.25), (6.75, 2.25), (6.75, 5.75), (2.25, 5.75)])]
    for k, xy in polygons:
        points = np.array([[x, y, k, 1] for x, y in xy]) @ geometry.affine_lps.T
        c = Dataset(); c.ContourGeometricType = kind; c.NumberOfContourPoints = len(xy)
        c.ContourData = points[:, :3].ravel().tolist()
        uid = next(u for u, v in geometry.sop_to_slice.items() if v == k)
        c.ContourImageSequence = Sequence([sop_ref(uid)])
        item.ContourSequence.append(c)
    rt.ROIContourSequence = Sequence([item])
    return rt


def test_physical_landmark_sort_and_ras():
    g = inspect_headers(ct_headers())
    assert g.shape == (10, 9, 4)
    assert g.report['spacing_xyz_mm'] == pytest.approx([1, 2, 3])
    assert [str(h.SOPInstanceUID) for h in g.headers] == ['1.2.3.1', '1.2.3.2', '1.2.3.3', '1.2.3.4']
    # Landmark x=4,y=3,k=2: origin + (4,0,0) + (0,4.8,3.6) + (0,-3.6,4.8).
    assert g.affine_lps @ [4, 3, 2, 1] == pytest.approx([14, 21.2, 38.4, 1])
    assert g.affine_ras @ [4, 3, 2, 1] == pytest.approx([-14, -21.2, 38.4, 1])


@pytest.mark.parametrize('problem', ['missing', 'duplicate_sop', 'duplicate_position', 'mixed_series',
                                    'mixed_frame', 'orientation', 'HU', 'irregular', 'tilt'])
def test_geometry_gate_rejects(problem):
    h = ct_headers(); expected = {str(r.SOPInstanceUID) for r in h}
    if problem == 'missing': h.pop()
    elif problem == 'duplicate_sop': h[0].SOPInstanceUID = h[1].SOPInstanceUID
    elif problem == 'duplicate_position': h[0].ImagePositionPatient = h[1].ImagePositionPatient
    elif problem == 'mixed_series': h[0].SeriesInstanceUID = '1.2.99'
    elif problem == 'mixed_frame': h[0].FrameOfReferenceUID = '1.2.99'
    elif problem == 'orientation': h[0].ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    elif problem == 'HU': del h[0].RescaleIntercept
    elif problem == 'irregular': h[0].ImagePositionPatient = [10, 16.22, 35.04]
    elif problem == 'tilt': h[0].ImagePositionPatient[0] += .2
    with pytest.raises(ValueError): inspect_headers(h, expected_sops=expected)


def test_pixels_and_nifti_world_coordinates(tmp_path):
    paths = []
    headers = ct_headers()
    for h in headers:
        meta = FileMetaDataset(); meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = CTImageStorage; meta.MediaStorageSOPInstanceUID = h.SOPInstanceUID
        d = FileDataset(None, h, file_meta=meta, preamble=b'\0'*128)
        d.SamplesPerPixel = 1; d.PhotometricInterpretation = 'MONOCHROME2'
        d.BitsAllocated = 16; d.BitsStored = 16; d.HighBit = 15; d.PixelRepresentation = 1
        pixels = np.arange(90, dtype=np.int16).reshape(9, 10)
        d.PixelData = pixels.tobytes()
        path = tmp_path / f'{h.SOPInstanceUID}.dcm'; d.save_as(path, enforce_file_format=True); paths.append(path)
    g = inspect_headers(headers, paths); volume = load_volume(g)
    assert volume[4, 3, :].tolist() == [-966, -932, -898, -864]
    save_nifti(volume, g.affine_ras, tmp_path / 'ct.nii.gz')
    import nibabel as nib
    image = nib.load(tmp_path / 'ct.nii.gz')
    assert image.header.get_xyzt_units()[0] == 'mm'
    np.testing.assert_allclose(image.affine @ [4, 3, 2, 1], [-14, -21.2, 38.4, 1], atol=1e-5)


def test_oblique_polygon_has_expected_centers_and_no_shift():
    g = inspect_headers(ct_headers()); mask, report, _ = rasterize_heart(rt_for(g), g)
    expected = np.zeros((10, 9, 4), np.uint8); expected[3:7, 3:6, 1] = 1
    np.testing.assert_array_equal(mask, expected)
    assert report['heart_voxels'] == 12
    assert report['volume_ml'] == pytest.approx(.072)


def test_xor_hole():
    g = inspect_headers(ct_headers())
    polygons = [(1, [(1.25, 1.25), (7.75, 1.25), (7.75, 6.75), (1.25, 6.75)]),
                (1, [(3.25, 3.25), (5.75, 3.25), (5.75, 4.75), (3.25, 4.75)])]
    mask, _, _ = rasterize_heart(rt_for(g, polygons, 'CLOSEDPLANAR_XOR'), g)
    expected = np.zeros(g.shape, np.uint8); expected[2:8, 2:7, 1] = 1; expected[4:6, 4, 1] = 0
    np.testing.assert_array_equal(mask, expected)


@pytest.mark.parametrize('problem', ['frame', 'series', 'sop', 'plane', 'outside', 'empty', 'open'])
def test_rt_gate_rejects(problem):
    g = inspect_headers(ct_headers()); rt = rt_for(g)
    c = rt.ROIContourSequence[0].ContourSequence[0]
    if problem == 'frame': rt.StructureSetROISequence[0].ReferencedFrameOfReferenceUID = '1.2.99'
    elif problem == 'series': rt.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0].RTReferencedSeriesSequence[0].SeriesInstanceUID = '1.2.99'
    elif problem == 'sop': c.ContourImageSequence[0].ReferencedSOPInstanceUID = '1.2.99'
    elif problem == 'plane': c.ContourData[2] += 1
    elif problem == 'outside': c.ContourData[0] = -10
    elif problem == 'empty': rt.ROIContourSequence[0].ContourSequence = Sequence()
    elif problem == 'open': c.ContourGeometricType = 'OPEN_PLANAR'
    with pytest.raises(ValueError): rasterize_heart(rt, g)


@pytest.mark.parametrize('syntax', [ImplicitVRLittleEndian, ExplicitVRLittleEndian])
def test_stream_extract_complete_item_matches_full_parser(syntax):
    rt = rt_for(inspect_headers(ct_headers())); rt.file_meta.TransferSyntaxUID = syntax
    other = copy.deepcopy(rt.ROIContourSequence[0]); other.ReferencedROINumber = 2
    rt.ROIContourSequence.append(other)
    buffer = io.BytesIO(); rt.save_as(buffer, enforce_file_format=True); payload = buffer.getvalue()
    stream = BoundedStream(io.BytesIO(payload)); extracted, receipt = extract_heart_stream(stream)
    full = pydicom.dcmread(io.BytesIO(payload))
    assert extracted.ROIContourSequence[0] == full.ROIContourSequence[0]
    assert len(extracted.ROIContourSequence) == 1
    assert receipt['heart_contour_item_complete'] is True
    assert receipt['original_full_RTSTRUCT_downloaded'] is False
    assert len(stream.buffer) < len(payload)


def test_stream_rejects_truncated_heart():
    rt = rt_for(inspect_headers(ct_headers()))
    buffer = io.BytesIO(); rt.save_as(buffer, enforce_file_format=True)
    payload = buffer.getvalue()
    with pytest.raises((ValueError, EOFError, AttributeError)):
        extract_heart_stream(BoundedStream(io.BytesIO(payload[:-90])))
