"""Synthetic clinical import tests; no patient files in Git."""
import json
import numpy as np
import pydicom
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, CTImageStorage
from heart3d.dicom.ct import inspect_headers
from scripts.audit_external_ct import scan, sha256, numbers, classify_series, cardiac_phase_tokens, position_repetitions, sr_structure


def write_ct(root, study, series, slice_index, age=None, image_type=None):
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = f'{series}.{slice_index+1}'
    h = FileDataset(None, {}, file_meta=meta, preamble=b'\0'*128)
    h.SOPClassUID = CTImageStorage
    h.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    h.StudyInstanceUID = study
    h.SeriesInstanceUID = series
    h.FrameOfReferenceUID = f'{study}.99'
    h.Modality = 'CT'
    h.PatientID = 'synthetic_private_id'
    h.PatientName = 'Synthetic^PrivateName'
    h.PatientBirthDate = '20000101'
    if age is not None: h.PatientAge = age
    h.ImageType = image_type or ['ORIGINAL','PRIMARY','AXIAL']
    h.Rows = 4; h.Columns = 5
    h.PixelSpacing = [2,1]
    h.ImageOrientationPatient = [1,0,0,0,1,0]
    h.ImagePositionPatient = [10,20,30+slice_index*3]
    h.SliceThickness = 7
    h.RescaleSlope = 1; h.RescaleIntercept = -1000
    h.RescaleType = 'HU'; h.InstanceNumber = 100-slice_index
    h.SamplesPerPixel = 1; h.PhotometricInterpretation = 'MONOCHROME2'
    h.BitsAllocated = 16; h.BitsStored = 16; h.HighBit = 15; h.PixelRepresentation = 1
    h.PixelData = np.arange(20,dtype=np.int16).reshape(4,5).tobytes()
    path = root/f'{series}_{slice_index}.bin'
    h.save_as(path,enforce_file_format=True)
    return path,h


def test_scan_separates_two_studies_and_multiple_series_without_mutation(tmp_path):
    incoming=tmp_path/'incoming'; incoming.mkdir()
    paths=[]
    for study,series,age in [('1.2.3','1.2.3.10',None),('1.2.3','1.2.3.11','010Y'),
                             ('1.2.4','1.2.4.10','020Y')]:
        for k in [2,0,1]:paths.append(write_ct(incoming,study,series,k,age)[0])
    before={p:sha256(p) for p in paths}
    report=scan(incoming,tmp_path/'audit',reserve_gb=0)
    assert report['study_count']==2 and report['dicom_files']==9
    assert [len(s['series']) for s in report['studies']]==[2,1]
    assert all(s['geometry_status']=='passed_header_geometry' for c in report['studies'] for s in c['series'])
    assert report['studies'][0]['series'][0]['geometry']['spacing_xyz_mm']==[1,2,3]
    assert {p:sha256(p) for p in paths}==before
    for filename in ['audit_summary.json','source_locators.private.json']:
        text=(tmp_path/'audit'/filename).read_text()
        assert 'Synthetic^PrivateName' not in text and 'synthetic_private_id' not in text
        assert '20000101' not in text
    assert report['clinical_status']=='unknown'


def test_absent_age_does_not_imply_invalid_geometry(tmp_path):
    headers=[write_ct(tmp_path,'1.2.3','1.2.3.10',k)[1] for k in range(3)]
    with pytest.raises(ValueError,match='PatientAge'):inspect_headers(headers)
    geometry=inspect_headers(headers,require_demographic_metadata=False)
    assert geometry.report['patient_age'] is None
    assert geometry.report['demographic_metadata_status']['PatientAge']=='missing'
    assert geometry.report['physical_geometry_confirmed_from_DICOM']
    del headers[0].RescaleSlope
    with pytest.raises(ValueError,match='HU gate'):
        inspect_headers(headers,require_demographic_metadata=False)


def test_duplicate_sop_stays_review_and_is_not_silently_fixed(tmp_path):
    incoming=tmp_path/'incoming';incoming.mkdir()
    paths=[write_ct(incoming,'1.2.3','1.2.3.10',k)[0] for k in range(3)]
    (incoming/'duplicate.dcm').write_bytes(paths[0].read_bytes())
    report=scan(incoming,tmp_path/'audit',reserve_gb=0)
    series=report['studies'][0]['series'][0]
    assert series['duplicate_sop_instances']==1
    assert series['geometry_status']=='requires_review'
    assert series['geometry_error']=='Duplicated SOP instance'


def test_missing_slice_gap_is_rejected_without_age_metadata(tmp_path):
    hs=[write_ct(tmp_path,'1.2.3','1.2.3.10',k)[1] for k in [0,1,3]]
    with pytest.raises(ValueError,match='Irregular spacing'):
        inspect_headers(hs,require_demographic_metadata=False)


def test_null_numeric_tags_and_localizer_handled(tmp_path):
    assert numbers(None) is None
    assert numbers([None,None]) is None
    h=write_ct(tmp_path,'1.2.3','1.2.3.10',0,image_type=['ORIGINAL','PRIMARY','LOCALIZER'])[1]
    assert classify_series([h])=='localizer'


def test_outputs_cannot_be_written_into_incoming(tmp_path):
    incoming=tmp_path/'incoming'; incoming.mkdir()
    with pytest.raises(ValueError,match='outside'):
        scan(incoming,incoming/'audit',reserve_gb=0)


def test_multiphase_evidence_without_guessing_or_exporting_free_text(tmp_path):
    hs=[write_ct(tmp_path,'1.2.3','1.2.3.10',k)[1] for k in range(3)]
    hs[0].SeriesDescription='private clinical text 0%-90%'
    hs[1].ProtocolName='private protocol 23% 999%'
    assert cardiac_phase_tokens(hs)==[0,23,90]
    stats=position_repetitions(hs+hs)
    assert stats['distinct_IPP']==3 and stats['multiplicity_histogram']=={2:3}
    assert stats['phase_separation_performed'] is False


def test_sr_inventory_omits_free_text_person_and_date_values():
    from pydicom.dataset import Dataset
    from pydicom.sequence import Sequence
    sr=Dataset();sr.ValueType='CONTAINER'
    child=Dataset();child.ValueType='TEXT';child.TextValue='private diagnosis'
    person=Dataset();person.ValueType='PNAME';person.PersonName='Private^Observer'
    date=Dataset();date.ValueType='DATETIME';date.DateTime='20261009123456'
    concept=Dataset();concept.CodeValue='TEST';concept.CodingSchemeDesignator='99TEST';concept.CodeMeaning='Finding'
    child.ConceptNameCodeSequence=Sequence([concept])
    sr.ContentSequence=Sequence([child,person,date])
    report=sr_structure(sr)
    assert report['value_type_counts']=={'CONTAINER':1,'TEXT':1,'PNAME':1,'DATETIME':1}
    serialized=json.dumps(report)
    assert 'private diagnosis' not in serialized and 'Private^Observer' not in serialized
    assert '20261009123456' not in serialized
    assert report['free_text_and_person_values_exported'] is False
