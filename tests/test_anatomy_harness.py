import json
from pathlib import Path
import nibabel as nib
import numpy as np
import pytest
from heart3d.anatomy_harness import prepare, render_plane, inspect_runtime_receipt
from heart3d.storage import sha256_file

PROTOCOL = Path(__file__).parents[1]/"configs/anatomy_review_protocol_v1.json"


def admitted_source(tmp_path):
    a = np.arange(6*8*10,dtype=np.float32).reshape(6,8,10)
    affine = np.diag([-.7,-1.1,2.,1.]);affine[:3,3]=[20,30,-40]
    image=nib.Nifti1Image(a,affine);image.header.set_xyzt_units('mm')
    path=tmp_path/'ct.nii.gz';nib.save(image,path)
    admission={k:True for k in ('user_attested_permission_and_basis','local_only','geometry_verified',
        'independent_reader_verified','metadata_deidentified','pixel_identifier_screening_passed')}
    admission['ct_sha256']=sha256_file(path);ap=tmp_path/'admission.json';ap.write_text(json.dumps(admission))
    return a,affine,path,ap


@pytest.mark.parametrize('view,axis,index',[('AX',2,4),('COR',1,3),('SAG',0,2)])
def test_context_packet_preserves_source_plane_physical_mapping_and_orientation(tmp_path,view,axis,index):
    a,affine,_,_=admitted_source(tmp_path)
    im,info=render_plane(a,affine,[.7,1.1,2.],view,index,600,150)
    x,y,w,h=info['content_rectangle_xywh'];matrix=np.array(info['source_ijk_from_png_pixel_centre'])
    centre=matrix@np.array([x+(w-1)/2,y+(h-1)/2,1])
    assert centre[axis]==pytest.approx(index)
    np.testing.assert_allclose(centre,info['locator']['anchor_ijk'])
    lps=np.diag([-1,-1,1,1])@affine
    np.testing.assert_allclose(nib.affines.apply_affine(lps,centre),info['locator']['focal_point_lps_mm'])
    assert info['locator']['overlay_visible'] is False
    assert info['orientation_markers']['top']==('A' if view=='AX' else 'S')
    assert im.width==info['png_width']


def test_preparation_creates_only_unreviewed_raw_context_and_does_not_invent_findings(tmp_path):
    _,_,ct,ap=admitted_source(tmp_path);digest=sha256_file(ct)
    result=prepare(ct,ap,PROTOCOL,tmp_path/'review',4)
    assert result['observations']==[] and result['structure_tracks']==[] and result['target_annotation'] is None
    assert not result['all_planes_reviewed'] and not result['target_anatomically_selected']
    assert {p['locator']['view'] for p in result['packets']}=={'AX','COR','SAG'}
    assert all(p['review_state']=='not_reviewed' and p['evidence_kind']=='raw_CT' for p in result['packets'])
    assert all(sha256_file(tmp_path/'review'/p['file'])==p['sha256'] for p in result['packets'])
    assert sha256_file(ct)==digest
    with pytest.raises(ValueError,match='new private'):prepare(ct,ap,PROTOCOL,tmp_path/'review')


def test_missing_admission_stops_before_creating_context_images(tmp_path):
    _,_,ct,ap=admitted_source(tmp_path);record=json.loads(ap.read_text());record['local_only']=False;ap.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='admitted'):prepare(ct,ap,PROTOCOL,tmp_path/'blocked')
    assert not (tmp_path/'blocked').exists()


def test_unknown_or_cloud_runtime_receipt_cannot_trigger_image_submission():
    for receipt in ({},{'remote_host':'https://example.org','capabilities':['vision'],'cloud_enabled':True}):
        result=inspect_runtime_receipt(receipt)
        assert not result['ready_for_transport'] and result['execution_enabled'] is False
        assert result['image_submission_performed'] is False and not result['medical_validity_verified']


def test_followup_requests_render_contiguous_context_without_changing_review_status(tmp_path):
    _,_,ct,ap=admitted_source(tmp_path);request=tmp_path/'request.json'
    record={'ct_sha256':sha256_file(ct),'planes':[{'view':'AX','indices':[3,4,5],'window':'wide'}]}
    request.write_text(json.dumps(record))
    result=prepare(ct,ap,PROTOCOL,tmp_path/'followup',request_path=request)
    assert result['phase']=='requested_followup' and len(result['packets'])==3
    assert [p['locator']['native_index'] for p in result['packets']]==[3,4,5]
    assert result['observations']==[]
    record['planes'][0]['url']='https://not-a-local-image.invalid'
    request.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='unsupported'):prepare(ct,ap,PROTOCOL,tmp_path/'badrequest',request_path=request)
    assert not (tmp_path/'badrequest').exists()
