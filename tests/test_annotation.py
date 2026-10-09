import json
import nibabel as nib
import numpy as np
import pytest
from heart3d.annotation import MaskEditor
from heart3d.storage import sha256_file

def make(mask=None,affine=None):
    return MaskEditor(np.zeros((11,13,9),dtype=np.uint8) if mask is None else mask,
                      np.diag([.5,1.,2.,1.]) if affine is None else affine,{1:"Heart"},"ct","mask",True)

@pytest.mark.parametrize("axis",[0,1,2])
def test_plane_edit_and_undo_leave_source_unchanged(axis):
    source=np.zeros((11,13,9),dtype=np.uint8);editor=make(source)
    center=[source.shape[i]//2 for i in range(3) if i!=axis]
    n=editor.paint(axis,3,center,1.1,1)
    assert n>0 and editor.mask.sum()==n and source.sum()==0
    assert np.count_nonzero(editor.mask.take(3,axis))==n
    assert editor.undo() and not editor.mask.any()
    assert not editor.undo()

def test_brush_uses_mm_with_anisotropy():
    e=make();e.paint(2,4,(5,6),1.1,1)
    assert e.mask[7,6,4]==1 and e.mask[5,7,4]==1 and e.mask[5,8,4]==0

def test_world_selection_full_affine_oblique_roundtrip():
    angle=.3;c,s=np.cos(angle),np.sin(angle)
    affine=np.array([[c,-s,0,40],[s,c,0,-20],[0,0,2,10],[0,0,0,1.]])
    e=make(affine=affine)
    point=nib.affines.apply_affine(affine,[3,7,5])
    assert np.array_equal(e.world_to_voxel(point),[3,7,5])
    with pytest.raises(ValueError):e.world_to_voxel([10000,0,0])

def test_geometry_gate_and_label_namespace():
    with pytest.raises(ValueError):MaskEditor(np.zeros((3,3,3),dtype=np.uint8),np.eye(4),{1:"Heart"},"a","b")
    with pytest.raises(ValueError):make(np.full((3,3,3),2,dtype=np.uint8))
    with pytest.raises(ValueError):make().paint(2,0,(2,2),1,7)

def test_immutable_version_restores_original_grid_and_checks_sources(tmp_path):
    ct=tmp_path/"ct.nii.gz";mask=tmp_path/"mask.nii.gz";aff=np.diag([.5,1.,2.,1.])
    for path,data in [(ct,np.zeros((11,13,9),dtype=np.float32)),(mask,np.zeros((11,13,9),dtype=np.uint8))]:
        image=nib.Nifti1Image(data,aff);image.header.set_xyzt_units("mm");nib.save(image,path)
    before=(sha256_file(ct),sha256_file(mask))
    e=MaskEditor(np.zeros((11,13,9),dtype=np.uint8),aff,{1:"Heart"},*before,True)
    e.paint(2,4,(5,6),1.1,1)
    a=e.save(tmp_path/"versions",ct,mask,"unverified research",reserve_GB=0)
    b=e.save(tmp_path/"versions",ct,mask,reserve_GB=0)
    assert a!=b
    result=nib.load(a/"research_mask.nii.gz")
    assert np.array_equal(result.dataobj,e.mask) and np.allclose(result.affine,aff)
    report=json.loads((a/"annotation.json").read_text())
    assert report["annotation_status"]=="research_edited" and not report["expert_approval"]
    assert before==(sha256_file(ct),sha256_file(mask))
    ct.write_bytes(b"changed")
    with pytest.raises(ValueError):e.save(tmp_path/"versions",ct,mask,reserve_GB=0)
