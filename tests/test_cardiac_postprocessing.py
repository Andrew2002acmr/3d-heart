import json
from pathlib import Path
import numpy as np
import pytest
import nibabel as nib
from heart3d.ml.postprocessing import small_changes,apply_changes
from scripts.postprocess_cardiac_validation import gate,score


@pytest.fixture
def config():
    return json.loads(Path('configs/cardiac_chd68_postprocessing_v1.json').read_text())


def test_small_island_removed_but_largest_and_vessels_retained(config):
    p=np.zeros((40,40,40),np.uint8);p[2:22,2:22,2:22]=1;p[30,30,30]=1
    p[35,35,35]=6;p[36,35,35]=7;original=p.copy()
    removal,fill,_=small_changes(p,config);result=apply_changes(p,removal,fill,'combined')
    assert result[30,30,30]==0 and np.all(result[2:22,2:22,2:22]==1)
    assert result[35,35,35]==6 and result[36,35,35]==7
    assert np.array_equal(p,original)


def test_26_connectivity_preserves_diagonal_appendage(config):
    p=np.zeros((35,35,35),np.uint8);p[2:22,2:22,2:22]=1;p[22,22,22]=1
    removal,_,_=small_changes(p,config)
    assert not removal.any()


def test_enclosed_background_hole_filled_without_overwriting_other_labels(config):
    p=np.zeros((35,35,35),np.uint8);p[2:22,2:22,2:22]=1
    p[10,10,10]=0;p[12,12,12]=2
    removed,fill,_=small_changes(p,config);result=apply_changes(p,removed,fill,'fill_small')
    assert result[10,10,10]==1 and result[12,12,12]==2
    assert not np.any((p>0)&(p!=result))


def test_large_cavity_and_open_boundary_not_filled(config):
    p=np.zeros((35,35,35),np.uint8);p[2:22,2:22,2:22]=1
    p[8:13,8:13,8:13]=0;p[2,10,10]=0
    removal,fill,_=small_changes(p,config)
    assert not fill.any()


def test_myocardium_chamber_cavity_not_filled(config):
    p=np.zeros((35,35,35),np.uint8);p[2:22,2:22,2:22]=5;p[10,10,10]=0
    _,fill,_=small_changes(p,config)
    assert not fill.any()


def test_gate_rejects_test_before_protocol_access():
    with pytest.raises(ValueError,match='Only validation'):
        gate({'partition':'validation'},{'partition':'test'})


def test_invalid_labels_and_unsafe_policy_rejected(config):
    with pytest.raises(ValueError):small_changes(np.full((4,4,4),8,np.uint8),config)
    config['remove_labels'].append(7)
    with pytest.raises(ValueError,match='Vessels'):small_changes(np.zeros((4,4,4),np.uint8),config)


def test_prediction_roundtrip_and_chunked_metrics(config,tmp_path):
    p=np.zeros((40,40,40),np.uint8);p[2:22,2:22,2:22]=1;p[30,30,30]=1
    rem,fill,_=small_changes(p,config);result=apply_changes(p,rem,fill,'remove_small')
    affine=np.diag([.7,.8,1.5,1]);path=tmp_path/'mask.nii.gz'
    nib.save(nib.Nifti1Image(result,affine),path);loaded=nib.load(path)
    assert np.array_equal(np.asanyarray(loaded.dataobj),result)
    assert np.allclose(loaded.affine,affine)
    metrics=score(result,p)
    assert metrics['classes']['LV']['target_voxels']==8001
    assert metrics['classes']['LV']['predicted_voxels']==8000
    assert metrics['classes']['LV']['Dice']==16000/16001
