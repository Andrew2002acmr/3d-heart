"""Synthetic fixtures validate mathematics only, never dataset research results."""
import numpy as np
import nibabel as nib
import pytest
import torch
from ct_lab.core import (window, normalize, geometry_matches, orthogonal_pair, crop_pad,
                         mask_analysis, line_profile, load_pair, resize_dimensions, cardiac_target, fragment_comparison)
from ct_lab.model_inspector import inspect
from heart3d.ml.infer_cardiac import prepare_volume, protocol
from heart3d.ml.cardiac_data import index_neighbors, resize_plane, index_transform


def test_window_and_normalization():
    x=np.array([-2.,0,5,10,12])
    clipped, fractions=window(x,0,10)
    np.testing.assert_equal(clipped,[0,0,5,10,10])
    assert fractions=={'below_fraction':.2,'above_fraction':.2}
    np.testing.assert_allclose(normalize(x,'Fixed-window',0,10),[-1,-1,0,1,1])
    z=normalize(x,'Z-score')
    assert abs(z.mean())<1e-6 and abs(z.std()-1)<1e-6
    np.testing.assert_equal(normalize(np.ones(4),'Min-Max'),np.zeros(4))
    assert np.isnan(normalize(np.array([0.,1.,np.nan]),'Min-Max')[-1])
    with pytest.raises(ValueError): window(x,10,0)
    with pytest.raises(ValueError): normalize(x,'Saved train Z-score')


def test_geometry_includes_affine_and_units():
    x=np.zeros((5,6,7),np.float32)
    image=nib.Nifti1Image(x,np.eye(4)); mask=nib.Nifti1Image(x,np.eye(4))
    assert geometry_matches(image,mask)
    mask.affine[0,3]=1
    assert not geometry_matches(image,mask)
    with pytest.raises(ValueError,match='geometry mismatch'): orthogonal_pair(image,mask)
    with pytest.raises(ValueError,match='geometry mismatch'): resize_dimensions(image,[3,3,3],mask)
    mask=nib.Nifti1Image(x,np.eye(4));mask.header.set_xyzt_units('mm')
    assert not geometry_matches(image,mask)


def test_oblique_reslice_preserves_discrete_labels_and_world_position():
    angle=.2
    affine=np.array([[np.cos(angle),-2*np.sin(angle),0,10],[np.sin(angle),2*np.cos(angle),0,-4],[0,0,3,2],[0,0,0,1]])
    coords=np.indices((8,9,10));x=(coords[0]+2*coords[1]+coords[2]).astype(np.float32)
    y=(x>15).astype(np.uint8)*7
    im=nib.Nifti1Image(x,affine);mask=nib.Nifti1Image(y,affine)
    result,seg=orthogonal_pair(im,mask,[1,1,2])
    assert geometry_matches(result,seg)
    assert set(np.unique(np.asanyarray(seg.dataobj))) <= {0,7}
    np.testing.assert_allclose(result.affine[:3,:3],np.diag([1,1,2]))
    crop=crop_pad(im,[-2,3,1],[7,4,5])
    np.testing.assert_allclose(crop.affine[:3,3],nib.affines.apply_affine(affine,[-2,3,1]))
    np.testing.assert_equal(np.asanyarray(crop.dataobj)[2,0,0],x[0,3,1])


@pytest.mark.parametrize('variant',['v1','v2'])
@pytest.mark.parametrize('center',[0,3,6])
def test_inspector_exact_production_tensor_and_training_preparation(variant,center):
    rng=np.random.default_rng(7)
    x=rng.uniform(0,4095,(11,9,7)).astype(np.float32)
    result=inspect(nib.Nifti1Image(x,np.eye(4)),variant,center)
    pre=protocol(variant)[2]
    full,t=prepare_volume(x,pre)
    neighbors,_=index_neighbors(center,7,pre['context_offsets_indices'])
    assert torch.equal(result['tensor'],torch.from_numpy(full[neighbors].copy()).unsqueeze(0))
    # Independent reconstruction of the existing training script's raw resize,
    # clipping, explicit pad reset, then context gather.
    t=index_transform(x.shape,pre['input_size']);lo,hi=pre['clip_source_intensity']
    train=[]
    rx,ry=t['resized_xy'];px,py=t['pad_xy']
    for z in neighbors:
        plane=resize_plane(x[:,:,z],t,1,0)
        plane=np.clip((plane-lo)/(hi-lo),0,1)*2-1
        valid=np.zeros_like(plane,dtype=bool);valid[py:py+ry,px:px+rx]=True
        plane[~valid]=-1;train.append(plane)
    np.testing.assert_array_equal(result['tensor'].numpy()[0],np.stack(train))


def test_components_2d_3d_and_volume_units():
    y=np.zeros((5,5,5),np.uint8);y[1,1,1]=1;y[3,3,3]=1
    result=mask_analysis(y,np.diag([2,3,4,1]))
    row=next(r for r in result['classes'] if r['label']==1)
    assert row['components_3d_26']==2
    assert row['volume_grid_units3']==pytest.approx(48)
    assert row['volume_mm3_verified'] is None
    assert row['components_2d_8_per_axis']['2']==[0,1,0,1,0]


def test_profile_distance_and_boundary():
    x=np.tile(np.arange(10,dtype=np.float32),(4,1));y=(x>=5).astype(np.uint8)
    p=line_profile(x,x*2,y,[1,0],[1,9],np.array([3,2]))
    assert p['distance'][-1]==18
    np.testing.assert_allclose(p['gradient'],.5,atol=1e-5)
    assert np.count_nonzero(np.diff(p['label']))==1


def test_reject_misaligned_pair(tmp_path):
    image=tmp_path/'a_image.nii.gz';mask=tmp_path/'a_label.nii.gz'
    nib.save(nib.Nifti1Image(np.zeros((3,3,3)),np.eye(4)),image)
    a=np.eye(4);a[2,3]=4
    nib.save(nib.Nifti1Image(np.zeros((3,3,3)),a),mask)
    with pytest.raises(ValueError,match='geometry mismatch'): load_pair({'image':str(image),'mask':str(mask)})


def test_resize_dimensions_retains_world_endpoints():
    affine=np.diag([2.,3.,4.,1.]);affine[:3,3]=[10,20,30]
    x=np.ones((6,8,10),np.float32);y=np.ones_like(x,dtype=np.uint8)*3
    image=nib.Nifti1Image(x,affine);mask=nib.Nifti1Image(y,affine)
    res,seg=resize_dimensions(image,[4,5,6],mask)
    np.testing.assert_allclose(nib.affines.apply_affine(res.affine,np.array(res.shape)-1),
                               nib.affines.apply_affine(image.affine,np.array(image.shape)-1))
    assert geometry_matches(res,seg)
    assert set(np.unique(np.asanyarray(seg.dataobj)))=={3}


def test_extra_source_labels_ignored_in_model_metrics_not_background():
    target=cardiac_target(np.array([0,1,7,8,13,15],dtype=np.int8))
    np.testing.assert_equal(target,[0,1,7,255,255,255])


def test_fragment_location_and_wrong_anatomy_overlap():
    prediction=np.zeros((5,5,5),np.uint8);prediction[1,1,1:3]=1;prediction[4,4,4]=1
    gt=np.zeros_like(prediction);gt[1,1,1:3]=1;gt[4,4,4]=2
    rows=fragment_comparison(prediction,gt,np.diag([2,3,4,1]))
    largest=next(r for r in rows if r['largest']);satellite=next(r for r in rows if not r['largest'])
    assert largest['gt_same_class_fraction']==1
    assert satellite['gt_overlap_voxels']=={'2':1}
    np.testing.assert_equal(satellite['centroid_world_ras'],[8,12,16])
