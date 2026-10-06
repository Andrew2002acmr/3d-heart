import numpy as np
import pytest
from heart3d.ml.geometry import make_transform,resample_xyz,physical_context,restore_probability


def test_physical_neighbors_interpolate_different_native_spacing_and_edges():
    low,high,w,outside=physical_context(np.arange(12)*.625,5,[-4,-2,0,2,4])
    reconstructed=(low*(1-w)+high*w)*.625
    assert np.allclose(reconstructed,np.clip(3.125+np.array([-4,-2,0,2,4]),0,6.875))
    assert outside.tolist()==[True,False,False,False,True]
    _,_,w2,_=physical_context(np.arange(8)*2.,3,[-4,-2,0,2,4])
    assert np.allclose(w2,0)


def test_single_plane_context_is_explicitly_replicated():
    low,high,w,outside=physical_context([0],0,[-2,0,2])
    assert not low.any() and not high.any() and not w.any()
    assert outside.tolist()==[True,False,True]


def test_image_mask_identity_roundtrip_and_native_affine():
    image=np.arange(16*16*8,dtype=np.float32).reshape(16,16,8)
    affine=np.diag([-1.,-1.,2.,1.]);affine[:3,3]=[20,30,-40]
    t=make_transform(image.shape,affine,16,2.)
    transformed=resample_xyz(image,t,1)
    assert np.array_equal(restore_probability(transformed,t),image)
    mask=(image>900).astype(np.uint8)
    assert np.array_equal(restore_probability(resample_xyz(mask,t,0),t),mask)
    assert t['original_affine_ras']==affine.tolist()


def test_fit_pad_anisotropic_FOV_and_inverse_landmark():
    shape=(16,8,5); affine=np.diag([-1.,-1.,2.,1.])
    t=make_transform(shape,affine,32,2.)
    assert t['resized_xy']==[32,16] and t['pad_xy']==[0,8]
    mask=np.zeros(shape,dtype=np.uint8);mask[6:10,2:6,1:4]=1
    result=resample_xyz(mask,t,0)
    assert set(np.unique(result))=={0,1}
    restored=restore_probability(result,t)>=.5
    assert np.array_equal(restored,mask)


def test_restore_rejects_wrong_grid_and_neighbor_order():
    t=make_transform((16,16,8),np.eye(4),16,1)
    with pytest.raises(ValueError):restore_probability(np.zeros((7,16,16)),t)
    with pytest.raises(ValueError):physical_context([0,2,1],1,[0])
