"""No neural training. Synthetic mathematical fixtures are never research results."""
import inspect
import numpy as np
import nibabel as nib
import pytest
from scipy import ndimage as ndi
from ct_lab.spatial_features import (extract,columns,feature_sets,z_context,reflect_indices,plane_coordinates,from_plane,check_affine)
from ct_lab.filtering import feature_maps
from ct_lab.spatial_run import freeze,predict
from ct_lab.core import fingerprint


def test_plane_click_coordinates_and_affine_are_one_position():
    xyz=np.array([3,5,7]);affine=np.diag([2.,3.,4.,1.]);affine[:3,3]=[11,-6,20]
    for plane in ('Axial XY','Coronal XZ','Sagittal YZ'):
        u,v,index=plane_coordinates(xyz,plane)
        restored=from_plane(u,v,index,plane)
        np.testing.assert_array_equal(restored,xyz)
        np.testing.assert_allclose(nib.affines.apply_affine(affine,restored),[17,9,48])
    oblique=affine.copy();oblique[0,1]=.1
    with pytest.raises(ValueError):check_affine(oblique)


def test_neighbors_replicate_edges_and_keep_flags():
    volume=np.broadcast_to(np.arange(5,dtype=np.float32),(3,4,5));xyz=np.array([[1,2,0],[1,2,4]])
    values,flags=z_context(volume,xyz,3)
    np.testing.assert_array_equal(values,[[0,0,0,0,1,2,3],[1,2,3,4,4,4,4]])
    np.testing.assert_array_equal(flags.sum(1),[3,3])
    b=extract(volume,xyz,np.eye(4),include_gaussians=False)
    np.testing.assert_allclose(columns(b,['Z1 Gradient'])[:,0],.5)
    np.testing.assert_array_equal(reflect_indices(np.arange(-2,7),5),[1,0,0,1,2,3,4,4,3])


def test_xy_features_match_stage01_and_ids():
    rng=np.random.default_rng(8);vol=rng.normal(1000,100,(12,13,14)).astype(np.float32)
    points=np.array([[0,0,0],[11,12,13],[5,7,8],[3,4,6]])
    bank=extract(vol,points,np.eye(4));names=feature_sets()['B 2D']
    expected=np.stack([feature_maps(vol[:,:,z].T)[y,x] for x,y,z in points])
    np.testing.assert_allclose(columns(bank,names),expected,rtol=2e-5,atol=.001)
    ids=np.ravel_multi_index(points.T,vol.shape)
    np.testing.assert_array_equal(np.array(np.unravel_index(ids,vol.shape)).T,bank['xyz'])


def test_3d_features_match_dense_and_chunk_halos():
    rng=np.random.default_rng(4);vol=rng.normal(1000,100,(13,15,17)).astype(np.float32)
    points=np.array([[0,0,0],[12,14,16],[5,7,8],[3,4,6]])
    bank=extract(vol,points,np.eye(4),chunk_size=1);other=extract(vol,points,np.eye(4),chunk_size=3)
    np.testing.assert_allclose(bank['values'],other['values'],rtol=1e-6,atol=.001)
    mean=ndi.uniform_filter(vol.astype(float),5,mode='reflect')
    std=np.sqrt(np.maximum(ndi.uniform_filter(vol.astype(float)**2,5,mode='reflect')-mean**2,0))
    np.testing.assert_allclose(columns(bank,['3D Mean'])[:,0],mean[tuple(points.T)],rtol=1e-6)
    np.testing.assert_allclose(columns(bank,['3D Std'])[:,0],std[tuple(points.T)],rtol=1e-6)
    for sigma in (.5,1.,2.):
        dense=ndi.gaussian_filter(vol.astype(float),sigma,mode='reflect')
        np.testing.assert_allclose(columns(bank,[f'3D Gaussian {sigma:g}'])[:,0],dense[tuple(points.T)],rtol=1e-6)
    for dim in (2,3):
        config={'method':'Gaussian','sigma':2.,'dimensions':dim}
        filtered=extract(vol,points,np.eye(4),filter_config=config,include_gaussians=False,chunk_size=2)
        dense=ndi.gaussian_filter(vol,[2,2,2 if dim==3 else 0],mode='reflect')
        np.testing.assert_allclose(columns(filtered,['Intensity'])[:,0],dense[tuple(points.T)],rtol=1e-6)
        reference=extract(dense,points,np.eye(4),include_gaussians=False)
        np.testing.assert_allclose(filtered['values'],reference['values'],rtol=1e-5,atol=.002)


def test_bilateral_3d_is_coherent_and_source_not_modified(tmp_path):
    vol=np.zeros((8,9,10),np.float32);vol[4:]=1000;vol[3,4,5]=600
    path=tmp_path/'source.nii.gz';nib.save(nib.Nifti1Image(vol,np.eye(4)),path);before=fingerprint(path);original=vol.copy()
    point=np.array([[3,4,5]])
    for dim in (2,3):
        b=extract(vol,point,np.eye(4),include_gaussians=False,filter_config={'method':'Bilateral','dimensions':dim})
        offsets=np.array(list(__import__('itertools').product(range(-3,4),range(-3,4),range(-3,4) if dim==3 else [0])))
        indices=np.stack([reflect_indices(point[0,a]+offsets[:,a],vol.shape[a]) for a in range(3)])
        neighbors=vol[tuple(indices)];w=np.exp(-.5*(offsets**2).sum(1)-.5*((neighbors-600)/100)**2)
        np.testing.assert_allclose(columns(b,['Intensity'])[0,0],(w*neighbors).sum()/w.sum(),rtol=1e-5)
    np.testing.assert_array_equal(vol,original);assert fingerprint(path)==before


def test_frozen_sample_plan_has_no_test_or_group_overlap():
    plan,records=freeze()
    assert len(plan['feature_sets'])==8 and plan['test_CT_read'] is False
    assert not {records[c]['group_id'] for c in plan['train_cases']}&{records[c]['group_id'] for c in plan['validation_cases']}
    assert 'NOT independent final' in plan['validation_status']
    assert 'mask' not in inspect.signature(extract).parameters
    for names in feature_sets().values():assert not any('GT' in n or 'distance' in n.lower() or 'label' in n.lower() for n in names)


def test_no_coordinate_or_gt_feature_and_train_scaling():
    vol=np.ones((9,9,9),np.float32)*17
    bank=extract(vol,np.array([[1,2,3],[6,7,5]]),np.eye(4))
    np.testing.assert_allclose(bank['values'][0],bank['values'][1],atol=1e-5)
    train=np.array([[0.],[1.],[10.],[11.]]);labels=np.array([1,1,5,5])
    from ct_lab.filtering import fit_scaler
    scaler=fit_scaler(train)
    assert scaler['mean']==[5.5]
    np.testing.assert_array_equal(predict(train,labels,np.array([[0.],[11.]]),scaler,k=1),[1,5])


def test_actual_stage01_voxel_ids_and_sample_counts_are_identical():
    from ct_lab.spatial_run import OUT,PREVIOUS,previous_hashes
    import json
    if not (OUT/'case_ct_1001.npz').exists():pytest.skip('Real extraction not run yet')
    with np.load(PREVIOUS/'samples_ct_1001.npz') as old,np.load(OUT/'case_ct_1001.npz') as new:
        names=json.loads((OUT/'case_ct_1001.json').read_text())['names']
        for p in (0,1):
            for key in ('xyz','voxel_ids','labels','train'):np.testing.assert_array_equal(old[f'p{p}_{key}'],new[f'p{p}_{key}'])
            for feature_names in feature_sets().values():assert columns({'names':names,'values':new[f'p{p}_values']},feature_names).shape[0]==len(old[f'p{p}_labels'])
    assert previous_hashes()==json.loads((OUT/'preserved_stage01.json').read_text())


def test_common_projection_is_same_measured_space_not_zero_padding():
    from ct_lab.spatial_projection import project
    rng=np.random.default_rng(8);values=rng.normal(size=(30,5));fold=np.arange(30)<24
    embed,info=project(values[:,[0]],fold,'PCA')
    assert embed.shape==(30,2)
    np.testing.assert_array_equal(embed[:,1],0)
    np.testing.assert_allclose(info['scaler']['mean'],values[fold,0].mean().reshape(1))
    a,_=project(values,fold,'PCA');b,_=project(values.copy(),fold,'PCA')
    np.testing.assert_array_equal(a,b)
