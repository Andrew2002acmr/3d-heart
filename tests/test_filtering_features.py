"""Mathematical fixtures only; research statistics use real release files."""
import inspect
import numpy as np
import nibabel as nib
import pytest
from scipy import ndimage as ndi
import ct_lab.launch
from ct_lab.core import fingerprint
from ct_lab.filtering import (CONFIGS,apply_filter,feature_maps,sample_voxels,matrix,fixed_regions,
                              measurements,fit_scaler,projection,knn_score,selected_voxel_ids)
from ct_lab.filtering_run import protocol


def test_gaussian_grid_units_and_source_copy():
    x=np.zeros((21,21),np.float32);x[10,10]=1000;original=x.copy()
    after,meta=apply_filter(x,{'method':'Gaussian','sigma':2.},[2,1])
    np.testing.assert_allclose(after,ndi.gaussian_filter(x,[1,2],mode='reflect'))
    np.testing.assert_equal(x,original)
    assert not np.shares_memory(after,x) and meta['clipped_fraction']==0


def test_bilateral_uses_source_intensity_sigma():
    x=np.tile(np.arange(12,dtype=np.float32)*100,(12,1))
    a,_=apply_filter(x,dict(method='Bilateral',spatial_sigma=1.,intensity_sigma=50.))
    b,_=apply_filter(x*10,dict(method='Bilateral',spatial_sigma=1.,intensity_sigma=500.))
    np.testing.assert_allclose(b,a*10,rtol=2e-5,atol=.01)
    with pytest.raises(ValueError,match='isotropic'):apply_filter(x,CONFIGS[4],[2,1])


def test_clahe_records_fixed_mapping_and_clipping():
    x=np.linspace(-10,4105,100,dtype=np.float32).reshape(10,10)
    a,meta=apply_filter(x,dict(method='CLAHE',clip_limit=.01,kernel_size=4,input_range=[0,4095]))
    assert 0<=a.min()<=a.max()<=4095
    assert meta['clipped_fraction']==pytest.approx(.02)
    assert 'fixed-range clip and unit conversion' in meta['sequence']


def test_gradient_channel_never_replaces_source_and_has_no_gt_input():
    x=np.tile(np.arange(20,dtype=np.float32),(15,1))
    a,_=apply_filter(x,dict(method='Gradient Magnitude'))
    np.testing.assert_equal(a,x)
    maps=feature_maps(a,[1,2]);assert maps.shape==(15,20,5)
    np.testing.assert_allclose(maps[:,:,1],.5)
    assert 'mask' not in inspect.signature(feature_maps).parameters
    assert 'mask' not in inspect.signature(apply_filter).parameters
    constant=feature_maps(np.ones((15,20),np.float32)*1000)
    np.testing.assert_equal(constant[:,:,3],0)


def test_stratified_sampling_preserves_original_voxel_ids_after_roi():
    shape=(20,16,9);mask=np.where(np.indices((16,20))[1]<10,1,5)
    s=sample_voxels(mask,(1,5),'ct_example',4,shape,(3,14,4,17),20)
    t=sample_voxels(mask,(1,5),'ct_example',4,shape,(3,14,4,17),20)
    np.testing.assert_equal(s['voxel_ids'],t['voxel_ids'])
    np.testing.assert_equal(np.array(np.unravel_index(s['voxel_ids'],shape)).T,s['xyz'])
    assert len(np.unique(s['voxel_ids']))==40
    assert np.count_nonzero(s['labels']==1)==20 and np.count_nonzero(s['train'])==32
    values=np.indices(mask.shape)[0]*100+np.indices(mask.shape)[1]
    f=feature_maps(values)
    np.testing.assert_equal(matrix(f,s)[:,0],values[s['rows'],s['cols']])
    event={'selection':{'points':[{'customdata':[4,3,4,int(s['voxel_ids'][0]),1]}, {'customdata':[4,3,4,-999,1]}]}}
    np.testing.assert_equal(selected_voxel_ids(event,s['voxel_ids']),s['voxel_ids'][:1])


def test_metrics_use_fixed_regions_and_match_baseline():
    rng=np.random.default_rng(3);mask=np.ones((40,40),np.uint8);mask[:,20:]=5
    x=np.where(mask==1,1000.,1300.)+rng.normal(0,30,mask.shape)
    baseline=feature_maps(x);regions=fixed_regions(baseline,mask,(1,5),(0,40,0,40))
    smooth,_=apply_filter(x,dict(method='Gaussian',sigma=1.))
    a=measurements(baseline,mask,(1,5),regions);b=measurements(feature_maps(smooth),mask,(1,5),regions)
    assert a['boundary_voxels']==80 and a['homogeneous_voxels']>0
    assert b['homogeneous_local_variance_median']<a['homogeneous_local_variance_median']
    assert b['expert_interface_gradient_mean']<a['expert_interface_gradient_mean']
    assert 0<=a['histogram_overlap']<=1


def test_original_nifti_unchanged(tmp_path):
    path=tmp_path/'source.nii.gz';data=np.arange(400,dtype=np.int16).reshape(20,20)
    nib.save(nib.Nifti1Image(data[:,:,None],np.eye(4)),path);before=fingerprint(path)
    x=nib.load(path).get_fdata(dtype=np.float32)[:,:,0]
    for config in CONFIGS:
        result,_=apply_filter(x,config);feature_maps(result)
        np.testing.assert_equal(x,data)
    assert fingerprint(path)==before


def test_frozen_group_plan_never_includes_test():
    plan,records=protocol()
    assert len(plan['train_cases'])==5 and len(plan['validation_cases'])==10
    assert not {records[c]['group_id'] for c in plan['train_cases']}&{records[c]['group_id'] for c in plan['validation_cases']}
    assert plan['test_CT_read'] is False


def test_training_only_standardization_and_pca():
    rng=np.random.default_rng(2);values=rng.normal(size=(40,5));train=np.arange(40)<30
    embed,info=projection(values,train,'PCA')
    np.testing.assert_allclose(info['scaler']['mean'],values[:30].mean(0))
    other=values.copy();other[30:]+=10000
    _,changed=projection(other,train,'PCA')
    np.testing.assert_allclose(changed['settings']['components'],info['settings']['components'])
    assert np.isfinite(embed).all()


def test_knn_group_macro_and_separate_feature_sets():
    x=np.array([[0,0],[.1,.1],[10,10],[10.1,10.1]])
    y=np.array([1,1,5,5]);z=np.array([[0,0],[10,10]])
    result=knn_score(x,y,z,np.array([1,5]),[0],k=1,groups=['case_a','case_b'])
    assert result['macro_f1']==1 and result['features']==['Intensity']
    assert result['scaler']['mean']==[float(x[:,0].mean())]
    assert result['equal_case_macro_f1_mean']==.5 # both predefined classes; one present in each case


def test_constant_features_pca_and_umap_train_only_seed():
    rng=np.random.default_rng(7);train=np.arange(80)<64
    const=np.ones((80,5));embedding,info=projection(const,train,'PCA')
    assert np.isfinite(embedding).all() and info['settings']['explained_variance_ratio']==[0.,0.]
    pytest.importorskip('umap')
    values=rng.normal(size=(80,5))
    a,meta=projection(values,train,'UMAP')
    changed=values.copy();changed[~train]+=100
    b,other=projection(changed,train,'UMAP')
    np.testing.assert_allclose(a[train],b[train],atol=1e-6)
    np.testing.assert_allclose(meta['scaler']['mean'],values[train].mean(0))
    assert meta['settings']['seed']==20261028 and np.isfinite(a).all()
