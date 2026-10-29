import json
import numpy as np
import torch
from scipy import ndimage as ndi
from ct_lab.multiclass_core import (FEATURES,SETS,augment,transformed_xyz,extract_points,FeatureEncoder,normalize,fit_scaler,load_model,OUT,freeze)
from ct_lab.multiclass_data import natural_ids,training_ids
from ct_lab.multiclass_metrics import classification,dense_metrics,bootstrap
from ct_lab.spatial_features import extract,columns


def test_feature_equivalence_edges_and_chunks():
    rng=np.random.default_rng(61);volume=rng.normal(1200,80,(19,17,15)).astype(np.float32)
    points=np.array([[0,0,0],[18,16,14],[1,16,7],[9,8,7],[4,2,13]])
    old=columns(extract(volume,points,np.eye(4)),FEATURES)
    new=extract_points(volume,points,np.eye(4),2)
    np.testing.assert_allclose(new,old,rtol=1e-6,atol=1e-4)
    np.testing.assert_array_equal(new,extract_points(volume,points,np.eye(4),5))


def test_sampling_natural_GT_independent_and_training_categories():
    shape=(20,20,24);ids=natural_ids(shape,1024,91);np.testing.assert_array_equal(ids,natural_ids(shape,1024,91))
    assert len(np.unique(ids))==1024
    mask=np.zeros(shape,np.uint8);mask[4:14,4:14,3:21]=1;mask[8:12,8:12,8:16]=2;mask[:2]=255
    config=freeze()[0]['sampling'];chosen=training_ids(mask,42,config);labels=mask.ravel()[chosen]
    assert 255 not in labels and {0,1,2}<=set(labels)
    assert len(np.unique(chosen%shape[2]))>7
    np.testing.assert_array_equal(chosen,training_ids(mask,42,config))


def test_source_augmentations_recompute_and_geometry_targets():
    x=np.arange(15**3,dtype=np.float32).reshape(15,15,15);xyz=np.array([[7,6,5],[0,0,0],[14,14,14]])
    spec={'translation':[1,-1,0]};mapped,valid=transformed_xyz(xyz,spec,x.shape);altered=augment(x,spec,19)
    np.testing.assert_equal(altered[tuple(mapped[valid].T)],x[tuple(xyz[valid].T)])
    mask=(x%8).astype(np.uint8);m=ndi.shift(mask,spec['translation'],order=0,mode='constant',cval=255,prefilter=False)
    np.testing.assert_equal(m[tuple(mapped[valid].T)],mask[tuple(xyz[valid].T)])
    noise=augment(x,{'noise':8},55);np.testing.assert_equal(noise,augment(x,{'noise':8},55));assert not np.array_equal(noise,x)
    assert not np.array_equal(extract_points(x,xyz,np.eye(4)),extract_points(noise,xyz,np.eye(4)))
    assert x[0,0,0]==0


def test_ignore_metrics_all_eight_and_dense_components():
    truth=np.array([0,1,255,7]);pred=np.array([0,1,3,2]);score=classification(truth,pred)
    assert score['valid_voxels']==3 and len(score['confusion_matrix'])==8
    assert np.array(score['confusion_matrix']).sum()==3 and score['classwise']['3']['predicted']==0
    t=np.zeros((9,9,9),np.uint8);t[2:5,2:5,2:5]=1;p=t.copy();p[7,7,7]=1
    d=dense_metrics(p,t)['dense_classwise']['1'];assert d['components_6connected']==2 and d['false_components_no_GT_overlap']==1
    assert d['false_component_voxels']==1 and d['dice']<1
    assert bootstrap([1,1,1])['case_bootstrap_CI95']==[1,1]


def test_protocol_no_leakage_folds_and_features():
    plan,records=freeze();train=set(plan['train_cases']);val=set(plan['validation_cases'])
    assert not train&val and len(train)==48 and len(val)==10 and len(plan['jobs'])==36
    seen=set()
    for fold in plan['CV_folds']:
        assert not set(fold)&seen;seen.update(fold)
        others=train-set(fold);assert not {records[c]['group_id'] for c in fold}&{records[c]['group_id'] for c in others}
    assert seen==train
    assert not any(any(s in n.lower() for s in ('mask','label','distance','gt','class')) for n in FEATURES)
    assert set(FEATURES)==set(n for names in SETS.values() for n in names)


def test_eight_class_forward_scaler_and_checkpoint(tmp_path):
    plan,_=freeze();x=np.array([[1,3,5,7,9],[3,5,7,9,11]],np.float32);sc=fit_scaler(x,plan['train_cases'])
    np.testing.assert_allclose(normalize(x,sc).mean(0),0)
    model=FeatureEncoder(5,16,8);logits,z=model(torch.from_numpy(x));assert logits.shape==(2,8) and z.shape==(2,16)
    path=tmp_path/'model.pt';torch.save({'format':'multiclass_feature_04_v1','state_dict':model.state_dict(),'features':SETS['C 3D-core'],
         'feature_set':'C 3D-core','classes':list(range(8)),'embedding_dim':16,'scaler':sc,'train_cases':plan['train_cases'],
         'validation_cases':plan['validation_cases'],'split_sha256':plan['split_sha256']},path)
    loaded,_=load_model(path);np.testing.assert_allclose(loaded(torch.from_numpy(x))[0].detach(),logits.detach())


def test_dense_halos_equivalent_to_sampled_stencils():
    from ct_lab.multiclass_dense import dense_features,blocks,roi_bounds
    rng=np.random.default_rng(77);x=rng.uniform(0,2500,(19,17,15)).astype(np.float32)
    expected=extract_points(x,np.column_stack(np.unravel_index(np.arange(x.size),x.shape)),np.eye(4)).reshape(*x.shape,len(FEATURES))
    actual=np.empty_like(expected)
    for start,stop in blocks(np.zeros(3,int),np.array(x.shape),(7,9,6)):
        block=dense_features(x,start,stop,np.eye(4),FEATURES).reshape(*tuple(stop-start),len(FEATURES))
        actual[tuple(slice(a,b) for a,b in zip(start,stop))]=block
    np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=.001)
    lo,hi=roi_bounds(x.shape,{'center_fraction':[.65,.45,.5],'size_xyz':[128,128,96]})
    assert np.array_equal(lo,[0,0,0]) and np.array_equal(hi,x.shape)


def test_dense_predictions_probabilities_and_native_crop_affine(tmp_path):
    import inspect
    import nibabel as nib
    from ct_lab.multiclass_dense import infer_region,save_native,cached_region
    assert not any(n in inspect.signature(infer_region).parameters for n in ('mask','target','GT','distance'))
    rng=np.random.default_rng(31);x=rng.normal(900,20,(13,12,11)).astype(np.float32);affine=np.diag([1.,1.,1.,1.]);affine[:3,3]=[3,4,5]
    features=SETS['E core+all'];points=np.column_stack(np.unravel_index(np.arange(x.size),x.shape))
    raw=extract_points(x,points,affine);sc=fit_scaler(raw[:,[FEATURES.index(n) for n in features]],['train_example'])
    model=FeatureEncoder(len(features),16,8);lower=np.array([1,2,3]);upper=np.array([12,11,10])
    pred=infer_region(x,affine,model,sc,features,lower,upper,tmp_path,(5,6,4))
    np.testing.assert_array_equal(cached_region(tmp_path,model,sc,features,lower,upper,x.shape),pred)
    probs=np.load(tmp_path/'probabilities.npy');np.testing.assert_allclose(probs.sum(-1),1,atol=.002)
    # Float16 storage can create ties; the authoritative mask uses float32 logits.
    chosen=np.take_along_axis(probs,pred[...,None],axis=-1)[...,0]
    np.testing.assert_allclose(chosen,probs.max(-1),atol=.00025)
    image=nib.Nifti1Image(x,affine);save_native(pred,image,lower,tmp_path/'crop.nii.gz');crop=nib.load(tmp_path/'crop.nii.gz')
    np.testing.assert_allclose(crop.affine[:3,3],[4,6,8]);assert crop.header.get_xyzt_units()[0]=='unknown'


def test_hd95_surface_tree_matches_exact_grid_distance(monkeypatch):
    target=np.zeros((17,18,19),np.uint8);target[3:10,4:12,5:14]=1
    pred=np.zeros_like(target);pred[5:12,5:13,4:13]=1;target[7:9,6:8,8:10]=255
    structure=ndi.generate_binary_structure(3,1);valid=target<8;trusted=ndi.binary_erosion(valid,structure,border_value=0)
    p=(pred==1)&valid;g=target==1;ps=p&~ndi.binary_erosion(p,structure,border_value=0)&trusted;gs=g&~ndi.binary_erosion(g,structure,border_value=0)&trusted
    expected=np.percentile(np.concatenate([ndi.distance_transform_edt(~gs)[ps],ndi.distance_transform_edt(~ps)[gs]]),95)
    actual=dense_metrics(pred,target)['dense_classwise']['1']['HD95_index_units'];assert abs(actual-expected)<1e-12
    import ct_lab.multiclass_metrics as metrics
    monkeypatch.setattr(metrics,'HD95_RASTER_THRESHOLD',0)
    raster=dense_metrics(pred,target)['dense_classwise']['1']['HD95_index_units'];assert abs(raster-expected)<1e-12
