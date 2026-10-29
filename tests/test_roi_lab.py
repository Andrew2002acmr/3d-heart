import inspect
import numpy as np
import torch
from ct_lab.roi_core import bbox,clip_box,expand_box,coverage,transform,prepare_plane,restore_plane,freeze,cardiac_target
from ct_lab.roi_models import ROISegmenter,localizer_loss
from ct_lab.roi_localizer import locate,augment_batch
from heart3d.ml.cardiac_data import restore_probability_plane
from heart3d.ml.runtime import seed_runtime
from heart3d.ml.losses import MulticlassSegmentationLoss


def test_bbox_coverage_ignore_and_edges():
    target=np.zeros((17,19,21),np.uint8);target[2:12,3:14,4:17]=1;target[4:8,6:9,10:20]=7;target[0]=255
    b=bbox((target>0)&(target<8));np.testing.assert_array_equal(b,[[2,3,4],[12,14,20]])
    score=coverage(target,b);assert score['foreground_coverage']==1 and score['all_structures_fully_covered']
    cropped=coverage(target,[[0,0,0],[10,19,18]]);assert cropped['foreground_coverage']<1 and cropped['per_class_coverage']['7']<1
    np.testing.assert_array_equal(expand_box(b,target.shape,1,8),[[0,0,0],list(target.shape)])
    mapped=cardiac_target(np.array([-1,0,1,7,8,255]));np.testing.assert_array_equal(mapped,[255,0,1,7,255,255])


def test_crop_probability_restore_matches_production_pixel_centers():
    shape=(37,29,17);b=np.array([[3,5,2],[32,26,15]]);t=transform(shape,b,32)
    rng=np.random.default_rng(32);p=rng.random((8,32,32),dtype=np.float32);p/=p.sum(0)
    expected=restore_probability_plane(p,t['resize']);actual=restore_plane(p,t).numpy()
    np.testing.assert_allclose(actual,expected,atol=1e-6)
    image=np.arange(np.prod(shape[:2])).reshape(shape[:2]).astype(np.float32)
    out=prepare_plane(image,t);assert out.shape==(32,32)
    assert t['bbox_xyz']==b.tolist()


def test_geometry_alignment_and_mask_namespace():
    cfg={'axial_rotation_degrees':0,'scale_delta':0,'translation_fraction':0,'intensity_shift':0,'noise_std':0}
    x=torch.zeros(1,1,12,12,12);target=torch.zeros(1,12,12,12,dtype=torch.uint8);x[:,:,3:8,4:10,2:9]=1;target[:,3:8,4:10,2:9]=1;target[:,0]=255
    a,b=augment_batch(x,target,np.random.default_rng(51),cfg)
    np.testing.assert_allclose(a,x,atol=1e-6);np.testing.assert_array_equal(b,target)
    cfg.update(axial_rotation_degrees=5,scale_delta=.05,translation_fraction=.03)
    a,b=augment_batch(x,target,np.random.default_rng(51),cfg)
    assert set(torch.unique(b).tolist())<={0,1,255} and torch.isfinite(a).all()
    assert ((a[:,0]>.5)&(b==1)).sum()/(b==1).sum()>.85


def test_no_GT_localizer_inference_and_box_mapping():
    assert not any(n in inspect.signature(locate).parameters for n in ('mask','target','GT'))
    class Constant(torch.nn.Module):
        def forward(self,x):
            logits=torch.full_like(x,-12.);logits[:,:,8:24,12:28,4:20]=12;return logits
    prediction,p=locate(np.zeros((32,32,32),np.float32),np.array([128,96,64]),Constant(),[[0,0,0],[1,1,1]],0,0)
    np.testing.assert_array_equal(prediction['bbox'],[[32,36,8],[96,84,40]])
    assert prediction['GT_input'] is False and not prediction['fallback_fixed_ROI']


def test_split_case_groups_and_equal_architecture():
    plan,_=freeze();seen=set()
    for fold in plan['CV_folds']:assert not seen&set(fold);seen.update(fold)
    assert seen==set(plan['train_cases']) and not seen&set(plan['validation_cases'])
    base=ROISegmenter();other=ROISegmenter();branch=ROISegmenter(True)
    assert sum(p.numel() for p in base.parameters())==sum(p.numel() for p in other.parameters())==489112
    assert sum(p.numel() for p in branch.parameters())>489112


def test_seed_reproducible_CPU_training_and_ignore():
    losses=[];weights=[]
    for repeat in range(2):
        seed_runtime(771,1,'cpu');model=ROISegmenter();optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        x=torch.rand(1,5,32,32);y=torch.randint(0,8,(1,32,32));y[:,:4]=255
        loss=MulticlassSegmentationLoss()(model(x),y);loss.backward();optimizer.step();losses.append(loss.item());weights.append(model.base.output.weight.detach().clone())
    assert losses[0]==losses[1];assert torch.equal(*weights)
    logits=torch.randn(1,1,5,5,5,requires_grad=True);target=torch.zeros(1,5,5,5,dtype=torch.uint8);target[:,:2]=255
    loss=localizer_loss(logits,target);loss.backward();assert torch.isfinite(logits.grad).all() and torch.count_nonzero(logits.grad[:,0,:2])==0


def test_batched_resize_matches_original_plane_helpers():
    from ct_lab.roi_data import resize_stack
    rng=np.random.default_rng(31);x=rng.uniform(0,4095,(37,29,7)).astype(np.float32);y=rng.integers(0,8,x.shape,dtype=np.uint8);y[:2]=255
    t=transform(x.shape,[[1,2,0],[34,28,7]],32);indices=np.array([0,3,6])
    expected=np.stack([np.clip(prepare_plane(x[:,:,z],t),0,2015)/2015*2-1 for z in indices])
    np.testing.assert_allclose(resize_stack(x,indices,t),expected,atol=1e-5)
    expected_masks=np.stack([prepare_plane(y[:,:,z],t,0,255) for z in indices]).astype(np.uint8)
    np.testing.assert_array_equal(resize_stack(y,indices,t,mask=True),expected_masks)


def test_predicted_box_forbidden_GT_and_oracle_gate(monkeypatch):
    import pytest
    import ct_lab.roi_data as data
    from ct_lab.roi_core import OUT
    if not (OUT/'predicted_boxes.json').exists():pytest.skip('Localizer receipt not available')
    plan,_=freeze();case=plan['validation_cases'][0];reads=[]
    def forbidden(*args,**kwargs):reads.append(args);raise AssertionError('GT input to automatic localization')
    monkeypatch.setattr(data.nib,'load',forbidden)
    assert data.case_box(case,'predicted').shape==(2,3) and not reads
    with pytest.raises(ValueError):data.case_box(case,'oracle')
    assert not reads


def test_training_loader_rejects_outer_and_fold_holdouts():
    import pytest
    from ct_lab.roi_data import TrainingData
    plan,_=freeze()
    with pytest.raises(ValueError):TrainingData([plan['validation_cases'][0]],'full',32)
    with pytest.raises(ValueError):TrainingData([plan['CV_folds'][0][0]],'full',32,'fold0')


def test_native_full_volume_reconstruction_no_GT_input():
    from ct_lab.roi_evaluate import predict_with_box
    assert not any(n in inspect.signature(predict_with_box).parameters for n in ('mask','target','GT'))
    class Constant(torch.nn.Module):
        def forward(self,x,core=None):
            result=torch.zeros(len(x),8,*x.shape[-2:]);result[:,1]=8;return result
    x=np.ones((17,19,9),np.float32)*1200;box=np.array([[3,4,2],[14,16,7]])
    pred,t=predict_with_box(x,np.eye(4),Constant(),{'size':32,'branch':False},box)
    expected=np.zeros(x.shape,np.uint8);expected[3:14,4:16,2:7]=1
    np.testing.assert_array_equal(pred,expected);assert t['source_shape_xyz']==list(x.shape)


def test_native_topology_exact_components_ignore_and_surface_distance():
    from ct_lab.roi_topology import native_metrics
    target=np.zeros((16,16,16),np.uint8);target[4:8,4:8,4:8]=1
    prediction=np.zeros_like(target);prediction[5:9,4:8,4:8]=1
    prediction[11:13,11:13,11:13]=1;prediction[1,1,1]=2;target[1,1,1]=255
    result=native_metrics(prediction,target);c=result['topology']['1']
    assert c['components_6connected']==2 and c['GT_components_6connected']==1
    assert c['component_sizes_desc']==[64,8] and c['largest_component_voxels']==64
    assert c['false_components_no_GT_overlap']==1 and c['false_component_voxels']==8
    assert result['topology']['2']['components_6connected']==0
    assert result['BG_to_foreground_voxels']==24
    assert result['classes']['LV']['Dice']==96/136
    assert c['HD95_index_units']>1 and result['components_removed'] is False
    # Remove only the artificial island; translated cube has HD95 exactly one index.
    prediction[11:13,11:13,11:13]=0
    assert native_metrics(prediction,target)['topology']['1']['HD95_index_units']==1


def test_chunked_native_core_matches_direct_neighborhoods(tmp_path):
    from ct_lab.roi_evaluate import core_volume
    from ct_lab.multiclass_dense import dense_features
    from ct_lab.roi_core import OUT,resize_plane
    import os
    os.environ['NUMBA_CACHE_DIR']=str(OUT/'numba_cache')
    rng=np.random.default_rng(45);x=rng.uniform(0,2015,(19,17,21)).astype(np.float32)
    box=np.array([[2,3,0],[18,15,21]]);t=transform(x.shape,box,32)
    # A left-over array without its completion receipt must never be a cache hit.
    np.save(tmp_path/'core_input.npy',np.zeros((21,4,32,32),np.float16))
    names=['3D Mean','3D Std','3D Gradient','3D LoG'];actual=core_volume(x,np.eye(4),box,t,tmp_path)
    # Positions on either side of the eight-slice chunk boundary and source edges.
    for z in (0,7,8,15,16,20):
        lo=box[0].copy();hi=box[1].copy();lo[2]=z;hi[2]=z+1
        features=dense_features(x,lo,hi,np.eye(4),names).reshape(*(hi-lo),4)[:,:,0]
        reference=[]
        for c in range(4):
            plane=features[:,:,c]/(2015 if c==0 else 512)
            if c==0:plane=plane*2-1
            reference.append(resize_plane(plane,t['resize'],1,-1 if c==0 else 0))
        np.testing.assert_allclose(actual[z],np.stack(reference),atol=1e-3,rtol=1e-3)
    old=np.array(actual);del actual
    changed=core_volume(x+5,np.eye(4),box,t,tmp_path)
    assert np.any(changed!=old) and (tmp_path/'core_input.json').exists()


def test_HD95_exact_KD_and_EDT_agree_with_ignore_and_empty_surfaces():
    from ct_lab.roi_topology import surface_hd95
    p=np.zeros((31,29,27),bool);g=p.copy();valid=np.ones_like(p)
    p[4:20,5:18,6:16]=True;g[5:21,7:20,5:15]=True;valid[:6,:6,:6]=False
    kd,_=surface_hd95(p,g,valid,'KD');edt,_=surface_hd95(p,g,valid,'EDT')
    assert abs(kd-edt)<1e-12 and kd>0
    assert surface_hd95(np.zeros_like(p),g,valid)==(None,None)
