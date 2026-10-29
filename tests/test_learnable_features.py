"""Feature encoder tests, not training or changing any production segmentator."""
import inspect
import numpy as np
import pytest
import torch
from ct_lab.learnable_core import (FeatureEncoder,supcon_loss,spatial_loss,admissible_neighbors,class_indices,
                                   fit_scaler,normalize,seed_all,load_encoder,SETS,REPO)
from ct_lab.learnable_data import freeze,extract_case
from ct_lab.learnable_train import balanced_pools
from ct_lab.core import fingerprint


def test_supcon_positive_negative_mask_and_gradients():
    z=torch.tensor([[1.,0],[1.,0],[0,1.],[0,1.]],requires_grad=True);y=torch.tensor([0,0,1,1])
    loss=supcon_loss(z,y,.1)
    assert loss.item()==pytest.approx(np.log1p(2*np.exp(-10)),abs=1e-6)
    loss.backward();assert torch.isfinite(z.grad).all()
    assert supcon_loss(z,torch.tensor([0,1,0,1])).item()>loss.item()+1
    assert supcon_loss(z,torch.arange(4)).item()==0
    assert supcon_loss(z[:1],y[:1]).item()==0


def test_true_spatial_pairs_never_cross_classes_or_replicate_self():
    mask=np.ones((4,5,6),np.uint8);mask[2:]=5;mask[0,0,0]=7
    xyz=np.array([[1,2,3],[3,4,5],[0,0,0]]);labels=mask[tuple(xyz.T)]
    points,valid=admissible_neighbors(xyz,labels,mask,np.random.default_rng(4))
    assert valid.tolist()==[True,True,False]
    np.testing.assert_equal(np.abs(points[valid]-xyz[valid]).sum(1),1)
    np.testing.assert_equal(mask[tuple(points[valid].T)],labels[valid])
    np.testing.assert_equal(points[~valid],-1)
    same,flags=admissible_neighbors(xyz,labels,mask,np.random.default_rng(4));np.testing.assert_equal(same,points)
    assert spatial_loss(torch.ones(2,8),torch.ones(2,8)).item()==0


def test_architecture_activations_class_mapping_and_future_multiclass():
    seed_all(17);model=FeatureEncoder(8,16,2);x=torch.randn(5,8)
    logits,z=model(x);trace=model(x,trace=True)
    assert trace['linear1'].shape==(5,64) and trace['linear2'].shape==(5,32)
    torch.testing.assert_close(logits,trace['logits']);torch.testing.assert_close(z,trace['embedding'])
    torch.testing.assert_close(trace['probabilities'].sum(1),torch.ones(5))
    np.testing.assert_equal(class_indices([1,5,1],[1,5]),[0,1,0])
    with pytest.raises(ValueError):class_indices([7],[1,5])
    assert FeatureEncoder(5,8,8)(torch.randn(2,5))[0].shape==(2,8)
    assert set(inspect.signature(model.forward).parameters)=={'x','trace'}
    assert not any('GT' in n or 'distance' in n.lower() or 'label' in n.lower() for names in SETS.values() for n in names)


def test_train_only_normalization_and_seed():
    train=np.array([[0,3],[2,3]],np.float32);scaler=fit_scaler(train,['train_case'])
    np.testing.assert_equal(scaler['mean'],[1,3]);np.testing.assert_equal(scaler['scale'],[1,1])
    np.testing.assert_equal(normalize([[1000,3]],scaler),[[999,0]])
    seed_all(99);a=FeatureEncoder(5,8).state_dict();seed_all(99);b=FeatureEncoder(5,8).state_dict()
    for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
    pools=balanced_pools(np.array([0,0,0,0]),np.array(['a','b','b','b']))
    indices,w=pools[0];assert w[0]==pytest.approx(.5) and w[1:].sum()==pytest.approx(.5)


def test_checkpoint_strict_loading_and_frozen_split(tmp_path):
    plan,_=freeze();model=FeatureEncoder(5,8)
    saved={'format':'learnable_feature_03_v1','state_dict':model.state_dict(),'features':SETS['C 3D-core'],
           'classes':[1,5],'embedding_dim':8,'scaler':fit_scaler(np.ones((3,5)),plan['train_cases']),
           'training':{'train_cases':plan['train_cases'],'validation_cases':plan['validation_cases']},'split_sha256':plan['split_sha256']}
    path=tmp_path/'encoder.pt';torch.save(saved,path);loaded,_=load_encoder(path,SETS['C 3D-core'])
    x=torch.randn(2,5);torch.testing.assert_close(model(x)[0],loaded(x)[0])
    with pytest.raises(ValueError):load_encoder(path,['wrong'])
    saved['scaler']['fit_partition']='validation';torch.save(saved,path)
    with pytest.raises(ValueError,match='scaler'):load_encoder(path)


def test_split_and_validation_neighbor_gates():
    plan,records=freeze()
    assert len(plan['train_cases'])==48 and len(plan['validation_cases'])==10
    assert not {records[c]['group_id'] for c in plan['train_cases']}&{records[c]['group_id'] for c in plan['validation_cases']}
    with pytest.raises(ValueError):extract_case(plan['validation_cases'][0],'train',{},plan)
    with pytest.raises(ValueError):extract_case('ct_1022','test',{},plan)


def test_perturbations_match_per_voxel_without_modifying_features():
    from ct_lab.learnable_diagnostics import perturb
    features=SETS['C 3D-core'];x=np.ones((2,5),np.float32)
    a=perturb(x,'Gaussian .05 SD',features,['a','b'],[10,20]);b=perturb(x[::-1],'Gaussian .05 SD',features,['b','a'],[20,10])
    np.testing.assert_equal(a,b[::-1]);np.testing.assert_equal(x,1)
    shifted=perturb(x,'Intensity +.1 SD',features,['a','b'],[10,20])
    np.testing.assert_allclose(shifted[:,0],1.1);np.testing.assert_equal(shifted[:,1:],1)
    dropped=perturb(x,'Drop 3D LoG',features,['a','b'],[10,20]);np.testing.assert_equal(dropped[:,-1],0)


def test_real_cache_no_validation_neighbors_and_source_ids():
    from ct_lab.learnable_core import OUT
    import json
    if not (OUT/'data/ct_1027.npz').exists():pytest.skip('Real data extraction not completed')
    for case,role in [('ct_1001','train'),('ct_1027','validation')]:
        info=json.loads((OUT/'data'/f'{case}.json').read_text())
        assert info['training_neighbor_pairs_built']==(role=='train')
        with np.load(OUT/'data'/f'{case}.npz') as data:
            for p in (0,1):
                xyz=data[f'p{p}_xyz'];np.testing.assert_equal(np.ravel_multi_index(xyz.T,info['shape_xyz']),data[f'p{p}_voxel_ids'])
                assert np.isfinite(data[f'p{p}_features']).all()
                if role=='validation':assert not any('neighbor' in k for k in data.files)
                else:
                    valid=data[f'p{p}_neighbor_valid'];neighbor=data[f'p{p}_neighbor_xyz'][valid]
                    np.testing.assert_equal(np.abs(neighbor-xyz[valid]).sum(1),1)
