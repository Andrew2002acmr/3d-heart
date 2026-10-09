import torch
import pytest
from heart3d.ml.model import HeartUNet25D
from heart3d.ml.losses import MulticlassSegmentationLoss,multiclass_dice_loss

def test_binary_default_and_multiclass_shape():
    torch.set_num_threads(2)
    assert HeartUNet25D()(torch.zeros(1,5,31,33)).shape==(1,1,31,33)
    model=HeartUNet25D(output_channels=8)
    assert model(torch.zeros(1,5,31,33)).shape==(1,8,31,33)
    assert sum(p.numel() for p in model.parameters())==489112

def test_multiclass_ignore_is_invariant_and_has_zero_gradient():
    target=torch.tensor([[[0,1],[2,255]]]);logits=torch.randn(1,3,2,2,requires_grad=True)
    criterion=MulticlassSegmentationLoss();a=criterion(logits,target)
    shifted=logits.detach().clone();shifted[:,:,-1,-1]=1000
    assert torch.allclose(a,criterion(shifted,target))
    a.backward();assert logits.grad[:,:,-1,-1].abs().sum()==0
    assert logits.grad[:,:,:1,:].abs().sum()>0

def test_perfect_dice_and_wrong_label_namespace():
    target=torch.tensor([[[0,1],[2,1]]])
    logits=torch.nn.functional.one_hot(target,3).permute(0,3,1,2).float()*40-20
    assert multiclass_dice_loss(logits,target)<1e-5
    with pytest.raises(ValueError):multiclass_dice_loss(logits,target+5)
    logits.requires_grad_()
    ignored=target*0+255
    loss=MulticlassSegmentationLoss()(logits,ignored)
    assert loss==0;loss.backward();assert logits.grad.abs().sum()==0


def test_noncardiac_signed_int8_labels_are_explicitly_ignored():
    import numpy as np
    from scripts.multiclass_sanity import cardiac_target
    assert np.array_equal(cardiac_target(np.array([0,1,7,14],dtype=np.int8)),[0,1,7,255])
