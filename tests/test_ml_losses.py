import math
import pytest
import torch
from heart3d.ml.losses import soft_dice_loss,SegmentationLoss


def test_dice_known_value_and_bce_combination():
    logits=torch.zeros(1,1,2,2);target=torch.tensor([[[[1.,1.],[0.,0.]]]])
    expected=1-(2.+1e-6)/(4.+1e-6)
    assert float(soft_dice_loss(logits,target))==pytest.approx(expected)
    assert float(SegmentationLoss('bce')(logits,target))==pytest.approx(math.log(2))
    assert float(SegmentationLoss()(logits,target))==pytest.approx(math.log(2)+expected)


def test_empty_foreground_and_per_image_dice():
    empty=torch.zeros(1,1,4,4);certain_empty=torch.full_like(empty,-60.)
    assert float(soft_dice_loss(certain_empty,empty))==pytest.approx(0,abs=1e-5)
    positive=torch.ones_like(empty);certain_positive=torch.full_like(empty,60.)
    joined=soft_dice_loss(torch.cat([certain_empty,certain_positive]),torch.cat([empty,positive]))
    assert float(joined)==pytest.approx(0,abs=1e-5)
    assert float(soft_dice_loss(certain_positive,empty))>.99


@pytest.mark.parametrize('variant',['bce','dice','bce_dice'])
def test_losses_have_finite_nonzero_gradients(variant):
    logits=torch.zeros(2,1,8,8,requires_grad=True);target=torch.zeros_like(logits)
    target[:,:,2:6,2:6]=1
    loss=SegmentationLoss(variant)(logits,target);loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all() and logits.grad.abs().sum()>0


def test_loss_rejects_wrong_shape():
    with pytest.raises(ValueError):soft_dice_loss(torch.zeros(1,2,8,8),torch.zeros(1,2,8,8))
