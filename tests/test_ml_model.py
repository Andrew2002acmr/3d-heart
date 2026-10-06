import pytest
import torch

from heart3d.ml.benchmark import seed_everything, train_step
from heart3d.ml.losses import SegmentationLoss
from heart3d.ml.model import HeartUNet25D


def test_own_model_preserves_center_target_shape_and_has_expected_parameters():
    seed_everything(20261006, 1)
    model = HeartUNet25D()
    assert sum(p.numel() for p in model.parameters()) == 488993
    for shape in [(2, 5, 64, 64), (1, 5, 67, 73)]:
        with torch.no_grad():
            logits = model(torch.randn(shape))
        assert logits.shape == (shape[0], 1, shape[2], shape[3])
        assert torch.isfinite(logits).all()


def test_own_model_backward_and_optimizer_update_with_real_segmentation_loss():
    seed_everything(20261006, 1)
    model = HeartUNet25D()
    before = model.encoder[0].layers[0].weight.detach().clone()
    batch = {'image': torch.randn(2, 5, 32, 32), 'target': torch.zeros(2, 1, 32, 32)}
    batch['target'][:, :, 10:22, 10:22] = 1
    loss, norm = train_step(model, torch.optim.Adam(model.parameters(), lr=.001),
                            SegmentationLoss('bce_dice'), batch)
    assert loss > 0 and norm > 0
    assert not torch.equal(before, model.encoder[0].layers[0].weight)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_invalid_model_configuration_or_input_is_rejected():
    with pytest.raises(ValueError):
        HeartUNet25D(encoder_channels=[16, 32, 64])
    with pytest.raises(ValueError):
        HeartUNet25D(encoder_channels=[15, 32, 64, 128])
    with pytest.raises(ValueError):
        HeartUNet25D()(torch.zeros(1, 3, 32, 32))
