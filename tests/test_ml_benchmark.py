import json

import pytest
import torch

from heart3d.ml.benchmark import run, seed_everything, train_step
from heart3d.ml.losses import SegmentationLoss


def test_seed_repeats_random_initialization_and_optimizer_reduces_loss():
    seed_everything(23, 1)
    model = torch.nn.Conv2d(5, 1, 1)
    initial = model.weight.detach().clone()
    seed_everything(23, 1)
    assert torch.equal(initial, torch.nn.Conv2d(5, 1, 1).weight)
    batch = {'image': torch.ones(2, 5, 8, 8), 'target': torch.ones(2, 1, 8, 8)}
    loss_fn = SegmentationLoss('bce_dice')
    before = float(loss_fn(model(batch['image']), batch['target']).detach())
    optimizer = torch.optim.Adam(model.parameters(), lr=.03)
    for _ in range(10):
        loss, norm = train_step(model, optimizer, loss_fn, batch)
        assert norm > 0 and loss > 0
    after = float(loss_fn(model(batch['image']), batch['target']).detach())
    assert after < before
    assert not torch.equal(initial, model.weight)


def test_nonfinite_loss_is_rejected_before_optimizer_step():
    model = torch.nn.Conv2d(5, 1, 1)
    initial = model.weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters())
    batch = {'image': torch.full((1, 5, 8, 8), float('nan')), 'target': torch.zeros(1, 1, 8, 8)}
    with pytest.raises(ValueError, match='Non-finite'):
        train_step(model, optimizer, SegmentationLoss('bce'), batch)
    assert torch.equal(initial, model.weight)


@pytest.mark.parametrize('batches,run_name', [(201, 'valid'), (50, '..'), (50, '../overwrite')])
def test_benchmark_limits_before_any_data_access(batches, run_name):
    with pytest.raises(ValueError):
        run('missing.json', batches, run_name)


def test_configuration_cannot_turn_benchmark_into_long_training(tmp_path):
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'benchmark_warmup_batches': 5, 'sanity_batches': 10000}))
    with pytest.raises(ValueError, match='bounded'):
        run(config, 50, 'safe')
