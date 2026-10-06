import json
from pathlib import Path
import numpy as np
import pytest
import torch
from test_ml_dataset import toy_experiment
from heart3d.ml.bundle import pack, unpack, verify, safe_relative
from heart3d.ml.metrics import overlap_metrics, physical_metrics, aggregate
from heart3d.ml.runtime import load_model
from heart3d.ml.train import train, check_benchmark
from heart3d.pediatric import write_json


def test_metrics_empty_and_physical_surface_distance():
    p = np.zeros((20,20,20), bool); t = p.copy()
    p[5:10,5:10,5:10] = True; t[7:12,5:10,5:10] = True
    result = physical_metrics(p,t,[3,1,1],True)
    assert result['Dice'] == .6 and result['HD95_mm'] == 6
    assert 0 < result['ASSD_mm'] <= 6
    exact = physical_metrics(p,p,[3,1,1],True)
    assert exact['Dice'] == 1 and exact['HD95_mm'] == exact['ASSD_mm'] == 0
    empty = physical_metrics(p, np.zeros_like(p), [3,1,1], True)
    assert empty['Dice'] == 0 and empty['HD95_mm'] is None
    assert overlap_metrics(np.zeros_like(p),np.zeros_like(p))['Dice'] == 1
    with pytest.raises(ValueError,match='confirmed'): physical_metrics(p,t,[3,1,1],False)
    assert aggregate([result,exact])['all']['Dice']['mean'] == .8


def test_bundle_verified_roundtrip_and_refuses_corruption(tmp_path):
    source = tmp_path/'source'; source.mkdir(); (source/'sample.json').write_text('{"ok":true}')
    archive = tmp_path/'bundle.tar.gz'; manifest = tmp_path/'manifest.json'
    pack(source,['sample.json'],archive,manifest)
    destination = tmp_path/'destination'; unpack(archive,destination)
    assert (destination/'sample.json').read_bytes() == (source/'sample.json').read_bytes()
    (destination/'sample.json').write_text('corrupt')
    with pytest.raises(ValueError,match='integrity'): verify(destination,destination/'bundle_manifest.json')
    with pytest.raises(ValueError,match='existing'): unpack(archive,destination)
    for value in ('../x','/root/x','C:/key','a\\b'):
        with pytest.raises(ValueError,match='Unsafe'): safe_relative(value)


def test_full_trainer_scratch_best_last_and_checkpoint_provenance(toy_experiment):
    path,split,cohort = toy_experiment
    config = json.loads(path.read_text()); config.update(
        epochs=2, full_training_authorized=True, reserve_GB=0,
        outputs_relative_path='outputs', checkpoint_relative_path='checkpoints',
        model={'input_channels':5,'encoder_channels':[4,8,16,32],'norm_groups':4},
        batch_size=2,cpu_threads=1,num_workers=0,learning_rate=.001,
        loss={'variant':'bce_dice'})
    write_json(path,config)
    result = train(path,'unit',device='cpu',require_benchmark=False)
    assert result['epochs'] == 2 and not result['test_evaluated']
    root = Path(config['data_root'])
    best = root/'checkpoints/unit/best.pt'; last = root/'checkpoints/unit/last.pt'
    model, checkpoint = load_model(best,config,'cpu')
    assert checkpoint['training_scope'] == 'full_scratch_baseline'
    assert torch.load(last,weights_only=True)['epoch'] == 2
    assert 1 <= checkpoint['epoch'] <= 2
    history = json.loads((root/'outputs/unit/history.json').read_text())
    assert {r['patient_id'] for h in history for r in h['validation_patients']} == {'p2','p3'}
    with pytest.raises(ValueError,match='exists'): train(path,'unit',device='cpu',require_benchmark=False)
    with pytest.raises(ValueError,match='CUDA'): check_benchmark(path,path,'cpu')
