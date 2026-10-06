"""Bounded actual CUDA benchmark and learning sanity before scratch training."""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from heart3d.ml.dataset import HeartDataset
from heart3d.ml.losses import SegmentationLoss
from heart3d.ml.model import HeartUNet25D
from heart3d.ml.resources import ResourceMonitor, environment
from heart3d.ml.runtime import seed_runtime, step, provenance
from heart3d.pediatric import write_json


def benchmark(config_path, output, data_root=None, batches=50):
    if not 1 <= batches <= 200:
        raise ValueError('GPU benchmark limited to 1..200 measured batches')
    config = json.loads(Path(config_path).read_text()); output = Path(output)
    if output.exists():
        raise ValueError('Benchmark output exists')
    seed_runtime(config['seed'], config['cpu_threads'], 'cuda')
    if '4090' not in torch.cuda.get_device_name(0):
        raise ValueError('The authorized RTX 4090 was not found')
    train = HeartDataset(config_path, 'train', data_root)
    sanity = HeartDataset(config_path, 'train', data_root, augmentation=False)
    loader = DataLoader(train, batch_size=config['batch_size'], shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(config['seed']))
    warmup = 5
    if batches+warmup > len(loader):
        raise ValueError('Benchmark exceeds one training epoch')
    model = HeartUNet25D(**config['model']).cuda()
    criterion = SegmentationLoss(**config['loss'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    patients = list(sanity.provenance)[:2]
    indices = [sanity.indices.index((pid, sanity.provenance[pid]['positive_indices'][len(sanity.provenance[pid]['positive_indices'])//2])) for pid in patients]
    fixed = default_collate([sanity[i] for i in indices])
    initial = [p.detach().clone() for p in model.parameters()]
    def fixed_loss():
        model.eval()
        with torch.no_grad():
            return float(criterion(model(fixed['image'].cuda()), fixed['target'].cuda()))
    before = fixed_loss(); timings = []; history = []; monitor = ResourceMonitor()
    torch.cuda.reset_peak_memory_stats(); iterator = iter(loader)
    with monitor:
        for i in range(batches+warmup):
            torch.cuda.synchronize(); start = time.perf_counter()
            batch = next(iterator)
            loss, norm = step(model, optimizer, criterion, batch, 'cuda')
            torch.cuda.synchronize(); elapsed = time.perf_counter()-start
            history.append({'batch': i, 'loss': loss, 'gradient_norm': norm, 'seconds': elapsed,
                            'patient_ids': batch['patient_id'], 'warmup': i < warmup})
            if i >= warmup: timings.append(elapsed)
            if (i+1)%10 == 0: print('GPU_BENCHMARK', i+1, round(elapsed,4), round(loss,5), flush=True)
        for _ in range(20): step(model, optimizer, criterion, fixed, 'cuda')
        after = fixed_loss()
    delta = sum(float((p.detach()-old).abs().sum()) for p,old in zip(model.parameters(),initial))
    mean = float(np.mean(timings))
    result = {**provenance(config_path, config), 'device': 'cuda', 'GPU_name': torch.cuda.get_device_name(0),
              'environment': environment('cuda'), 'CUDA_version': torch.version.cuda, 'precision': 'FP32',
              'measured_batches': batches, 'warmup_batches': warmup, 'batch_size': config['batch_size'],
              'mean_sec_per_batch': mean, 'p95_sec_per_batch': float(np.quantile(timings,.95)),
              'samples_per_second': config['batch_size']/mean,
              'GPU_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'GPU_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
              'resources': monitor.summary(), 'fixed_train_patient_ids': patients,
              'fixed_loss_before': before, 'fixed_loss_after': after, 'weights_absolute_delta': delta,
              'sanity_passed': bool(after < before and delta > 0),
              'estimated_train_epoch_sec': math.ceil(len(train)/config['batch_size'])*mean,
              'estimate_excludes': 'validation and checkpoint I/O',
              'training_scope': 'train-only technical benchmark, no scientific test metrics'}
    output.mkdir(parents=True); write_json(output/'history.json', history); write_json(output/'summary.json', result)
    print('GPU_SUMMARY', json.dumps(result), flush=True)
    if not result['sanity_passed']: raise ValueError('GPU learning sanity failed; full training forbidden')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data', type=Path); p.add_argument('--batches', type=int, default=50)
    a = p.parse_args(); benchmark(a.config, a.output, a.data, a.batches)
