"""Full scratch baseline, enabled only by a successful matching CUDA benchmark."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from heart3d.ml.dataset import HeartDataset
from heart3d.ml.inference import predict_patient
from heart3d.ml.losses import SegmentationLoss
from heart3d.ml.metrics import overlap_metrics
from heart3d.ml.model import HeartUNet25D
from heart3d.ml.resources import ResourceMonitor, environment
from heart3d.ml.runtime import experiment_paths, provenance, revision, seed_runtime, step
from heart3d.pediatric import write_json
from heart3d.storage import require_space, sha256_file


def check_benchmark(path, config_path, device):
    receipt = json.loads(Path(path).read_text())
    if device != 'cuda' or not torch.cuda.is_available():
        raise ValueError('Full cloud baseline requires verified CUDA')
    if receipt['config_SHA256'] != sha256_file(config_path) or not receipt['sanity_passed'] or receipt['device'] != 'cuda':
        raise ValueError('A successful GPU benchmark of this exact config is required')
    if receipt['GPU_name'] != torch.cuda.get_device_name(0):
        raise ValueError('Benchmark GPU differs from training GPU')
    if receipt['git_revision'] != revision():
        raise ValueError('Code changed after the benchmark')


def validate(model, dataset, config, device):
    """All validation slices, patient mean on the fixed preprocessing grid."""
    records = []
    for row in dataset.rows:
        directory = dataset.root/config['cache_relative_path']/row['patient_id']
        probability, _ = predict_patient(model, directory, dataset.preproc['context_offsets_mm'], config['batch_size'], device)
        target = np.load(directory/'mask.npy', mmap_mode='r')
        try:
            records.append({'patient_id': row['patient_id'], **overlap_metrics(probability >= .5, target)})
        finally:
            target._mmap.close()
    return float(np.mean([r['Dice'] for r in records])), records


def train(config_path, run_name, benchmark_path=None, data_root=None, device='cuda', *, require_benchmark=True):
    config = json.loads(Path(config_path).read_text())
    if not config.get('full_training_authorized') or not 1 <= config['epochs'] <= 30:
        raise ValueError('Full baseline must be explicitly enabled and bounded to <=30 epochs')
    if require_benchmark:
        check_benchmark(benchmark_path, config_path, device)
    seed_runtime(config['seed'], config['cpu_threads'], device)
    root, output, checkpoints = experiment_paths(config, run_name, data_root)
    if output.exists() or checkpoints.exists():
        raise ValueError('Run already exists; no implicit resume or overwriting')
    require_space(root, 1_000_000_000, int(config['reserve_GB']*1e9))
    dataset = HeartDataset(config_path, 'train', data_root)
    validation = HeartDataset(config_path, 'validation', data_root, augmentation=False)
    # Dataset epoch-dependent indices/augmentation stay in the main process.
    if config['num_workers'] != 0:
        raise ValueError('Baseline v1 uses workers=0 for explicit epoch state')
    model = HeartUNet25D(**config['model']).to(device)  # From scratch, never from benchmark weights.
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = SegmentationLoss(**config['loss'])
    info = provenance(config_path, config)
    output.mkdir(parents=True); checkpoints.mkdir(parents=True)
    env = environment(device); env['device'] = device
    if device == 'cuda':
        env.update(GPU_name=torch.cuda.get_device_name(0), CUDA_version=torch.version.cuda,
                   GPU_total_bytes=torch.cuda.get_device_properties(0).total_memory,
                   TF32=False, precision='FP32', deterministic_cuda='warn_only')
    write_json(output/'config.json', config); write_json(output/'environment.json', env)
    write_json(output/'run_provenance.json', {**info, 'benchmark_SHA256': sha256_file(benchmark_path) if benchmark_path else None,
               'initialization': 'scratch, seeded random weights; benchmark checkpoint not loaded',
               'checkpoint_selection': 'maximum patient mean validation Dice on preprocessing grid',
               'test_used_during_training': False})
    pre = json.loads(Path(config['preprocessing']).read_text())
    history = []; best = -1.; start = time.perf_counter(); monitor = ResourceMonitor()
    with monitor:
        for epoch in range(config['epochs']):
            epoch_start = time.perf_counter(); dataset.set_epoch(epoch)
            loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0,
                                generator=torch.Generator().manual_seed(config['seed']+epoch))
            total = 0.; count = 0
            for batch in loader:
                value, _ = step(model, optimizer, criterion, batch, device)
                total += value*len(batch['image']); count += len(batch['image'])
            training_seconds = time.perf_counter()-epoch_start
            validation_start = time.perf_counter()
            score, records = validate(model, validation, config, device)
            record = {'epoch': epoch+1, 'train_loss': total/count, 'validation_mean_patient_Dice': score,
                      'train_seconds': training_seconds, 'validation_seconds': time.perf_counter()-validation_start,
                      'samples': count, 'learning_rate': optimizer.param_groups[0]['lr'], 'validation_patients': records}
            history.append(record)
            is_best = score > best
            best = max(best, score)
            checkpoint = {**info, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                          'torch_rng_state': torch.get_rng_state(), 'config': config, 'preprocessing': pre,
                          'epoch': epoch+1, 'best_validation_Dice': best, 'history': history,
                          'training_scope': 'full_scratch_baseline'}
            if is_best:
                torch.save(checkpoint, checkpoints/'best.pt')
            torch.save(checkpoint, checkpoints/'last.pt')
            write_json(output/'history.json', history)
            print('EPOCH', epoch+1, 'loss', round(record['train_loss'],5), 'validation_Dice', round(score,5),
                  'train_sec', round(training_seconds,2), 'val_sec', round(record['validation_seconds'],2), flush=True)
    result = {**info, 'created_utc': datetime.now(timezone.utc).isoformat(), 'epochs': len(history),
              'seconds': time.perf_counter()-start, 'best_validation_Dice_preprocessing_grid': best,
              'best_epoch': int(np.argmax([r['validation_mean_patient_Dice'] for r in history]))+1,
              'resources': monitor.summary(), 'test_evaluated': False,
              'checkpoints': {name: {'bytes': (checkpoints/name).stat().st_size,
                                   'SHA256': sha256_file(checkpoints/name)} for name in ('best.pt','last.pt')}}
    write_json(output/'training_summary.json', result)
    write_json(output/'resource_samples.json', monitor.samples)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True); p.add_argument('--run-name', required=True)
    p.add_argument('--benchmark', type=Path, required=True); p.add_argument('--data', type=Path)
    a = p.parse_args(); train(a.config, a.run_name, a.benchmark, a.data)
