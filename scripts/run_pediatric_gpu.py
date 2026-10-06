"""Verified bundle -> CUDA benchmark -> one scratch baseline -> one test evaluation."""
import argparse
import json
from pathlib import Path
import subprocess
from heart3d.ml.bundle import pack, verify
from heart3d.ml.evaluate import evaluate
from heart3d.ml.gpu_benchmark import benchmark
from heart3d.ml.runtime import experiment_paths
from heart3d.ml.train import train
from heart3d.pediatric import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True); p.add_argument('--run-name', required=True)
    p.add_argument('--export', type=Path, required=True); a = p.parse_args()
    config = json.loads(a.config.read_text())
    root, output, checkpoints = experiment_paths(config, a.run_name, a.data)
    verify(root, root/'bundle_manifest.json', Path.cwd())
    receipt_dir = root/config['outputs_relative_path']/(a.run_name+'_gpu_benchmark')
    benchmark(a.config, receipt_dir, root, batches=50)
    train(a.config, a.run_name, receipt_dir/'summary.json', root)
    # Hyperparameters and threshold are already fixed. Test is evaluated once.
    evaluate(a.config, checkpoints/'best.pt', output/'test', 'test', root, 'cuda')
    freeze = subprocess.check_output(['python','-m','pip','freeze'], text=True)
    (output/'pip_freeze.txt').write_text(freeze, encoding='utf-8')
    paths = [p.relative_to(root).as_posix() for directory in (output, checkpoints, receipt_dir)
             for p in directory.rglob('*') if p.is_file()]
    a.export.mkdir(parents=True, exist_ok=True)
    pack(root, paths, a.export/(a.run_name+'_results.tar.gz'), a.export/(a.run_name+'_manifest.json'), reserve_bytes=int(10e9))
    write_json(a.export/(a.run_name+'_complete.json'), {'training_completed': True,
               'test_evaluated_once': True, 'results_packed': True,
               'local_download_and_verification_still_required': True})
    print('RUN_COMPLETE', a.run_name, flush=True)


if __name__ == '__main__': main()
