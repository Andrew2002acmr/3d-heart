"""Shared device, provenance and checkpoint rules for the scratch baseline."""
import os
import random
import re
import subprocess
from pathlib import Path
import numpy as np
import torch
from heart3d.ml.model import HeartUNet25D
from heart3d.storage import sha256_file


def seed_runtime(seed, threads, device):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if str(device).startswith('cuda'):
        if not torch.cuda.is_available():
            raise ValueError('CUDA unavailable; GPU experiment must not fall back to CPU')
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    # Bilinear CUDA backward may lack a deterministic implementation in some
    # torch versions. Warn explicitly; do not claim bitwise GPU reproducibility.
    torch.use_deterministic_algorithms(True, warn_only=str(device).startswith('cuda'))


def revision():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()


def experiment_paths(config, run_name, data_root=None):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', run_name) or '..' in run_name:
        raise ValueError('Run name must be a safe directory component')
    root = Path(data_root or config['data_root'])
    return root, root/config['outputs_relative_path']/run_name, root/config['checkpoint_relative_path']/run_name


def provenance(config_path, config):
    return {'git_revision': revision(), 'config_SHA256': sha256_file(config_path),
            'split_SHA256': sha256_file(config['split']),
            'preprocessing_SHA256': sha256_file(config['preprocessing']),
            'cohort_SHA256': sha256_file(config['cohort']), 'model_version': 'own_25d_unet_v1'}


def load_model(checkpoint_path, config, device):
    # Only use project-produced checkpoints with independently verified SHA.
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if checkpoint['split_SHA256'] != sha256_file(config['split']) or checkpoint['preprocessing_SHA256'] != sha256_file(config['preprocessing']):
        raise ValueError('Checkpoint belongs to a different split/preprocessing')
    if checkpoint['config']['model'] != config['model']:
        raise ValueError('Checkpoint architecture mismatch')
    model = HeartUNet25D(**config['model']).to(device)
    model.load_state_dict(checkpoint['model'])
    return model, checkpoint


def step(model, optimizer, criterion, batch, device):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    image = batch['image'].to(device)
    target = batch['target'].to(device)
    loss = criterion(model(image), target)
    if not torch.isfinite(loss):
        raise ValueError('Non-finite training loss')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
    if norm <= 0:
        raise ValueError('Missing or zero gradients')
    optimizer.step()
    return float(loss.detach()), float(norm)
