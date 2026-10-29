#!/usr/bin/env bash
# Optional reproducible execution on an already provisioned pod. Does NOT create/bill a pod.
set -euo pipefail
: "${ROI05_DATA_ROOT:?Set directory containing the SHA-verified public train/development NIfTI pairs}"
: "${ROI05_WEIGHTS_ROOT:?Set directory containing immutable chd68_v1_best.pt and chd68_v2_best.pt}"
python -c "import torch; assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name(0), 'This script is for a pre-provisioned RTX4090'; print(torch.cuda.get_device_name(0))"
# Run from the existing repository checkout with scientific/CT Lab dependencies installed.
# Use a fresh outputs/heart_localization_roi_05 for independent reproduction; never delete old experiments.
python -m ct_lab.roi_analysis
python -m ct_lab.roi_smoke
python -m ct_lab.roi_localizer
python -m ct_lab.roi_train
python -m ct_lab.roi_baselines
python -m ct_lab.roi_topology
python -m ct_lab.roi_analysis
python -m ct_lab.roi_report
