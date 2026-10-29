#!/usr/bin/env bash
# MANUAL command on an existing pod. No provider API, no paid infrastructure creation.
set -euo pipefail
: "${DYNAMICS06_DATA_ROOT:?Set58 allowed train/development public NIfTI pairs directory}"
: "${DYNAMICS06_WEIGHTS_ROOT:?Set immutable historical v1/v2 checkpoints directory}"
python -c "import torch; assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name(0); print(torch.cuda.get_device_name(0))"
# Run from the existing repository with scientific dependencies installed.
# Copy local completed stage06 outputs to reuse baseline, or use fresh outputs in a separate checkout.
python -m ct_lab.dynamics_audit
python -m ct_lab.dynamics_data
python -m ct_lab.dynamics_smoke
python -m ct_lab.dynamics_suite --prepare-only
# Defaults to baseline seeds only; choose sampling/data/losses/all deliberately.
python -m ct_lab.dynamics_suite --group "${DYNAMICS06_GROUP:-baseline}"
python -m ct_lab.dynamics_topology --all-runs
