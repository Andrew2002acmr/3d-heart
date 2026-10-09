#!/usr/bin/env bash
# Verify a minimal public-data bundle and run a bounded GPU benchmark only.
set -euo pipefail
ARCHIVE="${1:?usage: bash scripts/runpod_cardiac_benchmark.sh ARCHIVE RECEIPT [DATA_ROOT] [RUN_NAME]}"
RECEIPT="${2:?archive receipt is required}"
DATA_ROOT="${3:-/workspace/cardiac_chd68_v1}"
RUN_NAME="${4:-gpu_benchmark_v1}"
CONFIG="configs/cardiac_chd68_runpod_v1.json"
test "$(git branch --show-current)" = "feature/pediatric-heart-segmentation"
test -z "$(git status --porcelain)"
git rev-parse HEAD
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA'); assert torch.cuda.is_available(); assert '4090' in torch.cuda.get_device_name(0)"
df -h /workspace
findmnt -T /workspace
mountpoint -q /workspace || { echo '/workspace is not a separate mounted persistent volume'; exit 1; }
nproc
free -h
export TMPDIR=/workspace/tmp/cardiac_chd68_v1
export PIP_CACHE_DIR=/workspace/cache/pip
export PYTHONUTF8=1
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"
python -m venv --system-site-packages /workspace/venvs/cardiac_chd68_v1
source /workspace/venvs/cardiac_chd68_v1/bin/activate
python -m pip install -r requirements-runpod.txt
python - "$ARCHIVE" "$RECEIPT" <<'PY'
import hashlib,json,sys
from pathlib import Path
archive=Path(sys.argv[1]);receipt=json.loads(Path(sys.argv[2]).read_text())
h=hashlib.sha256()
with archive.open('rb') as stream:
    for block in iter(lambda: stream.read(4<<20),b''): h.update(block)
assert archive.stat().st_size==receipt['archive_bytes']
assert h.hexdigest()==receipt['archive_SHA256'], 'Transferred archive SHA-256 mismatch'
print('ARCHIVE_SHA256_VERIFIED')
PY
python -m heart3d.ml.bundle unpack --archive "$ARCHIVE" --destination "$DATA_ROOT" --reserve-GB 10
python -m heart3d.ml.bundle verify --root "$DATA_ROOT" --manifest "$DATA_ROOT/bundle_manifest.json" --repo .
python -m heart3d.ml.cardiac_train benchmark --config "$CONFIG" --data "$DATA_ROOT" --output "$DATA_ROOT/experiments/cardiac_chd68_v1/$RUN_NAME" --batches 50
echo 'GPU benchmark completed. Full training was not started.'
echo 'Inspect the benchmark receipt before running the separately documented train command.'
