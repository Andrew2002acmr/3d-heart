#!/usr/bin/env bash
# Authorized full run AFTER a matching GPU benchmark; validation only, no test.
set -euo pipefail
REPO="${1:?repository under /workspace}"
DATA_ROOT="${2:?data root under /workspace}"
VENV="${3:?CUDA venv under /workspace}"
RUN_NAME="${4:?new unique run name}"
BENCHMARK="${5:?matching successful GPU benchmark summary}"
EXPECTED_COMMIT="${6:?pinned full commit SHA}"
PREFIX="${7:?new absolute export prefix under /workspace}"
python - "$REPO" "$DATA_ROOT" "$VENV" "$BENCHMARK" "$PREFIX" <<'PY'
from pathlib import Path
import sys
for value in sys.argv[1:]:
    assert Path(value).is_absolute() and Path(value).resolve().is_relative_to(Path("/workspace")), "All paths must stay under /workspace"
PY
cd "$REPO"
export TMPDIR=/workspace/tmp/cardiac_chd68_resolution384_v2
export PIP_CACHE_DIR=/workspace/cache/pip
export PYTHONUTF8=1
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$(dirname "$PREFIX")"
source "$VENV/bin/activate"
state() {
  python - "$PREFIX" "$1" "$2" "$EXPECTED_COMMIT" <<'STATE'
import datetime,json,os,pathlib,sys
p=pathlib.Path(sys.argv[1]+"_status.json")
s={"stage":sys.argv[2],"detail":sys.argv[3],"updated_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
   "execution_commit":sys.argv[4],"full_training_explicitly_authorized":True,"test_evaluated":False}
t=p.with_suffix(".tmp");t.write_text(json.dumps(s,indent=2)+"\n");os.replace(t,p)
STATE
}
exec 9>"$PREFIX.lock"
flock -n 9 || { echo "Another runner owns this export prefix"; exit 1; }
test ! -e "$PREFIX_status.json" || { echo "Export status exists; inspect it and use a new prefix"; exit 1; }
trap 'rc=$?; state failed "exit code $rc; preserve checkpoints and inspect log"; exit "$rc"' ERR
test "$(git branch --show-current)" = feature/pediatric-heart-segmentation
test -z "$(git status --porcelain)"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
mountpoint -q /workspace
python -c "import torch; assert torch.cuda.is_available(); assert '4090' in torch.cuda.get_device_name(0)"
CONFIG=configs/cardiac_chd68_resolution384_v2.json
BASELINE_CKPT="$(python - "$CONFIG" "$DATA_ROOT" "$RUN_NAME" "$PREFIX" <<'PY'
from pathlib import Path
import sys
from heart3d.ml.cardiac_data import read_json
from heart3d.ml.runtime import experiment_paths
from heart3d.storage import sha256_file
config=read_json(sys.argv[1]);root,run,checkpoints=experiment_paths(config,sys.argv[3],sys.argv[2]);prefix=Path(sys.argv[4])
assert config["allowed_partitions"]==["train","validation"]
for path in (run,checkpoints,root/"evaluation"/(sys.argv[3]+"_baseline_validation"),root/"evaluation"/(sys.argv[3]+"_validation"),root/"evaluation"/(sys.argv[3]+"_comparison.json"),Path(str(prefix)+"_results.tar.gz"),Path(str(prefix)+"_results_manifest.json")):
    assert not path.exists(), f"Do not overwrite existing result: {path}"
p=read_json(root/config["cache_relative_path"]/"prepared_manifest.json")
ref=p["validation_reference"];checkpoint=root/ref["checkpoint_relative_path"]
assert sha256_file(checkpoint)==ref["checkpoint_SHA256"]
print(checkpoint)
PY
)"
state training "30 epochs from scratch; frozen 48train/10validation; sealed test"
python -u -m heart3d.ml.cardiac_train train --config "$CONFIG" --data "$DATA_ROOT" --run-name "$RUN_NAME" --benchmark "$BENCHMARK" --allow-full-training
state baseline_validation "v1 best checkpoint23 on original-grid validation; no test"
python -u -m heart3d.ml.cardiac_train evaluate --config configs/cardiac_chd68_runpod_v1.json --data "$DATA_ROOT" --checkpoint "$BASELINE_CKPT" --partition validation --output "$DATA_ROOT/evaluation/${RUN_NAME}_baseline_validation"
state validation "v2 best validation checkpoint on same original-grid validation"
python -u -m heart3d.ml.cardiac_train evaluate --config "$CONFIG" --data "$DATA_ROOT" --checkpoint "$DATA_ROOT/checkpoints/cardiac_chd68_resolution384_v2/$RUN_NAME/best.pt" --partition validation --output "$DATA_ROOT/evaluation/${RUN_NAME}_validation"
python -m scripts.compare_cardiac_validation --baseline "$DATA_ROOT/evaluation/${RUN_NAME}_baseline_validation/metrics.json" --variant "$DATA_ROOT/evaluation/${RUN_NAME}_validation/metrics.json" --output "$DATA_ROOT/evaluation/${RUN_NAME}_comparison.json"
state export "allowlisted checkpoints, histories, environments, validation predictions/metrics"
python - "$DATA_ROOT" "$RUN_NAME" "$REPO" "$PREFIX" "$BENCHMARK" <<'PACK'
from pathlib import Path
import shutil,sys
from heart3d.ml.bundle import pack
workspace=Path("/workspace");data=Path(sys.argv[1]);name=sys.argv[2];repo=Path(sys.argv[3]);prefix=Path(sys.argv[4]);benchmark=Path(sys.argv[5])
paths=[]
for directory in (data/"experiments/cardiac_chd68_resolution384_v2"/name,data/"checkpoints/cardiac_chd68_resolution384_v2"/name,data/"evaluation"/(name+"_baseline_validation"),data/"evaluation"/(name+"_validation")):
    paths.extend(p.relative_to(workspace).as_posix() for p in directory.rglob("*") if p.is_file())
paths.append((data/"evaluation"/(name+"_comparison.json")).relative_to(workspace).as_posix())
paths.extend(p.relative_to(workspace).as_posix() for p in benchmark.parent.glob("*.json"))
for filename in ("configs/cardiac_chd68_resolution384_v2.json","configs/cardiac_chd68_preprocessing_resolution384_v2.json","configs/cardiac_chd68_runpod_v1.json","configs/cardiac_chd68_preprocessing_v1.json","configs/splits/cardiac_chd68_v1.json","metadata/pediatric/cardiac_chd68_cohort_v1.json"):
    paths.append((repo/filename).relative_to(workspace).as_posix())
snapshot=Path(str(prefix)+"_compute.log");shutil.copyfile(Path(str(prefix)+".log"),snapshot);paths.append(snapshot.relative_to(workspace).as_posix())
pack(workspace,paths,Path(str(prefix)+"_results.tar.gz"),Path(str(prefix)+"_results_manifest.json"),reserve_bytes=10_000_000_000)
PACK
state completed "training, original-grid paired validation comparison and SHA export complete; awaiting verified local backup"
echo RESOLUTION384_FULL_RUN_VALIDATION_EXPORT_COMPLETED
