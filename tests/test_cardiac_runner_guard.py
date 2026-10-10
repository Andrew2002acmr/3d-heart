"""Exercise the real Bash guard before any training or failure-state mutation."""
from pathlib import Path
import shutil
import subprocess
import pytest


def test_existing_run_status_is_preserved(tmp_path):
    bash = Path("C:/Program Files/Git/bin/bash.exe")
    if not bash.exists():
        located = shutil.which("bash")
        if located is None:
            pytest.skip("Bash unavailable")
        bash = Path(located)
    source = Path("scripts/runpod_cardiac_resolution_full.sh").read_text()
    guard = next(line for line in source.splitlines() if line.startswith("test ! -e "))
    assert source.index(guard) < source.index("trap '")
    prefix = tmp_path / "run"
    status = tmp_path / "run_status.json"
    status.write_text('{"stage":"training"}')
    command = 'set -eu; PREFIX="$1"; ' + guard
    result = subprocess.run([str(bash), "-c", command, "guard", prefix.as_posix()], capture_output=True, text=True)
    assert result.returncode == 1
    assert "Export status exists" in result.stdout
    assert "unbound variable" not in result.stderr
    assert status.read_text() == '{"stage":"training"}'
    result = subprocess.run([str(bash), "-c", command, "guard", (tmp_path / "fresh").as_posix()], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
