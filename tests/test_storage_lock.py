import pytest
from heart3d.storage import exclusive_file_lock


def test_process_lock_refuses_concurrent_writer_and_releases(tmp_path):
    path=tmp_path/'case.lock'
    with exclusive_file_lock(path):
        with pytest.raises(RuntimeError,match='Another process'):
            with exclusive_file_lock(path):pass
    with exclusive_file_lock(path):pass
    # No automatic removal of data or stale-marker guesswork.
    assert path.exists()
