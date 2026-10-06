import json
from pathlib import Path
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from verify_pediatric_source_cache import verify
from heart3d.pediatric import write_json
from heart3d.storage import sha256_file


def source_pair(root):
    directory = root / 'Pediatric-CT-SEG-test'
    directory.mkdir()
    receipt = {'patient_id': directory.name}
    for key, filename in [('CT', 'ct.zip'), ('RTSTRUCT', 'rtstruct.zip')]:
        with zipfile.ZipFile(directory / filename, 'w') as archive:
            archive.writestr('synthetic.txt', 'unit test only')
        receipt[key] = {'sha256': sha256_file(directory / filename)}
    write_json(directory / 'full_download_receipt.json', receipt)
    return directory, receipt


def test_corruption_fails_without_modifying_source_or_publishing_success(tmp_path):
    directory, _ = source_pair(tmp_path)
    archive = directory / 'ct.zip'
    archive.write_bytes(archive.read_bytes() + b'changed')
    before = archive.read_bytes()
    output = tmp_path / 'summary.json'
    with pytest.raises(ValueError, match='hash mismatch'):
        verify(tmp_path, output)
    assert archive.read_bytes() == before
    assert not output.exists()


def test_recovery_retains_historical_failure_and_source_hashes(tmp_path):
    directory, receipt = source_pair(tmp_path)
    failure = directory / 'cohort_failure.json'
    write_json(failure, {'reason': 'historical interrupted download'})
    failure_digest = sha256_file(failure)
    incident = tmp_path / 'incident.json'
    write_json(incident, {'missing_receipts_to_retry': [directory.name], 'data_deleted': False})
    result = verify(tmp_path, tmp_path / 'summary.json', incident)
    assert result['verified_archives'] == 2 and result['hash_mismatches'] == []
    assert sha256_file(directory / 'ct.zip') == receipt['CT']['sha256']
    assert sha256_file(failure) == failure_digest
    assert json.loads(incident.read_text())['pending_retry_patient_ids'] == []
    assert json.loads((directory / 'download_failure_resolution.json').read_text())['download_resolved']
