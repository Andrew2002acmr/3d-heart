import json
import pytest
from heart3d.storage import copy_verified, sha256_file


def test_copy_and_reuse_with_inventory(tmp_path):
    source = tmp_path/'source'; source.mkdir()
    (source/'case').mkdir(); (source/'case'/'scan.dcm').write_bytes(b'original')
    inventory = tmp_path/'inventory.json'; inventory.write_text('[]')
    target = tmp_path/'target'
    summary, records = copy_verified(source, target, reserve_bytes=0, inventory=inventory)
    assert summary['files_verified'] == 2 and summary['all_destination_SHA256_match']
    assert (target/'case'/'scan.dcm').read_bytes() == b'original'
    assert all(r['destination_hash_verified'] for r in records)
    again, _ = copy_verified(source, target, reserve_bytes=0, inventory=inventory)
    assert again['bytes_copied'] == 0 and again['files_reused'] == 2
    assert (source/'case'/'scan.dcm').read_bytes() == b'original'


def test_corrupt_source_receipt_prevents_any_copy(tmp_path):
    source = tmp_path/'source'; (source/'case'/'census').mkdir(parents=True)
    path = source/'case'/'census'/'heart_extract.dcm'; path.write_bytes(b'good')
    (path.parent/'heart_extract_receipt.json').write_text(json.dumps({'derived_sha256':sha256_file(path)}))
    path.write_bytes(b'tampered')
    with pytest.raises(ValueError, match='recorded download'):
        copy_verified(source, tmp_path/'target', reserve_bytes=0)
    assert not (tmp_path/'target').exists()


def test_destination_conflict_prevents_any_copy(tmp_path):
    source = tmp_path/'source'; source.mkdir(); (source/'scan').write_bytes(b'source')
    target = tmp_path/'target'; target.mkdir(); (target/'scan').write_bytes(b'user file')
    with pytest.raises(ValueError, match='overwrite'):
        copy_verified(source, target, reserve_bytes=0)
    assert (target/'scan').read_bytes() == b'user file'


def test_reserve_gate_before_copy(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr('heart3d.storage.shutil.disk_usage', lambda _: SimpleNamespace(free=100))
    source = tmp_path/'source'; source.mkdir(); (source/'scan').write_bytes(b'0123456789')
    with pytest.raises(ValueError, match='Insufficient storage'):
        copy_verified(source, tmp_path/'target', reserve_bytes=95)
    assert not (tmp_path/'target').exists()


def test_nested_roots_rejected(tmp_path):
    with pytest.raises(ValueError, match='non-nested'):
        copy_verified(tmp_path, tmp_path/'child', reserve_bytes=0)
