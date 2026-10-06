"""Copy external data with hash preflight, no overwrite and a free-space reserve."""
import hashlib
import json
from pathlib import Path
import shutil
import os
from contextlib import contextmanager


@contextmanager
def exclusive_file_lock(path):
    """OS-released process lock; persistent marker is not a stale exclusive lock."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as handle:
        handle.seek(0,2)
        if not handle.tell():handle.write(b'0');handle.flush()
        handle.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f'Another process is using this data preparation root: {path}') from error
        try:yield
        finally:
            handle.seek(0)
            if os.name=='nt':msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(handle.fileno(),fcntl.LOCK_UN)


def redirected_path(path):
    return path.is_symlink() or getattr(path, 'is_junction', lambda: False)()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while block := handle.read(4 << 20):
            digest.update(block)
    return digest.hexdigest()


def require_space(directory, additional_bytes, reserve_bytes):
    if additional_bytes < 0 or reserve_bytes < 0:
        raise ValueError('Storage budgets must be nonnegative')
    existing = Path(directory).resolve()
    while not existing.exists():
        existing = existing.parent
    free = shutil.disk_usage(existing).free
    if free - additional_bytes < reserve_bytes:
        raise ValueError(f'Insufficient storage: free={free}, additional={additional_bytes}, reserve={reserve_bytes}')
    return free


def known_pediatric_hashes(root):
    """Expected hashes from previous downloads, not hashes invented during copy."""
    root = Path(root)
    expected = {}
    census = root / 'census.json'
    if census.exists():
        for row in json.loads(census.read_text())['records']:
            if row.get('CT_probe_sha256'):
                expected[f"{row['patient_id']}/census/ct_probe.dcm"] = row['CT_probe_sha256']
    for path in root.glob('*/census/heart_extract_receipt.json'):
        expected[(path.parent / 'heart_extract.dcm').relative_to(root).as_posix()] = json.loads(path.read_text())['derived_sha256']
    for path in root.glob('*/full_download_receipt.json'):
        receipt = json.loads(path.read_text())
        for key, archive in [('CT', 'ct.zip'), ('RTSTRUCT', 'rtstruct.zip')]:
            expected[(path.parent / archive).relative_to(root).as_posix()] = receipt[key]['sha256']
    for path in root.glob('*/reference_review.json'):
        expected[(path.parent / 'reference_review_rt.zip').relative_to(root).as_posix()] = json.loads(path.read_text())['original_archive_receipt']['sha256']
    return expected


def copy_verified(source, destination, *, reserve_bytes, inventory=None):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not source.is_dir() or source == destination or source.is_relative_to(destination) or destination.is_relative_to(source):
        raise ValueError('Need separate, non-nested source and destination directories')
    # Reject symlinks/junctions rather than copying through redirected data paths.
    entries = sorted(source.rglob('*'))
    if any(redirected_path(p) for p in entries):
        raise ValueError('Source contains redirected paths')
    mapping = {p.relative_to(source).as_posix(): p for p in entries if p.is_file()}
    if inventory is not None:
        relative = 'cache/tcia_series_v1.json'
        if relative in mapping:
            raise ValueError('Inventory target already included in source')
        mapping[relative] = Path(inventory).resolve()
    expected = known_pediatric_hashes(source)
    if set(expected) - set(mapping):
        raise ValueError('Download receipt refers to a missing source file')
    records = []
    for relative, path in mapping.items():
        digest = sha256_file(path)
        if relative in expected and digest != expected[relative]:
            raise ValueError(f'Source differs from recorded download SHA-256: {relative}')
        target = destination / relative
        for parent in [target, *target.parents]:
            if redirected_path(parent):
                raise ValueError(f'Redirected destination path: {relative}')
        present = target.exists()
        if present and (not target.is_file() or sha256_file(target) != digest):
            raise ValueError(f'Refuse to overwrite different destination: {relative}')
        records.append({'path': relative, 'bytes': path.stat().st_size, 'sha256': digest,
                        'download_receipt_hash_checked': relative in expected, 'already_present': present})
    additional = sum(r['bytes'] for r in records if not r['already_present'])
    free_before = require_space(destination, additional, reserve_bytes)
    print(f'Preflight: {len(records)} source hashes verified; {additional} bytes to copy', flush=True)
    destination.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(records):
        target = destination / record['path']
        if not record['already_present']:
            require_space(destination, record['bytes'], reserve_bytes)
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            # Exclusive creation: never overwrite a file that appeared after preflight.
            with mapping[record['path']].open('rb') as original, target.open('xb') as copied:
                while block := original.read(4 << 20):
                    digest.update(block)
                    copied.write(block)
            if digest.hexdigest() != record['sha256']:
                raise ValueError(f'Source changed during copy: {record["path"]}')
        if sha256_file(target) != record['sha256']:
            raise ValueError(f'Destination SHA-256 mismatch: {record["path"]}')
        record['destination_hash_verified'] = True
        if (index + 1) % 500 == 0:
            print(f'Copied/verified {index + 1}/{len(records)} files', flush=True)
    summary = {'source': str(source), 'destination': str(destination),
               'files_verified': len(records), 'source_bytes': sum(r['bytes'] for r in records),
               'bytes_copied': additional, 'files_reused': sum(r['already_present'] for r in records),
               'download_receipt_hashes_verified': len(expected), 'reserve_bytes': reserve_bytes,
               'free_bytes_before': free_before, 'free_bytes_after': require_space(destination, 0, reserve_bytes),
               'all_destination_SHA256_match': True, 'source_files_modified_or_deleted': False}
    return summary, records
