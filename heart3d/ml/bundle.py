"""Minimal allowlisted training/results bundles; SHA-256 verification, no raw DICOM."""
import argparse
import gzip
import json
from pathlib import Path, PurePosixPath
import tarfile
from heart3d.ml.splits import load_frozen_split
from heart3d.pediatric import write_json
from heart3d.storage import sha256_file, require_space


def safe_relative(value):
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or '\\' in value or ':' in value:
        raise ValueError('Unsafe bundle path')
    return path.as_posix()


def verify(root, manifest_path, repo=None):
    root = Path(root); manifest = json.loads(Path(manifest_path).read_text())
    paths = [safe_relative(row['path']) for row in manifest['files']]
    if len(paths) != len(set(paths)): raise ValueError('Duplicate manifest paths')
    for record in manifest['files']:
        path = root/safe_relative(record['path'])
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('Redirected bundle file')
        if path.stat().st_size != record['bytes'] or sha256_file(path) != record['SHA256']:
            raise ValueError(f'Bundle integrity failure: {record["path"]}')
    if repo is not None:
        for row in manifest.get('repository_files', []):
            if sha256_file(Path(repo)/safe_relative(row['path'])) != row['SHA256']:
                raise ValueError('Repository frozen/config file differs from bundle')
    print('SHA256_VERIFIED', len(paths), 'files', flush=True)
    return manifest


def pack(root, paths, archive, manifest_path, repository_files=None, reserve_bytes=0):
    root = Path(root); archive = Path(archive); manifest_path = Path(manifest_path)
    if archive.exists() or manifest_path.exists(): raise ValueError('Bundle/archive exists; use a new name')
    paths = sorted(set(safe_relative(p) for p in paths))
    records = [{'path': p, 'bytes': (root/p).stat().st_size, 'SHA256': sha256_file(root/p)} for p in paths]
    manifest = {'version': 1, 'files': records, 'repository_files': repository_files or [],
                'source_bytes': sum(r['bytes'] for r in records), 'raw_DICOM_included': False}
    require_space(archive.parent, manifest['source_bytes']+1_000_000, reserve_bytes)
    archive.parent.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, manifest)
    with archive.open('xb') as handle, gzip.GzipFile(fileobj=handle, mode='wb', compresslevel=1) as compressed:
        with tarfile.open(fileobj=compressed, mode='w|') as tar:
            tar.add(manifest_path, arcname='bundle_manifest.json', recursive=False)
            for path in paths: tar.add(root/path, arcname=path, recursive=False)
    receipt = {'archive_bytes': archive.stat().st_size, 'archive_SHA256': sha256_file(archive),
               'manifest_SHA256': sha256_file(manifest_path), 'files': len(paths), 'source_bytes': manifest['source_bytes']}
    write_json(archive.with_suffix('.receipt.json'), receipt)
    print('BUNDLE', json.dumps(receipt), flush=True)
    return receipt


def training_bundle(config_path, data_root, archive, manifest):
    config = json.loads(Path(config_path).read_text()); root = Path(data_root)
    split = load_frozen_split(config['split'], config['cohort']); paths = []
    for partition, rows in split['partitions'].items():
        paths.append(f'{config["cache_relative_path"]}/{partition}_index.json')
        for row in rows:
            directory = root/config['cache_relative_path']/row['patient_id']
            p = json.loads((directory/'provenance.json').read_text())
            if p['partition'] != partition or p['split_SHA256'] != sha256_file(config['split']) or p['preprocessing_SHA256'] != sha256_file(config['preprocessing']):
                raise ValueError('Stale/misassigned cache')
            for name,key in [('image.npy','image_SHA256'),('mask.npy','mask_SHA256')]:
                if sha256_file(directory/name) != p[key]: raise ValueError('Cache SHA mismatch')
                paths.append((directory/name).relative_to(root).as_posix())
            paths.append((directory/'provenance.json').relative_to(root).as_posix())
            if partition != 'train':
                if sha256_file(root/row['mask_relative_path']) != row['review']['mask_SHA256']:
                    raise ValueError('Reviewed original GT changed')
                paths.append(row['mask_relative_path'])
    repository_files = [{'path': str(path).replace('\\','/'), 'SHA256': sha256_file(path)}
                        for path in [config_path, config['split'], config['cohort'], config['preprocessing']]]
    return pack(root, paths, archive, manifest, repository_files, int(80e9))


def unpack(archive, destination, reserve_bytes=0):
    """Allowlisted extraction, no links, no traversal, no different-file overwrite."""
    destination = Path(destination)
    with tarfile.open(archive, 'r:gz') as tar:
        member = tar.getmember('bundle_manifest.json')
        manifest = json.load(tar.extractfile(member))
        allowed = {safe_relative(r['path']): r for r in manifest['files']}
        if len(allowed) != len(manifest['files']): raise ValueError('Duplicate manifest paths')
        members = tar.getmembers()
        if len({m.name for m in members}) != len(members): raise ValueError('Duplicate tar members')
        require_space(destination, sum(r['bytes'] for r in allowed.values()), reserve_bytes)
        for m in members:
            name = safe_relative(m.name)
            if not m.isfile() or (name != 'bundle_manifest.json' and name not in allowed):
                raise ValueError('Unexpected/non-file tar member')
            target = destination/name
            if not target.resolve().is_relative_to(destination.resolve()) or target.is_symlink():
                raise ValueError('Redirected extraction path')
            if target.exists():
                expected = allowed.get(name)
                if expected is None or sha256_file(target) != expected['SHA256']:
                    raise ValueError('Refuse existing/different target')
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(m) as source, target.open('xb') as output:
                while block := source.read(4 << 20): output.write(block)
    return verify(destination, destination/'bundle_manifest.json')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    build = sub.add_parser('training'); build.add_argument('--config', type=Path, required=True)
    build.add_argument('--data', type=Path, required=True); build.add_argument('--archive', type=Path, required=True)
    build.add_argument('--manifest', type=Path, required=True)
    check = sub.add_parser('verify'); check.add_argument('--root', type=Path, required=True)
    check.add_argument('--manifest', type=Path, required=True); check.add_argument('--repo', type=Path)
    extract = sub.add_parser('unpack'); extract.add_argument('--archive', type=Path, required=True)
    extract.add_argument('--destination', type=Path, required=True); extract.add_argument('--reserve-GB', type=float, default=10)
    a = p.parse_args()
    if a.command == 'training': training_bundle(a.config, a.data, a.archive, a.manifest)
    elif a.command == 'verify': verify(a.root, a.manifest, a.repo)
    else: unpack(a.archive, a.destination, int(a.reserve_GB*1e9))
