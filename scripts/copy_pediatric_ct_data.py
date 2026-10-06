"""Copy verified pediatric data to an external CLI-selected root; never move/delete."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from heart3d.storage import copy_verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, help='Optional existing public inventory snapshot')
    parser.add_argument('--reserve-gb', type=float, default=80, help='Minimum free decimal GB after copy')
    parser.add_argument('--summary', type=Path, required=True, help='Small summary, allowed in Git')
    args = parser.parse_args()
    summary, records = copy_verified(args.source, args.destination,
                                     reserve_bytes=int(args.reserve_gb * 10**9), inventory=args.inventory)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    summary['date_utc'] = timestamp
    receipt = args.destination / 'cache' / f'copy_verification_{timestamp}.json'
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open('x', encoding='utf-8') as handle:
        json.dump({'summary': summary, 'files': records}, handle, ensure_ascii=False, indent=2)
    summary['external_receipt_relative_path'] = receipt.relative_to(args.destination).as_posix()
    for name in ('preprocessing', 'predictions', 'checkpoints', 'meshes', 'screenshots', 'experiments', 'temporary'):
        (args.destination / name).mkdir(exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
