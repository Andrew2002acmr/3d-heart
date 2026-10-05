"""Cache the public source inventory outside Git for reproducible download helpers."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from heart3d.dicom.tcia import json_get
from heart3d.pediatric import write_json


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise ValueError('Inventory already exists; choose a new versioned snapshot path')
    rows=json_get('getSeries',Collection='Pediatric-CT-SEG')
    if any(r['Collection']!='Pediatric-CT-SEG' for r in rows):raise ValueError('Unexpected collection')
    write_json(args.out,rows)
    print('Saved public series inventory:',len(rows))
