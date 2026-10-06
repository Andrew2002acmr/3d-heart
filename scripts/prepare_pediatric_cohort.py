"""Bounded, resumable full CT/RT download + technical QA; never auto-approve."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from heart3d.cohort import make_queue, automatic_review_status
from heart3d.storage import require_space, sha256_file, exclusive_file_lock
from heart3d.dicom.tcia import fetch_full_series
from heart3d.pediatric import write_json
from prepare_pediatric_ct_pilot import prepare_case


def download_case(row, root, reserve):
    directory = root/row['patient_id']
    receipt = directory/'full_download_receipt.json'
    if receipt.exists():
        data = json.loads(receipt.read_text())
        for key, name in [('CT', 'ct.zip'), ('RTSTRUCT', 'rtstruct.zip')]:
            if sha256_file(directory/name) != data[key]['sha256']:
                raise ValueError('Cached source archive differs from recorded SHA-256')
        return row
    ct = fetch_full_series(row['ct_series_uid'], directory/'ct', reserve_bytes=reserve)
    rt = fetch_full_series(row['rt_series_uid'], directory/'rtstruct', reserve_bytes=reserve)
    write_json(receipt, {'patient_id':row['patient_id'], 'CT':ct, 'RTSTRUCT':rt})
    print('DOWNLOADED',row['patient_id'],flush=True)
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--registry',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--queue',type=Path,required=True)
    p.add_argument('--start',type=int,default=0)
    p.add_argument('--count',type=int,default=60)
    p.add_argument('--download-workers',type=int,choices=range(1,5),default=2)
    p.add_argument('--age-group',action='append',choices=['2-5','6-11','12-17'])
    p.add_argument('--download-only',action='store_true')
    p.add_argument('--include-development',action='store_true')
    args=p.parse_args()
    if args.start<0 or args.count<1:raise ValueError('Positive bounded wave required')
    with exclusive_file_lock(args.data/'cache'/'cohort_preparation.lock'):
        run_wave(args)


def run_wave(args):
    config=json.loads(args.config.read_text())
    registry=json.loads(args.registry.read_text())['records']
    dev=config['development_pilot_patient_ids']
    if args.queue.exists():
        queue=json.loads(args.queue.read_text())
        if queue['registry_sha256']!=sha256_file(args.registry):
            raise ValueError('Registry changed after queue creation')
    else:
        queue={'seed':config['seed'],'registry_sha256':sha256_file(args.registry),
               'selection_uses_mask_volume_or_visual_quality':False,
               'development_patient_ids':dev,
               'records':make_queue(registry,config['seed'],set(dev),config['stratum_targets'])}
        write_json(args.queue,queue)
    remaining=queue['records'][args.start:]
    if args.age_group:
        remaining=[r for r in remaining if r['age_group'] in args.age_group]
    rows=remaining[:args.count]
    if args.include_development:
        by={r['patient_id']:r for r in registry}
        rows=[by[pid] for pid in dev]+rows
    # Registry uses slices; legacy pilot helper names the same count expected_slices.
    rows=[dict(r,expected_slices=r['slices']) for r in rows]
    reserve=int(config['storage']['minimum_free_reserve_GB']*1e9)
    missing=[r for r in rows if not (args.data/r['patient_id']/'full_download_receipt.json').exists()]
    raw=sum(r['CT_published_bytes']+r['RT_published_bytes'] for r in missing)
    # Conservative unsqueezed array budget for this wave, plus archive/extracted DICOM.
    prepared=sum(r['slices']*512*512*5 for r in rows)
    need=int(2.1*raw)+prepared+1_000_000_000
    free=require_space(args.data,need,reserve)
    suffix='_'.join(args.age_group or ['all'])
    mode='download_only' if args.download_only else 'full_QA'
    write_json(args.data/'cache'/f'cohort_wave_{args.start}_{args.count}_{suffix}_{mode}_budget.json',
               {'raw_DICOM_bytes':raw,'prepared_uncompressed_estimate_bytes':prepared,
                'additional_budget_bytes':need,'free_bytes':free,'reserve_bytes':reserve,
                'queue_start':args.start,'age_filter':args.age_group,'download_workers':args.download_workers,
                'mode':mode,
                'patients':[r['patient_id'] for r in rows]})
    print('WAVE',len(rows),'additional budget',need,'free',free,flush=True)
    with ThreadPoolExecutor(max_workers=args.download_workers) as pool:
        futures=[pool.submit(download_case,r,args.data,reserve) for r in rows]
        for index,(row,future) in enumerate(zip(rows,futures)):
            pid=row['patient_id']; directory=args.data/pid
            try:
                future.result()
                if args.download_only:
                    print('RAW READY FOR QA',index+1,'/',len(rows),pid,flush=True)
                    continue
                path=directory/'pilot_qa.json'
                old=json.loads(path.read_text()) if path.exists() else {}
                if old.get('mask_rasterized') and old.get('qa_sampling_version')==2:
                    report=old
                else:
                    report=prepare_case(row,args.data)
                report['review_status']=automatic_review_status(report)
                if report.get('mask_rasterized'):
                    report['prepared_CT_sha256']=sha256_file(directory/'prepared'/'ct_original.nii.gz')
                    report['prepared_mask_sha256']=sha256_file(directory/'prepared'/'heart_gt_original.nii.gz')
                write_json(path,report)
                print('FULL QA',index+1,'/',len(rows),pid,report['review_status'],
                      report.get('Heart',{}).get('volume_ml'),report.get('reason',''),flush=True)
            except Exception as error:
                write_json(directory/'cohort_failure.json',{'patient_id':pid,'status':'download_or_integrity_failure','reason':str(error)})
                print('FAILED',pid,str(error),flush=True)


if __name__=='__main__':main()
