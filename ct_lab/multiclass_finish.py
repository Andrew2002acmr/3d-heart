"""Same-task analysis pipeline after the training and concurrent CHD jobs finish."""
import json
import time
from ct_lab.multiclass_core import OUT


def ready(path,minimum=0):
    try:
        value=json.loads(path.read_text());return len(value)>=minimum
    except (OSError,ValueError):return False


def run():
    print('Waiting for completed training receipts',flush=True)
    while not ready(OUT/'results.json'):time.sleep(5)
    from ct_lab.multiclass_analysis import run as analyze
    from ct_lab.multiclass_projection import run as projections
    from ct_lab.multiclass_dense import run as dense
    analyze();projections()
    print('Waiting for concurrent CHD inference completion',flush=True)
    while not ready(OUT/'dense_summary.json',8):time.sleep(5)
    dense();print('STAGE04 ANALYSIS/DENSE/PROJECTIONS COMPLETE',flush=True)


if __name__=='__main__':run()
