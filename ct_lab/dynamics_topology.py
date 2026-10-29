"""Bounded asynchronous source-grid topology at preregistered control checkpoints."""
import argparse,gc,time
import nibabel as nib
from ct_lab.dynamics_core import *
from ct_lab.roi_topology import native_metrics
os.environ['NUMBA_CACHE_DIR']=str(OUT/'numba_cache')

def measure(path):
    case=path.parent.name;allowed(case)
    prediction=np.asanyarray(nib.load(path.with_name('prediction.nii.gz')).dataobj)
    target=cardiac_target(np.asanyarray(nib.load(DATA/f'{case}_label.nii.gz').dataobj));result=native_metrics(prediction,target)
    existing=json.loads(path.read_text());assert abs(result['macro_foreground_Dice']-existing['macro_foreground_Dice'])<1e-12
    assert result['confusion_matrix']==existing['native_confusion']
    result.update(case=case,role=existing['role'],checkpoint_SHA256=existing['checkpoint_SHA256']);write(path.with_name('topology.json'),result)
    print('DYNAMICS TOPOLOGY',path.parent.parent.name,case,result['macro_foreground_Dice'],flush=True);return result

def run(all_runs=False):
    while True:
        directories=list((OUT/'runs').glob('*')) if all_runs else [OUT/'runs/baseline_s20261009_n48']
        paths=[p for directory in directories for p in (directory/'evaluation').glob('*/*/metrics.json')]
        for p in paths:
            if p.with_name('topology.json').exists():continue
            measure(p);gc.collect()
        if directories and all((d/'result.json').exists() for d in directories):
            expected=[]
            for d in directories:
                cfg=json.loads((d/'config.json').read_text());result=json.loads((d/'result.json').read_text())
                checkpoints=[f'step{s:06d}' for s in cfg['stage06']['control_steps'] if s<=result['steps']]+['selected']
                cases=result['evaluation']['records'];expected.extend(d/'evaluation'/c/r['case']/'topology.json' for c in checkpoints for r in cases)
            if all(p.exists() for p in expected):
                write(OUT/'topology_completion.json',{'runs':[d.name for d in directories],'sets':len(expected),'all_expected_complete':True,'GT_cleanup':False,'units':'index grid, not mm'});break
        time.sleep(5)
    print('DYNAMICS ALL TOPOLOGY COMPLETE',len(expected),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--all-runs',action='store_true');args=parser.parse_args();run(args.all_runs)
