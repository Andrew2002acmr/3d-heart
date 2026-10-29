"""Finish the already running local study after its numerical workers complete.

Does not start/resume model training, create infrastructure or schedule a task.
"""
import argparse,time,traceback,zipfile
from ct_lab.dynamics_core import *

def results_archive():
    files=[]
    for directory in [OUT/'runs/baseline_s20261009_n48',OUT/'references',OUT/'figures',OUT/'tables']:
        for path in directory.rglob('*'):
            if not path.is_file() or path.name in ['last.pt','best_candidate.pt'] or '.partial' in path.name:continue
            if '__pycache__' in path.parts:continue
            if path.suffix=='.pt' and path.stem!='selected' and not path.stem.startswith('step'):continue
            files.append(path)
    files += [p for p in OUT.glob('*.json') if p.name not in ['previous_files.json','protected_files.json']]
    # Source/private data and prepared arrays are excluded; NIfTI here are predictions only.
    names={p.relative_to(REPO).as_posix():p for p in files}
    report=REPO/'TRAINING_DYNAMICS_GENERALIZATION_06.md';names[report.name]=report
    manifest={'scope':'completed first local baseline, common references and diagnostics; no raw CT/GT',
      'files':[{'path':name,'SHA256':fingerprint(path),'bytes':path.stat().st_size} for name,path in sorted(names.items())]}
    dest=OUT/'local_baseline_completed_results.zip';temp=dest.with_suffix('.partial.zip')
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,path in sorted(names.items()):z.write(path,name)
        z.writestr('MANIFEST_RESULTS06.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(temp) as z:assert z.testzip() is None
    os.replace(temp,dest);return {'path':str(dest),'SHA256':fingerprint(dest),'bytes':dest.stat().st_size,'CRC_verified':True,'raw_CT_GT_included':False}

def finish(wait=False):
    result=OUT/'runs/baseline_s20261009_n48/result.json'
    print('Waiting for the existing baseline and topology workers; no model training started here',flush=True)
    while not result.exists() or not (OUT/'topology_completion.json').exists():
        if not wait:raise ValueError('Baseline and114 control/case topologies must finish first')
        time.sleep(5)
    from ct_lab.dynamics_suite import fidelity_gate
    try:gate=fidelity_gate()
    except ValueError:gate=json.loads((OUT/'baseline_fidelity_gate.json').read_text())
    # A failed engineering reproduction still receives an honest report; no ablation is started.
    from ct_lab.dynamics_report import build
    from ct_lab.dynamics_integrity import run as integrity
    from ct_lab.dynamics_bundle import build as bundle
    build(final=True);integrity(final=True);bundle();archive=results_archive()
    write(OUT/'completion.json',{'status':'completed_one_local_baseline','additional_training_started':False,'fidelity_gate':gate,
      'manual_RunPod_bundle':str(OUT/'runpod_manual_bundle.zip'),'results_archive':archive,'report':str(REPO/'TRAINING_DYNAMICS_GENERALIZATION_06.md')})
    print('LOCAL STAGE06 COMPLETE; ALL OTHER TRAINING MANUAL RUNPOD PENDING',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');a=p.parse_args()
    try:finish(a.wait)
    except Exception as e:
        write(OUT/'completion_error.json',{'error':str(e),'traceback':traceback.format_exc(),'additional_training_started':False});raise
