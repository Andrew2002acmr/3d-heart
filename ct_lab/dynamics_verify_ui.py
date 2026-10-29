"""Real checkpoint changes retain native CT/Z and never write model weights."""
import sys,time
from ct_lab.dynamics_core import *
from streamlit.testing.v1 import AppTest

def run():
    seed_runtime(20261009,1,'cpu');directory=OUT/'runs/baseline_s20261009_n48'
    if '--wait-second' in sys.argv:
        while not (directory/'evaluation/step003840/ct_1001/metrics.json').exists():time.sleep(5)
    paths=[directory/'step000480.pt',WEIGHTS/'chd68_v1_best.pt',WEIGHTS/'chd68_v2_best.pt']
    if (directory/'step003840.pt').exists():paths.append(directory/'step003840.pt')
    before={str(p):fingerprint(p) for p in paths};app=AppTest.from_file(str(REPO/'ct_lab/app.py'),default_timeout=240).run();checks=[]
    def check(name):
        errors=[e.message for e in app.exception];checks.append({'check':name,'exceptions':errors});print(name,errors,flush=True);assert not errors
    def choose(label,value):next(s for s in app.selectbox if s.label==label).set_value(value).run()
    app.sidebar.radio[0].set_value('Training Dynamics Lab').run();check('Seven real-data training analysis panels')
    assert len(app.tabs)==7
    next(s for s in app.slider if s.label=='Dynamics native Z').set_value(120).run();check('Native source Z selected')
    for name in ['CHD68 v1 historical','CHD68 v2 historical','step000480']+(['step003840'] if (directory/'step003840.pt').exists() else []):
        choose('Dynamics checkpoint',name);assert next(s for s in app.slider if s.label=='Dynamics native Z').value==120;check('Checkpoint changes keep native Z: '+name)
    for curve in ['Loss','Prepared Dice','Native Dice','Gap','Learning Rate','Foreground Recall']:
        choose('Dynamics curve',curve);check('Measured curve '+curve)
    next(s for s in app.slider if s.label=='Dynamics mask opacity').set_value(.7).run();check('Mask opacity')
    next(s for s in app.multiselect if s.label=='Dynamics visible classes').set_value([1,5,7]).run();check('Class visibility')
    from ct_lab.core import discover
    cases=discover(DATA);split,_=cohort();choose('Исследование',next(i for i,r in enumerate(cases) if r['id']==split['validation'][0]));check('Development source CT checkpoint view')
    import ct_lab.dynamics_ui as ui
    original=ui.file_sha
    try:
        ui.file_sha=lambda *args:'mismatch';app.run();check('Source/GT identity guard');assert any('не совпадают' in e.value for e in app.error)
    finally:ui.file_sha=original
    import ct_lab.core as source
    loader=source.load_pair;reads=[]
    def forbidden(row):reads.append(row);raise AssertionError('Sealed test source read')
    source.load_pair=forbidden
    try:
        choose('Исследование',next(i for i,r in enumerate(cases) if r['id']==split['test'][0]));check('Sealed test gated before source I/O');assert not reads
    finally:source.load_pair=loader
    assert before=={str(p):fingerprint(p) for p in paths}
    write(OUT/'ui_verification.json',{'checks':checks,'checkpoint_SHA256_unchanged':before,'same_case_native_Z_across_checkpoints':True})

if __name__=='__main__':run()
