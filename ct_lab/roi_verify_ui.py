"""Real-data UI controls and sealed test guard; never starts training/inference."""
import json
import sys
import time
from ct_lab.roi_core import OUT,REPO,seed_runtime,SEED,cohort
import ct_lab.launch
from streamlit.testing.v1 import AppTest


def run():
    seed_runtime(SEED,1,'cpu');app=AppTest.from_file(str(REPO/'ct_lab/app.py'),default_timeout=240).run();checks=[]
    def check(name):
        errors=[e.message for e in app.exception];checks.append({'check':name,'exceptions':errors});print(name,errors,flush=True);assert not errors
    def choose(label,value):next(s for s in app.selectbox if s.label==label).set_value(value).run()
    app.sidebar.radio[0].set_value('Heart Localization & ROI Segmentation').run();check('Three source planes with automatic ROI')
    next(s for s in app.slider if s.label=='ROI mask opacity').set_value(.6).run();check('Mask opacity')
    next(s for s in app.multiselect if s.label=='ROI foreground classes').set_value([1,5,7]).run();check('Selected foreground classes')
    boxes=next(s for s in app.multiselect if s.label=='ROI boxes');boxes.set_value(boxes.options).run();check('Train fixed/predicted/oracle boxes')
    for model in ('Source','GT','CHD68_v1','CHD68_v2','B_full_256','D_predicted_256','C_oracle_256','B_full_384','D_predicted_384','D_core_256'):
        if '--wait-models' in sys.argv and model not in ('Source','GT','CHD68_v1','CHD68_v2'):
            deadline=time.monotonic()+3600;announced=False
            while True:
                try:
                    json.loads((OUT/'segmenters/all_train'/model/'result.json').read_text())
                    json.loads((OUT/'evaluation/all_train'/model/'ct_1001/scores.json').read_text());break
                except (FileNotFoundError,json.JSONDecodeError):
                    if not announced:print('Waiting for frozen native model:',model,flush=True);announced=True
                    if time.monotonic()>deadline:raise TimeoutError(model)
                    time.sleep(2)
            if announced:app.run()
        if '--partial' in sys.argv and model not in next(s for s in app.selectbox if s.label=='ROI segmentation model').options:continue
        choose('ROI segmentation model',model);check('Native source prediction / Heart3D '+model)
    for mode in ('False positives','False negatives','Class confusion','Segmentation'):
        choose('ROI error view',mode);check('Native error maps '+mode)
    from ct_lab.core import discover
    cases=discover(REPO.parent/'laboratories/data/chd68/pairs');split,_=cohort()
    choose('Исследование',next(i for i,r in enumerate(cases) if r['id']==split['validation'][0]));check('Development automatic ROI')
    assert 'C_oracle_256' not in next(s for s in app.selectbox if s.label=='ROI segmentation model').options
    assert 'Oracle GT ROI (diagnostic)' not in next(s for s in app.multiselect if s.label=='ROI boxes').options
    next(c for c in app.checkbox if c.label=='Oracle validation diagnostic — uses GT').check().run();check('Explicit development oracle diagnostic')
    assert 'Oracle GT ROI (diagnostic)' in next(s for s in app.multiselect if s.label=='ROI boxes').options
    if '--partial' not in sys.argv or 'C_oracle_256' in next(s for s in app.selectbox if s.label=='ROI segmentation model').options:
        assert 'C_oracle_256' in next(s for s in app.selectbox if s.label=='ROI segmentation model').options
        choose('ROI segmentation model','C_oracle_256');check('Explicit native oracle prediction')
    next(c for c in app.checkbox if c.label=='Oracle validation diagnostic — uses GT').uncheck().run();check('Oracle disabled again')
    assert 'C_oracle_256' not in next(s for s in app.selectbox if s.label=='ROI segmentation model').options
    import ct_lab.roi_ui as ui
    hasher=ui.source_fingerprint;predictor=ui.predictions
    def forbidden_prediction(*args):raise AssertionError('Cached prediction read for mismatched source')
    try:
        ui.source_fingerprint=lambda *args:'mismatched file';ui.predictions=forbidden_prediction
        app.run();check('Mismatched source/GT rejected before cached prediction access')
        assert any('Frozen' in e.value or 'frozen' in e.value for e in app.error)
    finally:ui.source_fingerprint=hasher;ui.predictions=predictor
    import ct_lab.core as source
    loader=source.load_pair;reads=[]
    def forbidden(row):reads.append(row);raise AssertionError('Sealed test source read')
    source.load_pair=forbidden
    try:
        choose('Исследование',next(i for i,r in enumerate(cases) if r['id']==split['test'][0]));check('Sealed test gate before source IO');assert not reads
    finally:source.load_pair=loader
    (OUT/('ui_preliminary.json' if '--partial' in sys.argv else 'ui_verification.json')).write_text(json.dumps(checks,indent=2,ensure_ascii=False),encoding='utf-8')


if __name__=='__main__':run()
