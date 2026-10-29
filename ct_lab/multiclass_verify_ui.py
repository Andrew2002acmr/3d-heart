"""Real-source AppTest: multiclass controls, dense overlays and sealed-test gate."""
import json
from ct_lab.multiclass_core import OUT,REPO,freeze,SEEDS,seed_all
import ct_lab.launch
from streamlit.testing.v1 import AppTest


def run():
    seed_all(SEEDS[0]);app=AppTest.from_file(str(REPO/'ct_lab/app.py'),default_timeout=240).run();checks=[]
    def check(name):
        errors=[e.message for e in app.exception];checks.append({'check':name,'exceptions':errors});print(name,errors,flush=True);assert not errors
    def choose(label,value):next(s for s in app.selectbox if s.label==label).set_value(value).run()
    app.sidebar.radio[0].set_value('Multiclass Feature Lab').run();check('Eight-class default CE combined model')
    next(c for c in app.checkbox if c.label=='Показывать background').check().run();check('Background and all seven foreground classes')
    for layer in ('Dense prediction','Errors','Entropy','Class probability'):
        choose('Multiclass CT overlay',layer);check('Native dense overlay '+layer)
    next(r for r in app.radio if r.label=='Dense inference scope').set_value('Full volume').run();check('Full-volume native maps and components')
    next(r for r in app.radio if r.label=='Dense inference scope').set_value('Fixed ROI').run()
    choose('Probability class',7);check('Eight-class probability selection')
    choose('CT исследовательское преобразование','noise12');check('Source CT noise and recomputed voxel features')
    choose('CT исследовательское преобразование','translateY+Z');check('Mapped translated source voxel and unchanged GT inference gate')
    choose('CT исследовательское преобразование','clean')
    choose('Multiclass representation','Inputs before training');check('Inputs clean/noisy common PCA basis')
    choose('Multiclass representation','Embedding');choose('Multiclass projection','UMAP');check('Train clean fit UMAP, noisy transform')
    choose('Multiclass projection','PCA');choose('Multiclass seed',SEEDS[1]);check('Independent seed checkpoint')
    choose('Multiclass model',('CE+SupCon','combined'));check('Augmented multiclass SupCon checkpoint')
    choose('Multiclass model',('CE+Spatial','clean'));check('Experimental multiclass Spatial checkpoint')
    choose('Multiclass feature set','F core+Z');check('Limited Z-context feature schema')
    from ct_lab.core import discover
    cases=discover(REPO.parent/'laboratories/data/chd68/pairs')
    choose('Исследование',next(i for i,r in enumerate(cases) if r['id']=='ct_1027'));check('Natural validation case and fixed ROI')
    import ct_lab.core as source
    loader=source.load_pair;reads=[]
    def forbidden(row):reads.append(row);raise AssertionError('Sealed source read')
    source.load_pair=forbidden
    try:
        split=json.loads((REPO/'configs/splits/cardiac_chd68_v1.json').read_text())['partitions']
        choose('Исследование',next(i for i,r in enumerate(cases) if r['id']==split['test'][0]));check('Sealed test gated BEFORE source IO');assert not reads
    finally:source.load_pair=loader
    choose('Исследование',next(i for i,r in enumerate(cases) if r['id']=='ct_1001'))
    app.sidebar.radio[0].set_value('Learnable Feature Explorer').run();choose('Feature learning task','Multiclass stage04');check('Existing Learnable Explorer multiclass extension')
    (OUT/'ui_verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')


if __name__=='__main__':run()
