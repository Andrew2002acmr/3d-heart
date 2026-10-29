"""Real-source UI regression checks; no training or sealed-test source access."""
import json
import ct_lab.learnable_core as core
import ct_lab.launch
from streamlit.testing.v1 import AppTest


def run():
    app=AppTest.from_file(str(core.REPO/'ct_lab/app.py'),default_timeout=180).run();checks=[]
    def check(name):
        errors=[e.message for e in app.exception];print(name,errors,flush=True)
        checks.append({'check':name,'exceptions':errors});assert not errors
    def choose(label,value):next(s for s in app.selectbox if s.label==label).set_value(value).run()
    app.sidebar.radio[0].set_value('Learnable Feature Explorer').run();check('kNN native coordinate viewer')
    choose('Learnable model','CE');check('CE e16 and actual forward activations')
    for layer in ('relu1','linear2','relu2','embedding','probabilities'):
        choose('Embedding inspector layer',layer);check('Forward '+layer)
    for dim in (8,32,16):choose('Embedding size',dim);check('Embedding '+str(dim))
    choose('Learnable anatomy pair','RV_PA');choose('Learnable input features','G core+all');check('RV PA Gaussian bank')
    for method in ('CE+SupCon','CE+SupCon+Spatial'):
        choose('Learnable model',method);check(method)
    choose('Learnable projection','UMAP');check('UMAP train-fit transform')
    choose('Selected voxel synthetic perturbation','Drop 3D LoG');check('Local ablation and stress test')
    from ct_lab.core import discover
    cases=discover(core.REPO.parent/'laboratories/data/chd68/pairs')
    choose('Исследование',next(i for i,r in enumerate(cases) if r['id']=='ct_1027'));check('Validation coordinates and stored metrics')
    import ct_lab.core as source
    loader=source.load_pair;reads=[]
    def forbidden(row):reads.append(row);raise AssertionError('Sealed source access')
    source.load_pair=forbidden
    try:
        split=json.loads((core.REPO/'configs/splits/cardiac_chd68_v1.json').read_text())
        choose('Исследование',next(i for i,r in enumerate(cases) if r['id']==split['partitions']['test'][0]))
        check('Sealed test gated before load');assert not reads
    finally:source.load_pair=loader
    (core.OUT/'ui_verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')


if __name__=='__main__':run()
