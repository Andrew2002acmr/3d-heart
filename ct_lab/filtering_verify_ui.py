"""Real-source integration checks for experiment 01 controls and exports."""
import json
from pathlib import Path
import ct_lab.launch
from streamlit.testing.v1 import AppTest


def main():
    repo=Path(__file__).resolve().parents[1]
    app=AppTest.from_file(str(repo/'ct_lab/app.py'),default_timeout=180).run()
    results=[]
    def check(name):
        errors=[e.message for e in app.exception];results.append(dict(feature=name,exceptions=errors));print(name,errors,flush=True)
        assert not errors
    app.sidebar.radio[0].set_value('Filtering & Feature Analysis').run();check('Baseline LV/MYO; PCA and scatter')
    next(s for s in app.selectbox if s.label=='Метод обработки').set_value('Gaussian').run();check('Gaussian linked images and fixed metrics')
    assert any(s.label=='Матрица ошибок k-NN' for s in app.selectbox);check('Automatic development k-NN and confusion matrix')
    next(r for r in app.radio if r.label=='Сравниваемые классы').set_value('RV / PA').run();check('Difficult RV/PA pair')
    next(s for s in app.selectbox if s.label=='Метод обработки').set_value('Bilateral').run();check('Source-scale bilateral')
    next(s for s in app.selectbox if s.label=='Метод обработки').set_value('CLAHE').run();check('CLAHE mapping')
    next(b for b in app.button if b.label=='Сохранить эксперимент фильтрации и интерактивный отчёт').click().run();check('Export real configuration and voxel identities')
    import ct_lab.core
    original_loader=ct_lab.core.load_pair;reads=[]
    def forbidden(row):
        reads.append(row['image']);raise AssertionError('Validation/test CT must not load in feature fitting UI')
    ct_lab.core.load_pair=forbidden
    try:
        from ct_lab.core import discover
        rows=discover(repo.parent/'laboratories/data/chd68/pairs')
        selector=next(s for s in app.sidebar.selectbox if s.label=='Исследование')
        selector.set_value(next(i for i,r in enumerate(rows) if r['id']=='ct_1027')).run();check('Validation: fixed results only, no source load')
        selector=next(s for s in app.sidebar.selectbox if s.label=='Исследование')
        selector.set_value(next(i for i,r in enumerate(rows) if r['id']=='ct_1022')).run();check('Sealed test: no source load')
        assert not reads
    finally:ct_lab.core.load_pair=original_loader
    output=repo/'outputs/filtering_feature_01';output.mkdir(parents=True,exist_ok=True)
    (output/'ui_verification.json').write_text(json.dumps(results,indent=2),encoding='utf-8')


if __name__=='__main__':main()
